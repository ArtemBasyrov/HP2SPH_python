"""D34: Krylov recycling -- harvest Ritz vectors from one solve, reuse on the next map.

Every attempt to SHRINK the coarse space failed because the near-null space is
collectively near-null: span(Z) needs about 400 of its 1366 directions at nside 32
before deflation helps at all.  Recycling is a different claim: the
RIGHT-HAND SIDE only excites part of that space, and section 7.1 shows it excites less
of it as nside grows (the plain count falls from 151 at nside 32 to 81 at nside 256).
So a few dozen directions harvested from an actual solve may carry most of the benefit.

Memory is k * n, with no matrix at all -- 168 MB for 40 vectors at nside 128 in
complex64, 336 MB at nside 256.

The harvest runs plain CG while accumulating the Lanczos tridiagonal that CG implies:

    T[j, j]     = 1 / alpha_j + beta_{j-1} / alpha_{j-1}
    T[j, j + 1] = sqrt(beta_j) / alpha_j

then the k smallest Ritz pairs give U and theta, and the deflated preconditioner is
M^-1 = I + U (diag(1/theta) - I) U^H.

CRUCIAL: the harvested vectors are tested on a DIFFERENT map from the one they came
from, otherwise the measurement is circular.
"""

import sys, time, contextlib, io
import numpy as np
import healpy as hp
from scipy.sparse.linalg import LinearOperator

import precond_common  # noqa: F401
from precond_common import capture, cg_count
from src.data_interpolation import transform_healpix_to_grid
from src.double_fourier_sphere import DFS, dfs_fold_plan


def sky_dfs(nside, seed, spin=2, slope=1.5):
    lmax = 2 * nside
    ell = np.arange(lmax + 1)
    np.random.seed(seed)
    aE = hp.synalm((1.0 + ell) ** (-2.0 * slope), lmax=lmax, new=True)
    aB = hp.synalm((1.0 + ell) ** (-2.0 * slope), lmax=lmax, new=True)
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, lmax)
    with contextlib.redirect_stdout(io.StringIO()):
        up, fc = transform_healpix_to_grid(Q + 1j * U)
        _, dfs = DFS(up, fc, spin=spin)
    return dfs


def harvest(A, b, maxiter, store_dtype=np.complex64, reorth=True):  # noqa
    """Plain CG, accumulating the Lanczos basis and tridiagonal it implies.

    Without full reorthogonalisation the Lanczos basis loses orthogonality after a few
    dozen steps and the Ritz vectors are meaningless -- measured: no iteration benefit at
    any k.  Reorthogonalising costs O(m^2 n), which is affordable for a one-off harvest.
    """
    n = A.shape[0]
    x = np.zeros(n, dtype=complex)
    r = b.copy()
    p = r.copy()
    rr = np.vdot(r, r).real
    V = []
    alphas, betas = [], []
    sign = 1.0
    for j in range(maxiter):
        nr = np.sqrt(rr)
        if nr == 0:
            break
        v = (sign * r / nr).astype(store_dtype)
        if reorth and V:
            Vm = np.asarray(V)
            for _ in range(2):  # twice is enough (Kahan)
                v -= (Vm.conj() @ v) @ Vm
            v /= np.linalg.norm(v)
        V.append(v)
        Ap = A.matvec(p)
        alpha = rr / np.vdot(p, Ap).real
        x += alpha * p
        r -= alpha * Ap
        rr_new = np.vdot(r, r).real
        beta = rr_new / rr
        alphas.append(alpha)
        betas.append(beta)
        p = r + beta * p
        rr = rr_new
        sign = -sign
        if np.sqrt(rr) <= 1e-9 * np.linalg.norm(b):
            break
    m = len(V)
    V = np.array(V)
    # The tridiagonal implied by the CG recurrences is NOT the Galerkin matrix once the
    # basis has been reorthogonalised, and is corrupted by orthogonality loss even when
    # it has not been.  Build V^H A V explicitly: m extra matrix-vector products, once.
    AV = np.empty_like(V)
    for j in range(m):
        AV[j] = A.matvec(V[j].astype(complex)).astype(V.dtype)
    T = (V.conj() @ AV.T).astype(np.complex128)
    T = 0.5 * (T + T.conj().T)
    return V, T, m


def ritz(V, T, k, A=None):
    """The k lowest Ritz pairs of the exact Galerkin matrix T = V^H A V."""
    theta, Y = np.linalg.eigh(T)
    Y, theta = Y[:, :k], theta[:k].real
    Q = (Y.T.astype(V.dtype) @ V).T.astype(np.complex128)  # (n, k)
    resid = None
    if A is not None:
        u = Q[:, 0] / np.linalg.norm(Q[:, 0])
        Au = A.matvec(u)
        lam = np.vdot(u, Au).real
        resid = (np.linalg.norm(Au - lam * u) / abs(lam), lam)
    return Q, theta, resid


def deflated_op(Q, A, n):
    """Additive coarse correction M^-1 = I + Q (Q^H A Q)^-1 Q^H.

    Uses the Galerkin matrix rather than diag(1/theta): the harvested vectors span the
    bad subspace but are NOT individual eigenvectors (measured Ritz residual 6.6), so the
    diagonal form is simply wrong.
    """
    k = Q.shape[1]
    AQ = np.empty_like(Q)
    for j in range(k):
        AQ[:, j] = A.matvec(Q[:, j])
    G = Q.conj().T @ AQ
    G = 0.5 * (G + G.conj().T)
    Gi = np.linalg.pinv(G, rcond=1e-12)
    Qc = Q.astype(np.complex64)
    Gc = Gi.astype(np.complex64)

    def apply(v):
        c = Qc.conj().T @ v.astype(np.complex64)
        return v + (Qc @ (Gc @ c)).astype(complex)

    return LinearOperator((n, n), matvec=apply, dtype=complex)


def main(
    nside, spin=2, tol=1e-2, rtol=1e-7, ks=(10, 20, 40, 80, 160), harvest_steps=400
):
    plan = dfs_fold_plan(nside, spin, tol)
    A, b0 = capture(sky_dfs(nside, 0), nside, plan, rtol=rtol)
    _, b1 = capture(sky_dfs(nside, 1), nside, plan, rtol=rtol)
    _, b2 = capture(sky_dfs(nside, 2), nside, plan, rtol=rtol)

    t = time.perf_counter()
    _, it0, _ = cg_count(A, b1, rtol=rtol, maxiter=20000)
    t0 = time.perf_counter() - t
    print(f"nside {nside}  n {A.shape[0]}   plain (map 1) {it0} its {t0:.3f} s")

    t = time.perf_counter()
    V, T, m = harvest(A, b0, maxiter=harvest_steps)
    th = time.perf_counter() - t
    print(
        f"   harvest on map 0: {m} Lanczos steps, {th:.1f} s, "
        f"basis {V.nbytes / 2**20:.0f} MB"
    )

    for k in ks:
        if k > m:
            continue
        Q, theta, resid = ritz(V, T, k, A)
        # a Ritz value can be a poor estimate; clamp to keep the preconditioner SPD
        rtxt = (
            ""
            if resid is None
            else (f"  ritz res {resid[0]:.1e} (rayleigh {resid[1]:.2e})")
        )
        M = deflated_op(Q, A, A.shape[0])
        res = []
        for tag, bb in (("map 1", b1), ("map 2", b2)):
            t = time.perf_counter()
            _, it, _ = cg_count(A, bb, M=M, rtol=rtol, maxiter=20000)
            dt = time.perf_counter() - t
            res.append((tag, it, dt))
        mem = Q.astype(np.complex64).nbytes / 2**20
        s = "   ".join(
            f"{tag} {it:4d} its {dt:6.3f} s ({t0 / dt:4.2f}x)" for tag, it, dt in res
        )
        print(f"   k {k:4d}  theta[0] {theta[0]:.2e}{rtxt}  mem {mem:5.1f} MB")
        print(f"          {s}")


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [128]:
        main(ns)
