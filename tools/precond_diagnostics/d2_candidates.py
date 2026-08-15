"""D2: candidate preconditioners / regularisations, measured on the dense operator.

0  none                       (baseline)
1  additive coarse correction with the ANALYTIC basis Z = A0^H W Psi
2  the same, compressed to r vectors by Rayleigh quotient
3  exact lowest eigenvectors                 (upper bound on deflation)
4  Tikhonov prior from mode_pole_envelope    (changes the operator, not the solver)
4+ prior AND deflation
"""

import sys
import numpy as np
import scipy.linalg as sla

from precond_common import make_spin_dfs, capture, densify, cg_count
from src.double_fourier_sphere import dfs_fold_plan
from src.data_interpolation import mode_pole_envelope, ring_fold_plan
from src.nuFFT import _upsampled_latitudes, compute_voronoi_weights_1d


def analytic_Z(nside, spin, tol, n_trans, N_modes):
    target, phase, keep = dfs_fold_plan(nside, spin, tol)
    x = _upsampled_latitudes(nside)
    w = np.abs(compute_voronoi_weights_1d(x))
    L = (N_modes - 1) // 2
    k = np.arange(-L, L + 1)
    E = np.exp(-1j * np.outer(x, k))
    rows, cols = np.nonzero(~keep)
    # null(D) has two kinds of generator:
    #   (a) relaxed entry that folds onto a DIFFERENT slot b: within the family block
    #       w u u^H, the direction conj(p) e_c - e_b is annihilated;
    #   (b) an entry whose equation was deleted outright (keep=False with target==c,
    #       which dfs_fold_plan does on the two pole rows): e_c itself is annihilated.
    cand = []
    for r, c in zip(rows, cols):
        b, p = target[r, c], phase[r, c]
        cand.append(
            (r, [(c, np.conj(p)), (b, -1.0 + 0j)] if b != c else [(c, 1.0 + 0j)])
        )
    Z = np.zeros((n_trans * N_modes, len(cand)), dtype=complex)
    for j, (r, terms) in enumerate(cand):
        for c, coeff in terms:
            Z[c * N_modes : (c + 1) * N_modes, j] += coeff * w[r] * E[r]
    return Z


def dfs_envelope(nside, spin, lmax=None):
    """mode_pole_envelope laid out on the DFS rows (as dfs_fold_plan does)."""
    env = mode_pole_envelope(nside, spin, lmax)
    n_lon = env.shape[1]
    pole = np.ones((1, n_lon))
    return np.concatenate((pole, env, pole, np.flip(env, axis=0)))


def prior_term(nside, spin, tol, n_trans, N_modes, cap=1.0):
    """A0^H Lambda A0 with Lambda_{r,c} = w_r * min(cap, (tol/env)^2) on RELAXED entries."""
    target, phase, keep = dfs_fold_plan(nside, spin, tol)
    env = dfs_envelope(nside, spin)
    x = _upsampled_latitudes(nside)
    w = np.abs(compute_voronoi_weights_1d(x))
    lam = np.minimum(cap, (tol / np.maximum(env, 1e-300)) ** 2)
    lam = np.where(~keep, lam, 0.0)  # only where the assertion was dropped
    L = (N_modes - 1) // 2
    k = np.arange(-L, L + 1)
    d = k[:, None] - k[None, :]
    P = np.zeros((n_trans * N_modes, n_trans * N_modes), dtype=complex)
    for c in range(n_trans):
        coef = lam[:, c] * w
        nz = np.nonzero(coef)[0]
        if nz.size == 0:
            continue
        blk = np.einsum(
            "r,rij->ij", coef[nz], np.exp(-1j * x[nz][:, None, None] * d[None, :, :])
        )
        P[c * N_modes : (c + 1) * N_modes, c * N_modes : (c + 1) * N_modes] = blk
    return P, lam


def coarse_M(N, Z):
    E = Z.conj().T @ N @ Z
    Ei = np.linalg.pinv(E, rcond=1e-12)
    ZEi = Z @ Ei

    def apply(v):
        return v + ZEi @ (Z.conj().T @ v)

    from scipy.sparse.linalg import LinearOperator

    return LinearOperator(N.shape, matvec=apply, dtype=complex)


def compress(N, Z, r):
    """Keep the r directions of span(Z) with the smallest Rayleigh quotient."""
    Q, _ = np.linalg.qr(Z)
    S = Q.conj().T @ N @ Q
    lam, W = np.linalg.eigh(0.5 * (S + S.conj().T))
    return Q @ W[:, :r]


def main(nside=8, spin=2, tol=1e-2, rtol=1e-7):
    n_trans, N_modes = 4 * nside, 4 * nside + 1
    dfs, _ = make_spin_dfs(nside, spin=spin)
    plan = dfs_fold_plan(nside, spin, tol)
    A, b = capture(dfs, nside, plan, rtol=rtol)
    N = densify(A)
    N = 0.5 * (N + N.conj().T)
    lam, V = np.linalg.eigh(N)
    print(f"=== nside {nside}, tol {tol}, rtol {rtol}, n = {N.shape[0]}")
    print(f"spectrum [{lam[0]:.3e}, {lam[-1]:.3e}]  cond {lam[-1] / lam[0]:.2e}")

    x0, it0, _ = cg_count(N, b, rtol=rtol)
    print(f"  0  none                          {it0:5d} its")
    ref = x0

    def report(tag, Nm, M, xref=None):
        x, it, info = cg_count(Nm, b, M=M, rtol=rtol)
        err = np.linalg.norm(x - (xref if xref is not None else ref)) / np.linalg.norm(
            ref
        )
        print(f"  {tag:32s} {it:5d} its   |dx|/|x| vs baseline {err:.2e}")
        return x

    Z = analytic_Z(nside, spin, tol, n_trans, N_modes)
    print(f"  analytic basis: {Z.shape[1]} vectors")
    report("1  analytic Z (full)", N, coarse_M(N, Z))
    for r in (10, 25, 50, 100):
        if r <= Z.shape[1]:
            report(
                f"2  analytic Z compressed to {r:3d}", N, coarse_M(N, compress(N, Z, r))
            )
    for nd in (25, 100):
        report(f"3  exact lowest {nd:3d} eigvecs", N, coarse_M(N, V[:, :nd]))

    # ---- Tikhonov prior: a different operator, so compare answers too -------------
    for cap in (1.0,):
        P, lamw = prior_term(nside, spin, tol, n_trans, N_modes, cap=cap)
        # match the scalar normalisation cg_nufft_forward applies
        scale = np.trace(N).real / (n_trans * N_modes)
        Np = N + P
        lp = np.linalg.eigvalsh(Np)
        Nm_rhs = b
        xp, itp, _ = cg_count(Np, b, rtol=rtol)
        print(
            f"  4  Tikhonov prior (cap {cap})     {itp:5d} its   "
            f"spectrum [{lp[0]:.3e}, {lp[-1]:.3e}]  cond {lp[-1] / lp[0]:.2e}"
            f"   |dx|/|x| {np.linalg.norm(xp - ref) / np.linalg.norm(ref):.2e}"
        )
        Zp = analytic_Z(nside, spin, tol, n_trans, N_modes)
        _, itpz, _ = cg_count(
            Np, b, M=coarse_M(Np, compress(Np, Zp, min(50, Zp.shape[1]))), rtol=rtol
        )
        print(f"  5  prior + deflation(50)          {itpz:5d} its")


if __name__ == "__main__":
    main(
        int(sys.argv[1]) if len(sys.argv) > 1 else 8,
        tol=float(sys.argv[2]) if len(sys.argv) > 2 else 1e-2,
    )
