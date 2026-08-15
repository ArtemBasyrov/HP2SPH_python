"""D1: is the near-null space of the folded operator the 'relaxed family' space?

Prediction (analytic): N = A0^H D A0 with D block-diagonal by latitude ROW; on a row
with an alias family {slot b, relaxed c_1..c_q} the block of D is w * u u^H (rank 1),
so D has exactly q zero directions there.  The near-null space of N is therefore the
band-limited projection of null(D), i.e.  Z = A0^H W Psi  with Psi an explicit sparse
basis of null(D).  Test that against the true eigenvectors.
"""

import numpy as np
from precond_common import make_spin_dfs, capture, densify
from src.double_fourier_sphere import dfs_fold_plan
from src.nuFFT import _upsampled_latitudes, compute_voronoi_weights_1d

np.set_printoptions(precision=3, suppress=True)


def null_D_basis(nside, spin=2, tol=1e-2):
    """Sparse basis of null(D) in position space, one vector per relaxed entry.

    Returns list of (row, [(col, coeff), ...]).  Within the family {b, c_1..c_q} on row r
    the block is w*u u^H with u = (1, p_1, ..., p_q)^H conj-transposed appropriately, so
    null(D) is spanned by  e_{c_i} - (p_i / 1) * ... ; concretely the vector
        v = conj(p_i) e_{c_i} - e_b        (up to scale)
    satisfies u^T v = 0 when u_c = p_{r,c}.
    """
    target, phase, keep = dfs_fold_plan(nside, spin, tol)
    M, ncol = target.shape
    out = []
    rows, cols = np.nonzero(~keep)
    for r, c in zip(rows, cols):
        b = target[r, c]
        p = phase[r, c]
        if b == c:
            continue
        out.append((r, [(c, np.conj(p)), (b, -1.0 + 0j)]))
    return out, (target, phase, keep)


def main(nside=8, spin=2, tol=1e-2):
    dfs, _ = make_spin_dfs(nside, spin=spin)
    plan = dfs_fold_plan(nside, spin, tol)
    A, b = capture(dfs, nside, plan)
    n = A.shape[0]
    N = densify(A)
    N = 0.5 * (N + N.conj().T)
    lam, V = np.linalg.eigh(N)
    n_trans = 4 * nside
    N_modes = 4 * nside + 1
    print(f"nside {nside}  n = {n} = n_trans {n_trans} x N_modes {N_modes}")
    print(f"eig range [{lam[0]:.3e}, {lam[-1]:.3e}]  cond {lam[-1] / lam[0]:.2e}")
    med = np.median(lam)
    small = lam < 0.1 * med
    print(f"eigenvalues below 0.1*median: {small.sum()}   (median {med:.3f})")

    # --- the analytic candidate space -------------------------------------------
    nullv, (target, phase, keep) = null_D_basis(nside, spin, tol)
    print(
        f"relaxed entries with a distinct target: {len(nullv)}"
        f"  (total relaxed {int((~keep).sum())})"
    )

    x = _upsampled_latitudes(nside)
    w = np.abs(compute_voronoi_weights_1d(x))
    M_samp = len(x)
    L = (N_modes - 1) // 2
    k = np.arange(-L, L + 1)
    # coefficient-space vector for (row r, column c): A0^H W e_{r,c}
    #   -> column c gets  w_r * exp(-i k x_r)
    E = np.exp(-1j * np.outer(x, k))  # (M, N_modes)

    Z = np.zeros((n, len(nullv)), dtype=complex)
    # unknown layout: cg reshapes as (n_trans, N_modes) -> index = col*N_modes + k
    for j, (r, terms) in enumerate(nullv):
        for c, coeff in terms:
            Z[c * N_modes : (c + 1) * N_modes, j] += coeff * w[r] * E[r]
    # orthonormalise
    Q, R, piv = __import__("scipy.linalg", fromlist=["qr"]).qr(
        Z, mode="economic", pivoting=True
    )
    rank = int(np.sum(np.abs(np.diag(R)) > 1e-10 * abs(R[0, 0])))
    Q = Q[:, :rank]
    print(f"numerical rank of the analytic basis: {rank}")

    # how well does span(Q) capture the true near-null eigenvectors?
    for nd in (5, 10, 25, 50, 100):
        Vs = V[:, :nd]
        cap = np.linalg.norm(Q.conj().T @ Vs, axis=0) ** 2
        print(
            f"  lowest {nd:3d} eigvecs: mean captured fraction {cap.mean():.4f}"
            f"   min {cap.min():.4f}"
        )

    # localisation of the true eigenvectors, in position space and in column space
    print("  lowest eigenvectors, column-space participation ratio and top columns:")
    for i in range(4):
        v = V[:, i].reshape(n_trans, N_modes)
        p = np.sum(np.abs(v) ** 2, axis=1)
        p /= p.sum()
        pr = 1.0 / np.sum(p**2)
        top = np.argsort(-p)[:4]
        print(
            f"    lam={lam[i]:.3e}  PR_cols={pr:5.1f}  top cols(m)="
            f"{[int(t - n_trans // 2) for t in top]}  mass={p[top].sum():.3f}"
        )


if __name__ == "__main__":
    import sys

    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
