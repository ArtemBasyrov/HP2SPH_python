"""D4: form the coarse (Galerkin) matrix E = Z^H N Z in CLOSED FORM, no NUFFTs.

With  Z = A0^H W Psi  and  N = A0^H F^H Wt F A0 / norm,

    E = Psi^H W G (F^H Wt F) G W Psi / norm,        G = A0 A0^H,

and G is the Dirichlet kernel evaluated at the latitude sample differences:

    G[r, r'] = sum_{k=-L..L} exp(i k (x_r - x_r')) = sin((L+1/2)u) / sin(u/2).

So the whole setup is O(M R^2) dense work on the LATITUDE grid -- no transform of the
map, no R matrix-vector products with N.  Verified here against the dense operator.
"""

import numpy as np
from precond_common import make_spin_dfs, capture, densify
from src.double_fourier_sphere import dfs_fold_plan
from src.nuFFT import _upsampled_latitudes, compute_voronoi_weights_1d


def dirichlet(x, L):
    u = x[:, None] - x[None, :]
    s = np.sin(0.5 * u)
    out = np.where(
        np.abs(s) < 1e-13,
        2.0 * L + 1.0,
        np.sin((L + 0.5) * u) / np.where(np.abs(s) < 1e-13, 1.0, s),
    )
    return out


def coarse_vectors(nside, spin, tol):
    """(row, [(col, coeff), ...]) for every generator of null(D)."""
    target, phase, keep = dfs_fold_plan(nside, spin, tol)
    rows, cols = np.nonzero(~keep)
    out = []
    for r, c in zip(rows, cols):
        b, p = int(target[r, c]), phase[r, c]
        out.append(
            (
                int(r),
                [(int(c), np.conj(p)), (b, -1.0 + 0j)]
                if b != c
                else [(int(c), 1.0 + 0j)],
            )
        )
    return out, (target, phase, keep)


def closed_form_E(nside, spin, tol, norm):
    vecs, (target, phase, keep) = coarse_vectors(nside, spin, tol)
    x = _upsampled_latitudes(nside)
    w = np.abs(compute_voronoi_weights_1d(x))
    M = len(x)
    n_lon = target.shape[1]
    L = 2 * nside
    G = dirichlet(x, L)
    R = len(vecs)

    # Y_j = G W Psi_j, stored sparsely: only the (<=2) longitude columns it touches.
    # Y_j[:, c] = coeff * w[r] * G[:, r]
    Yrow = np.array([v[0] for v in vecs])
    GW = G * w[None, :]  # (M, M): column r scaled by w_r
    base = GW[:, Yrow]  # (M, R)

    # position-space operator D = F^H Wt F, block diagonal by ROW, coupling columns
    # inside an alias family.  D[r][c, c'] = w_r * keep[r, t(r,c)] * conj(p_c) p_c'
    #                                        * delta(t(r,c) == t(r,c')).
    # E_ij = sum_r sum_{c,c'} conj(Y_i[r,c]) D[r][c,c'] Y_j[r,c'] / norm
    E = np.zeros((R, R), dtype=complex)
    # group vectors by the columns they touch
    cols_of = [[t[0] for t in v[1]] for v in vecs]
    coef_of = [[t[1] for t in v[1]] for v in vecs]
    for i in range(R):
        for ci, ai in zip(cols_of[i], coef_of[i]):
            ti = target[:, ci]
            ki = keep[np.arange(M), ti]
            pi = phase[:, ci]
            # contribution vector over rows for this (i, ci)
            vi = np.conj(ai) * np.conj(base[:, i]) * np.conj(pi) * w * ki
            for j in range(R):
                acc = 0.0
                for cj, aj in zip(cols_of[j], coef_of[j]):
                    same = target[:, cj] == ti
                    acc = acc + np.sum(vi * same * phase[:, cj] * aj * base[:, j])
                E[i, j] += acc
    return E / norm, vecs


def main(nside=8, spin=2, tol=1e-2):
    n_trans, N_modes = 4 * nside, 4 * nside + 1
    dfs, _ = make_spin_dfs(nside, spin=spin)
    A, b = capture(dfs, nside, dfs_fold_plan(nside, spin, tol))
    N = densify(A)
    N = 0.5 * (N + N.conj().T)

    # dense reference
    from d2_candidates import analytic_Z

    Z = analytic_Z(nside, spin, tol, n_trans, N_modes)
    E_ref = Z.conj().T @ N @ Z

    # the scalar `norm` cg_nufft_forward divides by
    target, phase, keep = dfs_fold_plan(nside, spin, tol)
    x = _upsampled_latitudes(nside)
    w = compute_voronoi_weights_1d(x)
    norm = (w[None, :] * keep.T).sum(axis=1).mean()

    E_cf, vecs = closed_form_E(nside, spin, tol, norm)
    rel = np.linalg.norm(E_cf - E_ref) / np.linalg.norm(E_ref)
    print(
        f"nside {nside}: R = {len(vecs)}, ||E_closedform - E_dense|| / ||E|| = {rel:.3e}"
    )


if __name__ == "__main__":
    import sys

    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
