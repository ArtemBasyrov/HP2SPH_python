"""D3: does the analytic two-level coarse space scale, and can E^-1 be cheap?

E = Z^H N Z is R x R with R = number of relaxed entries = O(nside^2), so a dense
factorisation walls out.  Test approximating E^-1 by its BLOCK diagonal, blocking the
coarse vectors by the alias pair (c, b) they came from -- vectors from different pairs
should barely overlap.
"""

import sys
import numpy as np
from scipy.sparse.linalg import LinearOperator

from precond_common import make_spin_dfs, capture, densify, cg_count
from src.double_fourier_sphere import dfs_fold_plan
from src.nuFFT import _upsampled_latitudes, compute_voronoi_weights_1d


def analytic_Z(nside, spin, tol, n_trans, N_modes):
    """Returns (Z, groups) with groups[i] = list of column indices of Z in block i."""
    target, phase, keep = dfs_fold_plan(nside, spin, tol)
    x = _upsampled_latitudes(nside)
    w = np.abs(compute_voronoi_weights_1d(x))
    L = (N_modes - 1) // 2
    k = np.arange(-L, L + 1)
    E = np.exp(-1j * np.outer(x, k))
    rows, cols = np.nonzero(~keep)
    Z = np.zeros((n_trans * N_modes, len(rows)), dtype=complex)
    key = {}
    groups = {}
    for j, (r, c) in enumerate(zip(rows, cols)):
        b, p = target[r, c], phase[r, c]
        terms = [(c, np.conj(p)), (b, -1.0 + 0j)] if b != c else [(c, 1.0 + 0j)]
        for cc, coeff in terms:
            Z[cc * N_modes : (cc + 1) * N_modes, j] += coeff * w[r] * E[r]
        g = (min(c, b), max(c, b))
        groups.setdefault(g, []).append(j)
    return Z, list(groups.values())


def coarse_M(N, Z, groups=None):
    """Additive coarse correction  M^-1 = I + Z E^-1 Z^H.

    ``groups`` not None -> approximate E^-1 by its block diagonal over those groups.
    """
    if groups is None:
        E = Z.conj().T @ (N @ Z)
        Ei = np.linalg.pinv(E, rcond=1e-12)
        ZEi = Z @ Ei

        def apply(v):
            return v + ZEi @ (Z.conj().T @ v)
    else:
        NZ = N @ Z
        blocks = []
        for g in groups:
            g = np.asarray(g)
            Eg = Z[:, g].conj().T @ NZ[:, g]
            blocks.append((g, np.linalg.pinv(Eg, rcond=1e-12)))

        def apply(v):
            out = v.copy()
            for g, Egi in blocks:
                out += Z[:, g] @ (Egi @ (Z[:, g].conj().T @ v))
            return out

    return LinearOperator(N.shape, matvec=apply, dtype=complex)


def main(nside, spin=2, tol=1e-2, rtol=1e-7):
    n_trans, N_modes = 4 * nside, 4 * nside + 1
    dfs, _ = make_spin_dfs(nside, spin=spin)
    plan = dfs_fold_plan(nside, spin, tol)
    A, b = capture(dfs, nside, plan, rtol=rtol)
    N = densify(A)
    N = 0.5 * (N + N.conj().T)
    n = N.shape[0]
    Z, groups = analytic_Z(nside, spin, tol, n_trans, N_modes)
    R = Z.shape[1]
    _, it0, _ = cg_count(N, b, rtol=rtol)
    _, it1, _ = cg_count(N, b, M=coarse_M(N, Z), rtol=rtol)
    _, it2, _ = cg_count(N, b, M=coarse_M(N, Z, groups), rtol=rtol)
    sizes = sorted(len(g) for g in groups)
    print(
        f"nside {nside:4d}  n {n:6d}  R {R:5d} ({100 * R / n:4.1f}% of n)  "
        f"blocks {len(groups):4d} (max {sizes[-1]:3d})   "
        f"CG: none {it0:4d} | dense E {it1:4d} ({it0 / it1:4.1f}x) | "
        f"block E {it2:4d} ({it0 / it2:4.1f}x)"
    )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [8, 16]:
        main(ns)
