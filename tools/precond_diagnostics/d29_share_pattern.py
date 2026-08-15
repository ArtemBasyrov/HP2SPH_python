"""D29: the structural pattern that the reproducing property predicts.

E_ij = w_i w_j sum_rho w_rho G(x_rho - x_i) G(x_rho - x_j) kappa_ij(rho).

If generators i and j SHARE a longitude column then kappa is nonzero at EVERY ring, and
the Dirichlet kernel is the reproducing kernel of the band-limited space, so the sum
telescopes to ~2 pi G(x_i - x_j): a single large entry decaying like 1 / |x_i - x_j|.
If they only collide on a few isolated polar rings, both G factors are off-peak there
and the entry is small.

So the pattern should be "shares a column, nearest in latitude", with no reference to E's
values at all.  Each column is touched by about 0.44 * nside generators, so capping at
the k nearest keeps it O(R k).
"""

import sys, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

import precond_common  # noqa: F401
from precond_common import make_spin_dfs, capture, cg_count
from precond_twolevel import TwoLevel
from d26_pattern import topk_lower, fsai
from d27_structural import recall
from src.double_fourier_sphere import dfs_fold_plan
from src.nuFFT import _upsampled_latitudes


def pattern_share(T, k, add_group=True):
    """For each i: the k generators j < i sharing a column, nearest in latitude."""
    R = T.R
    x = _upsampled_latitudes(T.nside)
    xr = x[T.rows]
    c0 = T.cols[:, 0].astype(int)
    c1 = T.cols[:, 1].astype(int)

    colmap = {}
    for j in range(R):
        colmap.setdefault(int(c0[j]), []).append(j)
        if c1[j] != c0[j]:
            colmap.setdefault(int(c1[j]), []).append(j)
    colmap = {u: np.asarray(v) for u, v in colmap.items()}

    grp = {}
    for j in range(R):
        grp.setdefault((int(c0[j]), int(c1[j])), []).append(j)
    grp = {u: np.asarray(v) for u, v in grp.items()}

    rows, cols = [], []
    for i in range(R):
        cand = [
            colmap.get(int(c0[i]), np.empty(0, int)),
            colmap.get(int(c1[i]), np.empty(0, int)),
        ]
        if add_group:
            cand.append(grp[(int(c0[i]), int(c1[i]))])
        sel = np.unique(np.concatenate(cand))
        sel = sel[sel < i]
        if len(sel) > k:
            d = np.abs(xr[sel] - xr[i])
            d = np.minimum(d, 2 * np.pi - d)
            sel = sel[np.argpartition(d, k)[:k]]
        sel = np.concatenate([np.sort(sel), [i]])
        rows.append(np.full(len(sel), i))
        cols.append(sel)
    return np.concatenate(rows), np.concatenate(cols)


def main(nside, spin=2, tol=1e-2, rtol=1e-7):
    dfs, _ = make_spin_dfs(nside, spin=spin)
    A, b = capture(dfs, nside, dfs_fold_plan(nside, spin, tol), rtol=rtol)
    t = time.perf_counter()
    x0, it0, _ = cg_count(A, b, rtol=rtol, maxiter=20000)
    t0 = time.perf_counter() - t
    T = TwoLevel(nside, spin, tol, sparse=True)
    Zs, ZsH = T.Zs, T.ZsH
    t = time.perf_counter()
    x1, it1, _ = cg_count(A, b, M=T.operator(), rtol=rtol, maxiter=20000)
    t1 = time.perf_counter() - t
    print(
        f"nside {nside}  R {T.R}  plain {it0} its {t0:.3f} s   "
        f"exact sparse E {it1} its {t1:.3f} s ({t0 / t1:.2f}x, "
        f"{T.E.nnz / T.R:.0f} nnz/row)"
    )

    oracle = topk_lower(T.E, 64)
    for k in (8, 16, 32, 64, 10**9):
        cand = pattern_share(T, k)
        rc = recall(oracle, cand, T.R)
        t = time.perf_counter()
        G = fsai(T.E, *cand)
        tf = time.perf_counter() - t
        GH = G.conj().T.tocsr()
        M = LinearOperator(
            (T.n, T.n), matvec=lambda u: u + Zs @ (GH @ (G @ (ZsH @ u))), dtype=complex
        )
        t = time.perf_counter()
        x, it, _ = cg_count(A, b, M=M, rtol=rtol, maxiter=20000)
        dt = time.perf_counter() - t
        lbl = "all" if k > 10**8 else f"{k:3d}"
        print(
            f"   share-a-column, k={lbl:>3s}  nnz(G)/row {G.nnz / T.R:6.1f}  "
            f"recall {100 * rc:4.1f}%   {it:5d} its  {dt:6.3f} s  "
            f"wall {t0 / dt:5.2f}x  build {tf:5.1f} s"
        )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [32]:
        main(ns)
