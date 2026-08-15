"""D30: stop theorising -- print the largest entries of E with full annotation.

Also: does the UNPRECONDITIONED iteration count actually grow with nside?  It is
81, 102, 151, 115, 92 at nside 8 to 128, i.e. it peaks at nside 32 and then FALLS.
If it stays flat, the preconditioner is a 10x optimisation rather than a necessity, and
the memory budget should be set accordingly.
"""

import sys, time
import numpy as np

import precond_common  # noqa: F401
from precond_common import make_spin_dfs, capture, cg_count
from precond_twolevel import TwoLevel
from src.double_fourier_sphere import dfs_fold_plan
from src.nuFFT import _upsampled_latitudes


def annotate(nside, spin=2, tol=1e-2, show=20):
    T = TwoLevel(nside, spin, tol, sparse=True)
    E = T.E.tocoo()
    d = np.sqrt(np.abs(T.E.diagonal()))
    n_rings = 4 * nside - 1
    x = _upsampled_latitudes(nside)
    r = T.rows.astype(int)
    ring = np.where(r <= n_rings, r, 2 * n_rings + 2 - r)
    ring = np.where(r == n_rings + 1, 0, ring)
    depth = np.minimum(ring, n_rings + 1 - ring)
    c0 = T.cols[:, 0].astype(int) - 2 * nside
    c1 = T.cols[:, 1].astype(int) - 2 * nside

    off = E.row != E.col
    val = np.abs(E.data[off]) / (d[E.row[off]] * d[E.col[off]])
    ii, jj = E.row[off], E.col[off]
    top = np.argsort(-val)[: show * 2]
    seen = set()
    print(f"nside {nside}: largest normalised |E_ij|")
    shown = 0
    for t in top:
        i, j = int(ii[t]), int(jj[t])
        if (j, i) in seen:
            continue
        seen.add((i, j))
        share = len({c0[i], c1[i]} & {c0[j], c1[j]}) > 0
        dx = abs(x[r[i]] - x[r[j]])
        dx = min(dx, 2 * np.pi - dx)
        print(
            f"   {val[t]:.4f}  i(depth {depth[i]:4d} m {c0[i]:+5d}->{c1[i]:+5d})  "
            f"j(depth {depth[j]:4d} m {c0[j]:+5d}->{c1[j]:+5d})  "
            f"share {str(share):5s}  |dx| {dx:.4f}  ddepth {abs(depth[i] - depth[j]):4d}"
            f"  dm {abs(c0[i] - c0[j]):4d}"
        )
        shown += 1
        if shown >= show:
            break

    # overall: what predicts a large entry?
    big = val > 0.05
    dxa = np.abs(x[r[ii]] - x[r[jj]])
    dxa = np.minimum(dxa, 2 * np.pi - dxa)
    sh = np.array([len({c0[a], c1[a]} & {c0[b], c1[b]}) > 0 for a, b in zip(ii, jj)])
    print(
        f"   entries with normalised value > 0.05: {big.sum()} of {len(val)} "
        f"({100 * big.mean():.2f}%), i.e. {big.sum() / T.R:.1f} per row"
    )
    print(
        f"      of those: share a column {100 * sh[big].mean():5.1f}%   "
        f"same depth {100 * np.mean(depth[ii[big]] == depth[jj[big]]):5.1f}%   "
        f"|dm| median {int(np.median(np.abs(c0[ii[big]] - c0[jj[big]])))}   "
        f"|dx| median {np.median(dxa[big]):.4f}"
    )
    print(
        f"      all entries:  share a column {100 * sh.mean():5.1f}%   "
        f"same depth {100 * np.mean(depth[ii] == depth[jj]):5.1f}%"
    )


def plain_iters(nside, spin=2, tol=1e-2, rtol=1e-7):
    dfs, _ = make_spin_dfs(nside, spin=spin)
    A, b = capture(dfs, nside, dfs_fold_plan(nside, spin, tol), rtol=rtol)
    t = time.perf_counter()
    x0, it, _ = cg_count(A, b, rtol=rtol, maxiter=20000)
    dt = time.perf_counter() - t
    print(
        f"nside {nside:5d}: plain CG {it:5d} its  {dt:8.3f} s  "
        f"({1e3 * dt / it:6.2f} ms/it)"
    )


if __name__ == "__main__":
    mode = sys.argv[1]
    for ns in [int(a) for a in sys.argv[2:]]:
        (annotate if mode == "look" else plain_iters)(ns)
