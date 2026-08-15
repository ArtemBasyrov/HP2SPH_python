"""D27: a STRUCTURAL FSAI pattern, computable without ever forming E.

d26's oracle pattern is not local in the DFS row index (|drow| median 58 of 256), but
that is misleading: the DFS layout is [Npole, rings, Spole, mirrored rings], so a north
cap ring, the matching south cap ring, and both mirrors sit far apart in row index while
being the SAME physical distance from a pole.  Rank generators by

    depth  = distance of the ring from the nearest pole
    mode   = the relaxed longitude mode c

and the pattern should become local again.  Measured here as recall against the oracle,
then used for real.
"""

import sys, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

import precond_common  # noqa: F401
from precond_common import make_spin_dfs, capture, cg_count
from precond_twolevel import TwoLevel
from d26_pattern import topk_lower, fsai
from src.double_fourier_sphere import dfs_fold_plan


def depth_of(T):
    """Distance of each generator's ring from the nearest pole."""
    n_rings = 4 * T.nside - 1
    r = T.rows.astype(int)
    ring = np.where(r <= n_rings, r, 2 * n_rings + 2 - r)
    ring = np.where(r == n_rings + 1, 0, ring)
    return np.minimum(ring, n_rings + 1 - ring)


def structural(T, dd, dm, share=True):
    """{j < i} within dd in depth and dm in mode, plus anything sharing a column."""
    R = T.R
    d = depth_of(T)
    c0 = T.cols[:, 0].astype(int)
    c1 = T.cols[:, 1].astype(int)

    bucket = {}
    for j in range(R):
        bucket.setdefault(int(d[j]), []).append(j)
    bucket = {k: np.asarray(v) for k, v in bucket.items()}

    colmap = {}
    for j in range(R):
        colmap.setdefault(int(c0[j]), []).append(j)
        colmap.setdefault(int(c1[j]), []).append(j)
    colmap = {k: np.asarray(v) for k, v in colmap.items()}

    rows, cols = [], []
    for i in range(R):
        cand = []
        for dep in range(int(d[i]) - dd, int(d[i]) + dd + 1):
            grp = bucket.get(dep)
            if grp is None:
                continue
            cand.append(grp[np.abs(c0[grp] - c0[i]) <= dm])
        if share:
            cand.append(colmap.get(int(c0[i]), np.empty(0, int)))
            cand.append(colmap.get(int(c1[i]), np.empty(0, int)))
        sel = np.unique(np.concatenate(cand + [np.array([i])]))
        sel = sel[sel <= i]
        rows.append(np.full(len(sel), i))
        cols.append(sel)
    return np.concatenate(rows), np.concatenate(cols)


def recall(oracle, cand, R):
    o = set(zip(oracle[0].tolist(), oracle[1].tolist()))
    c = set(zip(cand[0].tolist(), cand[1].tolist()))
    return len(o & c) / max(len(o), 1)


def main(nside, spin=2, tol=1e-2, rtol=1e-7):
    dfs, _ = make_spin_dfs(nside, spin=spin)
    A, b = capture(dfs, nside, dfs_fold_plan(nside, spin, tol), rtol=rtol)
    t = time.perf_counter()
    x0, it0, _ = cg_count(A, b, rtol=rtol, maxiter=20000)
    t0 = time.perf_counter() - t
    T = TwoLevel(nside, spin, tol, sparse=True)
    Zs, ZsH = T.Zs, T.ZsH
    x1, it1, _ = cg_count(A, b, M=T.operator(), rtol=rtol, maxiter=20000)
    print(
        f"nside {nside}  R {T.R}  plain {it0} its {t0:.3f} s   exact sparse E {it1} its"
    )

    oracle = topk_lower(T.E, 64)
    for dd, dm in ((1, 8), (2, 16), (3, 24), (4, 32)):
        cand = structural(T, dd, dm)
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
        print(
            f"   structural dd {dd} dm {dm:2d}  nnz(G)/row {G.nnz / T.R:5.1f}  "
            f"recall of oracle {100 * rc:4.1f}%   {it:5d} its  {dt:6.3f} s  "
            f"wall {t0 / dt:5.2f}x  build {tf:5.1f} s"
        )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [32]:
        main(ns)
