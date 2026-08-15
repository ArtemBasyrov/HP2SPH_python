"""D26: what does a good FSAI pattern for E look like, structurally?

FSAI needs a LOWER-TRIANGULAR pattern, so ranking entries over the whole row and then
discarding the upper half (as d25 did) gives an erratic effective pattern size.  Rank
inside the lower triangle instead.

Then read off the geometry of that oracle pattern -- distance in latitude row, distance
in longitude mode, whether the two generators share a column -- so the same pattern can
be written down structurally at an nside where E can never be formed.
"""

import sys, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

import precond_common  # noqa: F401
from precond_common import make_spin_dfs, capture, cg_count
from precond_twolevel import TwoLevel
from src.double_fourier_sphere import dfs_fold_plan


def topk_lower(E, k):
    """For each row i, the k largest |E[i, j]| with j < i, plus the diagonal."""
    R = E.shape[0]
    Ec = E.tocsr()
    rows, cols = [], []
    for i in range(R):
        s, e = Ec.indptr[i], Ec.indptr[i + 1]
        idx = Ec.indices[s:e]
        low = idx < i
        idx, val = idx[low], np.abs(Ec.data[s:e][low])
        if len(idx) > k:
            idx = idx[np.argpartition(-val, k)[:k]]
        sel = np.concatenate([idx, [i]])
        rows.append(np.full(len(sel), i))
        cols.append(sel)
    return np.concatenate(rows), np.concatenate(cols)


def fsai(E, rows, cols):
    """G lower triangular with M^-1 = G^H G; reads E only on the pattern."""
    R = E.shape[0]
    Ecsr = E.tocsr()
    order = np.argsort(rows, kind="stable")
    rows, cols = rows[order], cols[order]
    bnd = np.searchsorted(rows, np.arange(R + 1))
    gdata, gi, gj = [], [], []
    for i in range(R):
        P = np.unique(np.concatenate([cols[bnd[i] : bnd[i + 1]], [i]]))
        sub = Ecsr[P][:, P].toarray()
        pos = int(np.searchsorted(P, i))
        rhs = np.zeros(len(P), dtype=complex)
        rhs[pos] = 1.0
        try:
            y = np.linalg.solve(sub, rhs)
        except np.linalg.LinAlgError:
            y = rhs / max(sub[pos, pos].real, 1e-300)
        d = y[pos].real
        y = y / np.sqrt(max(d, 1e-300))
        gdata.append(y)
        gi.append(np.full(len(P), i))
        gj.append(P)
    return sp.csr_matrix(
        (np.concatenate(gdata), (np.concatenate(gi), np.concatenate(gj))), shape=(R, R)
    )


def describe(T, rows, cols):
    """Geometry of a pattern: latitude-row distance, mode distance, shared columns."""
    r = T.rows.astype(int)
    c0 = T.cols[:, 0].astype(int)
    c1 = T.cols[:, 1].astype(int)
    off = rows != cols
    i, j = rows[off], cols[off]
    dr = np.abs(r[i] - r[j])
    dc = np.abs(c0[i] - c0[j])
    share = (c0[i] == c0[j]) | (c0[i] == c1[j]) | (c1[i] == c0[j]) | (c1[i] == c1[j])
    print(
        f"      pattern geometry over {len(i)} off-diagonal entries: "
        f"share a column {100 * share.mean():5.1f}%   "
        f"|drow| med {int(np.median(dr))} p90 {int(np.percentile(dr, 90))}   "
        f"|dmode| med {int(np.median(dc))} p90 {int(np.percentile(dc, 90))}"
    )


def main(nside, spin=2, tol=1e-2, rtol=1e-7):
    dfs, _ = make_spin_dfs(nside, spin=spin)
    A, b = capture(dfs, nside, dfs_fold_plan(nside, spin, tol), rtol=rtol)
    t = time.perf_counter()
    x0, it0, _ = cg_count(A, b, rtol=rtol, maxiter=20000)
    t0 = time.perf_counter() - t
    T = TwoLevel(nside, spin, tol, sparse=True)
    Zs, ZsH = T.Zs, T.ZsH
    print(
        f"nside {nside}  R {T.R}  E nnz/row {T.E.nnz / T.R:.0f}  "
        f"plain {it0} its {t0:.3f} s"
    )
    x1, it1, _ = cg_count(A, b, M=T.operator(), rtol=rtol, maxiter=20000)
    print(f"   exact sparse E   {it1:5d} its  (reference)")

    for k in (4, 8, 16, 32, 64, 128):
        r, c = topk_lower(T.E, k)
        t = time.perf_counter()
        G = fsai(T.E, r, c)
        tf = time.perf_counter() - t
        GH = G.conj().T.tocsr()
        M = LinearOperator(
            (T.n, T.n), matvec=lambda u: u + Zs @ (GH @ (G @ (ZsH @ u))), dtype=complex
        )
        t = time.perf_counter()
        x, it, _ = cg_count(A, b, M=M, rtol=rtol, maxiter=20000)
        dt = time.perf_counter() - t
        print(
            f"   FSAI k={k:4d}  nnz(G)/row {G.nnz / T.R:5.1f}  {it:5d} its  "
            f"{dt:6.3f} s  wall {t0 / dt:5.2f}x  build {tf:5.1f} s"
        )
        if k in (16, 64):
            describe(T, r, c)


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [32]:
        main(ns)
