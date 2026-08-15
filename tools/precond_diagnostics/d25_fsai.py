"""D25: an O(R) coarse solve -- factorised sparse approximate inverse of E.

nnz(E) = 0.13 R^2 with R ~ nside^2, so the exact sparse factorisation is O(n^2) memory:
30 M nonzeros at nside 128, ~6 GB at nside 256.  Unusable where HP2SPH is meant to win.

Thresholding E and factorising it EXACTLY fails -- 793 iterations at nside 64, worse
than no preconditioner -- because dropping entries from a Hermitian matrix does not
preserve positive definiteness.  FSAI
(Kolotilina & Yeremin) does not have that failure mode: it builds a sparse lower
triangular G with M^-1 = G^H G, which is positive definite for ANY pattern, and it only
ever reads the entries of E inside that pattern -- so the full E never has to exist.

This measures, at a resolution where the exact E is still affordable, how good FSAI is at
a fixed number of entries per row, and whether a purely STRUCTURAL pattern (needed at
high nside, where E cannot be formed to rank its entries) does as well as an oracle
pattern picked by magnitude.
"""

import sys, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, splu

import precond_common  # noqa: F401
from precond_common import make_spin_dfs, capture, cg_count
from precond_twolevel import TwoLevel
from src.double_fourier_sphere import dfs_fold_plan
from src.nuFFT import _upsampled_latitudes


def pattern_topk(E, k):
    """Oracle pattern: the k largest |E_ij| in each row (plus the diagonal)."""
    R = E.shape[0]
    Ec = E.tocsr()
    rows, cols = [], []
    for i in range(R):
        s, e = Ec.indptr[i], Ec.indptr[i + 1]
        idx = Ec.indices[s:e]
        val = np.abs(Ec.data[s:e])
        if len(idx) > k:
            sel = idx[np.argpartition(-val, k)[:k]]
        else:
            sel = idx
        sel = np.union1d(sel, [i])
        rows.append(np.full(len(sel), i))
        cols.append(sel)
    return np.concatenate(rows), np.concatenate(cols)


def pattern_structural(T, dr, dm):
    """Structural pattern: generators close in latitude row AND in longitude mode.

    Needs no knowledge of E's values, so it is computable at any nside.
    """
    x = _upsampled_latitudes(T.nside)
    xr = x[T.rows]
    c0 = T.cols[:, 0].astype(int)
    R = T.R
    order = np.lexsort((c0, np.round(xr, 12)))
    rows, cols = [], []
    # bucket by rounded latitude to keep it O(R * neighbours)
    key = {}
    for j in range(R):
        key.setdefault(int(T.rows[j]), []).append(j)
    rowvals = np.array(sorted(key))
    pos = {r: i for i, r in enumerate(rowvals)}
    for j in range(R):
        ri = pos[int(T.rows[j])]
        cand = []
        for rr in rowvals[max(0, ri - dr) : ri + dr + 1]:
            grp = np.asarray(key[int(rr)])
            near = grp[np.abs(c0[grp] - c0[j]) <= dm]
            cand.append(near)
        sel = np.union1d(np.concatenate(cand), [j])
        rows.append(np.full(len(sel), j))
        cols.append(sel)
    return np.concatenate(rows), np.concatenate(cols)


def fsai(E, rows, cols):
    """Lower-triangular FSAI factor G with M^-1 = G^H G.  Reads only E on the pattern."""
    R = E.shape[0]
    Ecsr = E.tocsr()
    keep = cols <= rows
    rows, cols = rows[keep], cols[keep]
    order = np.argsort(rows, kind="stable")
    rows, cols = rows[order], cols[order]
    bnd = np.searchsorted(rows, np.arange(R + 1))
    gdata, gi, gj = [], [], []
    for i in range(R):
        P = cols[bnd[i] : bnd[i + 1]]
        if len(P) == 0 or P[-1] != i:
            P = np.union1d(P, [i])
        sub = Ecsr[P][:, P].toarray()
        rhs = np.zeros(len(P), dtype=complex)
        rhs[np.searchsorted(P, i)] = 1.0
        try:
            y = np.linalg.solve(sub, rhs)
        except np.linalg.LinAlgError:
            y = np.zeros(len(P), dtype=complex)
            y[np.searchsorted(P, i)] = (
                1.0 / sub[np.searchsorted(P, i), np.searchsorted(P, i)].real
            )
        d = y[np.searchsorted(P, i)].real
        y = y / np.sqrt(max(d, 1e-300))
        gdata.append(y)
        gi.append(np.full(len(P), i))
        gj.append(P)
    G = sp.csr_matrix(
        (np.concatenate(gdata), (np.concatenate(gi), np.concatenate(gj))), shape=(R, R)
    )
    return G


def main(nside, spin=2, tol=1e-2, rtol=1e-7):
    dfs, _ = make_spin_dfs(nside, spin=spin)
    A, b = capture(dfs, nside, dfs_fold_plan(nside, spin, tol), rtol=rtol)
    t = time.perf_counter()
    x0, it0, _ = cg_count(A, b, rtol=rtol, maxiter=20000)
    t0 = time.perf_counter() - t
    T = TwoLevel(nside, spin, tol, sparse=True)
    Zs, ZsH = T.Zs, T.ZsH
    print(
        f"nside {nside}  R {T.R}  E nnz {T.E.nnz} ({T.E.nnz / T.R:.0f} per row)  "
        f"plain {it0} its {t0:.3f} s"
    )
    x1, it1, _ = cg_count(A, b, M=T.operator(), rtol=rtol, maxiter=20000)
    print(f"   exact sparse E            {it1:5d} its   (reference)")

    for k in (8, 16, 32, 64):
        r, c = pattern_topk(T.E, k)
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
            f"   FSAI top-{k:3d} per row  nnz(G) {G.nnz:8d} ({G.nnz / T.R:5.1f}/row)  "
            f"{it:5d} its  {dt:6.3f} s  wall {t0 / dt:5.2f}x  build {tf:5.1f} s"
        )

    for dr, dm in ((1, 4), (2, 8), (3, 16)):
        r, c = pattern_structural(T, dr, dm)
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
            f"   FSAI structural dr {dr} dm {dm:2d}  nnz(G) {G.nnz:8d} "
            f"({G.nnz / T.R:5.1f}/row)  {it:5d} its  {dt:6.3f} s  "
            f"wall {t0 / dt:5.2f}x  build {tf:5.1f} s"
        )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [32]:
        main(ns)
