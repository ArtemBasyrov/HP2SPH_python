"""D31: an O(R) FSAI on a pattern that is local in (latitude, mode).

d30's annotation: only 11.8 entries per row exceed 0.05 of the diagonal scale, their
median |dx| is 0.054 (adjacent rings), and the very largest are SAME-ROW pairs whose
mode difference is 4 or 8 -- multiples of the innermost ring size, i.e. alias partners of
the coarsest rings.  So the pattern is local in x and in m, which is exactly what can be
written down at any nside.

Also regularise each FSAI sub-block: E[P_i, P_i] can be badly conditioned for individual
rows, which is what made d25/d26 non-monotone in the pattern size.
"""

import sys, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

import precond_common  # noqa: F401
from precond_common import make_spin_dfs, capture, cg_count
from precond_twolevel import TwoLevel
from d26_pattern import topk_lower
from src.double_fourier_sphere import dfs_fold_plan
from src.nuFFT import _upsampled_latitudes


def pattern_local(T, dxmax, dm, mirror=True):
    """{j < i : circular |x_i -+ x_j| <= dxmax and |m_i - m_j| <= dm}."""
    R = T.R
    x = _upsampled_latitudes(T.nside)
    xr = x[T.rows]
    m = T.cols[:, 0].astype(int)
    order = np.argsort(m, kind="stable")
    ms = m[order]
    rows, cols = [], []
    for i in range(R):
        lo = np.searchsorted(ms, m[i] - dm, "left")
        hi = np.searchsorted(ms, m[i] + dm, "right")
        cand = order[lo:hi]
        d = np.abs(xr[cand] - xr[i])
        d = np.minimum(d, 2 * np.pi - d)
        ok = d <= dxmax
        if mirror:
            d2 = np.abs(xr[cand] + xr[i])
            d2 = np.minimum(d2, 2 * np.pi - d2)
            ok = ok | (d2 <= dxmax)
        sel = cand[ok]
        sel = np.sort(sel[sel < i])
        sel = np.concatenate([sel, [i]])
        rows.append(np.full(len(sel), i))
        cols.append(sel)
    return np.concatenate(rows), np.concatenate(cols)


def fsai_reg(E, rows, cols, reg=1e-8):
    """FSAI with each sub-block ridged, so a badly conditioned row cannot blow up."""
    R = E.shape[0]
    Ecsr = E.tocsr()
    order = np.argsort(rows, kind="stable")
    rows, cols = rows[order], cols[order]
    bnd = np.searchsorted(rows, np.arange(R + 1))
    gdata, gi, gj = [], [], []
    for i in range(R):
        P = np.unique(np.concatenate([cols[bnd[i] : bnd[i + 1]], [i]]))
        sub = Ecsr[P][:, P].toarray()
        sub[np.diag_indices_from(sub)] += reg * np.abs(np.diag(sub)).max()
        pos = int(np.searchsorted(P, i))
        rhs = np.zeros(len(P), dtype=complex)
        rhs[pos] = 1.0
        try:
            y = np.linalg.solve(sub, rhs)
        except np.linalg.LinAlgError:
            y = rhs / max(sub[pos, pos].real, 1e-300)
        d = y[pos].real
        if not np.isfinite(d) or d <= 0:
            y = rhs / max(sub[pos, pos].real, 1e-300)
            d = y[pos].real
        y = y / np.sqrt(max(d, 1e-300))
        gdata.append(y)
        gi.append(np.full(len(P), i))
        gj.append(P)
    return sp.csr_matrix(
        (np.concatenate(gdata), (np.concatenate(gi), np.concatenate(gj))), shape=(R, R)
    )


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
        f"nside {nside}  R {T.R}  E nnz/row {T.E.nnz / T.R:.0f}   "
        f"plain {it0} its {t0:.3f} s   exact E {it1} its {t1:.3f} s ({t0 / t1:.2f}x)"
    )

    # the natural latitude scale is the band-limit resolution pi / L
    res = np.pi / (2 * nside)
    for fx, dm in ((2, 8), (4, 16), (8, 16), (8, 32), (16, 32)):
        dxmax = fx * res
        r, c = pattern_local(T, dxmax, dm)
        t = time.perf_counter()
        G = fsai_reg(T.E, r, c)
        tf = time.perf_counter() - t
        GH = G.conj().T.tocsr()
        M = LinearOperator(
            (T.n, T.n), matvec=lambda u: u + Zs @ (GH @ (G @ (ZsH @ u))), dtype=complex
        )
        t = time.perf_counter()
        x, it, _ = cg_count(A, b, M=M, rtol=rtol, maxiter=20000)
        dt = time.perf_counter() - t
        print(
            f"   local dx<{fx:2d}*res dm<{dm:3d}  nnz(G)/row {G.nnz / T.R:6.1f}  "
            f"{it:5d} its  {dt:6.3f} s  wall {t0 / dt:5.2f}x  build {tf:5.1f} s"
        )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [32]:
        main(ns)
