"""D14: what does E look like, and is it compressible?

Three questions decide which of sparsity / hierarchical / approximate-inverse can work.

  1. cond(E), and how much of it is diagonal scaling (i.e. does Jacobi-scaling E help).
  2. Does |E_ij| decay with the latitude separation of the two generators?  That is what
     an H-matrix needs, and the ordering that would expose it.
  3. What is the numerical rank of the off-diagonal blocks under that ordering?  HODLR is
     only worth building if those ranks are small and grow slowly.
"""

import sys
import numpy as np

import precond_common  # noqa: F401
from precond_twolevel import TwoLevel
from src.nuFFT import _upsampled_latitudes


def offdiag_ranks(E, order, tol):
    """HODLR-style: numerical ranks of the two off-diagonal blocks at each level."""
    Ep = E[np.ix_(order, order)]
    out = []
    lo, hi = 0, len(order)
    level = 0
    segs = [(0, len(order))]
    while segs and level < 5:
        ranks = []
        nxt = []
        for a, b in segs:
            if b - a < 8:
                continue
            m = (a + b) // 2
            B = Ep[a:m, m:b]
            s = np.linalg.svd(B, compute_uv=False)
            r = int(np.sum(s > tol * s[0])) if s.size and s[0] > 0 else 0
            ranks.append((r, min(B.shape)))
            nxt += [(a, m), (m, b)]
        if not ranks:
            break
        out.append((level, ranks))
        segs = nxt
        level += 1
    return out


def main(nside, spin=2, tol=1e-2):
    T = TwoLevel(nside, spin, tol)
    E, R = T.E, T.R
    x = _upsampled_latitudes(nside)

    lam = np.linalg.eigvalsh(E)
    d = np.sqrt(np.abs(np.diag(E)))
    Es = E / np.outer(d, d)
    lam_s = np.linalg.eigvalsh(Es)
    print(f"nside {nside}  R {R}")
    print(
        f"  cond(E) = {lam[-1] / lam[0]:.2e}   after Jacobi scaling = "
        f"{lam_s[-1] / lam_s[0]:.2e}   (lam_min {lam[0]:.2e}, lam_max {lam[-1]:.2e})"
    )

    # 2. decay with latitude separation, using the scaled magnitudes
    xr = x[T.rows]
    dx = np.abs(xr[:, None] - xr[None, :])
    dx = np.minimum(dx, 2 * np.pi - dx)
    a = np.abs(Es)
    np.fill_diagonal(a, 0.0)
    bins = np.linspace(0, np.pi, 9)
    idx = np.digitize(dx.ravel(), bins) - 1
    print("  |E_ij| (Jacobi scaled) vs latitude separation:")
    for k in range(8):
        sel = idx == k
        if sel.sum():
            v = a.ravel()[sel]
            print(
                f"    dx in [{bins[k]:.2f},{bins[k + 1]:.2f})  n {sel.sum():8d}  "
                f"mean {v.mean():.2e}  p99 {np.percentile(v, 99):.2e}  max {v.max():.2e}"
            )

    # sparsity if we simply threshold
    for th in (1e-2, 1e-3, 1e-4):
        print(f"  scaled |E_ij| > {th:.0e}: density {100 * np.mean(a > th):5.2f}%")

    # 3. off-diagonal ranks, generators ordered by latitude
    order = np.argsort(xr, kind="stable")
    for rtol in (1e-3, 1e-6):
        print(f"  HODLR off-diagonal ranks, latitude ordering, tol {rtol:.0e}:")
        for level, ranks in offdiag_ranks(E, order, rtol):
            rs = [r for r, _ in ranks]
            ms = [m for _, m in ranks]
            print(
                f"    level {level}: block size {ms[0]:5d}  ranks "
                f"min {min(rs):4d} med {int(np.median(rs)):4d} max {max(rs):4d}"
            )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [32]:
        main(ns)
