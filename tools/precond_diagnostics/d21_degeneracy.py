"""D21: where does the near-degeneracy of the coarse space come from?

cond(Z^H Z) is 1.1e2 / 1.3e2 / 1.2e3 / 1.7e6 at nside 8 / 16 / 32 / 64 (d13), and E
inherits it.  Find the offending pairs and read off their structure: same group and
adjacent polar rings, or DFS mirror twins, or something else.

dfs_fold_plan lays the rows out as [north pole, rings, south pole, mirrored rings], so a
generator on original ring r has a mirror twin at row index (n_rings + 1) + (n_rings - r).
"""

import sys
import numpy as np

import precond_common  # noqa: F401
from precond_twolevel import TwoLevel
from src.nuFFT import _upsampled_latitudes


def main(nside, spin=2, tol=1e-2, show=8):
    T = TwoLevel(nside, spin, tol, E=np.eye(1))
    S = np.asarray((T.ZsH @ T.Zs).todense())
    S = 0.5 * (S + S.conj().T)
    d = np.sqrt(np.abs(np.diag(S)))
    C = np.abs(S) / np.outer(d, d)
    np.fill_diagonal(C, 0.0)
    lam = np.linalg.eigvalsh(S)
    x = _upsampled_latitudes(nside)
    n_rings = 4 * nside - 1

    def rowkind(r):
        if r == 0:
            return "Npole"
        if r == n_rings + 1:
            return "Spole"
        if r <= n_rings:
            return f"ring{r}"
        return f"mirr{2 * n_rings + 2 - r}"

    print(
        f"nside {nside}  R {T.R}  cond(S) {lam[-1] / lam[0]:.2e}  "
        f"lam_min {lam[0]:.3e}  lam_max {lam[-1]:.3e}"
    )
    print(
        f"  correlations above 0.99: {int(np.sum(C > 0.99) // 2)}   "
        f"above 0.999: {int(np.sum(C > 0.999) // 2)}"
    )
    flat = np.argsort(-C.ravel())[::2][:show]
    print("  most correlated generator pairs:")
    for f in flat:
        i, j = divmod(int(f), T.R)
        print(
            f"    corr {C[i, j]:.6f}   i: row {rowkind(T.rows[i]):8s} "
            f"x {x[T.rows[i]]:+.4f} cols {tuple(int(v) for v in T.cols[i])}"
            f"   j: row {rowkind(T.rows[j]):8s} x {x[T.rows[j]]:+.4f} "
            f"cols {tuple(int(v) for v in T.cols[j])}"
        )

    # is the coarse space closed under the DFS mirror, and are twins correlated?
    twin = {}
    for j in range(T.R):
        r = T.rows[j]
        if 1 <= r <= n_rings:
            twin[j] = (2 * n_rings + 2 - r, tuple(T.cols[j]))
    lookup = {(int(T.rows[k]), tuple(int(v) for v in T.cols[k])): k for k in range(T.R)}
    corr = []
    for j, (rt, cs) in twin.items():
        k = lookup.get((rt, tuple(int(v) for v in cs)))
        if k is not None:
            corr.append(C[j, k])
    if corr:
        corr = np.array(corr)
        print(
            f"  mirror twins present for {len(corr)} of {T.R} generators; "
            f"their correlation: median {np.median(corr):.4f}  max {corr.max():.4f}"
        )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [64]:
        main(ns)
