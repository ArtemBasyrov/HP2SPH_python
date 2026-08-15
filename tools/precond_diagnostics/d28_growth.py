"""D28: how do R and nnz(E) actually grow?  Everything about scaling hangs on this.

R is the number of relaxed entries, which needs only ring_fold_plan -- no solve, no E.
The structural density of E needs only the interaction test, not the values.
"""

import sys, time
import numpy as np

import precond_common  # noqa: F401
from src.data_interpolation import ring_fold_plan
from src.double_fourier_sphere import dfs_fold_plan


def relaxed_count(nside, spin=2, tol=1e-2):
    target, phase, keep = ring_fold_plan(nside, spin, tol)
    # dfs layout doubles the rings and adds two pole rows whose dropped columns are
    # counted separately; count them the way dfs_fold_plan does
    t2, p2, k2 = dfs_fold_plan(nside, spin, tol)
    return int((~keep).sum()), int((~k2).sum()), k2.shape


def per_ring_profile(nside, spin=2, tol=1e-2):
    target, phase, keep = ring_fold_plan(nside, spin, tol)
    rel = (~keep).sum(axis=1)
    n_rings = 4 * nside - 1
    caps = rel[: nside - 1]
    return caps


def main():
    print(" nside      R_rings        R_dfs    R/nside^1.5   R/nside^2   ratio")
    prev = None
    for nside in (8, 16, 32, 64, 128, 256, 512):
        t = time.perf_counter()
        rr, rd, shape = relaxed_count(nside)
        dt = time.perf_counter() - t
        ratio = "" if prev is None else f"{rd / prev:5.2f}"
        print(
            f"{nside:6d}  {rr:11d}  {rd:11d}   {rd / nside**1.5:10.2f}  "
            f"{rd / nside**2:9.4f}   {ratio}   ({dt:.1f} s)"
        )
        prev = rd

    print("\n relaxed modes per cap ring (is it O(r) or O(sqrt r)?)")
    for nside in (64, 256):
        caps = per_ring_profile(nside)
        r = np.arange(1, len(caps) + 1)
        ok = caps > 0
        if ok.sum() > 4:
            p = np.polyfit(np.log(r[ok]), np.log(caps[ok]), 1)[0]
            print(
                f"   nside {nside:4d}: count on ring r fits r^{p:.2f}   "
                f"(ring 10 -> {caps[9]}, ring {len(caps) // 2} -> {caps[len(caps) // 2]}, "
                f"last -> {caps[-1]})"
            )


if __name__ == "__main__":
    main()
