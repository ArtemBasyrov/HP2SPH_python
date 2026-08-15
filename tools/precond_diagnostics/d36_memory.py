"""D36: the definitive memory curve for the tapered coarse matrix.

Counting the surviving pattern needs no values, so it runs at nside where assembling E
would be hopeless.  The count is an upper bound: it applies the taper's latitude
restriction and the window intersection, but not the collision test, which only removes
entries.
"""

import sys
import numpy as np

import precond_common  # noqa: F401
from precond_twolevel import coarse_generators, _groups, _near_partners, _support_rows
from src.nuFFT import _upsampled_latitudes


def pattern_nnz(nside, spin=2, tol=1e-2, taper=None):
    rows, cols, coeffs, plan = coarse_generators(nside, spin, tol)
    R = len(rows)
    x = _upsampled_latitudes(nside)
    gidx, gcols = _groups(rows, cols, coeffs)
    P = len(gidx)
    if taper is None:
        return R, P, None, None
    delta = taper * np.pi / (2 * nside)
    partners = _near_partners(gidx, rows, x, 2.0 * delta)
    window = _support_rows(gidx, rows, x, delta)
    sizes = np.array([len(g) for g in gidx])
    nnz = 0
    npairs = 0
    for gi in range(P):
        for gj in partners[gi]:
            if np.intersect1d(window[gi], window[gj], assume_unique=True).size == 0:
                continue
            npairs += 1
            nnz += sizes[gi] * sizes[gj] * (1 if gi == gj else 2)
    return R, P, nnz, delta


def main():
    print(" nside      R       P    delta   nnz(E) upper bound   per row      memory")
    for nside in (32, 64, 128, 256, 512):
        for taper in (16, 8):
            delta = taper * np.pi / (2 * nside)
            R, P, nnz, _ = pattern_nnz(nside, taper=taper)
            mem = nnz * (16 + 8) / 2**30  # complex128 value + int32 indices
            print(
                f"{nside:6d} {R:7d} {P:7d}  {delta:7.4f}  {nnz:16d}  "
                f"{nnz / R:9.0f}  {mem:8.2f} GB"
            )


if __name__ == "__main__":
    main()
