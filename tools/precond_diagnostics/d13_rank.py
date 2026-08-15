"""D13: how much of R is genuine dimension and how much is redundancy in Z?

Two generators of the same alias pair sitting on ADJACENT polar rings pull back to
Dirichlet kernels centred a distance apart that is far below the band-limit resolution
pi/L, so they are nearly parallel.  If that is where R comes from, the coarse space can
be orthonormalised and truncated, and the R^2 / R^3 walls move.

Reports, per nside:
  * P, the number of (c, b) alias-pair groups, and the group size distribution
  * the spectrum of S = Z^H Z and the effective rank at several truncation levels
  * the same for the per-group Gram matrices, which is the local view
"""

import sys
import numpy as np

import precond_common  # noqa: F401
from precond_twolevel import TwoLevel, coarse_generators
from src.nuFFT import _upsampled_latitudes


def main(nside, spin=2, tol=1e-2):
    T = TwoLevel(nside, spin, tol, E=np.eye(1))  # E unused here; skip the setup
    rows, cols, coeffs = T.rows, T.cols, T.coeffs
    R = T.R
    x = _upsampled_latitudes(nside)
    L = 2 * nside

    key = {}
    for j in range(R):
        key.setdefault((int(cols[j, 0]), int(cols[j, 1])), []).append(j)
    sizes = np.array([len(v) for v in key.values()])
    P = len(key)

    S = np.asarray((T.ZsH @ T.Zs).todense())
    S = 0.5 * (S + S.conj().T)
    s = np.linalg.eigvalsh(S)[::-1]
    ranks = {t: int(np.sum(s > t * s[0])) for t in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10)}

    # spread of the rows inside a group, in units of the band-limit resolution pi/L
    spread = []
    for v in key.values():
        if len(v) > 1:
            xs = np.sort(x[rows[v]])
            spread.append((xs[-1] - xs[0]) / (np.pi / L))
    spread = np.array(spread) if spread else np.array([0.0])

    print(
        f"nside {nside:4d}  n {T.n:7d}  R {R:6d}   P {P:5d} groups  "
        f"R/P {R / P:4.2f}  group size max {sizes.max()} mean {sizes.mean():.2f}"
    )
    print(
        f"    row spread inside a group, in units of pi/L: "
        f"median {np.median(spread):.3f}  max {spread.max():.3f}"
    )
    print(
        f"    cond(S) = {s[0] / s[-1]:.2e}   effective rank of Z: "
        + "  ".join(f"{t:.0e}->{r}" for t, r in ranks.items())
    )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [8, 16, 32, 64]:
        main(ns)
