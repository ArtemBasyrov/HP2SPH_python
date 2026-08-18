"""Feichtinger, Gröchenig & Strohmer (1995), Prop. 2, applied to the HP2SPH grid.

cond(T_w) <= ((1 + 2*delta*M) / (1 - 2*delta*M))^2  for maximal gap delta < 1/(2M).

Their weights w_i = (t_{i+1} - t_{i-1})/2 are exactly compute_voronoi_weights_1d,
so the bound applies to our latitude system directly. It is an a-priori bound: it
needs only the ring colatitudes, so it holds at resolutions never solved.
"""

import numpy as np
from src.nuFFT import _upsampled_latitudes

print("nside   r      M      delta*M   cond bound")
for nside in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
    x = _upsampled_latitudes(nside)
    xs = np.sort(np.mod(x, 2 * np.pi))
    gaps = np.diff(np.r_[xs, xs[0] + 2 * np.pi])
    delta = gaps.max() / (2 * np.pi)  # period-normalised maximal gap
    M = 2 * nside  # latitude bandlimit |k| <= 2*nside
    dM = delta * M
    bound = ((1 + 2 * dM) / (1 - 2 * dM)) ** 2 if 2 * dM < 1 else np.inf
    print(f"{nside:5d} {len(x):6d} {M:6d}   {dM:.4f}   {bound:9.3f}")
