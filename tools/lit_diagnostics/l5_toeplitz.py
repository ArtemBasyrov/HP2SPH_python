"""Ruiz-Antolín & Townsend (2018) Sect. 3.3, the algorithm Drake & Wright (2020)
Sect. 3.2.2 specify for the latitude solve: the normal operator is Toeplitz, so a
matrix-vector product is one FFT pair rather than two NUFFTs.

Two questions, both answered here.
  1. Is our WEIGHTED normal operator Toeplitz? (Yes -- W is diagonal in sample
     space, so it preserves the structure.)
  2. Is the Toeplitz matvec faster than the finufft pair we ship? (No: parity to
     slightly worse, once the FFT length is chosen with next_fast_len and both
     sides are threaded.)
"""

import time
import numpy as np
import scipy.fft as sfft
from src.nuFFT import (
    compute_voronoi_weights_1d,
    _mirror_plan,
    _upsampled_latitudes,
    _BatchPlan,
)
from src._threads import default_workers

W = default_workers()


def tmin(f, n=3):
    best = np.inf
    for _ in range(n):
        t0 = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - t0)
    return best


print("--- structure: ||N - Toeplitz|| / ||N||, half domain, spin 0 ---")
for nside in (8, 16, 32):
    N_modes = 4 * nside + 1
    n_trans = 4 * nside
    x = _upsampled_latitudes(nside)
    _, rows, mult, _, _, _ = _mirror_plan(x, 0, n_trans, N_modes)
    xh = x[rows]
    w = compute_voronoi_weights_1d(x)[rows] * mult
    K = (N_modes - 1) // 2
    k = np.arange(-K, K + 1)
    A = np.exp(1j * np.outer(xh, k))
    Nrm = A.conj().T @ (w[:, None] * A)
    d = np.arange(-2 * K, 2 * K + 1)
    t = (w[None, :] * np.exp(1j * np.outer(d, xh))).sum(axis=1)
    T = t[(np.arange(N_modes)[None, :] - np.arange(N_modes)[:, None]) + 2 * K]
    print(f"  nside {nside:4d}: {np.linalg.norm(Nrm - T) / np.linalg.norm(Nrm):.2e}")

print(f"\n--- matvec cost, {W} threads, t_min of 3 ---")
for nside in (256, 512, 1024):
    N_modes = 4 * nside + 1
    n_trans = 4 * nside
    x = _upsampled_latitudes(nside)
    _, rows, mult, _, _, _ = _mirror_plan(x, 0, n_trans, N_modes)
    xh = np.ascontiguousarray(x[rows])
    w = compute_voronoi_weights_1d(x)[rows] * mult
    Mh, K = len(rows), (N_modes - 1) // 2
    d = np.arange(-2 * K, 2 * K + 1)
    t = (w[None, :] * np.exp(1j * np.outer(d, xh))).sum(axis=1)

    pf = _BatchPlan(2, N_modes, n_trans, 1, 1e-6, W)
    pa = _BatchPlan(1, N_modes, n_trans, -1, 1e-6, W)
    pf.setpts(xh)
    pa.setpts(xh)
    c = np.random.randn(n_trans, N_modes) + 0j
    g = np.zeros((n_trans, Mh), complex)
    t_nu = tmin(lambda: (pf.execute(c, g), g.__imul__(w), pa.execute(g, c)))

    L = sfft.next_fast_len(2 * N_modes - 1)
    lam = sfft.fft(
        np.r_[t[2 * K :], np.zeros(L - (4 * K + 1), complex), t[: 2 * K]], workers=W
    )
    buf = np.zeros((n_trans, L), complex)

    def toep():
        buf[:, :N_modes] = c
        buf[:, N_modes:] = 0
        F = sfft.fft(buf, axis=1, workers=W)
        F *= lam
        sfft.ifft(F, axis=1, workers=W)

    t_tp = tmin(toep)
    print(
        f"  nside {nside:4d}: 2x finufft(eps=1e-6) {t_nu * 1e3:7.1f} ms   "
        f"Toeplitz FFT(L={L}) {t_tp * 1e3:7.1f} ms   ratio {t_tp / t_nu:.2f}"
    )
