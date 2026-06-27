"""Spin-2 (polarization Q/U <-> E/B) transforms.

This wires the spin stages into an end-to-end ``forward_spin`` / ``backward_spin``
and follows SPIN2_PLAN.md. Two analysis routes exist:

* ``analysis="library"`` (default, the plan's documented fallback): resample the
  HEALPix ``(Q, U)`` onto the FastTransforms equiangular grid and use the library's
  own ``spinsph_analysis`` for the grid->bivariate-Fourier step, then
  ``fourier2spinsph`` + the E/B decode. This is exact in the grid->coefficients
  step; its accuracy floor is the HEALPix<->equiangular RESAMPLING (bilinear
  ``hp.get_interp_val``), so it is accurate only when the map is well oversampled
  (``nside`` well above ``lmax``). The decode (column/row layout, the ``(-1)^m``
  spin phase, the E/B combination) is verified against healpy in
  ``tests/test_spin_FSHT.py``.

* ``analysis="hp2sph"``: the hand-rolled DFS + latitude nuFFT analysis (the true
  HP2SPH method, no resampling). It is correct for ``m = 0`` but NOT yet for
  ``m != 0`` at spin 2: a spin field does not vanish/flatten at the poles
  (``d^l_{2,-2}(pi) != 0``), so the DFS even-mirror introduces a derivative kink at
  the pole that spreads a single harmonic across theta-frequencies. Fixing the
  spin pole boundary condition is the open gating item (SPIN2_PLAN.md, Phase 3).
  Provided here for development; do not rely on its ``m != 0`` output.

Ground truth throughout is healpy ``map2alm_spin`` / ``alm2map_spin`` with spin 2.
"""

import numpy as np
import healpy as hp

from .ft_sphere import (
    spinsph_analysis,
    fourier2spinsph,
    spinsph2fourier,
    spinsph_synthesis,
)
from .FSHT import (
    FSHT_spin,
    spin_to_EB,
    _spin_F_col,
    _spin_conv_phase,
)
from .data_interpolation import transform_healpix_to_grid
from .double_fourier_sphere import DFS
from .nuFFT import apply_nuFFT

SPIN = 2  # the polarization spin; the pipeline runs the +SPIN and -SPIN passes


def _equiangular_grid(lmax):
    """The FastTransforms equiangular grid for a degree-``lmax`` spin field."""
    N = lmax + 1
    M = 2 * N - 1
    theta = (2 * np.arange(N) + 1) / (2 * N) * np.pi
    phi = 2 * np.pi * np.arange(M) / M
    return N, M, theta, phi


def _resample_to_grid(hmap, theta, phi):
    """Bilinearly sample a HEALPix map at the (theta, phi) tensor-product grid."""
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    return hp.get_interp_val(np.asarray(hmap), TH.ravel(), PH.ravel()).reshape(TH.shape)


def _spin_F_library(Q, U, theta, phi, spin):
    """Equiangular (Q,U) resample -> library spin analysis -> spin-SH ``F`` array."""
    Qg = _resample_to_grid(Q, theta, phi)
    Ug = _resample_to_grid(U, theta, phi)
    z = Qg + 1j * Ug if spin > 0 else Qg - 1j * Ug
    return fourier2spinsph(spinsph_analysis(z, spin), spin)


def _spin_F_hp2sph(Q, U, spin):
    """Hand-rolled HP2SPH analysis -> spin-SH ``F`` array (m=0 correct; see module doc)."""
    z = Q + 1j * U if spin > 0 else Q - 1j * U
    upsampled, fft_coeff = transform_healpix_to_grid(z)
    _, dfs = DFS(upsampled, fft_coeff, spin=spin)
    fft_lat = apply_nuFFT(dfs)
    return FSHT_spin(fft_lat, spin)


def forward_spin(Q, U, lmax, analysis="library"):
    """HEALPix ``(Q, U)`` polarization map -> healpy-ordered ``(aE, aB)``.

    ``analysis`` selects the grid->coefficients route (see the module docstring):
    ``"library"`` (default, resample + exact library analysis) or ``"hp2sph"``
    (hand-rolled DFS+nuFFT, ``m != 0`` not yet correct).
    """
    Q = np.asarray(Q)
    U = np.asarray(U)
    if analysis == "library":
        _, _, theta, phi = _equiangular_grid(lmax)
        Fp = _spin_F_library(Q, U, theta, phi, +SPIN)
        Fm = _spin_F_library(Q, U, theta, phi, -SPIN)
        # library F is complex-SH normalized and the half-sample grid has no DFS
        # colatitude phase: scale 1, no real-SH sqrt(2), no (-1)^l.
        return spin_to_EB(
            Fp, Fm, lmax, scale=1.0, colat_phase=False, real_sh_norm=False
        )
    elif analysis == "hp2sph":
        Fp = _spin_F_hp2sph(Q, U, +SPIN)
        Fm = _spin_F_hp2sph(Q, U, -SPIN)
        return spin_to_EB(Fp, Fm, lmax)  # pipeline norms (1/2pi, sqrt(2), (-1)^l)
    raise ValueError(f"unknown analysis {analysis!r}; use 'library' or 'hp2sph'")


def _F_phase(m, spin):
    """Phase relating an F-array cell to the healpy spin coefficient: F = phase * s_a.

    For ``m >= 0`` this is the decode phase ``_spin_conv_phase`` (verified by the
    F->alm probes); for ``m < 0`` the F columns carry the spin coefficient with no
    extra sign (``+1``), measured the same way (tests/test_spin_FSHT.py covers the
    m>=0 decode; the m<0 columns are exercised by the backward round trip).
    """
    return _spin_conv_phase(m, spin) if m >= 0 else 1.0


def _build_spin_F(a_signed, lmax, N, M, spin):
    """Place healpy-ordered spin coefficients (all signed m) into an F array."""
    F = np.zeros((N, M), dtype=complex)
    s0 = abs(spin)
    for m in range(-lmax, lmax + 1):
        col = _spin_F_col(m)
        ph = _F_phase(m, spin)
        for ell in range(max(abs(m), s0), lmax + 1):
            F[ell - max(abs(m), s0), col] = ph * a_signed(ell, m)
    return F


def backward_spin(aE, aB, nside, lmax=None):
    """healpy-ordered ``(aE, aB)`` -> HEALPix ``(Q, U)`` map (library synthesis route).

    Builds the spin ``F`` array from ``(aE, aB)`` (all signed ``m`` via the reality
    of Q/U), synthesizes ``Q + iU`` on the equiangular grid with the library, and
    resamples back to HEALPix. Resampling-limited, like the ``"library"`` forward.
    ``lmax`` defaults to the band of ``aE``.
    """
    aE = np.asarray(aE)
    aB = np.asarray(aB)
    if lmax is None:
        lmax = hp.Alm.getlmax(len(aE))
    N, M, theta, phi = _equiangular_grid(lmax)

    def healpy_alm(arr, ell, m):
        # signed-m healpy coefficient via the reality a_{l,-m} = (-1)^m conj(a_{l,m})
        if m >= 0:
            return arr[hp.Alm.getidx(lmax, ell, m)]
        return ((-1.0) ** m) * np.conj(arr[hp.Alm.getidx(lmax, ell, -m)])

    def a_plus(ell, m):  # +2 a = -(aE + i aB)
        return -(healpy_alm(aE, ell, m) + 1j * healpy_alm(aB, ell, m))

    Fp = _build_spin_F(a_plus, lmax, N, M, +SPIN)
    zp = spinsph_synthesis(spinsph2fourier(Fp, +SPIN), +SPIN)  # grid Q + iU

    npix = hp.nside2npix(nside)
    th, ph = hp.pix2ang(nside, np.arange(npix))
    Q = _grid_interp(zp.real, theta, phi, th, ph)
    U = _grid_interp(zp.imag, theta, phi, th, ph)
    return Q, U


def _grid_interp(grid, theta, phi, th, ph):
    """Bilinear interpolation of a tensor-product (theta, phi) grid at points (th, ph)."""
    Nt, Np = grid.shape
    # nearest-lower indices with linear weights (phi periodic, theta clamped)
    ti = np.clip(np.searchsorted(theta, th) - 1, 0, Nt - 2)
    t0, t1 = theta[ti], theta[ti + 1]
    wt = (th - t0) / (t1 - t0)
    pp = (ph % (2 * np.pi)) / (2 * np.pi) * Np
    pi0 = np.floor(pp).astype(int) % Np
    pi1 = (pi0 + 1) % Np
    wp = pp - np.floor(pp)
    g00 = grid[ti, pi0]
    g01 = grid[ti, pi1]
    g10 = grid[ti + 1, pi0]
    g11 = grid[ti + 1, pi1]
    return (1 - wt) * ((1 - wp) * g00 + wp * g01) + wt * ((1 - wp) * g10 + wp * g11)
