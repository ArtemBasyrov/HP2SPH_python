"""Spin-2 (polarization Q/U <-> E/B) transforms.

This wires the spin stages into an end-to-end ``forward_spin`` / ``backward_spin``
and follows SPIN2_PLAN.md. Two analysis routes exist:

* ``analysis="hp2sph"`` (default): the hand-rolled DFS + latitude nuFFT analysis --
  the true HP2SPH method, no resampling anywhere. Beats healpy ``map2alm_spin`` by
  4-8x on the median per-l C_l error at every nside tested (8 to 64), and converges
  with nside (``tests/test_spin_paper_accuracy.py``).

* ``analysis="library"``: resample the HEALPix ``(Q, U)`` onto the FastTransforms
  equiangular grid and use the library's own ``spinsph_analysis`` for the
  grid->bivariate-Fourier step, then ``fourier2spinsph`` + the E/B decode. Exact in
  the grid->coefficients step, but its accuracy floor is the HEALPix<->equiangular
  RESAMPLING (bilinear ``hp.get_interp_val``), so it is only accurate when the map is
  well oversampled (``nside`` well above ``lmax``). Kept as an independent cross-check
  of the decode, which it validates against healpy in ``tests/test_spin_FSHT.py``.

The ``m != 0`` failure the plan lists as its open gating item is FIXED. It was never
the pole boundary condition the plan hypothesized (a spin field's latitude profile is
``sin^|m+s|(theta/2) cos^|m-s|(theta/2) P(cos theta)``, whose DFS mirror extension is
analytic -- there is no kink). Three separate bugs were responsible:

1. ``data_interpolation.ring_mode_mask`` -- the innermost polar rings have 4, 8, 12
   pixels and cannot resolve ``|m| = |s|``, which is exactly the mode a spin-s field
   carries at O(1) INTO the pole. The zero-padding asserted that content was zero.
2. ``double_fourier_sphere.DFS`` read the last original ring instead of the filled
   south pole row (an off-by-one), so half the polynomial pole fill was discarded.
3. ``FSHT.spin_g_to_library`` -- the pipeline's ``x = pi - theta`` reflection has to be
   undone in the bivariate domain, because the reflection FLIPS THE SPIN and so cannot
   be undone by a phase on the output coefficients.

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
from .double_fourier_sphere import DFS, dfs_mode_mask
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
    """Hand-rolled HP2SPH analysis -> spin-SH ``F`` array (no resampling).

    The latitude fit is masked (``dfs_mode_mask``): the innermost polar rings do not
    resolve ``|m| = |spin|``, which for a spin field is exactly where the field is O(1)
    at the pole, so those entries are dropped as missing rather than asserted to be
    zero. Without the mask the ``|m| = |spin|`` columns are wrong by 15-50%.
    """
    z = Q + 1j * U if spin > 0 else Q - 1j * U
    nside = hp.npix2nside(np.asarray(z).shape[0])
    upsampled, fft_coeff = transform_healpix_to_grid(z)
    _, dfs = DFS(upsampled, fft_coeff, spin=spin)
    fft_lat = apply_nuFFT(dfs, solver="lsmr", sample_mask=dfs_mode_mask(nside))
    return FSHT_spin(fft_lat, spin)


def forward_spin(Q, U, lmax, analysis="hp2sph"):
    """HEALPix ``(Q, U)`` polarization map -> healpy-ordered ``(aE, aB)``.

    ``analysis`` selects the grid->coefficients route (see the module docstring):
    ``"hp2sph"`` (default, the hand-rolled DFS+nuFFT, no resampling) or ``"library"``
    (resample + the library's exact analysis, resampling-limited).
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
        # FSHT_spin already converted the pipeline conventions away (FSHT.
        # spin_g_to_library), so this is the same decode the library route uses.
        return spin_to_EB(
            Fp, Fm, lmax, scale=1.0, colat_phase=False, real_sh_norm=False
        )
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
