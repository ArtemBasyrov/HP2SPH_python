"""Stage 4: fast spherical-harmonic transform (FastTransforms.jl bridge).

The ``preparation`` <-> ``convert_to_bivar_coeffs`` round trip is pure Python.
Tests that actually run ``fourier2sph`` / ``sph2fourier`` shell out to Julia and
are marked ``julia`` so they can be skipped with ``-m "not julia"``.
"""

import numpy as np
import pytest

from src.data_interpolation import transform_healpix_to_grid
from src.double_fourier_sphere import DFS
from src.nuFFT import apply_nuFFT
from src.FSHT import preparation, convert_to_bivar_coeffs, FSHT, inverse_FSHT


def _fft_lat(healpix_map):
    upsampled, fft_coeff = transform_healpix_to_grid(healpix_map)
    _, double_fft = DFS(upsampled, fft_coeff)
    return apply_nuFFT(double_fft)


def test_preparation_shapes(nside, healpix_map):
    fft_lat = _fft_lat(healpix_map)
    n_modes = fft_lat.shape[0]
    L = (n_modes - 1) // 2
    g = preparation(fft_lat)
    assert g.shape == (L + 1, 2 * L + 1)


def test_preparation_convert_roundtrip(nside, healpix_map, relerr):
    """bivar -> g -> bivar must recover the bivariate Fourier coefficients.

    ``preparation`` and ``convert_to_bivar_coeffs`` are an algebraic inverse
    pair, so for a real band-limited map this should be exact. It currently is
    NOT (~2.6e-3 at nside=4, ~8e-4 at nside=16): ``preparation`` deliberately
    zeros odd-m content at the top latitude band (Nyquist edge). The polar-ring
    normalization fix in ``data_interpolation`` reduced this leakage ~5x, but a
    residual band-edge truncation remains. Tight tolerance documents it.
    """
    fft_lat = _fft_lat(healpix_map)
    g = preparation(fft_lat)
    bivar = convert_to_bivar_coeffs(g, nside)
    assert bivar.shape == fft_lat.shape
    assert relerr(bivar, fft_lat) < 1e-9


@pytest.mark.julia
def test_fsht_inverse_roundtrip(nside, healpix_map, relerr):
    """fft_lat -> C (fourier2sph) -> fft_lat (sph2fourier) must round-trip.

    fourier2sph / sph2fourier are exact inverses, so the residual here is
    identical to the Python ``preparation``/``convert`` truncation and fails for
    the same reason (high-latitude-mode leakage from the nuFFT). Tight tolerance
    on purpose: it documents the defect and should go green once fixed.
    """
    fft_lat = _fft_lat(healpix_map)
    C = FSHT(fft_lat)
    _, C_back = inverse_FSHT(C, nside)
    assert relerr(C_back, fft_lat) < 1e-9
