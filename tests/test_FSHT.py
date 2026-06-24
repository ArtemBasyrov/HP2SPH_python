"""Stage 4: fast spherical-harmonic transform (libfasttransforms backend).

The ``preparation`` <-> ``convert_to_bivar_coeffs`` round trip is pure Python.
Tests that actually run ``fourier2sph`` / ``sph2fourier`` need the C library and
are marked ``ft`` so they can be skipped with ``-m "not ft"``.
"""

import pytest

# The FSHT module loads the C library on import; skip cleanly if it is missing.
pytest.importorskip("src.ft_sphere")

from src.data_interpolation import transform_healpix_to_grid  # noqa: E402
from src.double_fourier_sphere import DFS  # noqa: E402
from src.nuFFT import apply_nuFFT  # noqa: E402
from src.FSHT import (  # noqa: E402
    preparation,
    convert_to_bivar_coeffs,
    FSHT,
    inverse_FSHT,
)


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


def test_preparation_convert_are_consistent_inverses(nside, healpix_map, relerr):
    """``convert_to_bivar_coeffs`` must invert ``preparation`` exactly.

    ``preparation`` is a PROJECTION, not a bijection: it discards the band-edge
    content (odd-m at the top latitude band, |m|>2*nside) that cannot be
    represented as valid spherical harmonics. So ``convert(prep(x)) == x`` is a
    false invariant for arbitrary x. The correct invariant is that ``prep`` and
    ``convert`` are consistent on the represented subspace, i.e. the map
    ``P = convert . prep`` is idempotent (P(P(x)) == P(x)) to machine precision.
    This catches any factor/sign mismatch between the two (e.g. the T_0-row fix).
    """
    fft_lat = _fft_lat(healpix_map)
    Px = convert_to_bivar_coeffs(preparation(fft_lat), nside)
    PPx = convert_to_bivar_coeffs(preparation(Px), nside)
    assert Px.shape == fft_lat.shape
    assert relerr(PPx, Px) < 1e-12


@pytest.mark.ft
def test_fsht_inverse_roundtrip(nside, healpix_map, relerr):
    """C -> bivar (sph2fourier) -> C (fourier2sph) must round-trip exactly.

    A valid coefficient array ``C`` (the output of ``FSHT``) lives in the
    represented subspace, so pushing it back to bivariate Fourier coefficients
    and forward again must return it. fourier2sph/sph2fourier are exact inverses
    and ``preparation``/``convert`` are consistent, so this is machine-precise.
    (Going the other way, fft_lat->C->fft_lat, is lossy by the projection above.)
    """
    fft_lat = _fft_lat(healpix_map)
    C = FSHT(fft_lat)
    _, bivar = inverse_FSHT(C, nside)
    C_back = FSHT(bivar)
    assert relerr(C_back, C) < 1e-9
