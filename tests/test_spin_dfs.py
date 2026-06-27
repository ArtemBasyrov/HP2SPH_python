"""Phase 3: the spin Double Fourier Sphere stage.

``DFS``/``DFS_inverse`` gain a ``spin`` parameter. For a spin-``s`` field the
mirror across the pole is multiplied by ``(-1)^(m+s)`` (the scalar ``s=0`` case is
"flip every odd wavenumber"), and the pole-ring fill must carry complex (Q + iU)
content. For the even spins the polarization pipeline needs (``s = +-2``) the
mirror parity is the same columns as scalar, so the spin DFS reduces to a
complex-safe version of the scalar DFS; these tests pin that.

See SPIN2_PLAN.md (Phase 3).
"""

import numpy as np
import healpy as hp
import pytest

from src.data_interpolation import transform_healpix_to_grid
from src.double_fourier_sphere import DFS, DFS_inverse, _mirror_odd_mask


def _complex_field(nside, seed=0):
    rng = np.random.default_rng(seed)
    npix = hp.nside2npix(nside)
    return (rng.standard_normal(npix) + 1j * rng.standard_normal(npix)).astype(
        np.complex128
    )


@pytest.mark.parametrize("spin", [0, 2, -2])
def test_spin_dfs_complex_roundtrip(nside, spin, relerr):
    """DFS -> DFS_inverse recovers the north-hemisphere Fourier coefficients exactly.

    The doubling is a lossless rearrangement of the original rings (the mirror and
    the two pole rings are appended, never overwriting the originals), so the
    inverse is exact for a genuinely complex spin field, not just real I.
    """
    z = _complex_field(nside, seed=abs(spin) + 1)
    up, fft_coeff = transform_healpix_to_grid(z)
    _, double_fft = DFS(up, fft_coeff, spin=spin)
    recovered = DFS_inverse(double_fft, spin=spin)
    assert np.iscomplexobj(np.asarray(recovered))
    assert relerr(np.asarray(recovered), np.asarray(fft_coeff)) < 1e-12


@pytest.mark.parametrize("spin", [0, 2])
def test_spin_dfs_keeps_complex(nside, spin):
    """The doubled map/spectrum stay complex for a complex field (no .real drop)."""
    z = _complex_field(nside, seed=7)
    up, fft_coeff = transform_healpix_to_grid(z)
    double_map, double_fft = DFS(up, fft_coeff, spin=spin)
    assert np.iscomplexobj(np.asarray(double_map))
    assert np.iscomplexobj(np.asarray(double_fft))
    # the polar rings were filled (not left zero)
    assert np.any(np.abs(np.asarray(double_map)[0]) > 0)


def test_mirror_mask_matches_scalar_for_even_spin(nside):
    """For even spin the (m+spin)-odd mask is the scalar 'odd wavenumber' columns."""
    n_lon = 4 * nside
    scalar = np.zeros(n_lon, bool)
    scalar[1::2] = True  # the old `south_part[:, 1::2] *= -1`
    np.testing.assert_array_equal(_mirror_odd_mask(n_lon, 0), scalar)
    np.testing.assert_array_equal(_mirror_odd_mask(n_lon, 2), scalar)
    np.testing.assert_array_equal(_mirror_odd_mask(n_lon, -2), scalar)
    # odd spin flips the parity (the even columns instead)
    assert np.array_equal(_mirror_odd_mask(n_lon, 1), ~scalar)
