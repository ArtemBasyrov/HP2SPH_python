"""Phase 3: the spin Double Fourier Sphere stage.

``DFS``/``DFS_inverse`` gain a ``spin`` parameter. For a spin-``s`` field the
mirror across the pole is multiplied by ``(-1)^(m+s)`` (the scalar ``s=0`` case is
"flip every odd wavenumber"), and the pole-ring fill must carry complex (Q + iU)
content. For the even spins the polarization pipeline needs (``s = +-2``) the
mirror parity is the same columns as scalar, so the spin DFS reduces to a
complex-safe version of the scalar DFS; these tests pin that.
"""

import numpy as np
import healpy as hp
import pytest

from hp2sph.data_interpolation import transform_healpix_to_grid
from hp2sph.double_fourier_sphere import DFS, DFS_inverse, _mirror_odd_mask


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


def test_mirror_map_is_a_phi_shift(nside):
    """The DFS glide reflection shifts phi by pi; it does not reverse it.

    ``theta -> 2*pi - theta`` pairs with ``phi -> phi + pi``, which on the grid is a
    ROLL by half the longitude samples. The old code used ``jnp.flip(mp)`` with no
    axis, reversing both axes and so applying ``phi -> -phi``.
    """
    from hp2sph.double_fourier_sphere import _mirror_map

    z = _complex_field(nside, seed=11)
    up, _ = transform_healpix_to_grid(z)
    up = np.asarray(up)
    n_lon = up.shape[1]
    got = np.asarray(_mirror_map(up, 0))
    want = np.roll(np.flip(up, axis=0), n_lon // 2, axis=1)
    np.testing.assert_allclose(got, want, atol=1e-14)
    # odd spin additionally flips the sign; even spin does not
    np.testing.assert_allclose(np.asarray(_mirror_map(up, 1)), -want, atol=1e-14)


def test_dfs_south_pole_row_is_the_pole(nside):
    """The south-pole row of double_fft is the FILLED pole, not the last ring.

    ``interpolate_polar_rings`` returns [north pole, rings, south pole, mirror], so the
    south pole is row ``n_rings+1``. ``DFS`` read row ``n_rings`` -- the last original
    ring -- so the south pole never got the polynomial fill the north pole did.
    """
    z = _complex_field(nside, seed=13)
    up, fft_coeff = transform_healpix_to_grid(z)
    double_map, double_fft = DFS(up, fft_coeff, spin=0)
    n_rings = fft_coeff.shape[0]
    n_lon = fft_coeff.shape[1]

    def natural(row):
        """One map row -> the natural centred order ``DFS`` returns.

        Both ``|m| = 2*nside`` ends are carried, the sampled slot split half onto each;
        see ``double_fourier_sphere._shifted_into``.
        """
        shifted = np.fft.fftshift(np.fft.fft(row, n=n_lon, norm="forward"))
        out = np.append(shifted, 0.5 * shifted[0])
        out[0] *= 0.5
        return out

    pole_row = natural(np.asarray(double_map)[n_rings + 1])
    np.testing.assert_allclose(double_fft[n_rings + 1], pole_row, atol=1e-14)
    # and it is genuinely different from the last original ring
    last_ring = natural(np.asarray(double_map)[n_rings])
    assert not np.allclose(double_fft[n_rings + 1], last_ring, atol=1e-10)


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
