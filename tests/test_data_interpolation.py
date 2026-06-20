"""Stage 1: HEALPix rings <-> equiangular longitude grid."""

import numpy as np
import healpy as hp

from src.data_interpolation import (
    get_ring_indices,
    create_latitude_array,
    transform_healpix_to_grid,
    transform_grid_to_healpix,
)


def test_latitude_array_shape_and_order(nside):
    lat = create_latitude_array(nside)
    assert lat.shape == (4 * nside - 1,)
    # latitudes run monotonically from near +90 (north) to near -90 (south)
    assert np.all(np.diff(lat) < 0)
    assert np.all(np.abs(lat) < 90.0)


def test_latitude_array_symmetric(nside):
    lat = create_latitude_array(nside)
    # north/south rings are mirror images about the equator
    np.testing.assert_allclose(lat, -lat[::-1], atol=1e-12)
    # the central ring sits exactly on the equator
    assert abs(lat[2 * nside - 1]) < 1e-12


def test_ring_sizes_sum_to_npix(nside):
    ring_info = np.asarray(get_ring_indices(nside))
    starts, ends = ring_info[:, 0], ring_info[:, 1]
    sizes = ends - starts + 1
    assert sizes.sum() == hp.nside2npix(nside)
    # north pole ring has 4 pixels, equatorial rings have 4*nside
    assert sizes[0] == 4
    assert sizes[2 * nside - 1] == 4 * nside


def test_grid_shape(nside, healpix_map):
    upsampled, fft_coeff = transform_healpix_to_grid(healpix_map)
    assert upsampled.shape == (4 * nside - 1, 4 * nside)
    assert fft_coeff.shape == (4 * nside - 1, 4 * nside)


def test_roundtrip_is_exact(nside, healpix_map, relerr):
    """HEALPix -> grid -> HEALPix recovers the map to machine precision."""
    _, fft_coeff = transform_healpix_to_grid(healpix_map)
    recovered = transform_grid_to_healpix(fft_coeff, fft_coeff)
    assert relerr(recovered, healpix_map) < 1e-10
