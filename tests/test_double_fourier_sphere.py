"""Stage 2: Double Fourier Sphere mirror/extension."""

import numpy as np

from src.data_interpolation import transform_healpix_to_grid
from src.double_fourier_sphere import DFS, DFS_inverse, compute_ring_area_weights


def test_dfs_shapes(nside, healpix_map):
    upsampled, fft_coeff = transform_healpix_to_grid(healpix_map)
    n_rings = 4 * nside - 1
    double_map, double_fft = DFS(upsampled, fft_coeff)
    # the doubled grid adds the mirror plus the two interpolated pole rings
    assert double_map.shape == (2 * n_rings + 2, 4 * nside)
    assert double_fft.shape == (2 * n_rings + 2, 4 * nside)


def test_dfs_roundtrip_is_exact(nside, healpix_map, relerr):
    """DFS -> DFS_inverse recovers the original fft_coeff exactly."""
    upsampled, fft_coeff = transform_healpix_to_grid(healpix_map)
    _, double_fft = DFS(upsampled, fft_coeff)
    recovered = DFS_inverse(double_fft)
    assert relerr(recovered, fft_coeff) < 1e-12


def test_ring_area_weights_partition_sphere(nside):
    """The ring areas underlying the weights must tile the full sphere."""
    # compute_ring_area_weights asserts sum(ring_areas) == 4*pi internally;
    # here we also confirm the returned correction is finite and positive.
    correction = compute_ring_area_weights(nside)
    assert correction.shape == (4 * nside - 1 + 2,)  # poles + rings
    assert np.all(np.isfinite(correction))
    assert np.all(correction > 0)
