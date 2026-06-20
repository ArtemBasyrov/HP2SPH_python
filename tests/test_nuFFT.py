"""Stage 3: non-uniform FFT in latitude (FINUFFT + CG solver)."""

import numpy as np

from src.data_interpolation import transform_healpix_to_grid, create_latitude_array
from src.double_fourier_sphere import DFS
from src.nuFFT import (
    apply_nuFFT,
    inverse_nuFFT,
    compute_voronoi_weights_1d,
)


def _dfs_coeffs(healpix_map):
    upsampled, fft_coeff = transform_healpix_to_grid(healpix_map)
    _, double_fft = DFS(upsampled, fft_coeff)
    return double_fft


def test_nufft_output_shape(nside, healpix_map):
    dfs = _dfs_coeffs(healpix_map)
    m_samples = dfs.shape[0]  # = 2*(4*nside-1) + 2 ~ 8*nside
    fft_lat = apply_nuFFT(dfs)
    n_modes = fft_lat.shape[0]
    # smallest odd count >= m_samples
    assert n_modes >= m_samples
    assert n_modes % 2 == 1
    assert fft_lat.shape[1] == 4 * nside


def test_nufft_roundtrip(nside, healpix_map, relerr):
    """Forward (CG solve) then backward (type-2 eval) recovers the samples.

    The forward solve interpolates the samples, so direct re-evaluation must
    return them; this is limited only by the CG tolerance (rtol=1e-6).
    """
    dfs = _dfs_coeffs(healpix_map)
    fft_lat = apply_nuFFT(dfs)
    recovered = inverse_nuFFT(fft_lat)
    assert recovered.shape == dfs.shape
    assert relerr(recovered, dfs) < 1e-4


def test_voronoi_weights_sum_to_domain():
    """Voronoi cell widths must tile the whole [-pi, pi) latitude domain."""
    nside = 8
    latitudes = create_latitude_array(nside)
    samp = np.zeros(len(latitudes) * 2 + 2)
    samp[0] = 90
    samp[1 : len(latitudes) + 1] = latitudes
    samp[len(latitudes) + 1] = -90
    samp[len(latitudes) + 2 :] = -180 + latitudes
    samp = samp * np.pi / 180 + np.pi / 2

    w = compute_voronoi_weights_1d(samp)
    assert w.shape == samp.shape
    # cell widths partition the full 2*pi period
    np.testing.assert_allclose(np.sum(w), 2 * np.pi, rtol=1e-12)
    assert np.all(w > 0)
