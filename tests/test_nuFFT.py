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
    fft_lat = apply_nuFFT(dfs)
    n_modes = fft_lat.shape[0]
    # default = compact band-limited band |k| <= 2*nside -> 4*nside+1 modes (odd),
    # i.e. L = lmax = 2*nside for the FSHT.
    assert n_modes == 4 * nside + 1
    assert n_modes % 2 == 1
    assert fft_lat.shape[1] == 4 * nside

    # the square (exact-interpolation) band is the wider one
    sq = apply_nuFFT(dfs, solver="svd", solve_modes=8 * nside + 1)
    assert sq.shape[0] == 8 * nside + 1


def test_nufft_roundtrip_square_is_exact(nside, healpix_map, relerr):
    """SQUARE interpolation (one mode per sample) reproduces the samples exactly.

    With ``solve_modes = 8*nside+1`` the latitude system is square, so the forward
    solve interpolates the samples and direct re-evaluation returns them. The
    Vandermonde is ill-conditioned, so the dense SVD solver is used (CG on the
    normal equations would floor at high nside).
    """
    dfs = _dfs_coeffs(healpix_map)
    fft_lat = apply_nuFFT(dfs, solver="svd", solve_modes=8 * nside + 1)
    recovered = inverse_nuFFT(fft_lat)
    assert recovered.shape == dfs.shape
    assert relerr(recovered, dfs) < 1e-4


def test_nufft_roundtrip_default_is_bounded_projection(nside, healpix_map, relerr):
    """The default (well-conditioned 4*nside+1 band) is a bounded projection.

    It does NOT interpolate every sample -- it drops the above-band latitude
    content the clustered grid can't represent (polar aliasing) -- so the nuFFT
    round trip is a few percent, not machine zero. That is the price for the
    well-conditioned, scalable solve; the residual shrinks with nside.
    """
    dfs = _dfs_coeffs(healpix_map)
    recovered = inverse_nuFFT(apply_nuFFT(dfs))
    assert recovered.shape == dfs.shape
    assert relerr(recovered, dfs) < 0.2


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
