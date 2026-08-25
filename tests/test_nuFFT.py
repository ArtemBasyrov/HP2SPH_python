"""Stage 3: non-uniform FFT in latitude (FINUFFT + CG solver)."""

import numpy as np
import pytest

from hp2sph.data_interpolation import transform_healpix_to_grid, create_latitude_array
from hp2sph.double_fourier_sphere import DFS
from hp2sph.nuFFT import (
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


# --------------------------------------------------------------------------- #
# The latitude axis is PERIODIC                                                #
# --------------------------------------------------------------------------- #
# The DFS doubling makes latitude periodic with period 2*pi, so the samples sit on
# a circle and the first and last Voronoi cells meet at the seam. The old code
# clamped them to fixed boundaries (pi, -pi) instead of wrapping, which put the
# north pole -- which sits exactly ON the seam -- at half its true weight and gave
# the missing half to the southernmost mirrored ring.
def _dfs_samples(nside):
    from hp2sph.nuFFT import _upsampled_latitudes

    return _upsampled_latitudes(nside)


def _mirror_index(x):
    """For each sample, the index of its image under ``x -> -x`` (mod 2*pi)."""
    xx = np.mod(x, 2 * np.pi)
    return np.array([int(np.argmin(np.abs(np.mod(-xi, 2 * np.pi) - xx))) for xi in x])


@pytest.mark.parametrize("nside", [4, 8, 16])
def test_voronoi_weights_are_mirror_symmetric(nside):
    """A sample set symmetric under ``x -> -x`` must get symmetric weights.

    This is the property the old implementation broke, and it is a prerequisite for
    halving the latitude solve with the DFS mirror symmetry.
    """
    x = _dfs_samples(nside)
    w = compute_voronoi_weights_1d(x)
    idx = _mirror_index(x)
    # the sample set really is symmetric, otherwise the test proves nothing
    assert np.allclose(np.mod(-x, 2 * np.pi), np.mod(x[idx], 2 * np.pi), atol=1e-12)
    np.testing.assert_allclose(w, w[idx], rtol=0, atol=1e-14)


@pytest.mark.parametrize("nside", [4, 8, 16])
def test_pole_sample_gets_a_whole_cell_not_half(nside):
    """The north pole sits at ``x = pi``, exactly on the periodic seam.

    Its cell spans both sides of the wrap. Clamping to the domain edge gave it only
    the half below ``pi`` (measured 0.0510 against a true 0.1021 at nside 8) and
    handed the rest to the last sample.
    """
    x = _dfs_samples(nside)
    w = compute_voronoi_weights_1d(x)
    assert np.isclose(x[0], np.pi), "expected the north pole first, at x = pi"
    # both neighbours across the seam are the same distance away by symmetry, so the
    # pole's cell is the full gap, not half of it
    gap_below = x[0] - x[1]
    gap_above = x[-1] - x[0] + 2 * np.pi
    np.testing.assert_allclose(w[0], 0.5 * (gap_below + gap_above), rtol=1e-12)
    assert w[0] > 0.9 * gap_below, "pole cell collapsed to half -- the seam bug is back"


def test_voronoi_weights_handle_both_orientations():
    """Descending (what the pipeline passes) and ascending must agree."""
    x = _dfs_samples(8)
    w_desc = compute_voronoi_weights_1d(x)
    w_asc = compute_voronoi_weights_1d(x[::-1])
    np.testing.assert_allclose(w_desc, w_asc[::-1], rtol=0, atol=1e-15)


def test_voronoi_weights_reject_non_monotonic_input():
    """Unsorted points have no well-defined 1-D Voronoi cells; say so loudly."""
    with pytest.raises(ValueError, match="monotonic"):
        compute_voronoi_weights_1d(np.array([0.0, 1.0, 0.5, 2.0]))
