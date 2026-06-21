"""Stage 2: Double Fourier Sphere mirror/extension."""

from functools import partial

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np

from src.data_interpolation import transform_healpix_to_grid, create_latitude_array
from src.double_fourier_sphere import (
    DFS,
    DFS_inverse,
    compute_ring_area_weights,
    interpolate_polar_rings,
)


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


def test_polynomial_pole_fill_beats_linear(nside):
    """The polynomial pole extrapolation recovers an unsampled pole value far
    better than the old piecewise-linear fill, and converges with nside.

    The two DFS pole rings are not sampled by HEALPix; their value drives the m=0
    (zonal) accuracy because P_l(cos theta) peaks at the poles. On a smooth zonal
    test function (whose exact pole value is known from its m=0 coefficients) the
    degree-(2*npts-1) Lagrange fill is ~1-2 orders of magnitude closer to the
    truth than linear interpolation -- this is the fix for the dominant high-l
    forward error.
    """
    lmax = 2 * nside
    alm = np.zeros(hp.Alm.getsize(lmax), dtype=complex)
    for ell in range(lmax + 1):
        alm[hp.Alm.getidx(lmax, ell, 0)] = 1.0 / (1 + ell) ** 2  # smooth, nonzero pole
    mp = hp.alm2map(alm, nside=nside, lmax=lmax)
    upsampled, _ = transform_healpix_to_grid(mp)
    doubled = jnp.concatenate((upsampled, jnp.flip(upsampled)), axis=0)

    ells = np.arange(lmax + 1)
    a_l0 = np.array([alm[hp.Alm.getidx(lmax, ell, 0)].real for ell in ells])
    north_exact = np.sum(a_l0 * np.sqrt((2 * ells + 1) / (4 * np.pi)))

    # production (polynomial) fill: the m=0 component is the longitude mean
    north_poly = float(interpolate_polar_rings(doubled)[0].mean())

    # old linear fill (3 rings each side, bracketing linear interp) for comparison
    lat = create_latitude_array(nside)
    nth = jnp.concatenate((jnp.flip(lat[:3]), 180 - lat[:3]))
    nfp = jnp.concatenate((jnp.flip(doubled[:3]), jnp.flip(doubled[-3:]))).T
    north_lin = float(
        jax.vmap(partial(lambda fp, th: jnp.interp(90, th, fp), th=nth))(nfp).mean()
    )

    poly_err = abs(north_poly - north_exact) / abs(north_exact)
    lin_err = abs(north_lin - north_exact) / abs(north_exact)
    assert poly_err < 0.2 * lin_err, f"poly {poly_err:.2e} not << linear {lin_err:.2e}"
    assert poly_err < 2e-2, f"poly pole fill inaccurate: {poly_err:.2e}"


def test_ring_area_weights_partition_sphere(nside):
    """The ring areas underlying the weights must tile the full sphere."""
    # compute_ring_area_weights asserts sum(ring_areas) == 4*pi internally;
    # here we also confirm the returned correction is finite and positive.
    correction = compute_ring_area_weights(nside)
    assert correction.shape == (4 * nside - 1 + 2,)  # poles + rings
    assert np.all(np.isfinite(correction))
    assert np.all(correction > 0)
