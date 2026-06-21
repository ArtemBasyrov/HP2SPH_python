import numpy as np
import jax.numpy as jnp
import healpy as hp

from .data_interpolation import create_latitude_array

# Number of HEALPix rings on EACH side of a pole used to extrapolate the (unsampled)
# pole-ring value. Capped at the polar-cap size (nside-1) so the stencil never
# reaches into the equatorial belt, where the long extrapolation to the pole would
# be unstable. The Lagrange weights are well behaved (L1 ~ 3.4, sum = 1) and stable
# across nside, so 6 is safe everywhere. See ``interpolate_polar_rings``.
POLE_INTERP_NPTS = 6


def DFS(mp: jnp.array, fft_coeff: jnp.array) -> (jnp.array, jnp.array):
    south_part = jnp.flip(mp)
    double_map = jnp.concatenate((mp, south_part), axis=0)

    double_map = interpolate_polar_rings(double_map)

    south_part = np.flipud(np.array(fft_coeff))
    south_part[:, 1::2] *= -1  # flip every odd wave number in the mirrored part by -1

    # double the fft coefficients
    n_rings = fft_coeff.shape[0]
    double_fft = np.zeros((2 * n_rings + 2, fft_coeff.shape[1]), dtype=complex)
    double_fft[0] = np.fft.fft(double_map[0], n=fft_coeff.shape[1], norm="forward")
    double_fft[1 : n_rings + 1] = fft_coeff[:]
    double_fft[n_rings + 1] = np.fft.fft(
        double_map[n_rings], n=fft_coeff.shape[1], norm="forward"
    )
    double_fft[n_rings + 2 :] = south_part

    """# apply weights correction
    weights = compute_ring_area_weights(fft_coeff.shape[1] // 4) # both poles + original map
    double_fft[:n_rings+2] *= weights[:, np.newaxis]
    double_fft[n_rings+2:] *= np.flip(weights[1:-1])[:, np.newaxis] # flip weights for the mirrored part 
    """
    # apply FFT shift from numpy ordering to natural ordering
    double_fft = np.fft.fftshift(double_fft, axes=1)

    return double_map, double_fft


def DFS_inverse(double_fft: jnp.array) -> jnp.array:
    nside = double_fft.shape[1] // 4
    n_rings = 4 * nside - 1

    # selecting the upper part of the double map without added poles
    fft_coeff = double_fft[1 : n_rings + 1]

    # apply weights correction
    # weights = compute_ring_area_weights(nside) # both poles + original map
    # fft_coeff /= weights[1:-1][:, np.newaxis]

    # apply FFT shift from natural ordering to numpy ordering
    fft_coeff = np.fft.ifftshift(fft_coeff, axes=1)

    return fft_coeff


def _pole_lagrange_weights(nodes: np.ndarray, x0: float) -> np.ndarray:
    """Lagrange weights to evaluate, at ``x0``, the degree-(n-1) polynomial that
    interpolates values sampled at ``nodes``. The nodes are the same for every
    longitude column, so the whole pole ring is one matvec ``weights @ stencil``.
    """
    n = len(nodes)
    w = np.ones(n)
    for i in range(n):
        for j in range(n):
            if j != i:
                w[i] *= (x0 - nodes[j]) / (nodes[i] - nodes[j])
    return w


def interpolate_polar_rings(mp: jnp.array) -> jnp.array:
    """Fill the two (HEALPix-unsampled) pole rings of the DFS-doubled map.

    Each pole value is a polynomial extrapolation in latitude: fit a degree-(2*npts-1)
    polynomial through a stencil symmetric about the pole -- the ``npts`` rings nearest
    the pole and their mirror images across it -- and evaluate it AT the pole. This
    replaces the old piecewise-LINEAR ``jnp.interp`` fill, which was the dominant
    high-l forward error: the m=0 latitude profile P_l(cos theta) peaks at the poles,
    so a crude pole value injects a large zonal error that grows with l. The
    higher-order fit cuts the m=0 error ~5-15x at the band edge (now on par with /
    better than healpy ring weights). See tests/test_double_fourier_sphere.py.
    """
    nside = mp.shape[1] // 4
    n_rings = mp.shape[0] // 2
    npts = max(2, min(POLE_INTERP_NPTS, nside - 1))

    latitudes = create_latitude_array(nside)
    mp = np.asarray(mp)

    # North pole (latitude 90): stencil = the npts northmost rings (latitudes
    # `latitudes[:npts]`, just below the pole) and their mirror images across the
    # pole (the southern-hemisphere rings of the DFS doubling, at 180 - latitude).
    north_theta = np.concatenate((np.flip(latitudes[:npts]), 180 - latitudes[:npts]))
    north_fp = np.concatenate((np.flip(mp[:npts], axis=0), np.flip(mp[-npts:], axis=0)))
    north_pole_mp = _pole_lagrange_weights(north_theta, 90.0) @ north_fp

    # South pole (latitude -90): same construction, mirrored. The original code
    # interpolated on the negated-latitude axis to reuse x0 = 90; keep that.
    south_theta = np.concatenate((latitudes[-npts:], -180 - np.flip(latitudes[-npts:])))
    south_fp = np.concatenate(
        (mp[n_rings - npts : n_rings], mp[n_rings : n_rings + npts])
    )
    south_pole_mp = _pole_lagrange_weights(-south_theta, 90.0) @ south_fp

    # Add the polar rings to the map
    double_map = np.zeros((mp.shape[0] + 2, mp.shape[1]))
    double_map[0] = north_pole_mp
    double_map[1 : n_rings + 1] = mp[:n_rings]
    double_map[n_rings + 1] = south_pole_mp
    double_map[n_rings + 2 :] = mp[n_rings:]

    return double_map


def compute_ring_area_weights(nside):
    theta = create_latitude_array(nside)
    theta = np.concatenate(([90.0], theta, [-90.0]))  # [90, ..., -90]

    ring_borders = np.zeros(len(theta) + 1)
    ring_borders[1:-1] = theta[:-1] + np.diff(theta) / 2
    ring_borders[0] = 90
    ring_borders[-1] = -90
    ring_borders = np.deg2rad(ring_borders + 90.0)  # [90, -90] -> [pi, 0]

    ring_areas = np.zeros(len(ring_borders) - 1)
    ring_areas = -2 * np.pi * (np.cos(ring_borders[:-1]) - np.cos(ring_borders[1:]))

    assert np.isclose(np.sum(ring_areas), 4 * np.pi), (
        "Sum of ring areas should be equal to 4*pi"
    )

    hp_pix_area = hp.nside2pixarea(nside)
    pixel_area = ring_areas / (4 * nside)
    correction = pixel_area / hp_pix_area

    return correction
