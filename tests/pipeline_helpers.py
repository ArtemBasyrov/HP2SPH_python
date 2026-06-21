"""Side-effect-free wiring of the HP2SPH pipeline stages.

``main.forward`` / ``main.backward`` do FITS I/O and have import-time side
effects, so the tests compose the four stages directly through these helpers.
"""

import numpy as np

from src.data_interpolation import (
    transform_healpix_to_grid,
    transform_grid_to_healpix,
)
from src.double_fourier_sphere import DFS, DFS_inverse
from src.nuFFT import apply_nuFFT, inverse_nuFFT
from src.FSHT import FSHT, inverse_FSHT, to_healpy_alm, SCALE_2PI


def forward_C(healpix_map, **nufft_kw):
    """HEALPix intensity map -> raw FastTransforms coefficient array ``C``.

    ``C`` is the (L+1, 2L+1) triangular array straight out of ``fourier2sph``;
    it has *not* been converted to healpy ordering/normalization. ``nufft_kw`` is
    forwarded to ``apply_nuFFT`` (e.g. ``solver``/``solve_modes`` to pick the
    scalable well-conditioned band vs the bit-exact square interpolation).
    """
    upsampled, fft_coeff = transform_healpix_to_grid(healpix_map)
    _, fft_coeff_dfs = DFS(upsampled, fft_coeff)
    fft_lat = apply_nuFFT(fft_coeff_dfs, **nufft_kw)
    return FSHT(fft_lat)


def forward_alm(healpix_map, lmax, scale=SCALE_2PI, mono_factor=1.0, **nufft_kw):
    """Full forward transform to a healpy-ordered alm.

    ``scale`` defaults to the first-principles ``1/(2*pi)`` (see
    ``FSHT.to_healpy_alm``); pass ``calibrate_scale(...)`` only to verify.
    """
    C = forward_C(healpix_map, **nufft_kw)
    return to_healpy_alm(C, lmax=lmax, scale=scale, mono_factor=mono_factor)


def backward_map(C, nside):
    """Raw ``C`` array -> HEALPix intensity map (inverse pipeline)."""
    _, C2 = inverse_FSHT(C, nside)
    fft_lat = inverse_nuFFT(C2)
    fft_coeff = DFS_inverse(fft_lat)
    return transform_grid_to_healpix(fft_coeff, fft_coeff)


def calibrate_scale(nside, lmax, ell_probe=2, **nufft_kw):
    """Global gain mapping a unit zonal a_{l,0} onto C[l, 0].

    Synthesises a single zonal harmonic, runs it forward, and reads the gain off
    the corresponding C cell. Pure healpy + pipeline; no assumptions baked in.
    Must use the same ``nufft_kw`` as the forward it calibrates.
    """
    import healpy as hp

    alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
    alm[hp.Alm.getidx(lmax, ell_probe, 0)] = 1.0
    mp = hp.alm2map(alm, nside=nside, lmax=lmax)
    C = forward_C(mp, **nufft_kw)
    sign = (-1.0) ** ell_probe
    return sign * C[ell_probe, 0].real
