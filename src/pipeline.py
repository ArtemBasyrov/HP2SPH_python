"""Side-effect-free wiring of the four HP2SPH stages.

This is the composition every caller should use: ``main.py`` wraps it with FITS
I/O, the tests pin it, and the benchmarks time it. It lived in
``tests/pipeline_helpers.py`` until the benchmarks started importing it, which put
production code behind a test module; ``tests/pipeline_helpers`` now re-exports
from here so the test suite is unchanged.

Nothing here prints, saves, or reads a file. The stages are:

    data_interpolation -> double_fourier_sphere -> nuFFT -> FSHT

and their exact inverses on the way back.
"""

import numpy as np

from .data_interpolation import (
    transform_healpix_to_grid,
    transform_grid_to_healpix,
)
from .double_fourier_sphere import DFS, DFS_inverse
from .nuFFT import apply_nuFFT, inverse_nuFFT
from .FSHT import FSHT, inverse_FSHT, to_healpy_alm, SCALE_2PI

__all__ = ["forward_C", "forward_alm", "backward_map", "nside_from_C"]


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
    ``FSHT.to_healpy_alm``); pass ``tests.pipeline_helpers.calibrate_scale(...)``
    only to verify.
    """
    C = forward_C(healpix_map, **nufft_kw)
    return to_healpy_alm(C, lmax=lmax, scale=scale, mono_factor=mono_factor)


def nside_from_C(C):
    """Infer nside from a raw ``C`` array.

    With the compact default band the internal latitude band limit is
    ``L = lmax = 2*nside``, and ``C`` has ``L + 1`` rows.
    """
    return (np.shape(C)[0] - 1) // 2


def backward_map(C, nside=None):
    """Raw ``C`` array -> HEALPix intensity map (inverse pipeline).

    ``nside`` defaults to :func:`nside_from_C`, which is right for the compact
    band; pass it explicitly for a ``C`` solved in a wider band.
    """
    if nside is None:
        nside = nside_from_C(C)
    _, C2 = inverse_FSHT(C, nside)
    fft_lat = inverse_nuFFT(C2)
    fft_coeff = DFS_inverse(fft_lat)
    return transform_grid_to_healpix(fft_coeff, fft_coeff)
