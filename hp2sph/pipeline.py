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
    npix2nside,
    transform_healpix_to_grid,
    transform_grid_to_healpix,
)
from .double_fourier_sphere import DFS, DFS_inverse, pole_stencil_rows
from .nuFFT import apply_nuFFT, inverse_nuFFT
from .FSHT import FSHT, inverse_FSHT, to_healpy_alm, SCALE_2PI

__all__ = ["forward_C", "forward_alm", "backward_map", "nside_from_C"]

# NUFFT tolerance for the latitude ANALYSIS. The solve is a least squares whose own
# residual is orders above this, so 1e-6 is free: measured over nside 64/128/256 and
# the cosmology / flat / aliased scenarios, four seeds, the top-band C_l error is
# unchanged to four digits and the forward runs 1.06-1.13x faster. The SYNTHESIS keeps
# 1e-12 -- it is a plain evaluation whose eps sets the output accuracy directly.
ANALYSIS_EPS = 1e-6

# NUFFT tolerance for the latitude SYNTHESIS. Unlike the analysis this is a plain
# evaluation, so eps sets the output accuracy directly, scaled by the size of the
# coefficient vector being evaluated. See backward_map's ``eps`` argument.
SYNTHESIS_EPS = 1e-12


def forward_C(healpix_map, **nufft_kw):
    """HEALPix intensity map -> raw FastTransforms coefficient array ``C``.

    ``C`` is the (L+1, 2L+1) triangular array straight out of ``fourier2sph``;
    it has *not* been converted to healpy ordering/normalization. ``nufft_kw`` is
    forwarded to ``apply_nuFFT`` (e.g. ``solver``/``solve_modes`` to pick the
    scalable well-conditioned band vs the bit-exact square interpolation). The CG
    analysis defaults to ``eps=ANALYSIS_EPS``; pass ``eps`` to override it.
    """
    # spin=0 lets the CG path use the DFS mirror symmetry (see nuFFT._mirror_plan); it is
    # ignored by the other solvers and falls back if the array is not symmetric.
    nufft_kw.setdefault("spin", 0)
    if nufft_kw.get("solver", "cg") == "cg":
        nufft_kw.setdefault("eps", ANALYSIS_EPS)
    # Only the CG solver understands the half layout: svd and lsmr take the full
    # 8*nside latitude sample set, so the square-band route keeps the full path.
    # A fold plan or a sample mask is laid out for whichever domain the caller built it
    # in, and a wider band needs the full sample set, so those opt out too.
    half = (
        nufft_kw.get("solver", "cg") == "cg"
        and nufft_kw["spin"] == 0
        and nufft_kw.get("solve_modes") is None
        and nufft_kw.get("sample_mask") is None
        and nufft_kw.get("fold") is None
    )
    if half:
        # Only the rings the pole fill reads are brought back to map space, and only
        # the mirror-fundamental rows of the DFS are built; the solve restricts to
        # them anyway. Bit-identical to the full route, and the two arrays skipped are
        # the largest items in the transform's peak memory.
        nside = npix2nside(np.shape(healpix_map)[0])
        upsampled, fft_coeff = transform_healpix_to_grid(
            healpix_map, map_rows=pole_stencil_rows(nside)
        )
        _, fft_coeff_dfs = DFS(upsampled, fft_coeff, spin=0, half=True)
        fft_lat = apply_nuFFT(fft_coeff_dfs, half_domain=True, **nufft_kw)
    else:
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


def backward_map(C, nside=None, eps=SYNTHESIS_EPS):
    """Raw ``C`` array -> HEALPix intensity map (inverse pipeline).

    ``nside`` defaults to :func:`nside_from_C`, which is right for the compact
    band; pass it explicitly for a ``C`` solved in a wider band.

    ``eps`` is the latitude NUFFT tolerance for the synthesis. Synthesis is a plain
    evaluation, so its error enters the output multiplied by the norm of the
    coefficient vector: in the well-conditioned compact band that norm is O(1) and
    the default is far tighter than needed, but a wide, ill-conditioned band
    produces coefficients orders of magnitude larger than the map and needs a
    correspondingly tighter ``eps`` to invert. Values below ~1e-13 are refused by
    finufft at the default upsampling factor.
    """
    if nside is None:
        nside = nside_from_C(C)
    _, C2 = inverse_FSHT(C, nside)
    fft_lat = inverse_nuFFT(C2, eps=eps)
    fft_coeff = DFS_inverse(fft_lat)
    return transform_grid_to_healpix(fft_coeff, fft_coeff)
