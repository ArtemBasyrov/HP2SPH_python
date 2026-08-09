"""Test-side view of the pipeline composition.

The composition itself now lives in :mod:`src.pipeline` -- the benchmarks import
it too, and production code should not reach into ``tests/``. This module
re-exports it so existing tests keep working, and adds the one helper that is
genuinely test-only.
"""

import numpy as np

from src.pipeline import (  # noqa: F401  (re-exported for the test suite)
    forward_C,
    forward_alm,
    backward_map,
    nside_from_C,
)

__all__ = [
    "forward_C",
    "forward_alm",
    "backward_map",
    "nside_from_C",
    "calibrate_scale",
]


def calibrate_scale(nside, lmax, ell_probe=2, **nufft_kw):
    """Global gain mapping a unit zonal a_{l,0} onto C[l, 0].

    Synthesises a single zonal harmonic, runs it forward, and reads the gain off
    the corresponding C cell. Pure healpy + pipeline; no assumptions baked in.
    Must use the same ``nufft_kw`` as the forward it calibrates.

    Verification only: the production scale is the first-principles
    ``FSHT.SCALE_2PI`` and a best fit differs from it by ~5e-5 (see CLAUDE.md).
    """
    import healpy as hp

    alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
    alm[hp.Alm.getidx(lmax, ell_probe, 0)] = 1.0
    mp = hp.alm2map(alm, nside=nside, lmax=lmax)
    C = forward_C(mp, **nufft_kw)
    sign = (-1.0) ** ell_probe
    return sign * C[ell_probe, 0].real
