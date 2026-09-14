"""Shared pytest fixtures and environment setup for the HP2SPH test suite.

Run from the repo root with an env that has the pipeline deps (substitute its
interpreter for ``python``)::

    python -m pytest

The OpenMP guards are set below (and by the package itself on import), so no
env-var prefix is needed; ``-m "not ft"`` skips the tests that need the
libfasttransforms C library.
"""

import os

# Mirrors hp2sph/_bootstrap.py, which cannot run first here: conftest imports healpy
# below, and libomp reads its thread count when the image loads. One thread per
# OpenMP runtime is a correctness requirement, not tuning -- see hp2sph/_openmp.py.
# Deliberately duplicated rather than imported from hp2sph: importing the package here
# would pull in hp2sph/FSHT.py, which raises ImportError without libfasttransforms, and
# that would break `pytest -m "not ft"` for anyone who has not built the C library.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_threads = os.environ.get("HP2SPH_OMP_THREADS", "1").strip()
os.environ["OMP_NUM_THREADS"] = (
    str(os.cpu_count() or 1) if _threads.lower() == "auto" else _threads
)

import numpy as np
import healpy as hp
import pytest


# --------------------------------------------------------------------------- #
# Parametrisation                                                              #
# --------------------------------------------------------------------------- #
# Small nside values keep the suite fast while still exercising the polar /
# equatorial ring split (needs nside >= 2) and the FSHT round trip.
NSIDES = [4, 8, 16]


@pytest.fixture(params=NSIDES)
def nside(request):
    return request.param


@pytest.fixture
def rng():
    return np.random.default_rng(20260620)


@pytest.fixture
def lmax(nside):
    """The band limit the pipeline targets for a given nside."""
    return 2 * nside


@pytest.fixture
def random_alm(nside, lmax, rng):
    """A random band-limited (complex, healpy-ordered) alm.

    Real-map symmetry is enforced by construction: m=0 coefficients are real.
    """
    ncoeff = hp.Alm.getsize(lmax)
    alm = rng.standard_normal(ncoeff) + 1j * rng.standard_normal(ncoeff)
    # zero out m=0 imaginary part so the synthesised map is real
    m0 = np.array([hp.Alm.getidx(lmax, ell, 0) for ell in range(lmax + 1)])
    alm[m0] = alm[m0].real
    return alm.astype(np.complex128)


@pytest.fixture
def healpix_map(random_alm, nside, lmax):
    """A real HEALPix intensity map synthesised from ``random_alm``."""
    return hp.alm2map(random_alm, nside=nside, lmax=lmax)


@pytest.fixture
def iqu_map(healpix_map):
    """3-row (I, Q, U) map; Q/U are zero (the pipeline ignores them)."""
    return np.vstack(
        [healpix_map, np.zeros_like(healpix_map), np.zeros_like(healpix_map)]
    )


# --------------------------------------------------------------------------- #
# Helpers usable from tests                                                    #
# --------------------------------------------------------------------------- #
def rel_err(a, b):
    """Relative L2 error ||a - b|| / ||b||."""
    a = np.asarray(a)
    b = np.asarray(b)
    denom = np.linalg.norm(b.ravel())
    if denom == 0:
        return np.linalg.norm(a.ravel())
    return np.linalg.norm((a - b).ravel()) / denom


@pytest.fixture
def relerr():
    return rel_err


# --------------------------------------------------------------------------- #
# Test-only helpers                                                            #
# --------------------------------------------------------------------------- #
def calibrate_scale(nside, lmax, ell_probe=2, **nufft_kw):
    """Global gain mapping a unit zonal a_{l,0} onto C[l, 0].

    Synthesises a single zonal harmonic, runs it forward, and reads the gain off
    the corresponding C cell. Pure healpy + pipeline; no assumptions baked in.
    Must use the same ``nufft_kw`` as the forward it calibrates.

    Verification only: the production scale is the first-principles
    ``FSHT.SCALE_2PI`` and a best fit differs from it by ~5e-5.
    """
    # Imported here, not at module level: hp2sph.pipeline pulls in the C library,
    # and conftest must import without it for ``-m "not ft"``.
    from hp2sph.pipeline import forward_C

    alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
    alm[hp.Alm.getidx(lmax, ell_probe, 0)] = 1.0
    mp = hp.alm2map(alm, nside=nside, lmax=lmax)
    C = forward_C(mp, **nufft_kw)
    sign = (-1.0) ** ell_probe
    return sign * C[ell_probe, 0].real
