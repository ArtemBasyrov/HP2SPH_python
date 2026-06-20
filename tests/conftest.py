"""Shared pytest fixtures and environment setup for the HP2SPH test suite.

Run from the repo root with the ``s2fft`` micromamba env, e.g.::

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
        /Users/basyrov/micromamba/envs/s2fft/bin/python3 -m pytest

The two env vars are also set defensively below (they only take effect if the
relevant libraries have not been imported yet), and ``-m "not julia"`` skips the
tests that shell out to Julia / FastTransforms.jl.
"""

import os

# Must be set before finufft / jax pull in libomp. Harmless if already set.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import jax

# The pipeline needs float64; without this JAX silently runs float32 and the
# transforms diverge. Must happen before any jax array is created.
jax.config.update("jax_enable_x64", True)

import numpy as np
import healpy as hp
import pytest


# --------------------------------------------------------------------------- #
# Parametrisation                                                              #
# --------------------------------------------------------------------------- #
# Small nside values keep the suite fast while still exercising the polar /
# equatorial ring split (needs nside >= 2) and the Julia round trip.
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
