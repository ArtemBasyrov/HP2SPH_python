"""Phase 1: the in-process spin-weighted FastTransforms backend.

``src/ft_sphere.py`` gains ``fourier2spinsph`` / ``spinsph2fourier`` wrapping the
C library's ``ft_plan_spinsph2fourier(n, s)`` + spin execute routines (genuinely
complex, interleaved ft_complex buffers). These tests are self-contained: a pure
backend round trip plus a spin-0 == scalar sanity check. No Julia, no healpy.

See SPIN2_PLAN.md (Phase 1).
"""

import numpy as np
import pytest

ft_sphere = pytest.importorskip("src.ft_sphere")

if not getattr(ft_sphere, "_HAVE_SPIN", False):
    pytest.skip("libfasttransforms has no spin entry points", allow_module_level=True)


def _random_g(L, rng):
    """A random complex (L+1, 2L+1) coefficient array (FSHT g/C layout)."""
    n, m = L + 1, 2 * L + 1
    return rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))


@pytest.mark.parametrize("L", [4, 8, 16])
@pytest.mark.parametrize("spin", [1, 2])
def test_spin_library_roundtrip(L, spin):
    """spinsph2fourier(fourier2spinsph(g, s), s) recovers g (plans are inverses)."""
    rng = np.random.default_rng(20260624 + L + spin)
    g = _random_g(L, rng)
    recovered = ft_sphere.spinsph2fourier(ft_sphere.fourier2spinsph(g, spin), spin)
    assert recovered.shape == g.shape
    err = np.max(np.abs(recovered - g))
    assert err < 1e-10, f"L={L} spin={spin}: max |diff| = {err:.2e}"


def test_spin_plan_cache():
    """A second call at the same (n, s) reuses the cached plan pointer."""
    rng = np.random.default_rng(1)
    g = _random_g(8, rng)
    ft_sphere.fourier2spinsph(g, 2)
    n = g.shape[0]
    p1 = ft_sphere._spin_plans[(n, 2)]
    ft_sphere.spinsph2fourier(g, 2)  # same (n, s) -> same plan
    p2 = ft_sphere._spin_plans[(n, 2)]
    assert p1 == p2


def test_spin0_matches_scalar():
    """spin=0 spin-transform == scalar fourier2sph (up to the legacy conj).

    Spin-0 spin-weighted harmonics ARE the ordinary spherical harmonics, so the
    spin path at s=0 must reproduce the scalar transform. ``fourier2sph`` carries
    the legacy ``np.conj`` (Julia-pipeline contract); the new spin path does not,
    so compare against the conjugate.
    """
    rng = np.random.default_rng(7)
    g = _random_g(8, rng)
    spin0 = ft_sphere.fourier2spinsph(g, 0)
    scalar = np.conj(ft_sphere.fourier2sph(g))
    err = np.max(np.abs(spin0 - scalar))
    assert err < 1e-10, f"spin0 vs scalar: max |diff| = {err:.2e}"
