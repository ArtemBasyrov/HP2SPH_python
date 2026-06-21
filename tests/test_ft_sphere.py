"""The in-process libfasttransforms backend must match the Julia path exactly.

``src/ft_sphere.py`` replaces the Julia/JSON subprocess in the FSHT stage with a
direct ctypes call into ``libfasttransforms`` (the same C library FastTransforms.jl
wraps). These tests assert the two produce identical numbers -- a pure backend
equivalence check, independent of the alm conventions downstream.

Requires both the C library (skipped otherwise -- set ``FASTTRANSFORMS_LIB``) and
Julia / FastTransforms.jl (the reference; ``julia`` marker).
"""

import importlib

import numpy as np
import pytest

from src.data_interpolation import transform_healpix_to_grid
from src.double_fourier_sphere import DFS
from src.nuFFT import apply_nuFFT

# ``src`` re-exports the ``FSHT`` function, shadowing the submodule attribute, so
# import the module object explicitly.
FSHT = importlib.import_module("src.FSHT")

# Skip the whole module if the C library can't be loaded.
ft_sphere = pytest.importorskip("src.ft_sphere")


def _build_g(healpix_map):
    """Run the pipeline up to the FSHT input and return the (L+1, 2L+1) g array."""
    upsampled, fft_coeff = transform_healpix_to_grid(healpix_map)
    _, fft_coeff_dfs = DFS(upsampled, fft_coeff)
    fft_lat = apply_nuFFT(fft_coeff_dfs)
    return FSHT.preparation(fft_lat)


@pytest.mark.julia
def test_fourier2sph_matches_julia(healpix_map):
    """In-process fourier2sph == Julia julia_sph.jl, to machine precision."""
    g = _build_g(healpix_map)
    c_inproc = ft_sphere.fourier2sph(g)
    c_julia = np.array(FSHT.call_Julia(g, scriptname="src/julia_sph.jl"))
    assert c_inproc.shape == c_julia.shape
    assert np.allclose(c_inproc, c_julia, rtol=0, atol=1e-11), (
        f"max |diff| = {np.max(np.abs(c_inproc - c_julia)):.2e}"
    )


@pytest.mark.julia
def test_sph2fourier_matches_julia(healpix_map):
    """In-process sph2fourier == Julia julia_sph_inverse.jl, to machine precision."""
    g = _build_g(healpix_map)
    c = np.array(FSHT.call_Julia(g, scriptname="src/julia_sph.jl"))  # a valid C array
    f_inproc = ft_sphere.sph2fourier(c)
    f_julia = np.array(FSHT.call_Julia(c, scriptname="src/julia_sph_inverse.jl"))
    assert f_inproc.shape == f_julia.shape
    assert np.allclose(f_inproc, f_julia, rtol=0, atol=1e-11), (
        f"max |diff| = {np.max(np.abs(f_inproc - f_julia)):.2e}"
    )


def test_library_roundtrip(healpix_map):
    """sph2fourier(fourier2sph(g)) == g from the C library alone (no Julia needed)."""
    g = _build_g(healpix_map)
    # both directions are real-linear and carry the same conj, so it cancels here
    recovered = ft_sphere.sph2fourier(ft_sphere.fourier2sph(g))
    assert np.allclose(recovered, g, rtol=0, atol=1e-10), (
        f"max |diff| = {np.max(np.abs(recovered - g)):.2e}"
    )
