"""Backend self-consistency for the in-process libfasttransforms FSHT stage.

``src/ft_sphere.py`` runs the FSHT stage through a direct ctypes call into
``libfasttransforms``. This checks the forward/inverse C transforms round-trip --
a pure backend check, independent of the alm conventions downstream.

Requires the C library; the whole module is skipped if it cannot be loaded.
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


def test_library_roundtrip(healpix_map):
    """sph2fourier(fourier2sph(g)) == g from the C library."""
    g = _build_g(healpix_map)
    # both directions are real-linear and carry the same conj, so it cancels here
    recovered = ft_sphere.sph2fourier(ft_sphere.fourier2sph(g))
    assert np.allclose(recovered, g, rtol=0, atol=1e-10), (
        f"max |diff| = {np.max(np.abs(recovered - g)):.2e}"
    )
