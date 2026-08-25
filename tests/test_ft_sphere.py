"""Backend self-consistency for the in-process libfasttransforms FSHT stage.

``hp2sph/ft_sphere.py`` runs the FSHT stage through a direct ctypes call into
``libfasttransforms``. This checks the forward/inverse C transforms round-trip --
a pure backend check, independent of the alm conventions downstream.

Requires the C library; the whole module is skipped if it cannot be loaded.
"""

import importlib

import numpy as np
import pytest

from hp2sph.data_interpolation import transform_healpix_to_grid
from hp2sph.double_fourier_sphere import DFS
from hp2sph.nuFFT import apply_nuFFT

# ``src`` re-exports the ``FSHT`` function, shadowing the submodule attribute, so
# import the module object explicitly.
FSHT = importlib.import_module("hp2sph.FSHT")

# Skip the whole module if the C library can't be loaded.
ft_sphere = pytest.importorskip("hp2sph.ft_sphere")


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


@pytest.mark.parametrize("direction", ["fourier2sph", "sph2fourier"])
def test_overwrite_matches_the_allocating_path_bit_for_bit(healpix_map, direction):
    """``overwrite=True`` reuses the input buffer and must not change the answer.

    It exists only to drop a full-size array from the FSHT stage's peak, so
    equality here has to be exact rather than approximate -- anything else would
    mean the two paths do different arithmetic.
    """
    g = _build_g(healpix_map)
    fn = getattr(ft_sphere, direction)
    expected = fn(g.copy())
    scratch = g.copy()
    got = fn(scratch, overwrite=True)
    assert np.array_equal(got, expected)
    assert got is scratch, "overwrite=True must return the caller's own buffer"


def test_default_leaves_the_input_untouched(healpix_map):
    """Without ``overwrite`` the input survives; ``inverse_FSHT`` relies on this."""
    g = _build_g(healpix_map)
    before = g.copy()
    ft_sphere.fourier2sph(g)
    assert np.array_equal(g, before)


def test_non_contiguous_input_is_handled(healpix_map):
    """The input is only ever read through ``.real``/``.imag``, so strides are fine."""
    g = _build_g(healpix_map)
    padded = np.zeros((g.shape[0], g.shape[1] * 2), dtype=np.complex128)
    padded[:, ::2] = g
    view = padded[:, ::2]
    assert not view.flags["C_CONTIGUOUS"]
    assert np.array_equal(ft_sphere.fourier2sph(view), ft_sphere.fourier2sph(g))
