"""The two numpy stages split their rows across threads; the split must change nothing.

``transform_healpix_to_grid`` and the half-domain ``DFS`` hand contiguous row blocks to
a thread pool. The blocks are disjoint views, and numpy's pocketfft releases the GIL, so
the threaded result is not merely close to the serial one -- it is the same arithmetic
in the same order, and equality is what these tests assert.
"""

import numpy as np
import pytest

import src.data_interpolation as di
import src.double_fourier_sphere as dfs_mod
from src.double_fourier_sphere import DFS, _shifted_into, pole_stencil_rows


@pytest.fixture
def forced_workers(monkeypatch):
    """Force a worker count on both stages regardless of the size gate."""

    def apply(workers):
        monkeypatch.setattr(di, "default_workers", lambda n=None: workers)
        monkeypatch.setattr(dfs_mod, "default_workers", lambda n=None: workers)
        monkeypatch.setattr(di, "_MIN_THREADED_RINGS", 0)
        monkeypatch.setattr(dfs_mod, "_MIN_THREADED_RINGS", 0)

    return apply


@pytest.mark.parametrize("shape", [(3, 8), (1, 16), (5, 12), (2, 7)])
def test_shifted_into_matches_fftshift(shape, rng):
    """The shift folded into the copy is exactly ``fftshift`` along the column axis."""
    src = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    dst = np.empty_like(src)
    _shifted_into(dst, src)
    assert np.array_equal(dst, np.fft.fftshift(src, axes=1))


def test_transform_threaded_matches_serial(nside, healpix_map, forced_workers):
    forced_workers(1)
    up_s, fc_s = di.transform_healpix_to_grid(healpix_map)
    forced_workers(4)
    up_t, fc_t = di.transform_healpix_to_grid(healpix_map)
    assert np.array_equal(fc_s, fc_t)
    assert np.array_equal(up_s, up_t)


def test_half_dfs_threaded_matches_serial(nside, healpix_map, forced_workers):
    _, fft_coeff = di.transform_healpix_to_grid(
        healpix_map, map_rows=pole_stencil_rows(nside)
    )
    up, _ = di.transform_healpix_to_grid(healpix_map, map_rows=pole_stencil_rows(nside))
    forced_workers(1)
    _, serial = DFS(up, fft_coeff, spin=0, half=True)
    forced_workers(4)
    _, threaded = DFS(up, fft_coeff, spin=0, half=True)
    assert np.array_equal(serial, threaded)


def test_half_dfs_still_returns_the_shifted_array(nside, healpix_map):
    """Folding the shift into the copy leaves the array it used to build.

    The rows are assembled and then column-shifted; doing the shift during the copy
    saves a second full-size array, so the equivalence is asserted rather than assumed.
    """
    up, fft_coeff = di.transform_healpix_to_grid(
        healpix_map, map_rows=pole_stencil_rows(nside)
    )
    _, got = DFS(up, fft_coeff, spin=0, half=True)

    n_rings, n_lon = fft_coeff.shape
    north, south = dfs_mod._pole_stencils(up, 0)
    ref = np.empty((n_rings + 2, n_lon), dtype=complex)
    ref[0] = np.fft.fft(north, n=n_lon, norm="forward")
    ref[1 : n_rings + 1] = fft_coeff
    ref[n_rings + 1] = np.fft.fft(south, n=n_lon, norm="forward")
    assert np.array_equal(got, np.fft.fftshift(ref, axes=1))
