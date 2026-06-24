"""HP2SPH command-line entry point and FITS I/O helpers.

Run from the repo root, e.g.::

    python main.py path/to/sky_map.fits

The OpenMP guard and JAX float64 are enabled automatically on import (see
``src/_bootstrap.py``), so no environment variables need to be set by hand.
"""

import argparse  # noqa: E402
import time  # noqa: E402

# Import the package first: this sets the OpenMP env guards (before any library
# that links libomp loads) and enables JAX float64. Must precede healpy/numpy/jax.
from src import _bootstrap

_bootstrap.enable_x64()

import healpy as hp  # noqa: E402
import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402

from src.data_interpolation import (  # noqa: E402
    transform_healpix_to_grid,
    transform_grid_to_healpix,
)
from src.double_fourier_sphere import DFS, DFS_inverse  # noqa: E402
from src.nuFFT import apply_nuFFT, inverse_nuFFT  # noqa: E402
from src.FSHT import FSHT, inverse_FSHT  # noqa: E402


def save_to_fits(data, filename):
    """Save a complex numpy array to a FITS file (real + IMAGINARY HDUs)."""
    hdu_real = fits.PrimaryHDU(np.real(data))
    hdu_imag = fits.ImageHDU(np.imag(data), name="IMAGINARY")
    fits.HDUList([hdu_real, hdu_imag]).writeto(filename, overwrite=True)
    print(filename + " saved as FITS file")


def read_fits(filename):
    """Read a complex array saved by :func:`save_to_fits`."""
    with fits.open(filename) as hdul:
        return hdul[0].data + 1j * hdul["IMAGINARY"].data


def calc_cl(alm, lmax=None):
    if lmax is None:
        lmax = alm.shape[0] - 1

    # create ell array
    l_array = np.fft.fftfreq(alm.shape[1]) * alm.shape[1]
    l_array = np.fft.fftshift(l_array)
    sel = np.argsort(np.abs(l_array), kind="stable")
    l_array = np.abs(l_array[sel])
    l_array = np.full(alm.shape, l_array)
    for i in range(lmax + 1):
        l_array[i] += i

    # calculate cl
    cl = np.zeros(lmax + 1)
    for ell in range(lmax + 1):
        norm = 1.0 / (2.0 * ell + 1.0)
        alm_l = alm[(l_array == ell)]
        cl[ell] = np.sum(np.abs(alm_l) ** 2) * norm

    return cl


def forward(mp: np.array, save: bool = False) -> np.array:
    """Forward transform: HEALPix map -> spherical-harmonic coefficient array.

    ``mp`` is a 3-row (I, Q, U) map; only intensity (I) is transformed. Set
    ``save=True`` to also dump the result to ``alm_array_cg_<nside>.fits``.
    """
    assert mp.shape[0] == 3, (
        "Input map must have I, Q, U components"
    )  # temporary limit, Q and U do nothing
    NSIDE = hp.get_nside(mp[0])
    upsampled_map, fft_coeff = transform_healpix_to_grid(mp[0])

    start = time.time()
    _, fft_coeff_DFS = DFS(upsampled_map, fft_coeff)
    print("DFS time:", time.time() - start)

    start = time.time()
    fft_lat_nufft = apply_nuFFT(fft_coeff_DFS)
    print("nuFFT time:", time.time() - start)

    start = time.time()
    alm = FSHT(fft_lat_nufft)
    print("FSHT time:", time.time() - start)

    if save:
        save_to_fits(alm, "alm_array_cg_{0}.fits".format(NSIDE))

    return alm


def backward(alm: np.array, save: bool = False) -> np.array:
    """Inverse transform: coefficient array -> HEALPix map.

    ``alm``/``C`` is the FastTransforms triangular array of shape (L+1, 2*L+1).
    With the default compact band the internal band limit is L = lmax = 2*nside,
    so nside = (rows - 1) // 2. Set ``save=True`` to dump
    ``mp_array_cg_<nside>.fits``.
    """
    NSIDE = (alm.shape[0] - 1) // 2
    _, C = inverse_FSHT(alm, NSIDE)
    fft_lat = inverse_nuFFT(C)
    fft_coeff = DFS_inverse(fft_lat)
    mp = transform_grid_to_healpix(fft_coeff, fft_coeff)

    if save:
        hp.write_map("mp_array_cg_{0}.fits".format(NSIDE), mp, overwrite=True)
        print("mp_array_cg_{0}.fits saved".format(NSIDE))

    return mp


def main():
    parser = argparse.ArgumentParser(
        description="Run a HP2SPH forward (+ optional inverse) transform on a "
        "HEALPix map FITS file."
    )
    parser.add_argument(
        "maps",
        nargs="+",
        help="HEALPix map FITS file(s) with I, Q, U fields (Q/U are ignored).",
    )
    parser.add_argument(
        "--roundtrip",
        action="store_true",
        help="Also run the inverse transform back to a map.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write intermediate alm / map FITS files to the current directory.",
    )
    args = parser.parse_args()

    for filename in args.maps:
        print(filename, "being processed")
        mp = hp.read_map(filename, field=(0, 1, 2))
        alm = forward(mp, save=args.save)
        if args.roundtrip:
            backward(alm, save=args.save)
        print()


if __name__ == "__main__":
    main()
