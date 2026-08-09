"""HP2SPH command-line entry point and FITS I/O helpers.

Run from the repo root, e.g.::

    python main.py path/to/sky_map.fits

The OpenMP guards are set automatically on import (see ``src/_bootstrap.py``),
so no environment variables need to be set by hand.
"""

import argparse  # noqa: E402
import logging  # noqa: E402

# Import the package first: this sets the OpenMP env guards, before any library
# that links libomp loads. Must precede healpy/numpy/finufft.
from src import _bootstrap  # noqa: F401

import healpy as hp  # noqa: E402
import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402

from src.pipeline import (  # noqa: E402
    forward_C,
    backward_map,
    nside_from_C,
)

logger = logging.getLogger(__name__)


def save_to_fits(data, filename):
    """Save a complex numpy array to a FITS file (real + IMAGINARY HDUs)."""
    hdu_real = fits.PrimaryHDU(np.real(data))
    hdu_imag = fits.ImageHDU(np.imag(data), name="IMAGINARY")
    fits.HDUList([hdu_real, hdu_imag]).writeto(filename, overwrite=True)
    logger.info("%s saved as FITS file", filename)


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


def forward(mp: np.array, save: bool = False, **nufft_kw) -> np.array:
    """Forward transform: HEALPix intensity map -> coefficient array ``C``.

    ``mp`` is a SINGLE HEALPix map. It used to be a 3-row (I, Q, U) stack of which
    only I was transformed, which meant every other caller bypassed this function;
    pass ``mp[0]`` if you have an IQU stack, and use
    ``src.spin_transform.forward_spin`` for polarization.

    ``nufft_kw`` goes to ``apply_nuFFT``. Set ``save=True`` to also dump the result
    to ``alm_array_cg_<nside>.fits``.
    """
    mp = np.asarray(mp)
    if mp.ndim != 1:
        raise ValueError(
            f"forward() takes a single HEALPix map, got shape {mp.shape}. "
            "Pass mp[0] for the intensity row of an (I, Q, U) stack, or use "
            "src.spin_transform.forward_spin for polarization."
        )
    C = forward_C(mp, **nufft_kw)
    if save:
        save_to_fits(C, "alm_array_cg_{0}.fits".format(hp.get_nside(mp)))
    return C


def backward(alm: np.array, save: bool = False) -> np.array:
    """Inverse transform: coefficient array -> HEALPix map.

    ``alm``/``C`` is the FastTransforms triangular array of shape (L+1, 2*L+1).
    With the default compact band the internal band limit is L = lmax = 2*nside,
    so nside = (rows - 1) // 2. Set ``save=True`` to dump
    ``mp_array_cg_<nside>.fits``.
    """
    NSIDE = nside_from_C(alm)
    mp = backward_map(alm, NSIDE)

    if save:
        hp.write_map("mp_array_cg_{0}.fits".format(NSIDE), mp, overwrite=True)
        logger.info("mp_array_cg_%d.fits saved", NSIDE)

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
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log per-stage debug timings.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    for filename in args.maps:
        logger.info("%s being processed", filename)
        mp = hp.read_map(filename, field=0)  # intensity only; see forward()
        C = forward(mp, save=args.save)
        if args.roundtrip:
            backward(C, save=args.save)


if __name__ == "__main__":
    main()
