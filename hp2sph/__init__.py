"""HP2SPH -- fast HEALPix map <-> spherical-harmonic (``alm``) transforms.

Implements the algorithm of K. P. Drake & G. B. Wright, "A Fast and Accurate
Algorithm for Spherical Harmonic Analysis on HEALPix Grids with Applications to
the Cosmic Microwave Background Radiation" (arXiv:1904.10514), for both the
scalar intensity and the spin-2 polarization channels.

The transform is routed through a structured latitude-longitude grid in four
stages -- ring FFTs, a Double Fourier Sphere, a latitude non-uniform FFT, and
Slevinsky's fast spherical-harmonic transform -- each of which is also exposed
individually.
"""

# Set the OpenMP guards before any numerical library loads. This import must stay
# first: libomp reads its thread count when the image loads (see hp2sph/_openmp.py).
from . import _bootstrap  # noqa: F401  (sets KMP_DUPLICATE_LIB_OK on import)

from .data_interpolation import (  # noqa: E402
    get_ring_indices,
    transform_grid_to_healpix,
    transform_healpix_to_grid,
    create_latitude_array,
    create_upsampled_grid,
)
from .double_fourier_sphere import DFS, DFS_inverse  # noqa: E402
from .FSHT import FSHT, inverse_FSHT, to_healpy_alm  # noqa: E402
from .nuFFT import apply_nuFFT, inverse_nuFFT  # noqa: E402
from .pipeline import (  # noqa: E402
    forward_C,
    forward_alm,
    backward_map,
    nside_from_C,
)
from .spin_transform import forward_spin, backward_spin  # noqa: E402

__version__ = "0.1.0"
__author__ = "Artem Basyrov"

__all__ = [
    # high-level API
    "forward_C",
    "forward_alm",
    "backward_map",
    "nside_from_C",
    "forward_spin",
    "backward_spin",
    "to_healpy_alm",
    # individual stages
    "get_ring_indices",
    "transform_grid_to_healpix",
    "transform_healpix_to_grid",
    "create_latitude_array",
    "create_upsampled_grid",
    "DFS",
    "DFS_inverse",
    "FSHT",
    "inverse_FSHT",
    "apply_nuFFT",
    "inverse_nuFFT",
]
