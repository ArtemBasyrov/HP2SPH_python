from .src.data_interpolation import get_ring_indices, transform_grid_to_healpix, transform_healpix_to_grid, create_latitude_array, create_upsampled_grid
from .src.double_fourier_sphere import DFS, DFS_inverse
from .src.FSHT import FSHT
from .src.nuFFT import apply_nuFFT, inverse_nuFFT

# package version
__version__ = "0.1.0"
# package author
__author__ = "Artem Basyrov"

# Package documentation
__doc__ = """" 
"HP2SPH_python is a Python package for transforming HEALPix maps into spherical harmonics and vice versa.  \
"""

__all__ = [
    "get_ring_indices",
    "transform_grid_to_healpix",
    "transform_healpix_to_grid",
    "create_latitude_array",
    "create_upsampled_grid",
    "DFS",
    "DFS_inverse",
    "FSHT",
    "apply_nuFFT",
    "inverse_nuFFT"
]