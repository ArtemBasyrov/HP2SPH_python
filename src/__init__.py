# Set the OpenMP guard and enable JAX float64 before any numerical library loads.
from . import _bootstrap  # noqa: F401  (sets KMP_DUPLICATE_LIB_OK on import)

_bootstrap.enable_x64()

from .data_interpolation import (  # noqa: E402
    get_ring_indices,
    transform_grid_to_healpix,
    transform_healpix_to_grid,
    create_latitude_array,
    create_upsampled_grid,
)
from .double_fourier_sphere import DFS, DFS_inverse  # noqa: E402
from .FSHT import FSHT, inverse_FSHT  # noqa: E402
from .nuFFT import apply_nuFFT, inverse_nuFFT  # noqa: E402

__all__ = [
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
