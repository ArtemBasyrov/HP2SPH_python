"""Step 2's dense n x n connection matrix against the HODLR one the library ships.

`ft_plan_sph2fourier` calls the file-static DENSE `plan_legendre_to_chebyshev`
(four n x n arrays, of which only the upper triangle is ever touched). The library
also exports `ft_plan_legendre_to_chebyshev`, which returns a compressed
`ft_tb_eigen_FMM *`; `ft_summary_size_tb_eigen_FMM` reports its byte size.

Usage: python -u tools/fsht_diagnostics/f3_hodlr_size.py <n> [<n> ...]
"""

import ctypes, sys, time

sys.path.insert(0, ".")
from src import ft_sphere as fts  # noqa: E402

MB = 1 << 20
lib = fts._lib
lib.ft_plan_legendre_to_chebyshev.restype = ctypes.c_void_p
lib.ft_plan_legendre_to_chebyshev.argtypes = [ctypes.c_int] * 3
lib.ft_summary_size_tb_eigen_FMM.restype = ctypes.c_size_t
lib.ft_summary_size_tb_eigen_FMM.argtypes = [ctypes.c_void_p]

print(f"{'n':>8} {'dense MB':>10} {'HODLR MB':>10} {'ratio':>8} {'build s':>9}")
for a in sys.argv[1:]:
    n = int(a)
    t = time.perf_counter()
    F = lib.ft_plan_legendre_to_chebyshev(1, 0, n)
    dt = time.perf_counter() - t
    h = lib.ft_summary_size_tb_eigen_FMM(F) / MB
    dense = n * n * 8 / MB  # what the plan allocates, per matrix
    tri = dense / 2  # what it actually touches (upper triangle)
    print(f"{n:8d} {tri:10.1f} {h:10.1f} {tri / h:8.2f}x {dt:9.3f}")
