"""In-process FastTransforms sphere transforms (replaces the Julia subprocess).

The HP2SPH FSHT stage needs Slevinsky's bivariate Fourier <-> spherical-harmonic
connection (``fourier2sph`` / ``sph2fourier``). Historically this shelled out to
Julia's ``FastTransforms.jl`` over a JSON pipe (``src/julia_sph*.jl``); this module
calls the SAME underlying C library, ``libfasttransforms``, directly via ctypes --
no subprocess, no JSON serialization, no second runtime.

Library resolution (NO hardcoded paths -- nothing tied to a Julia artifact):

* ``FASTTRANSFORMS_LIB`` env var = full path to the shared library, else
* ``ctypes.util.find_library("fasttransforms")``, else
* bare names (``libfasttransforms.{dylib,so}``) on the loader path.

The library AND its dependencies (FFTW, MPFR, OpenBLAS, OpenMP) must be loadable;
a clean build/package (e.g. built from MikaelSlevinsky/FastTransforms, or a distro
package) resolves those via its own rpath. If nothing loads, importing this module
raises ``ImportError`` and ``FSHT.py`` falls back to the Julia subprocess.

Numerics are verified bit-for-bit against the legacy Julia path (relerr 0). The
``np.conj`` below is deliberate: the old ``.jl`` scripts conjugated their input via
the ``'`` adjoint in ``reduce(vcat, complex_array')``, and the downstream
conversion (``FSHT.to_healpy_alm`` / ``convert_to_bivar_coeffs``) was calibrated
against that. ``fourier2sph`` is real-linear (run on the real and imaginary parts
separately), so ``conj(input) -> conj(output)`` and reproducing the conjugation on
the output is exactly equivalent. (A future cleanup could drop the conj here and
re-derive the conversion factors without it.)

C API used (from libfasttransforms; n = matrix rows = L+1, the coeff layout is
(n, 2n-1)):

    ft_plan_struct* ft_plan_sph2fourier(int n)              # one plan, both dirs
    void ft_execute_fourier2sph(char T, plan*, double* A, int N, int M)  # forward
    void ft_execute_sph2fourier(char T, plan*, double* A, int N, int M)  # inverse
    void ft_destroy_harmonic_plan(plan*)
"""

import ctypes
import ctypes.util
import os

import numpy as np

_TRANS_N = ord("N")  # 'N': no transpose (the only mode we need)


def _load_library():
    candidates = []
    env = os.environ.get("FASTTRANSFORMS_LIB")
    if env:
        candidates.append(env)
    found = ctypes.util.find_library("fasttransforms")
    if found:
        candidates.append(found)
    candidates += [
        "libfasttransforms.dylib",
        "libfasttransforms.so",
        "libfasttransforms.2.dylib",
    ]
    errors = []
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError as exc:  # not found / unresolved deps
            errors.append(f"  {name}: {exc}")
    raise ImportError(
        "Could not load libfasttransforms. Set FASTTRANSFORMS_LIB to its full path, "
        "or install it (with its FFTW/MPFR/OpenBLAS/OpenMP deps) on the loader path.\n"
        + "\n".join(errors)
    )


_lib = _load_library()
_lib.ft_plan_sph2fourier.restype = ctypes.c_void_p
_lib.ft_plan_sph2fourier.argtypes = [ctypes.c_int]
for _name in ("ft_execute_sph2fourier", "ft_execute_fourier2sph"):
    _fn = getattr(_lib, _name)
    _fn.restype = None
    _fn.argtypes = [
        ctypes.c_int,  # TRANS
        ctypes.c_void_p,  # plan
        ctypes.c_void_p,  # A (in place, column-major)
        ctypes.c_int,  # N = rows
        ctypes.c_int,  # M = cols
    ]
_lib.ft_destroy_harmonic_plan.argtypes = [ctypes.c_void_p]

_plans = {}  # n (= L+1 = matrix rows) -> plan pointer; reused across calls/directions


def _plan(n):
    p = _plans.get(n)
    if p is None:
        p = _lib.ft_plan_sph2fourier(n)
        _plans[n] = p
    return p


def _apply(symbol, A):
    """Run a real sphere-plan execute on a complex (L+1, 2L+1) matrix.

    The C routine is in-place, column-major and Float64, so we pass Fortran-ordered
    real/imaginary buffers separately. The plan only constrains the row count, so a
    single cached plan per ``n`` serves both transform directions.
    """
    A = np.ascontiguousarray(A, dtype=np.complex128)
    n, m = A.shape  # n = L+1 rows; m = 2L+1 = 2n-1 cols
    fn = getattr(_lib, symbol)
    p = _plan(n)
    out = np.empty_like(A)
    for part in ("real", "imag"):
        buf = np.asfortranarray(getattr(A, part), dtype=np.float64)
        fn(_TRANS_N, p, buf.ctypes.data_as(ctypes.c_void_p), n, m)
        setattr(out, part, buf)
    return np.conj(out)  # match the legacy Julia pipeline contract (see module docs)


def fourier2sph(g):
    """Bivariate Fourier-Chebyshev ``g`` coefficients -> spherical-harmonic ``C``."""
    return _apply("ft_execute_fourier2sph", g)


def sph2fourier(C):
    """Spherical-harmonic ``C`` -> bivariate Fourier-Chebyshev ``g`` coefficients."""
    return _apply("ft_execute_sph2fourier", C)
