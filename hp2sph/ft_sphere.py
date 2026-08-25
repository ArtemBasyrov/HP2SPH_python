"""In-process FastTransforms sphere transforms.

The HP2SPH FSHT stage needs Slevinsky's bivariate Fourier <-> spherical-harmonic
connection (``fourier2sph`` / ``sph2fourier``). This module calls the C library
``libfasttransforms`` directly via ctypes -- no subprocess, no second runtime.

Library resolution. The shared library is a native dependency (it is not on
PyPI), so it is discovered at import time, in order, from:

1. ``$FASTTRANSFORMS_LIB`` -- full path to the library (explicit override),
2. a ``lib/`` directory next to this package or at the repo root (drop or
   symlink the built ``libfasttransforms.{dylib,so}`` there for a self-contained
   checkout),
3. the active conda / virtualenv ``lib`` dir and the OS loader path
   (``ctypes.util.find_library``) -- where a system / conda-forge install lands,
4. a prebuilt ``FastTransforms.jl`` artifact under ``~/.julia`` if one happens to
   be present (just a precompiled binary; no Julia runtime is used).

So no environment variable is required when the library is installed in any of
the usual places; ``FASTTRANSFORMS_LIB`` remains available only as an override.
The library AND its dependencies (FFTW, MPFR, OpenBLAS, OpenMP) must be loadable;
a clean build/package resolves those via its own rpath. If nothing loads,
importing this module raises ``ImportError`` (the FSHT stage then errors with a
build/install hint -- see ``README.md`` / ``FSHT.py``).

The ``np.conj`` below is deliberate: the downstream conversion
(``FSHT.to_healpy_alm`` / ``convert_to_bivar_coeffs``) is calibrated against the
conjugated output. ``fourier2sph`` is real-linear (run on the real and imaginary
parts separately), so ``conj(input) -> conj(output)`` and reproducing the
conjugation on the output is exactly equivalent. (A future cleanup could drop the
conj here and re-derive the conversion factors without it.)

C API used (from libfasttransforms; n = matrix rows = L+1, the coeff layout is
(n, 2n-1)):

    ft_plan_struct* ft_plan_sph2fourier(int n)              # one plan, both dirs
    void ft_execute_fourier2sph(char T, plan*, double* A, int N, int M)  # forward
    void ft_execute_sph2fourier(char T, plan*, double* A, int N, int M)  # inverse
    void ft_destroy_harmonic_plan(plan*)
"""

import ctypes
import ctypes.util
import glob
import os
import sys

import numpy as np

from . import _openmp

_TRANS_N = ord("N")  # 'N': no transpose (the only mode we need)

_LIB_NAMES = (
    "libfasttransforms.dylib",
    "libfasttransforms.so",
    "libfasttransforms.2.dylib",
)


def _candidate_paths():
    """Yield candidate library paths/names, most specific first (see module docs)."""
    # 1. explicit override
    env = os.environ.get("FASTTRANSFORMS_LIB")
    if env:
        yield env

    # 2. a lib/ dir shipped with the checkout (next to this package, or repo root)
    here = os.path.dirname(os.path.abspath(__file__))
    local_dirs = [os.path.join(here, "lib"), os.path.join(here, os.pardir, "lib")]
    for d in local_dirs:
        for name in _LIB_NAMES:
            yield os.path.join(d, name)

    # 3. active conda / virtualenv lib dir, then the OS loader path
    prefixes = [
        sys.prefix,
        os.environ.get("CONDA_PREFIX"),
        os.environ.get("VIRTUAL_ENV"),
    ]
    for prefix in filter(None, prefixes):
        for name in _LIB_NAMES:
            yield os.path.join(prefix, "lib", name)
    found = ctypes.util.find_library("fasttransforms")
    if found:
        yield found

    # 4. a prebuilt FastTransforms.jl artifact binary, if present (no Julia is run)
    for name in _LIB_NAMES:
        yield from sorted(
            glob.glob(os.path.expanduser(f"~/.julia/artifacts/*/lib/{name}"))
        )

    # 5. bare names on the default loader path (last resort)
    yield from _LIB_NAMES


def _load_library():
    errors = []
    seen = set()
    for name in _candidate_paths():
        if name in seen:
            continue
        seen.add(name)
        try:
            return ctypes.CDLL(name)
        except OSError as exc:  # not found / unresolved deps
            errors.append(f"  {name}: {exc}")
    raise ImportError(
        "Could not load libfasttransforms. Build or install it (with its "
        "FFTW/MPFR/OpenBLAS/OpenMP deps) -- see README.md -- and put it on the "
        "loader path, in a lib/ dir next to the package, or point "
        "FASTTRANSFORMS_LIB at it.\n" + "\n".join(errors)
    )


def _preload_finufft():
    """Import finufft BEFORE libfasttransforms is loaded. Order is not optional.

    libfasttransforms links one OpenMP runtime (Homebrew's libomp in this checkout)
    and the finufft wheel vendors another under ``finufft/.dylibs/`` -- see
    ``hp2sph/_openmp.py`` for the full picture. Whichever initializes first claims the
    process-wide runtime state, and the two orders are NOT symmetric. Measured on
    macOS 24.6 / arm64, no OpenMP env vars set:

    * finufft imported first  -> both libraries run, repeatedly, no error;
    * libfasttransforms first -> SIGSEGV on the first ``ft_execute_*``.

    ``KMP_DUPLICATE_LIB_OK`` does not help. It only suppresses the ``OMP: Error #15``
    message, which is a different symptom from the crash.

    Importing here rather than in ``hp2sph/__init__.py`` puts the guard in the module
    that actually loads the offending library, so ``import hp2sph.FSHT`` and
    ``import hp2sph.ft_sphere`` are protected too, not just ``import hp2sph``.

    Failure is tolerated: finufft is a pipeline dependency, not an FSHT one, so a
    caller who only wants ``fourier2sph`` should not be blocked by its absence.
    """
    try:
        import finufft  # noqa: F401
    except ImportError:
        pass


_preload_finufft()
_lib = _load_library()


def _pin_threads():
    """Pin every loaded OpenMP runtime before entering libfasttransforms.

    See ``hp2sph/_openmp.py`` for why this is a correctness requirement and not a
    tuning knob: three vendored copies of libomp are loaded at once, and unless each
    is held to one thread the process crashes or hangs. Called per execute rather
    than once at import because ``omp_set_num_threads`` sets the CALLING thread's
    ICV; the scan behind it is cached, so the repeat cost is a few microseconds.
    """
    _openmp.pin()


_pin_threads()

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

# Spin-weighted plan + executes. Unlike the scalar routines (real double*, run on
# the real and imaginary parts separately), the spin routines take a genuinely
# complex interleaved array (ft_complex = double[2]) and an integer spin s.
_HAVE_SPIN = hasattr(_lib, "ft_plan_spinsph2fourier")
if _HAVE_SPIN:
    _lib.ft_plan_spinsph2fourier.restype = ctypes.c_void_p
    _lib.ft_plan_spinsph2fourier.argtypes = [ctypes.c_int, ctypes.c_int]
    for _name in ("ft_execute_spinsph2fourier", "ft_execute_fourier2spinsph"):
        _fn = getattr(_lib, _name)
        _fn.restype = None
        _fn.argtypes = [
            ctypes.c_int,  # TRANS
            ctypes.c_void_p,  # plan
            ctypes.c_void_p,  # A (in place, column-major, interleaved complex)
            ctypes.c_int,  # N = rows
            ctypes.c_int,  # M = cols
        ]
    _lib.ft_destroy_spin_harmonic_plan.argtypes = [ctypes.c_void_p]

# Equiangular FFTW synthesis/analysis on the sphere with spin. These map between
# function VALUES on the equiangular grid (theta_n = (2n+1)*pi/(2N), phi_m =
# 2*pi*m/M) and the bivariate Fourier coefficients that fourier2spinsph consumes --
# the library's own counterpart to HP2SPH's stages 1-3. They are the documented
# fallback for the spin path: bypass the hand-rolled DFS by
# resampling onto this grid. Only bound when the entry points exist.
_HAVE_SPIN_FFTW = _HAVE_SPIN and hasattr(_lib, "ft_plan_spinsph_analysis")
_FFTW_ESTIMATE = 1 << 6  # do not clobber the input array while planning
if _HAVE_SPIN_FFTW:
    for _name in ("ft_plan_spinsph_analysis", "ft_plan_spinsph_synthesis"):
        _fn = getattr(_lib, _name)
        _fn.restype = ctypes.c_void_p
        _fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
    for _name in ("ft_execute_spinsph_analysis", "ft_execute_spinsph_synthesis"):
        _fn = getattr(_lib, _name)
        _fn.restype = None
        _fn.argtypes = [
            ctypes.c_int,  # TRANS
            ctypes.c_void_p,  # plan
            ctypes.c_void_p,  # X (in place, column-major, interleaved complex)
            ctypes.c_int,  # N
            ctypes.c_int,  # M
        ]

_plans = {}  # n (= L+1 = matrix rows) -> plan pointer; reused across calls/directions
_fftw_plans = {}  # (kind, N, M, s) -> spinsphere fftw plan pointer
_spin_plans = {}  # (n, s) -> spin plan pointer; reused across calls/directions


def _plan(n):
    p = _plans.get(n)
    if p is None:
        p = _lib.ft_plan_sph2fourier(n)
        _plans[n] = p
    return p


def _spin_plan(n, s):
    p = _spin_plans.get((n, s))
    if p is None:
        if not _HAVE_SPIN:
            raise RuntimeError("libfasttransforms has no spin-weighted entry points")
        p = _lib.ft_plan_spinsph2fourier(n, s)
        _spin_plans[(n, s)] = p
    return p


def _apply(symbol, A, overwrite=False):
    """Run a real sphere-plan execute on a complex (L+1, 2L+1) matrix.

    The C routine is in-place, column-major and Float64, so we pass Fortran-ordered
    real/imaginary buffers separately. The plan only constrains the row count, so a
    single cached plan per ``n`` serves both transform directions.

    With ``overwrite=True`` the input doubles as the output buffer and is left
    undefined on return; pass it only for an array the caller will not read again.
    """
    A = np.asarray(A, dtype=np.complex128)
    n, m = A.shape  # n = L+1 rows; m = 2L+1 = 2n-1 cols
    fn = getattr(_lib, symbol)
    _pin_threads()
    p = _plan(n)
    # Every array live here is (L+1, 2L+1), and at high nside this function sets the
    # whole FSHT stage's peak RSS, so the count of them is the thing to watch. Two
    # were removed: one real Fortran scratch is now reused across the real and
    # imaginary passes rather than reallocated per part, and the conjugation is
    # written in place instead of allocating the result. Measured peak, as a
    # multiple of the input array: 5.04x before, 3.54x after, 2.54x with overwrite
    # (nside 2048, fresh subprocess per variant). Time is unchanged.
    out = A if overwrite else np.empty_like(A)
    buf = np.empty((n, m), dtype=np.float64, order="F")
    ptr = buf.ctypes.data_as(ctypes.c_void_p)
    for part in ("real", "imag"):
        # A.imag is still the original on the second pass: overwrite touches only
        # out.real on the first, and out is A.
        np.copyto(buf, getattr(A, part))
        fn(_TRANS_N, p, ptr, n, m)
        setattr(out, part, buf)
    # downstream conversion is calibrated against conj (module docs)
    return np.conjugate(out, out=out)


def fourier2sph(g, overwrite=False):
    """Bivariate Fourier-Chebyshev ``g`` coefficients -> spherical-harmonic ``C``.

    ``overwrite=True`` reuses ``g`` as the output buffer, which avoids holding a
    second array of the same shape -- worth several GB at nside 2048 and above.
    ``g`` is left undefined on return, so pass it only for a temporary.
    """
    return _apply("ft_execute_fourier2sph", g, overwrite=overwrite)


def sph2fourier(C, overwrite=False):
    """Spherical-harmonic ``C`` -> bivariate Fourier-Chebyshev ``g`` coefficients.

    ``overwrite=True`` reuses ``C`` as the output buffer, leaving it undefined on
    return; see :func:`fourier2sph`.
    """
    return _apply("ft_execute_sph2fourier", C, overwrite=overwrite)


def _apply_spin(symbol, A, spin):
    """Run a spin-weighted sphere-plan execute on a complex (L+1, 2L+1) matrix.

    The spin C routine is in-place, column-major and genuinely complex
    (ft_complex = double[2], i.e. interleaved re/im), so -- unlike the scalar
    ``_apply`` which runs on the real and imaginary parts separately -- we pass a
    single Fortran-ordered complex128 buffer whose memory layout already matches
    ft_complex. The plan is keyed by (rows, spin).

    The scalar ``np.conj`` contract is NOT applied here: the downstream spin
    conversion (Phase 4) is calibrated directly against this output.
    """
    A = np.asfortranarray(A, dtype=np.complex128)
    n, m = A.shape  # n = L+1 rows; m = 2L+1 = 2n-1 cols
    fn = getattr(_lib, symbol)
    _pin_threads()
    p = _spin_plan(n, spin)
    out = A.copy(order="F")  # operate in place on a copy
    fn(_TRANS_N, p, out.ctypes.data_as(ctypes.c_void_p), n, m)
    return out


def fourier2spinsph(g, spin):
    """Bivariate Fourier ``g`` coefficients -> spin-``spin`` spherical-harmonic ``C``."""
    return _apply_spin("ft_execute_fourier2spinsph", g, spin)


def spinsph2fourier(C, spin):
    """Spin-``spin`` spherical-harmonic ``C`` -> bivariate Fourier ``g`` coefficients."""
    return _apply_spin("ft_execute_spinsph2fourier", C, spin)


def _fftw_plan(kind, N, M, spin):
    key = (kind, N, M, spin)
    p = _fftw_plans.get(key)
    if p is None:
        if not _HAVE_SPIN_FFTW:
            raise RuntimeError("libfasttransforms has no spinsph FFTW entry points")
        p = getattr(_lib, f"ft_plan_spinsph_{kind}")(N, M, spin, _FFTW_ESTIMATE)
        _fftw_plans[key] = p
    return p


def _spinsph_fftw(kind, values_NM, spin):
    """Run the in-place spinsph FFTW transform on an ``(N, M)`` complex array.

    ``kind`` is ``"analysis"`` (grid values -> bivariate Fourier coefficients) or
    ``"synthesis"`` (the inverse). The C routine is in place, column-major and
    interleaved complex; the plan is keyed by ``(kind, N, M, spin)``.
    """
    A = np.asfortranarray(values_NM, dtype=np.complex128)
    N, M = A.shape
    out = A.copy(order="F")
    _pin_threads()
    p = _fftw_plan(kind, N, M, spin)
    getattr(_lib, f"ft_execute_spinsph_{kind}")(
        _TRANS_N, p, out.ctypes.data_as(ctypes.c_void_p), N, M
    )
    return out


def spinsph_analysis(values, spin):
    """Equiangular grid VALUES -> bivariate Fourier coefficients (spin ``spin``)."""
    return _spinsph_fftw("analysis", values, spin)


def spinsph_synthesis(coeffs, spin):
    """Bivariate Fourier coefficients -> equiangular grid VALUES (spin ``spin``)."""
    return _spinsph_fftw("synthesis", coeffs, spin)
