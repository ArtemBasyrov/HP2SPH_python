"""Uniform wrappers around HP2SPH, healpy and ducc0.

This is the only module that knows any library's calling convention. Every
benchmark drives backends through the four methods below, so adding a competitor
is a one-file change and no benchmark can accidentally compare two backends on
different work.

Every backend agrees on the interface:

* intensity   ``forward_I(mp, nside, lmax) -> alm``      / ``backward_I(alm, nside, lmax) -> mp``
* polarization ``forward_P(Q, U, nside, lmax) -> (aE, aB)`` / ``backward_P(aE, aB, nside, lmax) -> (Q, U)``

``alm``/``aE``/``aB`` are always healpy-ordered, healpy-normalized complex
coefficient arrays, and maps are always HEALPix RING. That is what makes the
comparison meaningful; the per-backend convention shims live here.

Conventions verified numerically against healpy before this module was written:

* ``ducc0.sht.synthesis`` on HEALPix ring geometry reproduces ``hp.alm2map`` to
  7e-15, and ``adjoint_synthesis`` with a uniform ``4*pi/npix`` ring factor
  reproduces ``hp.map2alm(iter=0, use_weights=False)`` to 6e-15.
* ``hp.alm2map([0, aE, aB], pol=True)`` is *bit-identical* to
  ``hp.alm2map_spin([aE, aB], spin=2)``, and ``hp.map2alm(..., pol=True)``
  returns E/B in the same sign convention as ``hp.map2alm_spin``. This matters
  because ``map2alm_spin`` accepts neither ``iter`` nor quadrature weights, so
  the weighted healpy variants have to go through the IQU path.
"""

import os

import numpy as np
import healpy as hp

from benchmarks.common import quiet
from hp2sph.FSHT import to_healpy_alm, from_healpy_alm
from hp2sph.spin_transform import forward_spin, backward_spin
from hp2sph.pipeline import ANALYSIS_EPS, forward_C, backward_map

# ``hp2sph.pipeline`` is the repo's side-effect-free wiring of the four pipeline
# stages -- the same composition ``main.forward`` wraps with FITS I/O and the same
# one the test suite pins. This used to import from ``tests.pipeline_helpers``,
# which put production code behind a test module.


# --------------------------------------------------------------------------- #
# Backend interface                                                            #
# --------------------------------------------------------------------------- #
# How many threads every backend is asked to use. Set by ``set_threads`` before any
# transform runs. 1 keeps the historical single-threaded comparison.
THREADS = 1


def set_threads(n):
    """Ask every backend to use ``n`` threads, and report what each can honour.

    The three families thread by different mechanisms, and only two of them can be
    set from Python at all:

    * ducc0 takes an explicit ``nthreads`` argument, so it is exact.
    * HP2SPH splits its NUFFT batch over ``HP2SPH_NUFFT_WORKERS`` Python threads and
      threads its FastTransforms stage through OpenMP.
    * healpy 1.20 exposes NO thread argument on ``map2alm``/``alm2map``; its bundled
      libsharp reads ``OMP_NUM_THREADS``, which ``hp2sph/_bootstrap`` pins to 1. Lifting
      that pin needs ``HP2SPH_OMP_THREADS`` set BEFORE ``src`` is imported, which is
      the process's business, not this function's.

    Returns a dict recording what was actually applied, so the run metadata says how
    each backend was configured rather than implying they all match.
    """
    global THREADS
    THREADS = max(1, int(n))
    os.environ["HP2SPH_NUFFT_WORKERS"] = str(THREADS)
    omp = os.environ.get("HP2SPH_OMP_THREADS")
    return {
        "requested": THREADS,
        "ducc0": THREADS,
        "hp2sph_nufft_workers": THREADS,
        "openmp": omp or "1 (pinned by hp2sph/_bootstrap)",
        "healpy": (
            omp
            if omp
            else "1 -- healpy 1.20 has no nthreads argument and OMP is pinned"
        ),
    }


class Backend:
    """One (library, configuration) pair.

    ``max_nside_I`` / ``max_nside_P`` cap a backend that does not scale (the
    ill-conditioned square-band HP2SPH solve). ``available_at`` is the hook for a
    backend whose prerequisites are resolution dependent (healpy pixel weights).
    """

    key = "?"
    label = "?"
    family = "?"
    kind = "?"
    max_nside_I = None
    max_nside_P = None

    def available_at(self, nside, channel):
        cap = self.max_nside_I if channel == "I" else self.max_nside_P
        if cap is not None and nside > cap:
            return False, f"capped at nside {cap}"
        return True, None

    def forward_I(self, mp, nside, lmax):
        raise NotImplementedError

    def backward_I(self, alm, nside, lmax):
        raise NotImplementedError

    def forward_P(self, Q, U, nside, lmax):
        raise NotImplementedError

    def backward_P(self, aE, aB, nside, lmax):
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# HP2SPH                                                                       #
# --------------------------------------------------------------------------- #
class HP2SPH(Backend):
    """This repository.

    ``nufft_kw`` is a function of ``nside`` because the square-band option's
    ``solve_modes`` is resolution dependent.
    """

    family = "hp2sph"

    def __init__(self, key, label, kind, nufft_kw=None, max_nside=None, channels="IP"):
        self.key = key
        self.label = label
        self.kind = kind
        self._nufft_kw = nufft_kw or (lambda nside: {})
        self.max_nside_I = max_nside
        self.max_nside_P = max_nside
        self.channels = channels

    def available_at(self, nside, channel):
        if channel not in self.channels:
            return False, f"{self.key} has no distinct {channel}-channel configuration"
        return super().available_at(nside, channel)

    def forward_I(self, mp, nside, lmax):
        with quiet():
            C = forward_C(mp, **self._nufft_kw(nside))
        return to_healpy_alm(C, lmax=lmax)

    def backward_I(self, alm, nside, lmax):
        # The synthesis band is the pipeline's compact grid band L = 2*nside
        # regardless of which band the *analysis* was configured to solve, so
        # the square-band variant shares this path.
        C = from_healpy_alm(alm, lmax, L=2 * nside)
        with quiet():
            return backward_map(C, nside)

    def forward_P(self, Q, U, nside, lmax):
        with quiet():
            return forward_spin(Q, U, lmax)

    def backward_P(self, aE, aB, nside, lmax):
        with quiet():
            return backward_spin(aE, aB, nside, lmax=lmax)

    def native_roundtrip_I(self, mp, nside, lmax):
        """``map -> C -> map`` without passing through healpy alm ordering.

        The cross-backend round trip has to route through healpy-ordered alm,
        because that is the only representation every library shares -- and
        ``to_healpy_alm`` reads only the triangular part of ``C``, discarding the
        quadrature residue the forward leaves in the tail cells. That discard is
        not free: it is what stops the square-band variant from showing its
        bit-exact interpolation property (measured 1.5e-13 here versus 2.2e-3
        through the alm conversion). This measures the pipeline in its own
        representation so that property is visible rather than merely asserted.

        Intensity only. The spin pipeline already exchanges healpy-ordered
        ``(aE, aB)`` at its boundary, so it has no separate native form.
        """
        with quiet():
            C = forward_C(mp, **self._nufft_kw(nside))
            return backward_map(C, nside)

    def stage_times_I(self, mp, nside, lmax):
        """Per-stage wall time of the scalar forward, for the profile plot.

        Deliberately re-composes the four stages instead of calling
        ``forward_C``: an end-to-end timer cannot say *which* stage costs what,
        and the stage split is the only part of the speed story that is
        actionable.
        """
        import time

        from hp2sph.data_interpolation import transform_healpix_to_grid
        from hp2sph.double_fourier_sphere import DFS, pole_stencil_rows
        from hp2sph.nuFFT import apply_nuFFT
        from hp2sph.FSHT import FSHT

        # Same route selection as pipeline.forward_C: the default compact CG band
        # goes half-domain, the square-band SVD variant needs the full sample set.
        kw = self._nufft_kw(nside)
        half = kw.get("solver", "cg") == "cg" and kw.get("solve_modes") is None
        if half:
            kw.setdefault("eps", ANALYSIS_EPS)

        out = {}
        with quiet():
            t = time.perf_counter()
            upsampled, fft_coeff = transform_healpix_to_grid(
                mp, map_rows=pole_stencil_rows(nside) if half else None
            )
            out["data_interpolation"] = time.perf_counter() - t

            t = time.perf_counter()
            _, dfs = DFS(upsampled, fft_coeff, spin=0, half=half)
            out["DFS"] = time.perf_counter() - t

            t = time.perf_counter()
            fft_lat = apply_nuFFT(dfs, spin=0, half_domain=half, **kw)
            out["nuFFT"] = time.perf_counter() - t

            t = time.perf_counter()
            FSHT(fft_lat)
            out["FSHT"] = time.perf_counter() - t
        return out

    def stage_times_P(self, Q, U, nside, lmax):
        """Per-stage wall time of the SPIN forward, at the SHIPPED solver settings.

        The settings come from ``spin_transform._spin_nufft_kw`` rather than being
        repeated here: an earlier version spelled out ``solver="cg"`` and left the
        tolerance and stopping rule at ``apply_nuFFT``'s defaults, which runs the
        latitude solve fully converged at ``eps=1e-12`` and reported a profile 3.3x
        slower than the transform it claims to describe.

        The spin path is not the scalar one with a different flag: it builds only the
        mirror-fundamental half of the DFS array, uses a sparse alias-fold plan, and
        its latitude stage is an iterative solve rather than a few CG steps. So it
        gets its own profile rather than being read off the scalar one.
        """
        import time

        from hp2sph.data_interpolation import transform_healpix_to_grid
        from hp2sph.double_fourier_sphere import DFS, dfs_fold_sparse, pole_stencil_rows
        from hp2sph.nuFFT import apply_nuFFT
        from hp2sph.FSHT import FSHT_spin
        from hp2sph.spin_transform import SPIN, _spin_nufft_kw

        out = {}
        z = np.asarray(Q) + 1j * np.asarray(U)
        with quiet():
            t = time.perf_counter()
            up, fc = transform_healpix_to_grid(z, map_rows=pole_stencil_rows(nside))
            out["data_interpolation"] = time.perf_counter() - t

            t = time.perf_counter()
            _, dfs = DFS(up, fc, spin=SPIN, half=True)
            out["DFS"] = time.perf_counter() - t

            t = time.perf_counter()
            kw = _spin_nufft_kw()
            plan = dfs_fold_sparse(nside, SPIN, kw.pop("alias_tol"))
            out["fold_plan"] = time.perf_counter() - t

            t = time.perf_counter()
            fft_lat = apply_nuFFT(dfs, fold=plan, spin=SPIN, half_domain=True, **kw)
            out["nuFFT"] = time.perf_counter() - t

            t = time.perf_counter()
            FSHT_spin(fft_lat, SPIN)
            out["FSHT"] = time.perf_counter() - t
        return out

    def stage_memory_P(self, Q, U, nside, lmax):
        """Peak RSS ABOVE the entry baseline after each stage of the spin forward.

        Cumulative, not per-stage: peak RSS is a high-water mark, so what a stage
        "costs" in isolation is not a well-defined quantity once earlier arrays are
        still live. The question this answers is how the high-water mark climbs, which
        is what decides whether a resolution fits at all.
        """
        import gc
        import resource

        from hp2sph.data_interpolation import transform_healpix_to_grid
        from hp2sph.double_fourier_sphere import DFS, dfs_fold_sparse, pole_stencil_rows
        from hp2sph.nuFFT import apply_nuFFT
        from hp2sph.FSHT import FSHT_spin
        from hp2sph.spin_transform import SPIN, _spin_nufft_kw

        def rss_mb():
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

        gc.collect()
        base = rss_mb()
        out = {}
        z = np.asarray(Q) + 1j * np.asarray(U)
        with quiet():
            up, fc = transform_healpix_to_grid(z, map_rows=pole_stencil_rows(nside))
            out["data_interpolation"] = rss_mb() - base
            _, dfs = DFS(up, fc, spin=SPIN, half=True)
            del up, fc
            out["DFS"] = rss_mb() - base
            kw = _spin_nufft_kw()
            plan = dfs_fold_sparse(nside, SPIN, kw.pop("alias_tol"))
            out["fold_plan"] = rss_mb() - base
            fft_lat = apply_nuFFT(dfs, fold=plan, spin=SPIN, half_domain=True, **kw)
            out["nuFFT"] = rss_mb() - base
            FSHT_spin(fft_lat, SPIN)
            out["FSHT"] = rss_mb() - base
        return out

    def stage_memory_I(self, mp, nside, lmax):
        """The scalar counterpart of ``stage_memory_P``."""
        import gc
        import resource

        from hp2sph.data_interpolation import transform_healpix_to_grid
        from hp2sph.double_fourier_sphere import DFS, pole_stencil_rows
        from hp2sph.nuFFT import apply_nuFFT
        from hp2sph.FSHT import FSHT

        def rss_mb():
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

        # The square-band variant solves the full latitude sample set; the default
        # compact CG band takes the half-domain route, as pipeline.forward_C does.
        kw = self._nufft_kw(nside)
        half = kw.get("solver", "cg") == "cg" and kw.get("solve_modes") is None
        if half:
            kw.setdefault("eps", ANALYSIS_EPS)

        gc.collect()
        base = rss_mb()
        out = {}
        with quiet():
            upsampled, fft_coeff = transform_healpix_to_grid(
                mp, map_rows=pole_stencil_rows(nside) if half else None
            )
            out["data_interpolation"] = rss_mb() - base
            _, dfs = DFS(upsampled, fft_coeff, spin=0, half=half)
            del upsampled, fft_coeff
            out["DFS"] = rss_mb() - base
            fft_lat = apply_nuFFT(dfs, spin=0, half_domain=half, **kw)
            out["nuFFT"] = rss_mb() - base
            FSHT(fft_lat)
            out["FSHT"] = rss_mb() - base
        return out


# --------------------------------------------------------------------------- #
# healpy                                                                       #
# --------------------------------------------------------------------------- #
_PIXEL_WEIGHT_CACHE = {}


def pixel_weights_available(nside):
    """Whether healpy actually has full pixel weights for this ``nside``.

    healpy-data only ships them for nside >= 32, and ``use_pixel_weights=True``
    below that **silently falls back to no weights at all** after a failed
    download. Reporting that fallback under a "pixel weights" label would be a
    fabricated row, so every cell is probed: run the transform both ways and see
    whether the results are bit-identical. Cached per nside.
    """
    if nside in _PIXEL_WEIGHT_CACHE:
        return _PIXEL_WEIGHT_CACHE[nside]
    import warnings

    rng = np.random.default_rng(0)
    mp = rng.standard_normal(hp.nside2npix(nside))
    lmax = min(2 * nside, 32)
    with warnings.catch_warnings(), quiet():
        warnings.simplefilter("ignore")
        try:
            a_pix = hp.map2alm(mp, lmax=lmax, iter=0, use_pixel_weights=True, pol=False)
        except Exception:
            _PIXEL_WEIGHT_CACHE[nside] = False
            return False
        a_none = hp.map2alm(mp, lmax=lmax, iter=0, use_weights=False, pol=False)
    ok = not np.array_equal(a_pix, a_none)
    _PIXEL_WEIGHT_CACHE[nside] = ok
    return ok


class Healpy(Backend):
    """healpy (its own bundled libsharp -- this build does not use ducc0).

    ``spin_route="spin"`` uses ``map2alm_spin``, the direct spin-2 analysis with
    no weighting and no iteration. ``spin_route="pol"`` goes through the IQU
    ``pol=True`` path, which is the only healpy route that accepts quadrature
    weights and iteration for polarization -- at the cost of also transforming an
    all-zero intensity map, which its polarization timings therefore include.
    """

    family = "healpy"

    def __init__(
        self,
        key,
        label,
        kind,
        iterations=0,
        use_weights=False,
        use_pixel_weights=False,
        spin_route="pol",
    ):
        self.key = key
        self.label = label
        self.kind = kind
        self.iterations = iterations
        self.use_weights = use_weights
        self.use_pixel_weights = use_pixel_weights
        self.spin_route = spin_route

    def available_at(self, nside, channel):
        ok, why = super().available_at(nside, channel)
        if not ok:
            return ok, why
        if self.use_pixel_weights and not pixel_weights_available(nside):
            return False, f"healpy ships no pixel weights for nside {nside}"
        return True, None

    def _map2alm(self, maps, lmax, pol):
        import warnings

        with warnings.catch_warnings(), quiet():
            warnings.simplefilter("ignore")
            return hp.map2alm(
                maps,
                lmax=lmax,
                iter=self.iterations,
                use_weights=self.use_weights,
                use_pixel_weights=self.use_pixel_weights,
                pol=pol,
            )

    def forward_I(self, mp, nside, lmax):
        return self._map2alm(mp, lmax, pol=False)

    def backward_I(self, alm, nside, lmax):
        return hp.alm2map(alm, nside=nside, lmax=lmax, pol=False)

    def forward_P(self, Q, U, nside, lmax):
        if self.spin_route == "spin":
            aE, aB = hp.map2alm_spin([Q, U], 2, lmax=lmax)
            return aE, aB
        zero = np.zeros_like(Q)
        _, aE, aB = self._map2alm([zero, Q, U], lmax, pol=True)
        return aE, aB

    def backward_P(self, aE, aB, nside, lmax):
        # Bit-identical to alm2map_spin (verified), so both spin_route settings
        # share one synthesis and the round trip stays self-consistent.
        Q, U = hp.alm2map_spin([aE, aB], nside, 2, lmax)
        return Q, U


# --------------------------------------------------------------------------- #
# ducc0                                                                        #
# --------------------------------------------------------------------------- #
_SHT_INFO_CACHE = {}


def _healpix_sht_info(nside):
    """HEALPix RING geometry in the form ducc0's generic SHT entry points want."""
    if nside not in _SHT_INFO_CACHE:
        import ducc0

        _SHT_INFO_CACHE[nside] = ducc0.healpix.Healpix_Base(nside, "RING").sht_info()
    return _SHT_INFO_CACHE[nside]


class Ducc0(Backend):
    """ducc0's spherical harmonic transforms on HEALPix ring geometry.

    Two analyses, which are genuinely different algorithms rather than tunings:

    * ``mode="adjoint"`` -- ``adjoint_synthesis`` with the uniform ``4*pi/npix``
      pixel area as the ring factor. A single quadrature pass; the direct
      counterpart of ``hp.map2alm(iter=0, use_weights=False)``, which it matches
      to 6e-15, so the two serve as a cross-check on this harness.
    * ``mode="pseudo"`` -- ``pseudo_analysis``, an iterative (LSMR) least-squares
      solve for the coefficients whose synthesis best reproduces the map. This is
      ducc0's accurate analysis on a grid without exact quadrature weights, and
      the real competitor to HP2SPH.
    """

    family = "ducc0"

    def __init__(self, key, label, kind, mode, maxiter=20, epsilon=1e-6):
        self.key = key
        self.label = label
        self.kind = kind
        self.mode = mode
        self.maxiter = maxiter
        self.epsilon = epsilon
        self.last_iterations = None

    def _analysis(self, maps, nside, lmax, spin):
        import ducc0

        info = _healpix_sht_info(nside)
        if self.mode == "adjoint":
            w = 4.0 * np.pi / maps.shape[-1]
            return ducc0.sht.adjoint_synthesis(
                map=maps,
                lmax=lmax,
                spin=spin,
                nthreads=THREADS,
                ringfactor=np.full(info["theta"].size, w),
                **info,
            )
        out = ducc0.sht.pseudo_analysis(
            map=maps,
            lmax=lmax,
            spin=spin,
            nthreads=THREADS,
            maxiter=self.maxiter,
            epsilon=self.epsilon,
            **info,
        )
        alm = out[0] if isinstance(out, tuple) else out
        if isinstance(out, tuple) and len(out) > 2:
            # (alm, istop, itn, ...) -- keep the iteration count so the speed
            # numbers can be read against how hard the solver actually worked.
            self.last_iterations = int(out[2])
        return alm

    def _synthesis(self, alm, nside, lmax, spin):
        import ducc0

        return ducc0.sht.synthesis(
            alm=alm,
            lmax=lmax,
            spin=spin,
            nthreads=THREADS,
            **_healpix_sht_info(nside),
        )

    def forward_I(self, mp, nside, lmax):
        maps = np.ascontiguousarray(np.asarray(mp, dtype=np.float64)).reshape(1, -1)
        return self._analysis(maps, nside, lmax, spin=0)[0]

    def backward_I(self, alm, nside, lmax):
        alm = np.ascontiguousarray(np.asarray(alm, dtype=np.complex128)).reshape(1, -1)
        return self._synthesis(alm, nside, lmax, spin=0)[0]

    def forward_P(self, Q, U, nside, lmax):
        maps = np.ascontiguousarray(np.asarray([Q, U], dtype=np.float64))
        alm = self._analysis(maps, nside, lmax, spin=2)
        return alm[0], alm[1]

    def backward_P(self, aE, aB, nside, lmax):
        alm = np.ascontiguousarray(np.asarray([aE, aB], dtype=np.complex128))
        out = self._synthesis(alm, nside, lmax, spin=2)
        return out[0], out[1]


# --------------------------------------------------------------------------- #
# Registry                                                                     #
# --------------------------------------------------------------------------- #
ALL_BACKENDS = [
    HP2SPH("hp2sph", "HP2SPH (compact band)", kind="hp2sph"),
    HP2SPH(
        "hp2sph-square",
        "HP2SPH (square band, SVD)",
        kind="hp2sph",
        nufft_kw=lambda nside: {"solver": "svd", "solve_modes": 8 * nside + 1},
        # O(nside^3) and ill-conditioned; the latitude Vandermonde hits 1/eps by
        # nside ~128, so running it higher would report noise as a measurement.
        max_nside=64,
        # Intensity only: forward_spin builds its own latitude operator (the alias
        # fold, which the spin path requires) and ignores these nuFFT options, so a
        # P-channel entry here would silently duplicate plain hp2sph.
        channels="I",
    ),
    Healpy(
        "healpy-plain",
        "healpy (no weights)",
        kind="single-pass",
        iterations=0,
        spin_route="spin",
    ),
    Healpy(
        "healpy-ring",
        "healpy (ring weights)",
        kind="single-pass",
        iterations=0,
        use_weights=True,
    ),
    Healpy(
        "healpy-pixel",
        "healpy (pixel weights)",
        kind="single-pass",
        iterations=0,
        use_pixel_weights=True,
    ),
    Healpy(
        "healpy-iter3",
        "healpy (iter=3, default)",
        kind="iterative",
        iterations=3,
        use_weights=True,
    ),
    Ducc0(
        "ducc0-adjoint", "ducc0 (adjoint synthesis)", kind="single-pass", mode="adjoint"
    ),
    Ducc0("ducc0-pseudo", "ducc0 (pseudo analysis)", kind="iterative", mode="pseudo"),
]

BACKENDS = {b.key: b for b in ALL_BACKENDS}

DEFAULT_KEYS = [b.key for b in ALL_BACKENDS]


def select(keys=None):
    """Resolve backend keys to objects, preserving the registry's order."""
    if not keys:
        return list(ALL_BACKENDS)
    unknown = [k for k in keys if k not in BACKENDS]
    if unknown:
        raise SystemExit(f"unknown backend(s) {unknown}; available: {sorted(BACKENDS)}")
    return [b for b in ALL_BACKENDS if b.key in set(keys)]


# --------------------------------------------------------------------------- #
# Plot styling                                                                 #
# --------------------------------------------------------------------------- #
# Colour encodes the LIBRARY, linestyle and marker encode the configuration
# within it. Seven-plus arbitrary hues would not survive a colour-vision check;
# three do, with room to spare (worst all-pairs CVD dE 9.2, normal-vision 24.0).
FAMILY_COLOR = {
    "hp2sph": "#2a78d6",  # blue
    "healpy": "#eb6834",  # orange
    "ducc0": "#1baf7a",  # aqua
}

VARIANT_STYLE = {
    "hp2sph": ("-", "o"),
    "hp2sph-square": ("--", "D"),
    "healpy-plain": ("--", "s"),
    "healpy-ring": ("-", "v"),
    "healpy-pixel": (":", "^"),
    "healpy-iter3": ("-.", "P"),
    "ducc0-adjoint": ("--", "X"),
    "ducc0-pseudo": ("-", "*"),
}


def style_for(key):
    """``(color, linestyle, marker)`` for a backend key."""
    family = BACKENDS[key].family if key in BACKENDS else "hp2sph"
    ls, marker = VARIANT_STYLE.get(key, ("-", "o"))
    return FAMILY_COLOR.get(family, "#52514e"), ls, marker


def label_for(key):
    return BACKENDS[key].label if key in BACKENDS else key
