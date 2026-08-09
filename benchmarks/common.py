"""Shared machinery for the benchmark suite: setup, signals, metrics, result IO.

Everything here is library-agnostic. Anything that knows how to *call* HP2SPH,
healpy or ducc0 lives in :mod:`benchmarks.backends`.

**Import this module first in every benchmark script.** It runs ``src._bootstrap``
(the OpenMP guards) and enables JAX float64 before healpy / finufft /
libfasttransforms can load. Importing it late is not merely untidy: JAX silently
falls back to float32 and the transforms diverge, and on macOS the duplicate
libomp aborts the process with ``OMP: Error #15``.
"""

import contextlib
import io
import json
import os
import platform
import sys
import threading
import time
from datetime import datetime, timezone

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Must precede every numerical import below (see module docstring).
from src import _bootstrap  # noqa: E402, F401

import numpy as np  # noqa: E402
import healpy as hp  # noqa: E402

RESULTS_DIR = os.path.join(_REPO_ROOT, "benchmarks", "results")
FIGURES_DIR = os.path.join(_REPO_ROOT, "benchmarks", "figures")


# --------------------------------------------------------------------------- #
# Process hygiene                                                              #
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def quiet():
    """Swallow stdout.

    Kept as a backstop only. ``src.data_interpolation`` and ``src.nuFFT`` used to
    print per-call timing and solver diagnostics into the measured region; both now
    use ``logging``, so this catches nothing of ours. It still guards against a
    third-party library printing inside a timed call.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


class PeakRSS:
    """Sample the process RSS on a background thread; report the peak.

    ``resource.getrusage`` only exposes a monotone high-water mark for the whole
    process, which is useless once a large ``nside`` has already run. Sampling
    gives a per-call figure instead. Falls back to a single before/after reading
    when ``psutil`` is unavailable, which understates short-lived spikes.

    **This is a lower bound on the working set, not a measurement of it.** It
    reports how far RSS *grew* during the call, so once the allocator is holding
    enough previously freed memory to satisfy the call it reads 0 -- which is
    what happens for every backend at the top of a run's nside ladder. Read it
    as "this call forced the process to grow by at least this much". A true
    per-call figure needs each cell in its own process, which is not worth the
    machinery here: memory is a secondary field, recorded in the JSON and not
    plotted or quoted.
    """

    def __init__(self, interval=0.005):
        self.interval = interval
        self.peak = 0
        self.baseline = 0
        self._stop = threading.Event()
        self._thread = None
        try:
            import psutil

            self._proc = psutil.Process()
        except ImportError:
            self._proc = None

    def _rss(self):
        if self._proc is None:
            import resource

            raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # macOS reports bytes, Linux kilobytes.
            return raw if sys.platform == "darwin" else raw * 1024
        return self._proc.memory_info().rss

    def _run(self):
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, self._rss())

    def __enter__(self):
        self.baseline = self._rss()
        self.peak = self.baseline
        if self._proc is not None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=1.0)
        self.peak = max(self.peak, self._rss())
        return False

    @property
    def delta_mb(self):
        return (self.peak - self.baseline) / 1024**2


def time_call(fn, repeats=3, warmup=1):
    """Time a zero-argument callable; return ``(last_result, stats)``.

    The warm-up calls are discarded: the first call through any of these
    backends pays one-off costs (FFTW plans, the FastTransforms rotation
    precompute, JAX tracing, first-touch page faults) that are not part of the
    steady-state cost being compared. ``min`` is the headline -- it is the least
    contaminated by scheduler noise -- with the median and spread kept so a
    noisy machine is visible in the record rather than hidden.
    """
    for _ in range(max(0, warmup)):
        result = fn()
    times = []
    with PeakRSS() as rss:
        for _ in range(repeats):
            t0 = time.perf_counter()
            result = fn()
            times.append(time.perf_counter() - t0)
    times = np.asarray(times, dtype=float)
    stats = {
        "t_min": float(times.min()),
        "t_median": float(np.median(times)),
        "t_max": float(times.max()),
        "repeats": int(repeats),
        "peak_rss_mb": float(rss.delta_mb),
    }
    return result, stats


# --------------------------------------------------------------------------- #
# Test signals                                                                 #
# --------------------------------------------------------------------------- #
def random_alm(lmax, seed, slope=0.0, mmax_cap=None, lmin=0):
    """Random ``alm`` with amplitude spectrum ``sqrt(C_l) ~ (1+l)^-slope``.

    ``slope=0`` is flat -- white noise on the sphere, maximally rough, whose
    aliasing tail swamps the quadrature differences between methods.
    ``slope=1.5`` is a smooth field, which is the regime Drake & Wright's test
    functions live in and where each method's quadrature accuracy shows through.

    ``mmax_cap`` zeroes all ``m`` above the cap. Passing ``2*nside-1`` keeps
    ``hp.alm2map`` an *exact* sampler of the function: the nonzero azimuthal
    content stays strictly below the HEALPix grid's longitude Nyquist, so the
    map carries no synthesis aliasing and every recovered coefficient can be
    compared against a known truth. The array stays standard-packed, so
    ``hp.Alm.getidx`` works normally.
    """
    rng = np.random.default_rng(seed)
    ncoeff = hp.Alm.getsize(lmax)
    alm = rng.standard_normal(ncoeff) + 1j * rng.standard_normal(ncoeff)
    ells, ems = hp.Alm.getlm(lmax, np.arange(ncoeff))
    alm *= (1.0 + ells.astype(float)) ** (-slope)
    alm[ems == 0] = alm[ems == 0].real  # reality of the field
    if mmax_cap is not None:
        alm[ems > mmax_cap] = 0.0
    alm[ells < lmin] = 0.0
    return alm.astype(np.complex128)


def random_EB(lmax, seed, slope=0.0, mmax_cap=None):
    """A random polarization sky ``(aE, aB)``.

    ``l < 2`` is zeroed: a spin-2 field has no monopole or dipole, and healpy's
    ``alm2map_spin`` ignores those coefficients, so leaving them nonzero would
    put unrecoverable power in the "truth" and show up as a spurious error.
    """
    aE = random_alm(lmax, seed, slope=slope, mmax_cap=mmax_cap, lmin=2)
    aB = random_alm(lmax, seed + 10_000, slope=slope, mmax_cap=mmax_cap, lmin=2)
    return aE, aB


def restrict_to_band(alm_full, lmax_full, lmax_band):
    """The in-band coefficients of a higher-``lmax`` alm, in ``lmax_band`` packing.

    These are the *true* coefficients an analysis at ``lmax_band`` should return.
    Everything above the band is the aliasing source, not a target, so it must
    not appear in the truth vector.
    """
    out = np.zeros(hp.Alm.getsize(lmax_band), dtype=np.complex128)
    for ell in range(lmax_band + 1):
        for m in range(min(ell, lmax_band) + 1):
            out[hp.Alm.getidx(lmax_band, ell, m)] = alm_full[
                hp.Alm.getidx(lmax_full, ell, m)
            ]
    return out


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def per_ell_cl_error(alm_rec, alm_true, lmax):
    """Relative error of the angular power spectrum -- Drake & Wright's metric.

    Blind to coefficient phase, so it is the *forgiving* of the two metrics;
    reported alongside the alm error rather than instead of it.
    """
    cl_rec = hp.alm2cl(alm_rec, lmax=lmax)
    cl_true = hp.alm2cl(alm_true, lmax=lmax)
    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.abs(cl_rec - cl_true) / cl_true
    err[~np.isfinite(err)] = np.nan
    return err


def per_ell_alm_error(alm_rec, alm_true, lmax):
    """Relative L2 error of the recovered coefficients at each ``l``, over all ``m``.

    Phase sensitive, so it catches convention errors that a ``C_l`` comparison
    cannot see.
    """
    ells, _ = hp.Alm.getlm(lmax, np.arange(len(alm_true)))
    out = np.full(lmax + 1, np.nan)
    for ell in range(lmax + 1):
        sel = ells == ell
        den = np.linalg.norm(alm_true[sel])
        if den > 0:
            out[ell] = np.linalg.norm(alm_rec[sel] - alm_true[sel]) / den
    return out


def per_ell_leakage(alm_contaminant, alm_signal, lmax):
    """Per-``l`` power ratio of the parity channel that should be empty.

    Fed a pure-E sky, this is ``C_l^BB / C_l^EE`` of the *recovered* pair: the
    fraction of power the transform moved across the E/B split. This is the
    number that decides whether a pipeline can measure primordial B modes, and
    neither healpy nor the paper reports it directly.
    """
    cl_c = hp.alm2cl(alm_contaminant, lmax=lmax)
    cl_s = hp.alm2cl(alm_signal, lmax=lmax)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = cl_c / cl_s
    out[~np.isfinite(out)] = np.nan
    return out


def ell_bands(lmax, lmin=0, nbands=4):
    """Split ``[lmin, lmax]`` into contiguous bands: ``[(label, lo, hi), ...]``.

    The band summary is where the high-``l`` story lives -- a single number over
    the whole spectrum averages the interesting top of the band away against the
    bulk, where every method is near-exact.
    """
    edges = [int(round(lmin + (lmax - lmin) * i / nbands)) for i in range(nbands + 1)]
    bands = []
    for i in range(nbands):
        lo = edges[i] if i == 0 else edges[i] + 1
        hi = edges[i + 1]
        if hi >= lo:
            bands.append((f"{lo}-{hi}", lo, hi))
    return bands


def band_rms(per_ell, lo, hi):
    """RMS of a per-``l`` error curve over ``lo <= l <= hi`` (NaNs dropped)."""
    seg = np.asarray(per_ell, dtype=float)[lo : hi + 1]
    if seg.size == 0 or np.all(np.isnan(seg)):
        return float("nan")
    return float(np.sqrt(np.nanmean(seg**2)))


def band_l2(alm_rec, alm_true, lmax, lo, hi):
    """Total L2 coefficient error over a band, normalised by the band amplitude.

    Robust where the per-``l`` relative metric is not: it never divides by an
    individual small coefficient, so one unlucky near-zero ``a_{l,0}`` cannot
    manufacture a huge "error".
    """
    ells, _ = hp.Alm.getlm(lmax, np.arange(len(alm_true)))
    sel = (ells >= lo) & (ells <= hi)
    den = np.linalg.norm(alm_true[sel])
    if den == 0:
        return float("nan")
    return float(np.linalg.norm(alm_rec[sel] - alm_true[sel]) / den)


def rel_l2(a, b):
    """Relative L2 error of ``a`` against reference ``b`` (maps or any arrays)."""
    a = np.asarray(a)
    b = np.asarray(b)
    den = np.linalg.norm(b)
    if den == 0:
        return float("nan")
    return float(np.linalg.norm(a - b) / den)


def rel_max(a, b):
    """Relative max-norm error, normalised by the reference's peak amplitude."""
    a = np.asarray(a)
    b = np.asarray(b)
    den = np.max(np.abs(b))
    if den == 0:
        return float("nan")
    return float(np.max(np.abs(a - b)) / den)


# --------------------------------------------------------------------------- #
# Result storage                                                               #
# --------------------------------------------------------------------------- #
def env_metadata(extra=None):
    """Versions and host details, stamped into every result file.

    A benchmark number without the machine and library versions beside it is not
    reproducible and should not be quoted.
    """
    import importlib

    versions = {}
    for mod in ("numpy", "scipy", "healpy", "ducc0", "finufft", "psutil"):
        try:
            versions[mod] = getattr(importlib.import_module(mod), "__version__", "?")
        except ImportError:
            versions[mod] = None
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "versions": versions,
    }
    if extra:
        meta.update(extra)
    return meta


def _sanitize(obj):
    """Make a record JSON-portable: numpy scalars/arrays out, non-finite -> null.

    ``json.dump`` writes a bare ``NaN`` by default, which Python reads back but
    which is not valid JSON -- and these files are committed, so something other
    than Python will eventually read them. Every non-finite value becomes
    ``null`` instead, which ``np.array(..., dtype=float)`` turns straight back
    into ``nan`` on the plotting side.
    """
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_sanitize(v) for v in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if np.isfinite(f) else None
    return obj


class ResultStore:
    """Append-only JSON record store with resume support.

    Written out after every record, so a long run that is interrupted keeps
    everything it had finished, and ``--resume`` can skip completed cells rather
    than recomputing an hour of transforms to add one ``nside``.
    """

    def __init__(self, path, meta, key_fields):
        self.path = path
        self.key_fields = tuple(key_fields)
        self.records = []
        self.meta = meta
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    blob = json.load(fh)
                self.records = blob.get("records", [])
                # Keep the original creation stamp; refresh everything else.
                self.meta = {**blob.get("meta", {}), **meta}
            except (json.JSONDecodeError, OSError):
                pass  # corrupt or unreadable: start fresh rather than crash
        self._keys = {self._key(r) for r in self.records}

    def _key(self, record):
        return tuple(record.get(f) for f in self.key_fields)

    def has(self, **fields):
        return tuple(fields.get(f) for f in self.key_fields) in self._keys

    def add(self, record):
        key = self._key(record)
        if key in self._keys:  # replace an existing cell rather than duplicate it
            self.records = [r for r in self.records if self._key(r) != key]
        self.records.append(record)
        self._keys.add(key)
        self.flush()

    def flush(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        payload = _sanitize({"meta": self.meta, "records": self.records})
        with open(tmp, "w") as fh:
            json.dump(payload, fh, allow_nan=False)
        os.replace(tmp, self.path)  # atomic: never leave a half-written file

    def frame(self):
        """Records as a list of dicts (the plotting side's only entry point)."""
        return list(self.records)


def load_results(path):
    """Read a result file written by :class:`ResultStore`."""
    with open(path) as fh:
        blob = json.load(fh)
    return blob.get("meta", {}), blob.get("records", [])


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
def markdown_table(rows, headers, floatfmt="{:.3e}"):
    """Render rows as a GitHub markdown table (for pasting into the README)."""

    def cell(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "--"
        if isinstance(v, float):
            return floatfmt.format(v)
        return str(v)

    body = [[cell(v) for v in row] for row in rows]
    widths = [
        max(len(str(h)), *(len(r[i]) for r in body)) if body else len(str(h))
        for i, h in enumerate(headers)
    ]
    out = ["| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, widths)) + " |"]
    out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in body:
        out.append("| " + " | ".join(v.ljust(w) for v, w in zip(r, widths)) + " |")
    return "\n".join(out)
