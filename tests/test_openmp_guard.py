"""The pipeline must run as a plain script, with no OpenMP environment prefix.

Three dependencies each vendor their own copy of the LLVM OpenMP runtime (healpy,
finufft, and libfasttransforms via Homebrew's libomp) and all three load into one
process. Unless each is held to one thread the process crashes or hangs -- see
``src/_openmp.py`` for the measured failure modes. Two guards keep that from
happening, and both are load-bearing:

* ``src/_bootstrap.py`` FORCES ``OMP_NUM_THREADS=1`` before anything links libomp,
  because libomp reads the count when the image loads;
* ``src/_openmp.py`` pins every already-loaded runtime through its own handle,
  which is the only thing that helps when a library loaded before ``src`` did.

These tests have to run in a subprocess: the environment has to be wrong BEFORE
the first import, which cannot be arranged inside a running interpreter. Each one
asserts the child finishes rather than hanging, so a regression shows up as a
timeout, not a wrong number.
"""

import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Generous: the transforms below take well under a second. Anything near this is a
# deadlock, which is the failure being guarded against.
TIMEOUT_S = 120

_FORWARD = """
import sys, contextlib, io
sys.path.insert(0, {root!r})
{preamble}
import numpy as np, healpy as hp
from tests.pipeline_helpers import forward_C
mp = hp.alm2map(np.zeros(hp.Alm.getsize(32), dtype=complex) + 1.0, nside=16, lmax=32)
with contextlib.redirect_stdout(io.StringIO()):
    C = forward_C(mp)
assert np.isfinite(np.asarray(C)).all()
print("OK")
"""


def _run(preamble="", **env_overrides):
    """Run a forward transform in a child with a deliberately hostile environment."""
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "KMP_DUPLICATE_LIB_OK", "HP2SPH_OMP_THREADS"):
        env.pop(key, None)
    env.update(env_overrides)
    script = _FORWARD.format(root=REPO_ROOT, preamble=preamble)
    try:
        return subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"the transform hung for {TIMEOUT_S}s with env {env_overrides} -- the "
            "OpenMP guards in src/_bootstrap.py and src/_openmp.py are the thing to "
            "look at"
        )


def _check(result):
    assert result.returncode == 0, (
        f"exit {result.returncode} (negative = killed by a signal; -11 is the "
        f"OpenMP SIGSEGV)\nstdout: {result.stdout}\nstderr: {result.stderr[-2000:]}"
    )
    assert "OK" in result.stdout


@pytest.mark.ft
def test_forward_runs_with_no_openmp_env_at_all():
    """The headline requirement: no ``OMP_NUM_THREADS=1`` prefix needed."""
    _check(_run())


@pytest.mark.ft
@pytest.mark.parametrize("threads", ["4", "8", "16"])
def test_forward_survives_an_exported_omp_num_threads(threads):
    """A user with ``OMP_NUM_THREADS`` exported in their shell is the common case.

    Before ``src/_bootstrap.py`` started FORCING the value, this hung: the guard
    used ``setdefault``, so an exported value won and every runtime came up
    multithreaded.
    """
    _check(_run(OMP_NUM_THREADS=threads))


@pytest.mark.ft
def test_forward_survives_healpy_being_imported_first():
    """``src/_bootstrap.py`` cannot win here, so the in-process pin has to.

    healpy's libomp has already loaded and read its thread count by the time any
    ``src`` code runs, so only ``_openmp.pin()`` can still tame it.
    """
    preamble = (
        "import numpy as np, healpy as hp\n"
        "hp.alm2map(np.zeros(hp.Alm.getsize(16), dtype=complex) + 1.0,"
        " nside=8, lmax=16)\n"
    )
    _check(_run(preamble=preamble, OMP_NUM_THREADS="8"))


def test_pin_reports_the_openmp_runtimes_it_found():
    """``pin`` returns a count so a silent no-op is detectable.

    On this machine there are three (healpy, finufft, libfasttransforms). The
    assertion is only ``>= 1``: the count is a property of how the wheels were
    built, not of this repo.
    """
    healpy = pytest.importorskip("healpy")  # noqa: F841
    from src import _openmp

    if not _openmp.loaded_images()[1]:
        pytest.skip(f"no loaded-image probe for {sys.platform}")
    assert _openmp.pin() >= 1


def test_threading_is_refused_or_allowed_according_to_the_runtime_count():
    """Requesting threads must never be silently accepted into a crash.

    Which branch runs depends on how the environment was built, so the test
    asserts the rule rather than one outcome:

    * several runtimes (PyPI healpy/finufft wheels, each vendoring libomp) ->
      ``MultipleOpenMPRuntimes``, because threading there deadlocks or segfaults;
    * one runtime (a conda-forge stack sharing ``llvm-openmp``, with
      libfasttransforms built against it) -> threading is allowed.

    ``tools/build_fasttransforms.sh --prefix <env>`` is what produces the second.
    """
    from src import _openmp

    paths = _openmp.runtime_paths()
    if not paths:
        pytest.skip(f"no loaded-image probe for {sys.platform}")

    if len(paths) > 1:
        _openmp._cached_generation = None  # force the check to re-run
        with pytest.raises(_openmp.MultipleOpenMPRuntimes) as excinfo:
            _openmp.pin(8)
        # The message has to name the offending libraries; that is the whole
        # point of raising instead of crashing.
        for path in paths:
            assert path in str(excinfo.value)
    else:
        assert _openmp.pin(8) == 1

    _openmp._cached_generation = None
    assert _openmp.pin(1) == len(paths)  # 1 thread is always allowed


def test_pin_does_not_rescan_once_warm():
    """It runs per transform, so the cached path must not walk the image list.

    Only the cheap image COUNT is allowed on the warm path. Enumerating and
    decoding every loaded library's path on each transform cost ~3% of the forward
    at nside 128, which is why the count guard exists.
    """
    from src import _openmp

    if not _openmp.loaded_images()[1]:
        pytest.skip(f"no loaded-image probe for {sys.platform}")
    _openmp.pin()  # warm the cache
    assert _openmp._cached_setters, "setters were not cached"

    calls = []
    original = _openmp._image_names
    _openmp._image_names = lambda: calls.append(1) or original()
    try:
        for _ in range(50):
            assert _openmp.pin() == len(_openmp._cached_setters)
    finally:
        _openmp._image_names = original
    assert calls == [], f"rescanned {len(calls)} times on the warm path"


# --- threading works, and gives the same answer -----------------------------------
#
# The tests above pin the GUARD: threading is REFUSED when several OpenMP runtimes are
# loaded. They are equally happy when threading is refused forever, so on their own they
# cannot tell a healthy single-runtime stack from a broken one. The tests below cover the
# other half: where threading is possible it must actually happen, and it must not change
# the numbers.
#
# The failure mode they exist for: ``ft_sphere`` resolves ``FASTTRANSFORMS_LIB``, then a
# repo ``lib/``, then the active environment. If either override points at a build linked
# against a different OpenMP runtime from the rest of the stack, that runtime is loaded
# too, ``_openmp.pin`` refuses to thread, and the pipeline silently stays single-threaded
# on a machine that was set up for threads.
#
# Where threading is legitimately impossible -- for instance a wheels-only stack in which
# healpy and finufft each vendor their own libomp -- these skip rather than fail, because
# that is a property of how the stack was built, not a defect in this repo.

_RUNTIME_REPORT = """
import sys, contextlib, io
sys.path.insert(0, {root!r})
import numpy as np, healpy as hp
from src import _openmp
print("BEFORE", len(_openmp.runtime_paths()))
import src.ft_sphere  # noqa: F401  -- this is what resolves FASTTRANSFORMS_LIB
print("AFTER", len(_openmp.runtime_paths()))
from tests.pipeline_helpers import forward_C
mp = hp.alm2map(np.zeros(hp.Alm.getsize(32), dtype=complex) + 1.0, nside=16, lmax=32)
with contextlib.redirect_stdout(io.StringIO()):
    C = np.asarray(forward_C(mp))
np.save({out!r}, C)
print("RUNTIMES", len(_openmp.runtime_paths()))
print("PREFIX", sys.prefix)
for p in _openmp.runtime_paths():
    print("OMP", p)
for p in _openmp.loaded_images()[1]:
    if "fasttransforms" in p.lower():
        print("FT", p)
print("OK")
"""


_PROBE = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("_omp_probe", {probe!r})
_omp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_omp)
import numpy, healpy, finufft  # the deps that legitimately carry their own libomp
{extra}
print("RUNTIMES", len(_omp.runtime_paths()))
for p in _omp.runtime_paths():
    print("OMP", p)
print("OK")
"""


def _run_probe(extra=""):
    """Count OpenMP runtimes with and without libfasttransforms in the process."""
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "KMP_DUPLICATE_LIB_OK", "HP2SPH_OMP_THREADS"):
        env.pop(key, None)
    script = _PROBE.format(
        probe=os.path.join(REPO_ROOT, "src", "_openmp.py"), extra=extra
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )


def _run_report(out_path, **env_overrides):
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "KMP_DUPLICATE_LIB_OK", "HP2SPH_OMP_THREADS"):
        env.pop(key, None)
    env.update(env_overrides)
    script = _RUNTIME_REPORT.format(root=REPO_ROOT, out=str(out_path))
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"the transform hung for {TIMEOUT_S}s with env {env_overrides} -- a "
            "threaded deadlock is exactly what src/_openmp.py exists to prevent"
        )
    return result


def _parse(result):
    fields = {"OMP": [], "FT": []}
    for line in result.stdout.splitlines():
        key, _, value = line.partition(" ")
        if key in ("OMP", "FT"):
            fields[key].append(value)
        elif key in ("RUNTIMES", "PREFIX", "BEFORE", "AFTER"):
            fields[key] = value
    return fields


@pytest.mark.ft
def test_libfasttransforms_does_not_add_an_openmp_runtime():
    """Whichever copy is resolved, it must share the runtime already in the process.

    This is the failure the suite used to miss, and it must be caught however the
    library is selected -- so the invariant is not "the copy comes from the active
    environment" (pointing ``FASTTRANSFORMS_LIB`` elsewhere is a documented override)
    but "loading it does not INCREASE the OpenMP runtime count". That holds however the
    stack was built, and fails only for a library linked against a runtime nothing else
    in the process uses -- which is what makes threading unavailable.

    The baseline has to be a SEPARATE process: ``src/__init__.py`` imports ``FSHT``,
    which imports ``ft_sphere``, so any ``src`` import has already loaded the library.
    The probe therefore loads ``src/_openmp.py`` by path, bypassing the package.
    """
    from src import _openmp

    if not _openmp.loaded_images()[1]:
        pytest.skip(f"no loaded-image probe for {sys.platform}")
    base = _run_probe()
    assert base.returncode == 0, base.stderr[-2000:]
    with_ft = _run_probe(extra="import src.ft_sphere  # noqa: F401")
    assert with_ft.returncode == 0, with_ft.stderr[-2000:]
    before = int(_parse(base)["RUNTIMES"])
    after = int(_parse(with_ft)["RUNTIMES"])
    assert after == before, (
        f"importing src.ft_sphere took the OpenMP runtime count from {before} to "
        f"{after}, so it links an OpenMP runtime nothing else in the process uses "
        "and _openmp.pin will refuse to thread. Check FASTTRANSFORMS_LIB and any repo "
        "lib/ directory -- both are resolved BEFORE the active environment.\n"
        + "\n".join(_parse(with_ft)["OMP"])
    )


@pytest.mark.ft
def test_threading_is_usable_when_one_runtime_is_loaded(tmp_path):
    """Threading must not merely be *refused safely* -- where possible it must WORK."""
    from src import _openmp

    if not _openmp.loaded_images()[1]:
        pytest.skip(f"no loaded-image probe for {sys.platform}")
    if len(_openmp.runtime_paths()) != 1:
        pytest.skip("this stack has several OpenMP runtimes; threading is refused")
    result = _run_report(tmp_path / "c.npy", HP2SPH_OMP_THREADS="auto")
    _check(result)
    assert _parse(result)["RUNTIMES"] == "1", (
        "a threaded run pulled in a second OpenMP runtime:\n"
        + "\n".join(_parse(result)["OMP"])
    )


@pytest.mark.ft
def test_threaded_output_matches_single_threaded(tmp_path):
    """Threads are a speed change, so they must not move a digit.

    Not asserted bit-for-bit: the reductions inside finufft and libfasttransforms
    reassociate with the thread count, so the two agree to a few ulp rather than
    exactly. The tolerance is loose enough for that and far tighter than any real
    regression.
    """
    from src import _openmp

    if not _openmp.loaded_images()[1]:
        pytest.skip(f"no loaded-image probe for {sys.platform}")
    if len(_openmp.runtime_paths()) != 1:
        pytest.skip("this stack has several OpenMP runtimes; threading is refused")
    one, many = tmp_path / "one.npy", tmp_path / "many.npy"
    _check(_run_report(one, HP2SPH_OMP_THREADS="1"))
    _check(_run_report(many, HP2SPH_OMP_THREADS="auto"))
    import numpy as np

    a, b = np.load(one), np.load(many)
    assert a.shape == b.shape
    assert np.linalg.norm(b - a) <= 1e-12 * np.linalg.norm(a)


# --- the default asks for threads, so it must degrade where a request would raise ---


def _fake_two_runtimes(monkeypatch, calls):
    """Two loaded runtimes whose setters record the count they were given."""
    from src import _openmp

    paths = ["/fake/a/libomp.dylib", "/fake/b/libomp.dylib"]
    monkeypatch.setattr(_openmp, "runtime_paths", lambda: paths)
    monkeypatch.setattr(
        _openmp, "_setter", lambda path: lambda n: calls.append((path, n))
    )
    monkeypatch.setattr(_openmp, "_cached_generation", None)
    monkeypatch.setattr(_openmp, "_cached_setters", [])
    monkeypatch.setattr(_openmp, "_cached_threads", None)
    return paths


def test_the_default_count_degrades_to_one_on_a_multi_runtime_stack(monkeypatch):
    """The default is ``auto``, and on a stack that cannot thread it must not raise.

    A count the caller chose is honoured to the point of refusing to run, but nobody
    chose the default, so it takes the value it would have had before rather than
    turning a working install into an import-time failure.
    """
    from src import _openmp

    calls = []
    _fake_two_runtimes(monkeypatch, calls)
    monkeypatch.setattr(_openmp, "NUM_THREADS", 8)
    monkeypatch.setattr(_openmp, "EXPLICIT_THREADS", False)

    assert _openmp.pin() == 2  # no raise
    assert [n for _, n in calls] == [1, 1]


def test_an_explicit_count_still_raises_on_a_multi_runtime_stack(monkeypatch):
    """``HP2SPH_OMP_THREADS=8`` on a stack that deadlocks must still be refused."""
    from src import _openmp

    calls = []
    paths = _fake_two_runtimes(monkeypatch, calls)
    monkeypatch.setattr(_openmp, "NUM_THREADS", 8)
    monkeypatch.setattr(_openmp, "EXPLICIT_THREADS", True)

    with pytest.raises(_openmp.MultipleOpenMPRuntimes) as excinfo:
        _openmp.pin()
    for path in paths:
        assert path in str(excinfo.value)
    assert calls == []  # nothing is pinned before the check


def test_bootstrap_loads_at_one_thread_whatever_the_default_is():
    """``OMP_NUM_THREADS`` is the LOAD-time count and must stay 1.

    Threads are granted afterwards by ``pin``, which can count the runtimes first.
    Putting the real count in the environment instead would let a stack with several
    vendored runtimes fork a worker pool per runtime as its images load, which is
    the hang ``src/_openmp.py`` exists to prevent.
    """
    from src import _openmp

    assert _openmp.BOOTSTRAP_THREADS == 1
    assert os.environ["OMP_NUM_THREADS"] == "1"
