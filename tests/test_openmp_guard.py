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
