"""Pin every OpenMP runtime loaded into this process.

Why this module exists. Three of the pipeline's dependencies each ship their OWN
copy of the LLVM OpenMP runtime, and all three end up loaded at once:

    healpy            site-packages/healpy/.dylibs/libomp.dylib   (vendored)
    finufft           site-packages/finufft/.dylibs/libomp.dylib  (vendored)
    libfasttransforms /opt/homebrew/opt/libomp/lib/libomp.dylib   (Homebrew)

libomp keeps process-wide state, so a second copy initializing on top of the first
is unsupported. In practice, measured on macOS 24.6 / arm64 with no OpenMP
environment variables set, the failures are:

* ``import healpy`` then ``import finufft``  -> ``OMP: Error #15``, SIGABRT;
* libfasttransforms loaded before finufft    -> SIGSEGV on the first ``ft_execute_*``;
* all three loaded, none pinned              -> ``finufft.Plan.setpts`` hangs forever.

``KMP_DUPLICATE_LIB_OK=TRUE`` only silences the first of those. It does not prevent
the crash or the hang.

What makes it work is one thread per runtime, so none of them forks a worker pool.
That is what the ``OMP_NUM_THREADS=1`` prefix in the old docs was buying.

``src/_bootstrap.py`` sets that variable itself, which covers everything the
pipeline imports. It cannot cover a library that loaded BEFORE ``src`` did, because
libomp reads its thread count when the image loads. :func:`pin` closes that gap
from inside the process: it walks the loaded shared libraries, picks out the OpenMP
runtimes, and calls ``omp_set_num_threads`` on each through its own handle. No
environment variable is involved, so a stale one cannot defeat it.

Cost. ``omp_set_num_threads`` sets the CALLING thread's ICV, so :func:`pin` has to
run on whichever thread is about to enter a transform; the callers therefore invoke
it per transform rather than once at import. The scan is cached behind a cheap
image-count check, so the repeat cost is one ``omp_set_num_threads`` call per
runtime.

Set ``HP2SPH_OMP_THREADS`` to override the count. Only do that with a build where a
single OpenMP runtime is shared by every library; with the wheels above, anything
other than 1 reproduces the failures listed at the top of this docstring.
"""

import ctypes
import os
import sys


def _requested_threads():
    """Read ``HP2SPH_OMP_THREADS``. ``auto`` means every core, and is the DEFAULT.

    Returns ``(n, explicit)``. ``explicit`` records whether the caller asked for the
    count or merely accepted the default, which is what :func:`pin` uses to decide
    between raising and degrading when several OpenMP runtimes are loaded.

    A bad value raises here, at import, rather than at the first transform. Silently
    falling back would look like the setting worked and leave no trace.
    """
    raw = os.environ.get("HP2SPH_OMP_THREADS")
    explicit = raw is not None
    raw = (raw or "auto").strip()
    if raw.lower() == "auto":
        return (os.cpu_count() or 1), explicit
    try:
        n = int(raw)
    except ValueError:
        raise ValueError(
            f"HP2SPH_OMP_THREADS={raw!r} is not an integer or 'auto'"
        ) from None
    if n < 1:
        raise ValueError(f"HP2SPH_OMP_THREADS={raw!r} must be >= 1")
    return n, explicit


NUM_THREADS, EXPLICIT_THREADS = _requested_threads()

# What ``_bootstrap`` puts in ``OMP_NUM_THREADS``. It is ALWAYS 1, whatever
# ``NUM_THREADS`` is, because that variable is read when each runtime's image loads
# and a runtime that loads with a worker pool has already claimed its state -- on a
# stack with several vendored runtimes that is the hang this module exists to
# prevent, and nothing in Python runs early enough to undo it. Threads are granted
# afterwards by :func:`pin`, which only does so once the scan has proved there is a
# single runtime to grant them to.
BOOTSTRAP_THREADS = 1

_OMP_PREFIXES = ("libomp", "libiomp5", "libgomp")

_cached_generation = None
_cached_setters = []
_cached_threads = None


def _is_openmp(path: str) -> bool:
    return os.path.basename(path).startswith(_OMP_PREFIXES)


class _DlPhdrInfo(ctypes.Structure):
    # Only the first two fields are read; the rest of the struct is ignored.
    _fields_ = [("dlpi_addr", ctypes.c_void_p), ("dlpi_name", ctypes.c_char_p)]


_PHDR_CB = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(_DlPhdrInfo), ctypes.c_size_t, ctypes.c_void_p
)


def _make_probe():
    """Bind the platform's loaded-image walk once, at import.

    Returns ``(count, names)`` callables, or ``(None, None)`` where there is no
    probe. Binding here rather than per call matters: ``ctypes.CDLL(None)`` is a
    ``dlopen``, and :func:`pin` runs on every transform.
    """
    try:
        libc = ctypes.CDLL(None)
        if sys.platform == "darwin":
            libc._dyld_image_count.restype = ctypes.c_uint32
            libc._dyld_get_image_name.restype = ctypes.c_char_p
            libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]

            def count():
                return libc._dyld_image_count()

            def names():
                return [
                    libc._dyld_get_image_name(i).decode()
                    for i in range(libc._dyld_image_count())
                ]

            return count, names

        if sys.platform.startswith("linux"):

            def names():
                found = []

                def collect(info, size, data):
                    name = info.contents.dlpi_name
                    if name:
                        found.append(name.decode())
                    return 0

                libc.dl_iterate_phdr(_PHDR_CB(collect), None)
                return found

            def count():
                # glibc offers no cheap count, so this walks. Still much cheaper
                # than rebinding every runtime's entry point.
                return len(names())

            return count, names
    except Exception:
        pass
    return None, None


_image_count, _image_names = _make_probe()


def loaded_images():
    """``(count, paths)`` of every shared library currently loaded.

    Returns ``(0, [])`` where the platform has no probe. Callers treat that as
    "nothing to pin" and fall back on the ``OMP_NUM_THREADS`` value that
    ``src/_bootstrap.py`` sets.
    """
    if _image_names is None:
        return 0, []
    try:
        paths = _image_names()
    except Exception:
        return 0, []
    return len(paths), paths


def _setter(path):
    """``omp_set_num_threads`` bound to the runtime at ``path``, or None.

    ``ctypes.CDLL`` on an already-loaded path returns a handle to that same image
    rather than loading a second copy, so this reaches exactly one runtime.
    """
    try:
        fn = ctypes.CDLL(path).omp_set_num_threads
    except (OSError, AttributeError):
        return None
    fn.argtypes = [ctypes.c_int]
    fn.restype = None
    return fn


def runtime_paths():
    """Paths of the OpenMP runtimes currently loaded, deduplicated."""
    seen = {}
    for path in loaded_images()[1]:
        if _is_openmp(path):
            seen[os.path.realpath(path)] = path
    return list(seen.values())


class MultipleOpenMPRuntimes(RuntimeError):
    """Raised when threading is requested but more than one libomp is loaded."""


def _reject_threading(paths, n):
    raise MultipleOpenMPRuntimes(
        f"HP2SPH_OMP_THREADS={n} asks for {n} OpenMP threads, but {len(paths)} "
        "separate OpenMP runtimes are loaded into this process:\n"
        + "\n".join(f"  {p}" for p in paths)
        + "\n\nMore than one of them threaded crashes or hangs -- this raises "
        "instead of letting that happen. Either drop HP2SPH_OMP_THREADS (the "
        "default of 1 is always safe), or build a stack where every library "
        "shares ONE runtime; see the OpenMP section of README.md. The usual "
        "cause of an unexpected extra runtime is a libfasttransforms built "
        "against a different libomp than the rest of the environment -- check "
        "FASTTRANSFORMS_LIB and any lib/ directory in the repo root, which "
        "takes precedence over the active environment."
    )


def pin(num_threads: int = None) -> int:
    """Pin every loaded OpenMP runtime to ``num_threads`` on the calling thread.

    Returns how many runtimes were pinned. Cheap to call repeatedly: the library
    scan is redone only when the loader's image count has changed since the last
    call, which happens when a new extension module is imported.

    Raises :class:`MultipleOpenMPRuntimes` if more than one thread is requested
    while several runtimes are loaded. That combination does not merely run slowly,
    it deadlocks or segfaults, so failing with an explanation beats failing with a
    signal. Nothing is pinned before the check, and enumerating images touches no
    OpenMP entry point, so the raise happens before anything can go wrong.
    """
    global _cached_generation, _cached_setters, _cached_threads
    if _image_count is None:
        return 0
    try:
        generation = _image_count()
    except Exception:
        return 0
    n = NUM_THREADS if num_threads is None else num_threads
    if generation != _cached_generation:
        paths = runtime_paths()
        if n > 1 and len(paths) > 1:
            # Asking for threads on a multi-runtime stack is fatal. Whether that is
            # an error or a fact of the environment depends on who asked: a count
            # the caller set is honoured to the point of refusing to continue, while
            # the default quietly takes the safe value it would have had before.
            if EXPLICIT_THREADS or num_threads is not None:
                _reject_threading(paths, n)
            n = 1
        _cached_generation = generation
        _cached_setters = [fn for fn in map(_setter, paths) if fn is not None]
        _cached_threads = n
    else:
        n = _cached_threads if num_threads is None else n
    for fn in _cached_setters:
        fn(n)
    return len(_cached_setters)
