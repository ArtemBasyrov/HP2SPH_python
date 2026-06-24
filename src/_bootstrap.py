"""One-time process setup that must run *before* the numerical libraries load.

``finufft``, ``jax`` and ``libfasttransforms`` all link an OpenMP runtime
(``libomp``).  On macOS, loading more than one copy aborts the process with
``OMP: Error #15`` unless ``KMP_DUPLICATE_LIB_OK`` is set, and letting the
several runtimes each spin up threads segfaults unless the thread count is
pinned -- so ``OMP_NUM_THREADS`` defaults to 1 here.  The pipeline also needs
float64, which JAX only uses when its x64 mode is enabled (otherwise it silently
runs float32 and the transforms diverge).

Importing this module sets both OpenMP guards, and :func:`enable_x64` turns on
JAX double precision.  Both are done automatically when the package is imported
(see ``src/__init__.py``), so end users can just run ``python ...`` without
prefixing the env vars or calling ``jax.config.update`` by hand.

``setdefault`` is used throughout so an explicit environment value always wins
(e.g. set ``OMP_NUM_THREADS`` yourself if your build tolerates more threads).
"""

import os

# Must be set before anything pulls in libomp (finufft / jax / libfasttransforms).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Multiple OpenMP runtimes in one process segfault when each spawns threads;
# pinning to 1 thread is the safe default (override by exporting OMP_NUM_THREADS).
os.environ.setdefault("OMP_NUM_THREADS", "1")


def enable_x64() -> None:
    """Enable JAX float64. Safe to call repeatedly; must precede any jax array."""
    import jax

    jax.config.update("jax_enable_x64", True)
