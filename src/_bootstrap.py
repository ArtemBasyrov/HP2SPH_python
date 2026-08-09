"""One-time process setup that must run *before* the numerical libraries load.

``finufft`` and ``libfasttransforms`` both link an OpenMP runtime (``libomp``).
On macOS, loading more than one copy aborts the process with ``OMP: Error #15``
unless ``KMP_DUPLICATE_LIB_OK`` is set, and letting the several runtimes each
spin up threads segfaults unless the thread count is pinned -- so
``OMP_NUM_THREADS`` defaults to 1 here.

Importing this module sets both guards, which happens automatically when the
package is imported (see ``src/__init__.py``), so end users can just run
``python ...`` without prefixing the env vars.

The pipeline is pure numpy and therefore float64 by default; the module used to
also enable JAX's x64 mode, which existed only to stop jax silently computing the
whole transform in float32.

``setdefault`` is used throughout so an explicit environment value always wins
(e.g. set ``OMP_NUM_THREADS`` yourself if your build tolerates more threads).
"""

import os

# Must be set before anything pulls in libomp (finufft / libfasttransforms).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Multiple OpenMP runtimes in one process segfault when each spawns threads;
# pinning to 1 thread is the safe default (override by exporting OMP_NUM_THREADS).
os.environ.setdefault("OMP_NUM_THREADS", "1")
