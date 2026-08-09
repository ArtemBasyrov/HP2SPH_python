"""One-time process setup that must run *before* the numerical libraries load.

Importing this module is all that is needed to run the pipeline. No environment
variable has to be set by hand -- in particular the ``OMP_NUM_THREADS=1`` prefix
the old docs demanded is gone, because this module sets it itself.

Three of the dependencies each vendor their own copy of the LLVM OpenMP runtime
(healpy, finufft, and libfasttransforms via Homebrew's libomp), and all three end
up loaded at once. Two of them running worker pools in one process crashes or
hangs; ``src/_openmp.py`` documents the exact failures. One thread per runtime is
therefore a CORRECTNESS requirement here, not a tuning choice.

Both variables below are set before the first ``import`` of anything that links
libomp, because libomp reads its thread count when the image loads. That is also
why ``src/_openmp.py``'s in-process pin cannot replace this: by the time any Python
code could call ``omp_set_num_threads``, a runtime that loaded with 8 threads has
already claimed its state.

``OMP_NUM_THREADS`` is FORCED rather than defaulted. An exported
``OMP_NUM_THREADS=8`` in the user's shell is the single most common way to make the
pipeline segfault, and honouring it would mean honouring a value that cannot work.
Set ``HP2SPH_OMP_THREADS`` to override, and only with a build where one OpenMP
runtime is shared by every library.

``KMP_DUPLICATE_LIB_OK`` uses ``setdefault``, since it is a plain on/off guard
against the ``OMP: Error #15`` abort and there is no reason to override a user who
turned it off deliberately.
"""

import os

# Suppresses the OMP: Error #15 abort from having several libomp images loaded.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# One thread per OpenMP runtime. See the module docstring: this is forced, and it
# must happen before healpy / finufft / libfasttransforms are imported. _openmp is
# pure stdlib, so importing it here loads nothing that links libomp.
from ._openmp import NUM_THREADS as OMP_NUM_THREADS  # noqa: E402

os.environ["OMP_NUM_THREADS"] = str(OMP_NUM_THREADS)
