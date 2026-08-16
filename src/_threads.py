"""One thread-pool policy, shared by every stage that splits work across threads.

The pipeline threads by handing contiguous row blocks of an array to Python threads.
That works because the heavy calls underneath -- numpy's pocketfft and finufft --
release the GIL, and because the blocks are disjoint views, so no chunk can see another
chunk's writes.

Set ``HP2SPH_NUFFT_WORKERS`` to override the thread count; 1 disables every split.
"""

import os
from concurrent.futures import ThreadPoolExecutor

__all__ = [
    "WORKER_MIN_TRANSFORMS",
    "default_workers",
    "row_blocks",
    "run_blocks",
]

_EXECUTORS = {}


def _executor(workers):
    """A process-lifetime thread pool of the given size, created on first use."""
    ex = _EXECUTORS.get(workers)
    if ex is None:
        ex = ThreadPoolExecutor(max_workers=workers)
        _EXECUTORS[workers] = ex
    return ex


# Below this many rows the thread hand-off costs more than the split saves.
WORKER_MIN_TRANSFORMS = 128


def default_workers(n_trans: int = None) -> int:
    """How many threads a batch of ``n_trans`` independent rows should be split over.

    Set ``HP2SPH_NUFFT_WORKERS`` to override; 1 disables the split. Otherwise it is the
    core count capped at 7, above which the measured gain flattens.

    Small batches return 1: the per-call thread hand-off is fixed while the work per
    thread shrinks, so below ``WORKER_MIN_TRANSFORMS`` the split is a slowdown rather
    than a speed-up.
    """
    env = os.environ.get("HP2SPH_NUFFT_WORKERS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    if n_trans is not None and n_trans < WORKER_MIN_TRANSFORMS:
        return 1
    return max(1, min(7, os.cpu_count() or 1))


def row_blocks(n_rows, workers):
    """Split ``range(n_rows)`` into ``workers`` contiguous ``(lo, hi)`` blocks.

    Contiguous so each block is a view of the original array rather than a copy, and
    balanced to within one row.
    """
    workers = max(1, min(workers, n_rows))
    edges = [(n_rows * i) // workers for i in range(workers + 1)]
    return [(edges[i], edges[i + 1]) for i in range(workers) if edges[i + 1] > edges[i]]


def run_blocks(fn, n_rows, workers):
    """Call ``fn(lo, hi)`` on each contiguous row block, in parallel where it pays.

    ``fn`` must write only into rows ``lo:hi`` of whatever it touches; the blocks are
    disjoint, which is what makes the parallel version equal the serial one exactly.
    With one worker the calls run inline, so no pool is created and no thread is
    handed off.
    """
    blocks = row_blocks(n_rows, workers)
    if len(blocks) == 1:
        fn(*blocks[0])
        return
    list(_executor(len(blocks)).map(lambda b: fn(*b), blocks))
