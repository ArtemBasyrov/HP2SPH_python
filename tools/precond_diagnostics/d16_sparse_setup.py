"""D16: sparse assembly + sparse factorisation, end to end, including setup.

Checks that the sparse-assembled E equals the dense-assembled one, then reports setup
time, memory, iteration count and wall time.
"""

import sys, time
import numpy as np
import scipy.sparse as sp

import precond_common  # noqa: F401
from precond_common import make_spin_dfs, capture, cg_count
from precond_twolevel import TwoLevel
from src.double_fourier_sphere import dfs_fold_plan


def verify(nside, spin=2, tol=1e-2):
    t = time.perf_counter()
    Td = TwoLevel(nside, spin, tol, sparse=False)
    td = time.perf_counter() - t
    t = time.perf_counter()
    Ts = TwoLevel(nside, spin, tol, sparse=True)
    ts = time.perf_counter() - t
    d = np.abs(Ts.E.toarray() - Td.E).max() / np.abs(Td.E).max()
    print(
        f"nside {nside}  R {Td.R}   dense setup {td:6.2f} s   sparse setup {ts:6.2f} s"
        f"   ({td / ts:4.1f}x)   max|E_sparse - E_dense|/max|E| = {d:.2e}"
    )


def timed(nside, spin=2, tol=1e-2, rtol=1e-7):
    dfs, _ = make_spin_dfs(nside, spin=spin)
    A, b = capture(dfs, nside, dfs_fold_plan(nside, spin, tol), rtol=rtol)
    t = time.perf_counter()
    x0, it0, _ = cg_count(A, b, rtol=rtol, maxiter=20000)
    t0 = time.perf_counter() - t

    t = time.perf_counter()
    T = TwoLevel(nside, spin, tol, sparse=True)
    tset = time.perf_counter() - t
    nnzE = T.E.nnz
    nnzLU = T.lu.L.nnz + T.lu.U.nnz
    M = T.operator()
    t = time.perf_counter()
    x1, it1, _ = cg_count(A, b, M=M, rtol=rtol, maxiter=20000)
    t1 = time.perf_counter() - t
    print(
        f"nside {nside:4d}  n {T.n:7d}  R {T.R:6d}   "
        f"E nnz {nnzE:9d} ({100 * nnzE / T.R**2:5.2f}%)  LU nnz {nnzLU:9d}   "
        f"setup {tset:6.2f} s"
    )
    print(
        f"            plain {it0:5d} its {t0:7.3f} s   two-level {it1:4d} its "
        f"{t1:7.3f} s   iters {it0 / it1:5.1f}x   wall {t0 / t1:5.2f}x   "
        f"|dx|/|x| {np.linalg.norm(x1 - x0) / np.linalg.norm(x0):.1e}"
    )


if __name__ == "__main__":
    mode = sys.argv[1]
    for ns in [int(a) for a in sys.argv[2:]]:
        (verify if mode == "verify" else timed)(ns)
