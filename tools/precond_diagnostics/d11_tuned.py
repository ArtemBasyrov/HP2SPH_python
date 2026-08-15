"""D11: the coarse solve is memory bound, and a preconditioner needs no precision.

Store the Cholesky factor of E in complex64 and re-time.  Also reports the split
between the matrix-vector product and the preconditioner application.
"""

import sys, time
import numpy as np
import scipy.linalg as sla
from scipy.sparse.linalg import LinearOperator

import precond_common  # noqa: F401
from precond_common import make_spin_dfs, capture, cg_count
from src.double_fourier_sphere import dfs_fold_plan
from precond_twolevel import TwoLevel


def op(T, dtype):
    c, low = T.chol
    c = c.astype(dtype)
    Zs, ZsH = T.Zs, T.ZsH

    def apply(v):
        y = sla.cho_solve((c, low), (ZsH @ v).astype(dtype))
        return v + Zs @ y.astype(complex)

    return LinearOperator((T.n, T.n), matvec=apply, dtype=complex)


def timeit(f, n=5):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return min(ts)


def main(nside, spin=2, tol=1e-2, rtol=1e-7):
    dfs, _ = make_spin_dfs(nside, spin=spin)
    A, b = capture(dfs, nside, dfs_fold_plan(nside, spin, tol), rtol=rtol)
    t = time.perf_counter()
    x0, it0, _ = cg_count(A, b, rtol=rtol, maxiter=20000)
    t0 = time.perf_counter() - t
    T = TwoLevel(nside, spin, tol)
    v = np.random.default_rng(0).standard_normal(T.n) + 0j
    t_mv = timeit(lambda: A.matvec(v))
    print(
        f"nside {nside}  n {T.n}  R {T.R}  plain {it0} its {t0:.3f} s   "
        f"matvec {1e3 * t_mv:.2f} ms"
    )
    for dt in (np.complex128, np.complex64):
        M = op(T, dt)
        t_pc = timeit(lambda: M.matvec(v))
        t = time.perf_counter()
        x1, it1, _ = cg_count(A, b, M=M, rtol=rtol, maxiter=20000)
        t1 = time.perf_counter() - t
        print(
            f"   coarse factor {np.dtype(dt).name:10s}  precond apply "
            f"{1e3 * t_pc:6.2f} ms   {it1:4d} its  {t1:6.3f} s   "
            f"iters {it0 / it1:5.1f}x  wall {t0 / t1:5.2f}x   "
            f"|dx|/|x| {np.linalg.norm(x1 - x0) / np.linalg.norm(x0):.1e}"
        )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [64]:
        main(ns)
