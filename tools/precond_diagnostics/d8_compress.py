"""D8: compress the coarse space by the generalised eigenproblem  E v = lam S v,
S = Z^H Z, keeping the r smallest Rayleigh quotients.  One-off O(R^3), cacheable;
afterwards the coarse solve is O(r^2) and the coarse space is r dense vectors in the
generator basis, so the per-iteration cost is dominated by applying Z itself.
"""

import sys, time
import numpy as np
import scipy.linalg as sla
from scipy.sparse.linalg import LinearOperator

from precond_common import make_spin_dfs, capture, cg_count
from src.double_fourier_sphere import dfs_fold_plan
from precond_twolevel import TwoLevel


def compressed(T, r):
    S = np.asarray((T.ZsH @ T.Zs).todense())
    S = 0.5 * (S + S.conj().T)
    ridge = 1e-12 * np.trace(S).real / S.shape[0]
    lam, V = sla.eigh(T.E, S + ridge * np.eye(S.shape[0]))
    V = V[:, :r]
    lam = lam[:r]
    Zs = T.Zs

    def apply(v):
        c = V.conj().T @ (T.ZsH @ v)
        return v + Zs @ (V @ (c / lam))

    return LinearOperator((T.n, T.n), matvec=apply, dtype=complex), lam


def main(nside, spin=2, tol=1e-2, rtol=1e-7):
    dfs, _ = make_spin_dfs(nside, spin=spin)
    A, b = capture(dfs, nside, dfs_fold_plan(nside, spin, tol), rtol=rtol)
    t = time.perf_counter()
    x0, it0, _ = cg_count(A, b, rtol=rtol, maxiter=20000)
    t0 = time.perf_counter() - t
    T = TwoLevel(nside, spin, tol)
    print(
        f"nside {nside}  n {T.n}  R {T.R}  plain {it0} its {t0:.3f} s "
        f"({1e3 * t0 / it0:.2f} ms/it)"
    )
    for r in (25, 50, 100, 200, 400, 800, T.R):
        if r > T.R:
            continue
        Mop, lam = compressed(T, r)
        t = time.perf_counter()
        x1, it1, _ = cg_count(A, b, M=Mop, rtol=rtol, maxiter=20000)
        t1 = time.perf_counter() - t
        print(
            f"   r {r:5d}  lam[0] {lam[0]:.2e} lam[-1] {lam[-1]:.3f}  "
            f"{it1:4d} its  {1e3 * t1 / it1:6.2f} ms/it  iters {it0 / it1:5.1f}x"
        )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [32]:
        main(ns)
