"""D7: screen the coarse space by the per-generator Rayleigh quotient.

rho_j = E_jj / (Z^H Z)_jj  estimates how near-null generator j is.  The fine level is
already the identity to 1%, so a generator with rho ~ 1 is handled by it and only
inflates the coarse solve.  Keeping rho < rho_max costs O(R) to evaluate.
"""

import sys, time
import numpy as np
import scipy.linalg as sla
from scipy.sparse.linalg import LinearOperator

from precond_common import make_spin_dfs, capture, cg_count
from src.double_fourier_sphere import dfs_fold_plan
from precond_twolevel import TwoLevel


def screened(T, rho_max):
    ZtZ = np.asarray((T.Zs.multiply(T.Zs.conj())).sum(axis=0)).ravel().real
    rho = np.diag(T.E).real / ZtZ
    sel = np.nonzero(rho < rho_max * np.median(rho))[0]
    if sel.size == 0:
        return None, 0
    Es = T.E[np.ix_(sel, sel)]
    ridge = 1e-10 * np.trace(Es).real / len(sel)
    cho = sla.cho_factor(Es + ridge * np.eye(len(sel)), lower=True)
    Zs = T.Zs[:, sel].tocsr()
    ZsH = Zs.conj().T.tocsr()

    def apply(v):
        return v + Zs @ sla.cho_solve(cho, ZsH @ v)

    return LinearOperator((T.n, T.n), matvec=apply, dtype=complex), len(sel)


def main(nside, spin=2, tol=1e-2, rtol=1e-7):
    dfs, _ = make_spin_dfs(nside, spin=spin)
    A, b = capture(dfs, nside, dfs_fold_plan(nside, spin, tol), rtol=rtol)
    t = time.perf_counter()
    x0, it0, _ = cg_count(A, b, rtol=rtol, maxiter=20000)
    t0 = time.perf_counter() - t
    T = TwoLevel(nside, spin, tol)
    print(f"nside {nside}  R {T.R}  plain {it0} its {t0:.3f} s")
    for rho_max in (0.05, 0.15, 0.3, 0.6, 1.0, 1e9):
        Mop, r = screened(T, rho_max)
        if Mop is None:
            continue
        t = time.perf_counter()
        x1, it1, _ = cg_count(A, b, M=Mop, rtol=rtol, maxiter=20000)
        t1 = time.perf_counter() - t
        print(
            f"   rho<{rho_max:5.2f}*med  r {r:5d} ({100 * r / T.R:5.1f}% of R)  "
            f"{it1:4d} its  {t1:6.3f} s   iters {it0 / it1:5.1f}x  wall {t0 / t1:5.2f}x"
            f"   |dx|/|x| {np.linalg.norm(x1 - x0) / np.linalg.norm(x0):.1e}"
        )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [32]:
        main(ns)
