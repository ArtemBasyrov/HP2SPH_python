"""D33: does the coarse space need every relaxed entry at high nside?

R = 0.88 * nside^2 and nnz(E) ~ R^2, so halving R quarters the memory.  d7 showed that
screening generators by their Rayleigh quotient fails, but that is an algebraic
criterion.  Try a PHYSICAL one instead: keep a generator only where the pole envelope is
large, i.e. where the dropped zero-assertion was most wrong.  The operator keeps the full
fold at alias_tol; only the COARSE SPACE is restricted.
"""

import sys, time
import numpy as np

import precond_common  # noqa: F401
from precond_common import make_spin_dfs, capture, cg_count
import precond_twolevel as pt
from src.double_fourier_sphere import dfs_fold_plan


class Restricted(pt.TwoLevel):
    """TwoLevel whose coarse space uses only the generators of a coarser tolerance."""

    def __init__(self, nside, spin, tol, coarse_tol, **kw):
        orig = pt.coarse_generators

        def restricted(ns, sp_, t):
            rows, cols, coeffs, _ = orig(ns, sp_, coarse_tol)
            return rows, cols, coeffs, orig(ns, sp_, t)[3]

        pt.coarse_generators = restricted
        try:
            super().__init__(nside, spin, tol, **kw)
        finally:
            pt.coarse_generators = orig


def main(nside, spin=2, tol=1e-2, rtol=1e-7, ctols=(1e-2, 3e-2, 1e-1, 3e-1)):
    dfs, _ = make_spin_dfs(nside, spin=spin)
    A, b = capture(dfs, nside, dfs_fold_plan(nside, spin, tol), rtol=rtol)
    t = time.perf_counter()
    x0, it0, _ = cg_count(A, b, rtol=rtol, maxiter=20000)
    t0 = time.perf_counter() - t
    print(f"nside {nside}   plain {it0} its {t0:.3f} s")
    for ct in ctols:
        t = time.perf_counter()
        T = Restricted(nside, spin, tol, ct)
        tset = time.perf_counter() - t
        t = time.perf_counter()
        x, it, _ = cg_count(A, b, M=T.operator(), rtol=rtol, maxiter=20000)
        dt = time.perf_counter() - t
        print(
            f"   coarse_tol {ct:.0e}  R {T.R:6d}  E nnz {T.E.nnz:9d} "
            f"({T.E.nnz / T.R:6.1f}/row)  {it:5d} its  {dt:6.3f} s  "
            f"wall {t0 / dt:5.2f}x   setup {tset:6.1f} s"
        )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [64]:
        main(ns)
