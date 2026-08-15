"""D32: taper the Dirichlet kernel -- the SPD-guaranteed way to sparsify E.

E = (G W Psi)^H D (G W Psi) is a congruence of the positive semi-definite D, so
replacing G by any symmetric G~ leaves E positive semi-definite.  Tapering G to a finite
radius therefore bands E in latitude with no risk of the definiteness loss that killed
thresholding.  The radius is quoted in units of the band-limit resolution pi / L.
"""

import sys, time
import numpy as np

import precond_common  # noqa: F401
from precond_common import make_spin_dfs, capture, cg_count
from precond_twolevel import TwoLevel
from src.double_fourier_sphere import dfs_fold_plan


def main(nside, spin=2, tol=1e-2, rtol=1e-7, tapers=(None, 64, 32, 16, 8, 4)):
    dfs, _ = make_spin_dfs(nside, spin=spin)
    A, b = capture(dfs, nside, dfs_fold_plan(nside, spin, tol), rtol=rtol)
    t = time.perf_counter()
    x0, it0, _ = cg_count(A, b, rtol=rtol, maxiter=20000)
    t0 = time.perf_counter() - t
    print(f"nside {nside}   plain {it0} its {t0:.3f} s")
    for tp in tapers:
        t = time.perf_counter()
        T = TwoLevel(nside, spin, tol, sparse=True, taper=tp)
        tset = time.perf_counter() - t
        nnzLU = T.lu.L.nnz + T.lu.U.nnz
        t = time.perf_counter()
        x, it, _ = cg_count(A, b, M=T.operator(), rtol=rtol, maxiter=20000)
        dt = time.perf_counter() - t
        print(
            f"   taper {str(tp):>4s}  E nnz/row {T.E.nnz / T.R:7.1f}  "
            f"LU nnz/row {nnzLU / T.R:7.1f}  {it:5d} its  {dt:6.3f} s  "
            f"wall {t0 / dt:5.2f}x   setup {tset:6.1f} s"
        )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [32]:
        main(ns)
