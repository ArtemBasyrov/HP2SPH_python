"""D23: cut nnz(E) by restricting the coarse Galerkin to a subset of latitude rows.

E = Psi^H W G D G W Psi.  Replacing D by S D S for any diagonal S >= 0 keeps E positive
semi-definite, so unlike thresholding the assembled matrix this cannot destroy
definiteness.  The innermost polar rings alias every longitude mode into a handful of
slots, so they generate most of the coupling; drop them from the COARSE operator only
(the fine operator is untouched) and see what the pattern and the iteration count do.
"""

import sys, time
import numpy as np

import precond_common  # noqa: F401
from precond_common import make_spin_dfs, capture, cg_count
from precond_twolevel import TwoLevel
from src.double_fourier_sphere import dfs_fold_plan


def mask_inner(nside, k):
    """Zero the k innermost rings of each cap, their mirrors, and the two pole rows."""
    n_rings = 4 * nside - 1
    M = 2 * n_rings + 2
    m = np.ones(M)
    if k <= 0:
        return m
    m[0] = 0.0  # north pole row
    m[n_rings + 1] = 0.0  # south pole row
    for i in range(1, k + 1):
        m[i] = 0.0  # north cap ring i
        m[n_rings + 1 - i] = 0.0  # south cap ring
        m[n_rings + 1 + i] = 0.0  # mirror of the south cap ring
        m[M - i] = 0.0  # mirror of the north cap ring
    return m


def main(nside, spin=2, tol=1e-2, rtol=1e-7):
    dfs, _ = make_spin_dfs(nside, spin=spin)
    A, b = capture(dfs, nside, dfs_fold_plan(nside, spin, tol), rtol=rtol)
    t = time.perf_counter()
    x0, it0, _ = cg_count(A, b, rtol=rtol, maxiter=20000)
    t0 = time.perf_counter() - t
    print(f"nside {nside}   plain {it0} its {t0:.3f} s")
    for k in (0, 1, 2, 4, 8):
        rw = mask_inner(nside, k)
        t = time.perf_counter()
        T = TwoLevel(nside, spin, tol, sparse=True, row_weight=None if k == 0 else rw)
        tset = time.perf_counter() - t
        t = time.perf_counter()
        x, it, _ = cg_count(A, b, M=T.operator(), rtol=rtol, maxiter=20000)
        dt = time.perf_counter() - t
        print(
            f"   drop innermost {k:2d} rings: E nnz {T.E.nnz:9d} "
            f"({100 * T.E.nnz / T.R**2:5.2f}%)  LU nnz {T.lu.L.nnz + T.lu.U.nnz:9d}  "
            f"setup {tset:6.2f} s   {it:5d} its  {dt:6.3f} s  wall {t0 / dt:5.2f}x"
        )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [64]:
        main(ns)
