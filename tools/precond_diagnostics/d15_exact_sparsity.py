"""D15: is E EXACTLY sparse, and does a sparse direct factorisation of it fit?

E_ij is nonzero only if generators i and j touch longitude columns that collide on some
latitude row.  That is a combinatorial condition, not a decay condition, so E should have
STRUCTURAL zeros -- and exploiting those is exact, unlike thresholding, which is what
broke positive definiteness at nside 64 (fix_pass_3.md section 8).
"""

import sys, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, LinearOperator

import precond_common  # noqa: F401
from precond_common import make_spin_dfs, capture, cg_count
from precond_twolevel import TwoLevel
from src.double_fourier_sphere import dfs_fold_plan


def main(nside, spin=2, tol=1e-2, rtol=1e-7):
    T = TwoLevel(nside, spin, tol)
    E, R = T.E, T.R
    exact_nz = np.count_nonzero(E)
    print(
        f"nside {nside}  R {R}   E: structural density "
        f"{100 * exact_nz / R**2:5.2f}%  ({exact_nz} of {R**2})"
    )

    Esp = sp.csc_matrix(E)
    t = time.perf_counter()
    lu = splu(Esp, permc_spec="COLAMD")
    t_fac = time.perf_counter() - t
    fill = (lu.L.nnz + lu.U.nnz) / exact_nz
    print(
        f"  sparse LU: nnz(L)+nnz(U) = {lu.L.nnz + lu.U.nnz} "
        f"(fill-in {fill:.2f}x, {100 * (lu.L.nnz + lu.U.nnz) / R**2:5.2f}% of dense)  "
        f"factorise {t_fac:.2f} s"
    )

    # accuracy of the sparse solve against the dense Cholesky
    rng = np.random.default_rng(0)
    v = rng.standard_normal(R) + 1j * rng.standard_normal(R)
    import scipy.linalg as sla

    ref = sla.cho_solve(T.chol, v)
    got = lu.solve(v)
    print(
        f"  ||E^-1 v (sparse) - E^-1 v (dense)|| / ||.|| = "
        f"{np.linalg.norm(got - ref) / np.linalg.norm(ref):.2e}"
    )

    # end to end
    dfs, _ = make_spin_dfs(nside, spin=spin)
    A, b = capture(dfs, nside, dfs_fold_plan(nside, spin, tol), rtol=rtol)
    t = time.perf_counter()
    x0, it0, _ = cg_count(A, b, rtol=rtol, maxiter=20000)
    t0 = time.perf_counter() - t

    Zs, ZsH = T.Zs, T.ZsH
    Msp = LinearOperator(
        (T.n, T.n), matvec=lambda v: v + Zs @ lu.solve(ZsH @ v), dtype=complex
    )
    t = time.perf_counter()
    x1, it1, _ = cg_count(A, b, M=Msp, rtol=rtol, maxiter=20000)
    t1 = time.perf_counter() - t

    Md = T.operator()
    t = time.perf_counter()
    x2, it2, _ = cg_count(A, b, M=Md, rtol=rtol, maxiter=20000)
    t2 = time.perf_counter() - t

    print(f"  plain      {it0:5d} its  {t0:7.3f} s")
    print(f"  dense  E   {it2:5d} its  {t2:7.3f} s  ({t0 / t2:5.2f}x)")
    print(
        f"  sparse E   {it1:5d} its  {t1:7.3f} s  ({t0 / t1:5.2f}x)   "
        f"|dx| vs dense {np.linalg.norm(x1 - x2) / np.linalg.norm(x2):.1e}"
    )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [32]:
        main(ns)
