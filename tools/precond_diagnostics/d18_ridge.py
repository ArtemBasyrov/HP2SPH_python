"""D18: E goes numerically singular as nside grows, so the coarse solve needs a ridge.

cond(Z^H Z) measured by d13 is 1.1e2 / 1.3e2 / 1.2e3 / 1.7e6 at nside 8 / 16 / 32 / 64.
E = Z^H N Z inherits that, and by nside 128 the sparse LU of E returns a solve with
residual 2.75, i.e. worse than returning zero.

Factorising E + eps * mean(diag E) * I instead keeps the matrix sparse and Hermitian
positive definite.  This sweep asks what eps costs in iterations and what it buys in
the accuracy of the coarse solve.  The reference is a tightly converged plain CG.
"""

import sys, time, contextlib, io
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, LinearOperator

import precond_common  # noqa: F401
from precond_common import make_spin_dfs, capture, cg_count
from precond_twolevel import TwoLevel
from src.double_fourier_sphere import dfs_fold_plan


def ridged(T, eps):
    R = T.R
    shift = eps * (T.E.diagonal().real.mean())
    Er = (T.E + shift * sp.eye(R, format="csc")).tocsc()
    t = time.perf_counter()
    lu = splu(Er)
    tfac = time.perf_counter() - t
    rng = np.random.default_rng(0)
    v = rng.standard_normal(R) + 1j * rng.standard_normal(R)
    res = np.linalg.norm(T.E @ lu.solve(v) - v) / np.linalg.norm(v)
    Zs, ZsH = T.Zs, T.ZsH
    op = LinearOperator(
        (T.n, T.n), matvec=lambda u: u + Zs @ lu.solve(ZsH @ u), dtype=complex
    )
    return op, res, tfac


def main(nside, spin=2, tol=1e-2, rtol=1e-7, ref_rtol=1e-11, ref_max=8000):
    dfs, _ = make_spin_dfs(nside, spin=spin)
    A, b = capture(dfs, nside, dfs_fold_plan(nside, spin, tol), rtol=rtol)
    t = time.perf_counter()
    x0, it0, _ = cg_count(A, b, rtol=rtol, maxiter=20000)
    t0 = time.perf_counter() - t
    xref, itref, _ = cg_count(A, b, rtol=ref_rtol, maxiter=ref_max)
    nref = np.linalg.norm(xref)
    print(
        f"nside {nside}   plain {it0} its {t0:.3f} s, "
        f"error vs reference {np.linalg.norm(x0 - xref) / nref:.2e}  "
        f"(reference: {itref} its at rtol {ref_rtol:.0e})"
    )
    T = TwoLevel(nside, spin, tol, sparse=True)
    for eps in (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4):
        try:
            op, res, tfac = ridged(T, eps)
        except Exception as e:
            print(f"   eps {eps:.0e}  factorisation failed: {e}")
            continue
        t = time.perf_counter()
        x, it, _ = cg_count(A, b, M=op, rtol=rtol, maxiter=20000)
        dt = time.perf_counter() - t
        print(
            f"   eps {eps:.0e}  LU residual {res:.2e}  fac {tfac:5.2f} s  "
            f"{it:5d} its  {dt:6.3f} s  wall {t0 / dt:5.2f}x  "
            f"err vs reference {np.linalg.norm(x - xref) / nref:.2e}"
        )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [64]:
        main(ns)
