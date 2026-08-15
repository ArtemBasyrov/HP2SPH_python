"""D19: is E itself sound at nside 128, or is the sparse LU at fault?

Checks, per nside:
  * Hermiticity of the assembled E
  * the diagonal: any exactly-zero or negative entry makes E singular / indefinite
  * whether an ITERATIVE solve on the same E succeeds where splu does not
"""

import sys, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, cg as spcg, LinearOperator

import precond_common  # noqa: F401
from precond_twolevel import TwoLevel


def main(nside, spin=2, tol=1e-2):
    T = TwoLevel(nside, spin, tol, sparse=True)
    E = T.E.tocsr()
    R = T.R
    herm = abs(E - E.conj().T).max() / abs(E).max()
    d = E.diagonal()
    print(f"nside {nside:4d}  R {R:6d}  nnz {E.nnz}")
    print(f"    ||E - E^H||_max / ||E||_max = {herm:.2e}")
    print(
        f"    diagonal: min {d.real.min():.3e}  max {d.real.max():.3e}  "
        f"exact zeros {int(np.sum(d.real == 0))}  negatives {int(np.sum(d.real < 0))}"
        f"  max |imag| {np.abs(d.imag).max():.2e}"
    )
    empty = np.diff(E.indptr) == 0
    print(f"    structurally empty rows: {int(empty.sum())}")

    rng = np.random.default_rng(0)
    v = rng.standard_normal(R) + 1j * rng.standard_normal(R)

    t = time.perf_counter()
    lu = splu(T.E.tocsc())
    tfac = time.perf_counter() - t
    ylu = lu.solve(v)
    rlu = np.linalg.norm(E @ ylu - v) / np.linalg.norm(v)

    # Jacobi-preconditioned CG on the same E
    dd = np.where(np.abs(d) > 0, 1.0 / d, 1.0)
    Mj = LinearOperator((R, R), matvec=lambda u: dd * u, dtype=complex)
    t = time.perf_counter()
    ycg, info = spcg(E, v, M=Mj, rtol=1e-10, maxiter=5000)
    tcg = time.perf_counter() - t
    rcg = np.linalg.norm(E @ ycg - v) / np.linalg.norm(v)
    print(
        f"    splu: fac {tfac:6.2f} s  residual {rlu:.2e}   "
        f"| CG(jacobi): {tcg:6.2f} s  info {info}  residual {rcg:.2e}"
    )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [64]:
        main(ns)
