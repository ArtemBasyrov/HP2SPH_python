"""D17: why do the plain and two-level solutions differ by 720% at nside 128?

Two candidates:
  (a) the sparse LU of E is inaccurate at 3e7 nonzeros;
  (b) the system is genuinely near-singular, the plain solve leaves the near-null
      directions at ~0, and the converged solve fills them with something large.

(a) is checked by the coarse residual, (b) by the fine residual of both solutions and by
the physical C_l^EE error.  If (b), the question is which answer is physically right.
"""

import sys, time, contextlib, io
import numpy as np
import healpy as hp

import precond_common  # noqa: F401
from precond_common import make_spin_dfs, capture, cg_count
from precond_twolevel import TwoLevel
from src.double_fourier_sphere import dfs_fold_plan


def main(nside, spin=2, tol=1e-2, rtol=1e-7):
    lmax = 2 * nside
    ell = np.arange(lmax + 1)
    np.random.seed(0)
    aE = hp.synalm((1.0 + ell) ** -3.0, lmax=lmax, new=True)
    aB = hp.synalm((1.0 + ell) ** -3.0, lmax=lmax, new=True)
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, lmax)
    z = Q + 1j * U

    from src.data_interpolation import transform_healpix_to_grid
    from src.double_fourier_sphere import DFS

    with contextlib.redirect_stdout(io.StringIO()):
        up, fc = transform_healpix_to_grid(z)
        _, dfs = DFS(up, fc, spin=spin)

    A, b = capture(dfs, nside, dfs_fold_plan(nside, spin, tol), rtol=rtol)
    T = TwoLevel(nside, spin, tol, sparse=True)

    # (a) coarse solve accuracy
    rng = np.random.default_rng(0)
    v = rng.standard_normal(T.R) + 1j * rng.standard_normal(T.R)
    y = T._solve(v)
    res = np.linalg.norm(T.E @ y - v) / np.linalg.norm(v)
    print(f"nside {nside}: coarse LU residual ||E y - v||/||v|| = {res:.2e}")

    # (b) fine residuals
    x0, it0, _ = cg_count(A, b, rtol=rtol, maxiter=20000)
    x1, it1, _ = cg_count(A, b, M=T.operator(), rtol=rtol, maxiter=20000)
    for tag, x in (("plain    ", x0), ("two-level", x1)):
        r = np.linalg.norm(A.matvec(x) - b) / np.linalg.norm(b)
        print(
            f"  {tag}  its {(it0 if tag.startswith('plain') else it1):5d}  "
            f"||Ax-b||/||b|| {r:.3e}   ||x|| {np.linalg.norm(x):.4e}"
        )

    # a much tighter plain solve, as an independent reference
    x2, it2, _ = cg_count(A, b, rtol=1e-11, maxiter=8000)
    r2 = np.linalg.norm(A.matvec(x2) - b) / np.linalg.norm(b)
    print(
        f"  plain tight       its {it2:5d}  ||Ax-b||/||b|| {r2:.3e}  "
        f"||x|| {np.linalg.norm(x2):.4e}   "
        f"|x2-x1|/|x2| {np.linalg.norm(x2 - x1) / np.linalg.norm(x2):.2e}   "
        f"|x2-x0|/|x2| {np.linalg.norm(x2 - x0) / np.linalg.norm(x2):.2e}"
    )

    # (c) physical accuracy of each
    from src.FSHT import spin_alm_from_F, spin_g_to_library  # noqa: F401
    from src.spin_transform import forward_spin

    for label, kw in (("plain", dict()), ("two-level", dict())):
        pass
    print("  physical C_l^EE error is measured by d18_accuracy128.py")


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [64]:
        main(ns)
