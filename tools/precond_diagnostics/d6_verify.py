"""D6: verify the closed-form E against the dense operator, then time the solve."""

import sys, time
import numpy as np
from precond_common import make_spin_dfs, capture, densify, cg_count
from src.double_fourier_sphere import dfs_fold_plan
from precond_twolevel import TwoLevel


def verify(nside, spin=2, tol=1e-2):
    n_trans, N_modes = 4 * nside, 4 * nside + 1
    dfs, _ = make_spin_dfs(nside, spin=spin)
    A, b = capture(dfs, nside, dfs_fold_plan(nside, spin, tol))
    N = densify(A)
    N = 0.5 * (N + N.conj().T)
    T = TwoLevel(nside, spin, tol)
    Z = T.Zs.toarray()
    E_ref = Z.conj().T @ N @ Z
    rel = np.linalg.norm(T.E - E_ref) / np.linalg.norm(E_ref)
    _, it0, _ = cg_count(N, b, rtol=1e-7)
    _, it1, _ = cg_count(N, b, M=T.operator(), rtol=1e-7)
    print(
        f"nside {nside}: R {T.R}  ||E_closed - E_dense||/||E|| = {rel:.2e}   "
        f"CG {it0} -> {it1}"
    )


def timed(nside, spin=2, tol=1e-2, rtol=1e-7):
    dfs, _ = make_spin_dfs(nside, spin=spin)
    A, b = capture(dfs, nside, dfs_fold_plan(nside, spin, tol), rtol=rtol)
    t = time.perf_counter()
    x0, it0, _ = cg_count(A, b, rtol=rtol, maxiter=20000)
    t0 = time.perf_counter() - t
    t = time.perf_counter()
    T = TwoLevel(nside, spin, tol)
    tset = time.perf_counter() - t
    Mop = T.operator()
    t = time.perf_counter()
    x1, it1, _ = cg_count(A, b, M=Mop, rtol=rtol, maxiter=20000)
    t1 = time.perf_counter() - t
    print(
        f"nside {nside:4d}  R {T.R:6d}   plain {it0:5d} its {t0:7.3f} s   "
        f"two-level {it1:4d} its {t1:6.3f} s   iters {it0 / it1:5.1f}x  "
        f"solve {t0 / t1:5.2f}x   setup(closed form, cacheable) {tset:6.2f} s   "
        f"|dx|/|x| {np.linalg.norm(x1 - x0) / np.linalg.norm(x0):.1e}"
    )


if __name__ == "__main__":
    mode = sys.argv[1]
    for ns in [int(a) for a in sys.argv[2:]]:
        (verify if mode == "verify" else timed)(ns)
