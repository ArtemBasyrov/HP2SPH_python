"""D45: a stopping rule stated in the units of the theorem, and its calibration.

d44 showed two things that change the shape of the stopping rule.

The weighted data residual ``rho_k = ||b - A x_k||_W / ||b||_W`` does NOT fall to zero.
It stagnates at a floor ``rho_inf``, which is the least-squares residual and therefore
the modelling error ``delta`` that the discrepancy principle asks for. And ``rho_inf``
is NOT proportional to ``alias_tol``: it saturates at an nside-dependent floor as soon
as ``alias_tol <= 3e-2``, because below that the alias assertion is no longer the
dominant term in the model error.

So Morozov's rule cannot be applied with ``delta = tau * alias_tol * ||b||``: for the
shipped default that level is unreachable and the iteration would run to the cap. The
level has to be ``delta = rho_inf * ||b||_W``, which is not known before the solve. It
is, however, observable during it, because ``rho_k`` decreases monotonically to it.

This calibrates the resulting online rule,

    stop at the first k >= d with (rho_{k-d}^2 - rho_k^2) / rho_k^2 <= eta,

against the fully converged answer. ``eta`` is dimensionless and ``d`` is a delay in
iterations. The output is the cost of the rule (error relative to fully converged) and
its benefit (iterations saved), over a grid in nside, alias_tol and seed, so that one
``eta`` can be chosen and its worst case quoted.

Traces are cached under the scratch directory, one file per (nside, alias_tol, seed);
delete it to re-measure.

Usage: python d45_stagnation.py [nside ...]
"""

import os
import sys
import numpy as np

import precond_common  # noqa: F401
from d44_residual_knee import trace

CACHE = os.environ.get(
    "D45_CACHE",
    "/private/tmp/claude-502/-Users-basyrov-Documents-APC-SHT-HP2SPH-python/"
    "6932b40e-2af2-491d-aaa1-14bd1b160b29/scratchpad/d45",
)
ETAS = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
DELAY = 5


def get_trace(nside, seed, tol, maxiter, pure_E=False):
    os.makedirs(CACHE, exist_ok=True)
    tag = "E" if pure_E else "EB"
    f = os.path.join(CACHE, f"ns{nside}_t{tol:.0e}_s{seed}_{tag}.npy")
    if os.path.exists(f):
        return np.load(f)
    rows, _ = trace(nside, seed, tol, rtol=1e-9, maxiter=maxiter, pure_E=pure_E)
    np.save(f, rows)
    return rows


def apply_rule(rows, eta, d=DELAY):
    """Index of the first iterate satisfying the stagnation rule, or the last one."""
    rho = rows[:, 1]
    for i in range(d, len(rho)):
        if (rho[i - d] ** 2 - rho[i] ** 2) / rho[i] ** 2 <= eta:
            return i
    return len(rho) - 1


def main(nside, tols=(1e-1, 1e-2), seeds=(0, 1, 2, 3), maxiter=600, pure_E=True):
    print(f"\n=== nside {nside}, spin 2, cosmology (slope 1.5), delay d={DELAY} ===")
    print("cost = error at the stop / error fully converged; speedup = its saved")
    for tol in tols:
        traces = [get_trace(nside, s, tol, maxiter, pure_E) for s in seeds]
        n_full = np.median([t[-1, 0] for t in traces])
        e_full = [t[-1, 3] for t in traces]
        l_full = [t[-1, 4] for t in traces]
        print(
            f"\n  alias_tol {tol:.0e}: fully converged in {n_full:.0f} its, "
            f"C_l err {np.median(e_full):.3e}, leakage {np.median(l_full):.3e}"
        )
        print(
            "     eta       its   speedup   rho/rho_inf   C_l cost   worst   leak cost"
        )
        for eta in ETAS:
            its, cost, rr, leak = [], [], [], []
            for t, ef, lf in zip(traces, e_full, l_full):
                i = apply_rule(t, eta)
                its.append(t[i, 0])
                cost.append(t[i, 3] / ef)
                leak.append(t[i, 4] / lf)
                rr.append(t[i, 1] / t[-1, 1])
            print(
                f"    {eta:.0e}  {np.median(its):6.0f}  {n_full / np.median(its):6.1f}x"
                f"    {np.median(rr):9.6f}   {np.median(cost):7.3f}x "
                f"{np.max(cost):7.3f}x   {np.median(leak):7.3f}x"
            )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [32, 64]:
        main(ns)
