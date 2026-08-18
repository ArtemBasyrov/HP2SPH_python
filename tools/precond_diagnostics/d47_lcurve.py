"""D47: the L-curve corner as a parameter-free alternative to ``eta``.

Qu, Zhong, Zhang, Wang & Shen (2005), "Convergence behavior of iterative SENSE
reconstruction with non-Cartesian trajectories", MRM 54(4), 1040-1045, DOI
10.1002/mrm.20648, stop CG at the corner of the residual L-curve: with
``delta(k) = log Delta(k)``, the corner is the point of maximum curvature

    kappa(k) = delta'' / ( (delta')^2 + 1 )^{3/2} .

They reach that rule from the same premise this project reached from Nemirovskii
and Neubauer -- the iteration count IS the regularisation parameter -- so the two
rules are stopping the same phenomenon by different means.

Why it is worth testing here. ``ALIAS_ETA = 1e-3`` is a calibrated constant, and
CLAUDE.md carries the open debt that it was calibrated over nside 8-256 and is only
ASSUMED to carry to 1024 and 2048. The L-curve corner carries no constant at all.
If it lands where ``eta`` lands, it removes the debt for free.

Two differences from Qu et al. are deliberate. Their ``Delta(k)`` is the
NORMAL-equation residual; this uses the data residual ``rho_k``, which is the
quantity the discrepancy principle is stated for and which ``nuFFT._cg_monitored``
already computes. And the curvature is evaluated against ``log k`` rather than ``k``,
since the residual falls over decades of iteration count and the corner of a
log-log curve is the scale-free one.

This is an OFFLINE evaluation: it locates the corner from a completed trace. A
shipped version would need an online detector, which must run past the corner to
know it has passed it. See the closing note for what that costs.

Usage: python d47_lcurve.py [nside ...]
"""

import sys
import numpy as np

import precond_common  # noqa: F401
from d45_stagnation import get_trace, apply_rule, DELAY

ETA_SHIPPED = 1e-3
KNEE_SLACK = 1.05  # same definition of "converged" as d44


def lcurve_corner(rows, smooth=3, logk=True):
    """Index of the maximum-curvature point of ``log rho`` against ``k``.

    ``logk=False`` is Qu et al.'s rule as they state it, with the curvature taken
    against the iteration count itself. ``logk=True`` takes it against ``log k``,
    on the reasoning that the residual falls over decades of iteration count and
    the corner of a log-log curve is the scale-free one. Both are measured; see
    the results table for which is better.

    ``smooth`` is a centred moving-average width applied to the curvature before
    taking its maximum. The discrete second difference of a CG residual is noisy,
    and without it the maximum lands on a single-iteration wobble.
    """
    k = rows[:, 0].astype(float)
    rho = rows[:, 1].astype(float)
    good = (k > 0) & (rho > 0)
    k, rho = k[good], rho[good]
    if len(k) < 5:
        return len(rows) - 1
    t = np.log(k) if logk else k
    d = np.log(rho)
    d1 = np.gradient(d, t)
    d2 = np.gradient(d1, t)
    kappa = d2 / (d1**2 + 1.0) ** 1.5
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        kappa = np.convolve(kappa, kernel, mode="same")
    # the corner is where the curve bends from falling to flat, i.e. kappa is most
    # POSITIVE; the endpoints are excluded because np.gradient is one-sided there
    interior = slice(2, len(kappa) - 2)
    idx = int(np.argmax(kappa[interior])) + interior.start
    return int(np.flatnonzero(good)[idx])


def knee(rows, slack=KNEE_SLACK):
    """First iterate within ``slack`` of the best error the run ever reaches."""
    err = rows[:, 3]
    finite = np.isfinite(err)
    best = np.min(err[finite])
    for i in range(len(err)):
        if finite[i] and err[i] <= slack * best:
            return i
    return len(err) - 1


def main(nside, tols=(1e-1, 1e-2), seeds=(0, 1, 2, 3), maxiter=600, pure_E=True):
    print(f"\n=== nside {nside}, spin 2, cosmology (slope 1.5), 4 seeds ===")
    print("cost = C_l error at the stop, relative to fully converged")
    for tol in tols:
        traces = [get_trace(nside, s, tol, maxiter, pure_E) for s in seeds]
        n_full = np.median([t[-1, 0] for t in traces])
        print(f"\n  alias_tol {tol:.0e}: fully converged in {n_full:.0f} its")
        print("    rule            its   C_l cost  worst   leak cost  worst")
        rules = {
            f"eta={ETA_SHIPPED:.0e}": lambda t: apply_rule(t, ETA_SHIPPED, DELAY),
            "L-curve (log k)": lambda t: lcurve_corner(t, logk=True),
            "L-curve (k)": lambda t: lcurve_corner(t, logk=False),
            "knee (oracle)": knee,
        }
        for name, fn in rules.items():
            its, cost, leak = [], [], []
            for t in traces:
                i = fn(t)
                its.append(t[i, 0])
                cost.append(t[i, 3] / t[-1, 3])
                leak.append(t[i, 4] / t[-1, 4])
            print(
                f"    {name:14s} {np.median(its):5.0f}   "
                f"{np.median(cost):7.3f}x {np.max(cost):7.3f}x  "
                f"{np.median(leak):7.3f}x {np.max(leak):7.3f}x"
            )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [32, 64]:
        main(ns)
