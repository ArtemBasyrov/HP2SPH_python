"""D48: does an accelerated first-order method beat CG on the folded latitude solve?

The question is not whether Nesterov's fast gradient method (FGM) and the optimized
gradient method (OGM) APPLY -- the folded normal operator is Hermitian positive
definite, so they do -- but whether the combination "converges faster + stop earlier"
can win wall time against CG, which minimises the error in the N-norm over the same
Krylov space at every step.

Five configurations are compared, at equal MATRIX-VECTOR COUNT, which is the only
currency that matters: every method here costs exactly one application of
``A^H W A`` per iteration (two NUFFTs), and everything else is inner products.

    cg              plain CG, no early stop  (rtol only)
    cg + eta        the shipped stagnation rule
    fgm             plain FGM, no early stop
    fgm + eta       FGM with the same stagnation rule
    fgm + L-curve   FGM with the parameter-free corner rule of d47

plus OGM-1 and adaptively-restarted FGM, which are free once FGM is written, and
CG + L-curve for a complete grid.

Three things make this a fair test rather than a rigged one.

* ``eta`` and the L-curve are applied OFFLINE to a completed trace, identically for
  every solver, so no solver gets a differently-tuned rule.
* FGM's step size needs ``L = lambda_max(N)``, which is not free. The power iterations
  spent estimating it are counted and reported as an iteration offset.
* Every method is monitored at the point where its gradient is evaluated -- ``x_k`` for
  CG, the extrapolated ``y_k`` for FGM -- so the data residual identity of ``src.cg``
  stays exact and no method pays for an extra operator application.

The trace is the real pipeline (``forward_spin``), not a dense surrogate, so the
physical column is the top-band C_l^EE error against the input alm and the leakage
column is E->B leakage with the input B set to zero.

NOTE on ``eta``: ``d44.trace`` never passes ``eta=None``, so its traces stop at the
SHIPPED stagnation rule rather than running long. This module passes ``eta=None``
explicitly and re-traces; it does not use the d45 cache.

Usage: python d48_accel.py [nside ...]
"""

import contextlib
import io
import os
import sys
import time

import numpy as np
import healpy as hp

import precond_common  # noqa: F401
import src.nuFFT as _nu
from src.cg import cg_normal_equations
from src.FSHT import FSHT_spin, spin_to_EB_real
from src.spin_transform import forward_spin

CACHE = os.environ.get(
    "D48_CACHE",
    "/private/tmp/claude-502/-Users-basyrov-Documents-APC-SHT-HP2SPH-python/"
    "10b48c13-dfc3-4774-a3ee-6ba220787a2c/scratchpad/d48",
)
POWER_ITERS = 20  # matvecs spent estimating L before FGM/OGM can take a step
ETA_SHIPPED = 1e-3
DELAY = 5


# --------------------------------------------------------------------------- solvers


def _power_lmax(matvec, n, iters=POWER_ITERS, seed=0):
    """Largest eigenvalue of ``N`` by power iteration, and the matvecs it cost.

    ``matvec`` returns a view of a buffer it reuses, so every result is copied.
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        w = np.array(matvec(v), copy=True)
        lam = float(np.vdot(v, w).real)
        nw = np.linalg.norm(w)
        if nw == 0.0:
            break
        v = w / nw
    return lam, iters


def _accel_normal_equations(
    matvec,
    rhs,
    bw2,
    norm=1.0,
    rtol=1e-9,
    maxiter=None,
    level=None,
    eta=None,
    delay=5,
    monitor=None,
    variant="fgm",
    restart=False,
    lmax_pad=1.02,
):
    """FGM / OGM-1 on ``min_x 0.5 <x, N x> - Re<x, rhs>``, drop-in for ``cg_normal_equations``.

    Same signature, same stopping rules, same data-residual identity. The iterate the
    rules and the monitor see is the point where the gradient is evaluated: ``y_k`` for
    FGM, ``x_k`` for OGM-1. That keeps the cost at one operator application per
    iteration, which is what makes the comparison against CG at equal iteration count a
    comparison at equal work.

    ``restart`` is O'Donoghue & Candes adaptive restart (the gradient test), the
    standard cure for FGM's non-monotone residual.
    """
    n = len(rhs)
    if maxiter is None:
        maxiter = 10 * n
    b0 = np.sqrt(np.vdot(rhs, rhs).real)
    bnorm = np.sqrt(bw2) if bw2 > 0 else 1.0
    if b0 == 0.0:
        return np.zeros_like(rhs), 0

    L, _ = _power_lmax(matvec, n)
    L *= (
        lmax_pad  # power iteration underestimates from below; pad so 1/L is a safe step
    )
    step = 1.0 / L

    x = np.zeros_like(rhs)  # FGM: the gradient-step point; OGM: y_k
    y = np.zeros_like(rhs)  # the point the gradient is evaluated at
    theta = 1.0
    info = maxiter
    hist = [float(bw2)]
    sol = np.zeros_like(rhs)

    for k in range(1, maxiter + 1):
        Ny = np.array(matvec(y), copy=True)
        g = Ny - rhs  # gradient at y
        rs_new = np.vdot(g, g).real
        rrel = np.sqrt(rs_new) / b0
        # J(y) = ||b||_W^2 - norm * ( Re<y, rhs> + Re<y, rhs - N y> ), the identity of
        # src.cg with r = rhs - N y evaluated at the monitored point.
        J = bw2 - norm * (2.0 * np.vdot(y, rhs).real - np.vdot(y, Ny).real)
        rho = np.sqrt(max(J, 0.0)) / bnorm

        np.copyto(sol, y)
        if monitor is not None:
            monitor(k, rho, rrel, y)

        if level is not None and rho * bnorm <= level:
            info = 0
            break
        if eta is not None:
            hist.append(max(J, 0.0))
            if len(hist) > delay + 1:
                hist.pop(0)
            if len(hist) == delay + 1 and hist[-1] > 0:
                if (hist[0] - hist[-1]) / hist[-1] <= eta:
                    info = 0
                    break
        if rrel <= rtol:
            info = 0
            break

        x_new = y - step * g
        if restart and np.vdot(g, x_new - x).real > 0:
            theta = 1.0
        if variant == "fgm":
            th_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * theta * theta))
            y = x_new + ((theta - 1.0) / th_new) * (x_new - x)
        elif variant == "ogm":
            th_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * theta * theta))
            y = (
                x_new
                + ((theta - 1.0) / th_new) * (x_new - x)
                + (theta / th_new) * (x_new - y)
            )
        else:
            raise ValueError(variant)
        x, theta = x_new, th_new
    return sol, info


SOLVERS = {
    "cg": lambda *a, **kw: cg_normal_equations(*a, **kw),
    "fgm": lambda *a, **kw: _accel_normal_equations(*a, variant="fgm", **kw),
    "fgm-r": lambda *a, **kw: _accel_normal_equations(
        *a, variant="fgm", restart=True, **kw
    ),
    "ogm": lambda *a, **kw: _accel_normal_equations(*a, variant="ogm", **kw),
}
# iterations spent before the first step, charged to the method
OFFSET = {"cg": 0, "fgm": POWER_ITERS, "fgm-r": POWER_ITERS, "ogm": POWER_ITERS}


# ---------------------------------------------------------------------------- traces


def _sky(nside, seed, slope=1.5, pure_E=True):
    lmax = 2 * nside
    ell = np.arange(lmax + 1)
    np.random.seed(seed)
    aE = hp.synalm((1.0 + ell) ** (-2.0 * slope), lmax=lmax, new=True)
    aB = hp.synalm((1.0 + ell) ** (-2.0 * slope), lmax=lmax, new=True)
    if pure_E:
        aB = np.zeros_like(aB)
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, lmax)
    return aE, aB, Q, U, lmax


def trace(nside, seed, alias_tol, solver, maxiter, rtol=1e-12, pure_E=True):
    """One long run of ``solver``; per-iteration (k, rho, rrel, C_l error, leakage).

    ``eta`` is disabled so the trace runs past every stopping rule and the rules can be
    applied offline. ``rtol`` is set below anything reachable for the same reason.
    """
    aE, aB, Q, U, lmax = _sky(nside, seed, pure_E=pure_E)
    cin = hp.alm2cl(aE, lmax=lmax)[: lmax + 1]
    top = slice(3 * lmax // 4, lmax + 1)
    rows = []

    def monitor(k, rho, rrel, get_spectrum):
        F = FSHT_spin(get_spectrum(), 2)
        gE, gB = spin_to_EB_real(
            F, lmax, scale=1.0, colat_phase=False, real_sh_norm=False
        )
        cout = hp.alm2cl(np.ascontiguousarray(gE), lmax=lmax)
        rel = np.abs(cout - cin) / np.maximum(cin, 1e-300)
        err = np.sqrt(np.mean(rel[top] ** 2))
        cB = hp.alm2cl(np.ascontiguousarray(gB), lmax=lmax)
        leak = np.sqrt(np.mean((cB[top] / np.maximum(cin[top], 1e-300)) ** 2))
        rows.append((k, rho, rrel, err, leak))

    orig = _nu.cg_normal_equations
    _nu.cg_normal_equations = SOLVERS[solver]
    t = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            forward_spin(
                Q,
                U,
                lmax,
                alias_tol=alias_tol,
                rtol=rtol,
                maxiter=maxiter,
                eta=None,
                monitor=monitor,
            )
    finally:
        _nu.cg_normal_equations = orig
    return np.array(rows), time.perf_counter() - t


def get_trace(nside, seed, alias_tol, solver, maxiter, pure_E=True):
    os.makedirs(CACHE, exist_ok=True)
    tag = "E" if pure_E else "EB"
    f = os.path.join(
        CACHE, f"ns{nside}_t{alias_tol:.0e}_s{seed}_{solver}_m{maxiter}_{tag}.npy"
    )
    if os.path.exists(f):
        return np.load(f)
    rows, dt = trace(nside, seed, alias_tol, solver, maxiter, pure_E=pure_E)
    np.save(f, rows)
    return rows


# ----------------------------------------------------------------------------- rules


def rule_none(rows):
    return len(rows) - 1


def rule_eta(rows, eta=ETA_SHIPPED, d=DELAY):
    rho = rows[:, 1]
    for i in range(d, len(rho)):
        if (rho[i - d] ** 2 - rho[i] ** 2) / rho[i] ** 2 <= eta:
            return i
    return len(rho) - 1


def rule_lcurve(rows, smooth=3):
    """d47's corner rule: max curvature of log(rho) against log(k)."""
    k = rows[:, 0].astype(float)
    rho = rows[:, 1].astype(float)
    good = (k > 0) & (rho > 0)
    k, rho = k[good], rho[good]
    if len(k) < 5:
        return len(rows) - 1
    t = np.log(k)
    d = np.log(rho)
    d2 = np.gradient(np.gradient(d, t), t)
    d1 = np.gradient(d, t)
    kappa = d2 / (d1**2 + 1.0) ** 1.5
    if smooth > 1:
        kappa = np.convolve(kappa, np.ones(smooth) / smooth, mode="same")
    interior = slice(2, len(kappa) - 2)
    return int(np.flatnonzero(good)[int(np.argmax(kappa[interior])) + interior.start])


CONFIGS = [
    ("cg", "none", "cg  (no stop)"),
    ("cg", "eta", "cg  + eta       <- SHIPPED"),
    ("cg", "lcurve", "cg  + L-curve"),
    ("fgm", "none", "fgm (no stop)"),
    ("fgm", "eta", "fgm + eta"),
    ("fgm", "lcurve", "fgm + L-curve"),
    ("fgm-r", "eta", "fgm-restart + eta"),
    ("ogm", "eta", "ogm + eta"),
]
RULES = {"none": rule_none, "eta": rule_eta, "lcurve": rule_lcurve}


def main(nside, tol=1e-2, seeds=(0, 1, 2, 3), maxiter=400):
    print(
        f"\n=== nside {nside}, spin 2, cosmology (slope 1.5), alias_tol {tol:.0e}, "
        f"seeds {list(seeds)}, maxiter {maxiter} ==="
    )
    traces = {}
    for s in sorted(set(c[0] for c in CONFIGS)):
        traces[s] = [get_trace(nside, sd, tol, s, maxiter) for sd in seeds]

    # --- (a) do the accelerated methods converge faster? residual only, no physics.
    print("\n  (a) CONVERGENCE of the weighted data residual rho_k")
    # per SEED, since each sky has its own least-squares floor
    floors = [
        min(np.nanmin(traces[s][j][1:, 1]) for s in traces) for j in range(len(seeds))
    ]
    print("      rho floor per seed: " + ", ".join(f"{f:.4e}" for f in floors))
    marks = (1.5, 1.1, 1.02, 1.005)
    print("      its to reach rho <= f * rho_floor (median over seeds; '-' = never)")
    print(
        "      solver   "
        + "".join(f"  f={m:<7g}" for m in marks)
        + "   rho @ 50 its   rho @ 200 its"
    )
    for s in traces:
        cells = []
        for m in marks:
            hit = []
            for j, t in enumerate(traces[s]):
                idx = np.flatnonzero(t[:, 1] <= m * floors[j])
                hit.append(t[idx[0], 0] + OFFSET[s] if len(idx) else np.nan)
            cells.append(
                "      -  " if np.all(np.isnan(hit)) else f"{np.nanmedian(hit):8.0f}  "
            )

        def at(k):
            v = [t[min(k, len(t)) - 1, 1] for t in traces[s]]
            return np.median(v)

        print(f"      {s:8s} " + "".join(cells) + f"  {at(50):.6e}   {at(200):.6e}")

    # --- (b) the physical answer, per seed, against that seed's own best.
    print("\n  (b) PHYSICAL error at the stop, each seed against its own best iterate")
    # the first monitored row is x_1 for CG but the zero vector for FGM/OGM, so the
    # references skip it; a zero iterate has zero leakage and would be a false floor.
    ref_err = [
        min(np.nanmin(traces[s][j][1:, 3]) for s in traces) for j in range(len(seeds))
    ]
    ref_leak = [
        min(np.nanmin(traces[s][j][1:, 4]) for s in traces) for j in range(len(seeds))
    ]
    print(f"      best C_l error per seed: " + ", ".join(f"{e:.3e}" for e in ref_err))
    print("      its = matvecs at the stop, INCLUDING the L estimate for fgm/ogm\n")
    print(
        "      config                       its(med)  C_l err     cost   worst   "
        "leak cost  worst"
    )
    for solver, rule, label in CONFIGS:
        fn = RULES[rule]
        its, cost, lcost, raw = [], [], [], []
        for j, t in enumerate(traces[solver]):
            i = fn(t)
            its.append(t[i, 0] + OFFSET[solver])
            raw.append(t[i, 3])
            cost.append(t[i, 3] / ref_err[j])
            lcost.append(t[i, 4] / ref_leak[j])
        cost, lcost = np.array(cost), np.array(lcost)
        print(
            f"      {label:26s} {np.median(its):7.0f}   {np.median(raw):.4e}  "
            f"{np.median(cost):6.3f}x {np.max(cost):6.3f}x  "
            f"{np.median(lcost):6.3f}x {np.max(lcost):6.3f}x"
        )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [32, 64]:
        main(ns)
