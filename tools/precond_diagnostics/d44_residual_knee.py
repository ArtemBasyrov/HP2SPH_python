"""D44: where does the physical error bottom out, measured in DATA-residual units?

d43 located the stopping knee in SciPy's ``rtol`` units, i.e. on the NORMAL-equation
residual ``||A^H W (b - Ax)|| / ||A^H W b||``. The discrepancy principle is stated for
the DATA residual ``||b - Ax||_W``, so that calibration is in the wrong units and
carries a constant (1e-3) that has no theoretical meaning. ``nuFFT._cg_monitored`` now
evaluates the data residual exactly and for free, so the knee can be located in the
units of the theorem.

For each (nside, alias_tol, seed) this runs ONE long CG and records, per iteration,

    rho_k   = ||b - A x_k||_W / ||b||_W        (the data residual)
    rrel_k  = the SciPy-convention ratio
    err_k   = top-band RMS relative C_l^EE error against the input alm

The knee is located by a stated criterion rather than by eye: the first iterate whose
error is within ``KNEE_SLACK`` of the best error over the whole run.

Three numbers come out of it. ``rho_inf`` is the residual the iteration stagnates at,
which is the least-squares floor and therefore the modelling error delta itself.
``rho_knee`` is the residual at the knee. Their ratio is the Morozov ``tau``. If
``rho_inf`` tracks ``alias_tol`` and ``tau`` is stable across the grid, the stopping
rule is universal and needs no per-resolution calibration.

Usage: python d44_residual_knee.py [nside ...]
"""

import sys, contextlib, io, time
import numpy as np
import healpy as hp

import precond_common  # noqa: F401
from src.FSHT import FSHT_spin, spin_to_EB_real
from src.spin_transform import forward_spin

KNEE_SLACK = 1.05  # "converged" = within 5% of the best error the run ever reaches
STRIDE = 1  # evaluate the physical error every STRIDE iterations


def _sky(nside, seed, slope=1.5, signal_lmax=None, pure_E=False):
    """A cosmology-like spin-2 sky. ``pure_E`` zeroes B, so output BB is pure leakage."""
    lmax = 2 * nside
    slmax = signal_lmax or lmax
    ell = np.arange(slmax + 1)
    np.random.seed(seed)
    aE = hp.synalm((1.0 + ell) ** (-2.0 * slope), lmax=slmax, new=True)
    aB = hp.synalm((1.0 + ell) ** (-2.0 * slope), lmax=slmax, new=True)
    if pure_E:
        aB = np.zeros_like(aB)
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, slmax)
    return aE, aB, Q, U, lmax


def trace(
    nside, seed, alias_tol, rtol=1e-9, maxiter=4000, signal_lmax=None, pure_E=False
):
    """One long CG; returns per-iteration (k, rho, rrel, C_l error, E->B leakage).

    The leakage column is only meaningful with ``pure_E=True``, where the input B is
    zero and any output BB power is leakage from E.
    """
    aE, aB, Q, U, lmax = _sky(nside, seed, signal_lmax=signal_lmax, pure_E=pure_E)
    cin = hp.alm2cl(aE, lmax=lmax)[: lmax + 1]
    top = slice(3 * lmax // 4, lmax + 1)
    rows = []

    def monitor(k, rho, rrel, get_spectrum):
        if k % STRIDE and k > 1:
            rows.append((k, rho, rrel, np.nan, np.nan))
            return
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

    t = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        forward_spin(
            Q, U, lmax, alias_tol=alias_tol, rtol=rtol, maxiter=maxiter, monitor=monitor
        )
    dt = time.perf_counter() - t
    return np.array(rows), dt


def knee(rows, slack=KNEE_SLACK, col=3):
    """First iterate within ``slack`` of the best value ever reached in column ``col``."""
    ok = np.isfinite(rows[:, col])
    vals = rows[ok, col]
    best = vals.min()
    i = int(np.argmax(vals <= slack * best))
    return rows[ok][i]


def main(nside, tols=(1e-1, 1e-2, 1e-3), seeds=(0, 1, 2, 3)):
    print(
        f"\n=== nside {nside}, spin 2, cosmology (slope 1.5), seeds {list(seeds)} ==="
    )
    print(
        "  alias_tol   its_tot  k_knee   rho_inf     rho_knee    tau      "
        "rrel_knee   err_knee    err_best"
    )
    for atol in tols:
        acc = []
        for s in seeds:
            rows, dt = trace(nside, s, atol)
            kr = knee(rows)
            rho_inf = rows[-1, 1]
            acc.append(
                (
                    rows[-1, 0],
                    kr[0],
                    rho_inf,
                    kr[1],
                    kr[1] / rho_inf,
                    kr[2],
                    kr[3],
                    np.nanmin(rows[:, 3]),
                )
            )
        a = np.array(acc)
        med = np.median(a, axis=0)
        spread = (a[:, 4].max() - a[:, 4].min()) / med[4] if med[4] else 0.0
        print(
            f"   {atol:.0e}    {med[0]:6.0f}  {med[1]:6.0f}   {med[2]:.3e}  "
            f"{med[3]:.3e}  {med[4]:6.3f}  {med[5]:.2e}  {med[6]:.3e}  {med[7]:.3e}"
            f"   (tau spread {spread:.1%})"
        )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [32, 64]:
        main(ns)
