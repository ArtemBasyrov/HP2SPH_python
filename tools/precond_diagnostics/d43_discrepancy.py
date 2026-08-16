"""D43: is the right stopping level tied to alias_tol?  (the discrepancy principle)

Truncating CG on this system is iterative regularisation, and the classical rule is
Morozov's discrepancy principle: stop when the residual reaches the level of the error in
the data, ||A x - b|| <= tau * delta with tau >~ 1.  Nemirovskii and Hanke proved that CG
on the normal equations plus the discrepancy principle is a regularisation method with
order-optimal rates, and CGNE needs the fewest iterations of any Krylov method under that
rule.

Here delta is NOT statistical noise. The data are a map; what is wrong is the OPERATOR --
the alias fold asserts that unresolved modes below ``alias_tol`` of a mode's peak vanish.
That is a modelling error rather than data noise, so the classical results do not apply
directly and an extension to perturbed operators is needed. NOTE: this docstring
originally cited "Kaltenbacher et al. (2023)" for that extension while
``latitude_solve_theory.md`` §9 [3] cites Neubauer, J. Inverse Ill-Posed Probl. 30
(2022), 905-915. Both were written from memory and NEITHER has been checked against a
publisher record. Do not repeat either until one is verified.

So the principle predicts the stopping level should track ``alias_tol`` rather than being
a fixed constant, which would explain CLAUDE.md's note that the two "interact".  This
measures whether the knee of the error-vs-rtol curve actually MOVES with alias_tol.  If it
does, the shippable rule is rtol = f(alias_tol) and it needs no per-nside tuning.
"""

import sys, time, contextlib, io
import numpy as np
import healpy as hp

import precond_common  # noqa: F401
import src.nuFFT as _nu
from src.spin_transform import forward_spin

ITERS = []


def run(nside, seed, rtol, alias_tol):
    lmax = 2 * nside
    ell = np.arange(lmax + 1)
    np.random.seed(seed)
    aE = hp.synalm((1.0 + ell) ** -3.0, lmax=lmax, new=True)
    aB = hp.synalm((1.0 + ell) ** -3.0, lmax=lmax, new=True)
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, lmax)

    orig = _nu.cg

    def wrapped(A, b, **kw):
        cnt = [0]
        kw["callback"] = lambda x: cnt.__setitem__(0, cnt[0] + 1)
        out = orig(A, b, **kw)
        ITERS.append(cnt[0])
        return out

    _nu.cg = wrapped
    ITERS.clear()
    try:
        t = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            gE, _ = forward_spin(Q, U, lmax, alias_tol=alias_tol, rtol=rtol)
        dt = time.perf_counter() - t
    finally:
        _nu.cg = orig

    cin = hp.alm2cl(aE, lmax=lmax)
    rel = np.abs(hp.alm2cl(np.ascontiguousarray(gE), lmax=lmax) - cin) / cin
    top = slice(3 * lmax // 4, lmax + 1)
    return dt, ITERS[0], np.sqrt(np.mean(rel[top] ** 2))


def main(nside, seeds=(0, 1), tols=(1e-1, 1e-2, 1e-3), rtols=(1e-3, 1e-4, 1e-5, 1e-6)):
    print(
        f"nside {nside}: top-band C_l^EE error against the input, by (alias_tol, rtol)"
    )
    print("   alias_tol    rtol     its    time      error       vs its own best")
    for atol in tols:
        rows = []
        for rtol in rtols:
            r = [run(nside, s, rtol, atol) for s in seeds]
            rows.append(
                (
                    rtol,
                    int(np.median([v[1] for v in r])),
                    np.median([v[0] for v in r]),
                    np.median([v[2] for v in r]),
                )
            )
        best = min(v[3] for v in rows)
        for rtol, its, dt, err in rows:
            print(
                f"   {atol:.0e}     {rtol:.0e}  {its:6d}  {dt:6.2f} s  "
                f"{err:.4e}   {err / best:6.3f}x"
            )
        print()


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [64]:
        main(ns)
