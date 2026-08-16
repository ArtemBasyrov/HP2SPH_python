"""D41: does the PHYSICAL answer need all ~92 CG iterations?

The folded spin solve takes 80-150 iterations where the unfolded one takes 4, and the
extra iterations go into the plunge region -- directions the data do not determine.
d22 measured that content to be physically inert: at nside 128 the plain and two-level
latitude solutions differ by 720% as vectors while the alm agree to 2.9e-7.

If it is inert, CG does not need to resolve it, and truncating the iteration is a
zero-cost fix that works at every nside and needs no matrix at all.  Truncated CG on an
ill-posed problem is regularisation by iteration (Hanke, Nemirovskii), so this is a
principled stopping rule rather than a fudge -- but it has to be measured against the
INPUT sky, because the danger is semi-convergence: the error falls, bottoms out, and
then rises again as the noise-like directions come in.

Reports the top-band and band-edge C_l^EE error against the input alm as a function of
the iteration cap, plus E->B leakage, so the shape of that curve is visible.
"""

import sys, time, contextlib, io
import numpy as np
import healpy as hp

import precond_common  # noqa: F401
import src.nuFFT as _nu
from src.spin_transform import forward_spin

ITERS = []


def capped(maxiter):
    """Force scipy's CG to stop after ``maxiter`` iterations, whatever rtol says."""
    orig = _nu.cg

    def wrapped(A, b, **kw):
        kw["maxiter"] = maxiter
        kw["rtol"] = 0.0 if maxiter is not None else kw.get("rtol", 1e-7)
        cnt = [0]
        cb = kw.pop("callback", None)
        kw["callback"] = lambda x: (
            cnt.__setitem__(0, cnt[0] + 1),
            cb(x) if cb else None,
        )
        out = orig(A, b, **kw)
        ITERS.append(cnt[0])
        return out

    return orig, wrapped


def run(nside, seed, maxiter, tol=1e-2, rtol=1e-7, bmode=True):
    lmax = 2 * nside
    ell = np.arange(lmax + 1)
    np.random.seed(seed)
    aE = hp.synalm((1.0 + ell) ** -3.0, lmax=lmax, new=True)
    aB = (
        hp.synalm((1.0 + ell) ** -3.0, lmax=lmax, new=True)
        if bmode
        else np.zeros_like(aE)
    )
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, lmax)

    orig, wrapped = capped(maxiter)
    if maxiter is not None:
        _nu.cg = wrapped
    ITERS.clear()
    try:
        t = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            gE, gB = forward_spin(Q, U, lmax, alias_tol=tol, rtol=rtol)
        dt = time.perf_counter() - t
    finally:
        _nu.cg = orig

    cin = hp.alm2cl(aE, lmax=lmax)
    top = slice(3 * lmax // 4, lmax + 1)
    relE = np.abs(hp.alm2cl(np.ascontiguousarray(gE), lmax=lmax) - cin) / cin
    leak = (
        np.sqrt(
            np.mean((hp.alm2cl(np.ascontiguousarray(gB), lmax=lmax) / cin)[top] ** 2)
        )
        if not bmode
        else np.nan
    )
    its = ITERS[0] if ITERS else -1
    return (
        dt,
        its,
        np.sqrt(np.mean(relE[top] ** 2)),
        np.sqrt(np.mean(relE[lmax - 4 :] ** 2)),
        leak,
    )


def main(nside, seeds=(0, 1), caps=(4, 8, 12, 16, 24, 32, 48, 64, None)):
    print(f"nside {nside}  (RMS relative C_l^EE error vs the INPUT alm)")
    print("   cap    its    time      top band    band edge    E->B leakage")
    for cap in caps:
        r = [run(nside, s, cap) for s in seeds]
        lk = [run(nside, s, cap, bmode=False)[4] for s in seeds]
        tag = "full" if cap is None else str(cap)
        print(
            f"   {tag:>4s}  {int(np.median([v[1] for v in r])):5d}  "
            f"{np.median([v[0] for v in r]):6.2f} s   "
            f"{np.median([v[2] for v in r]):.4e}   "
            f"{np.median([v[3] for v in r]):.4e}   "
            f"{np.median(lk):.4e}"
        )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [64]:
        main(ns)
