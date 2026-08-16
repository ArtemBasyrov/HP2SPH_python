"""D42: which rtol reproduces the truncation sweet spot?

d41 shows the physical answer saturates after 12-24 CG iterations while the default
``rtol=1e-7`` runs 81-115. A fixed iteration cap would be a crude way to exploit that;
``rtol`` is the knob that already exists, is data adaptive, and needs no new code.

This maps rtol -> iterations -> physical error so a default can be chosen, and it also
checks the ALIASED regime, where CLAUDE.md records that tightening the solve made the
physical answer three orders WORSE. Early stopping should be safer there, not riskier,
and that has to be shown rather than assumed.
"""

import sys, time, contextlib, io
import numpy as np
import healpy as hp

import precond_common  # noqa: F401
import src.nuFFT as _nu
from src.spin_transform import forward_spin

ITERS = []


def run(nside, seed, rtol, signal_lmax=None, tol=1e-2):
    lmax = 2 * nside
    slmax = signal_lmax or lmax
    ell = np.arange(slmax + 1)
    np.random.seed(seed)
    aE = hp.synalm((1.0 + ell) ** -3.0, lmax=slmax, new=True)
    aB = hp.synalm((1.0 + ell) ** -3.0, lmax=slmax, new=True)
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, slmax)

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
            gE, _ = forward_spin(Q, U, lmax, alias_tol=tol, rtol=rtol)
        dt = time.perf_counter() - t
    finally:
        _nu.cg = orig

    cin = hp.alm2cl(aE, lmax=lmax)[: lmax + 1]
    cout = hp.alm2cl(np.ascontiguousarray(gE), lmax=lmax)
    rel = np.abs(cout - cin) / np.maximum(cin, 1e-300)
    top = slice(3 * lmax // 4, lmax + 1)
    return (
        dt,
        ITERS[0],
        np.sqrt(np.mean(rel[top] ** 2)),
        np.sqrt(np.mean(rel[lmax - 4 :] ** 2)),
    )


def sweep(nside, seeds, rtols, signal_lmax=None, label=""):
    print(f"  {label or 'cosmology'} nside {nside}")
    print("     rtol      its    time      top band     band edge")
    for rtol in rtols:
        r = [run(nside, s, rtol, signal_lmax) for s in seeds]
        print(
            f"     {rtol:.0e}  {int(np.median([v[1] for v in r])):5d}  "
            f"{np.median([v[0] for v in r]):6.2f} s   "
            f"{np.median([v[2] for v in r]):.4e}   "
            f"{np.median([v[3] for v in r]):.4e}"
        )


def main(nside, seeds=(0, 1)):
    sweep(nside, seeds, (1e-3, 1e-4, 1e-5, 1e-6, 1e-7))
    # the diagnostic-only regime where a tighter solve was recorded as much worse
    sweep(nside, seeds, (1e-3, 1e-5, 1e-7), signal_lmax=4 * nside, label="aliased")


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [64]:
        main(ns)
