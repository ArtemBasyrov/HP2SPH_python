"""D22: is the coefficient-space difference between the two solves really inert?

At nside 128 the plain and two-level latitude solutions differ by 720% as vectors while
the top-band C_l^EE error agrees to four digits.  One metric on one seed is not enough
to call that inert, so check the whole spectrum, both parities, the alm themselves, and
the E->B leakage.
"""

import sys, time, contextlib, io
import numpy as np
import healpy as hp
import scipy.sparse as sp
from scipy.sparse.linalg import splu, LinearOperator

import precond_common  # noqa: F401
import src.nuFFT as _nu
from src.spin_transform import forward_spin
from precond_twolevel import TwoLevel

STATE = {}
ITERS = []


def op_for(nside, tol, eps):
    key = (nside, tol)
    if key not in STATE:
        T = TwoLevel(nside, 2, tol, sparse=True)
        Er = (
            (T.E + eps * T.E.diagonal().real.mean() * sp.eye(T.R, format="csc")).tocsc()
            if eps > 0
            else T.E.tocsc()
        )
        lu = splu(Er)
        Zs, ZsH = T.Zs, T.ZsH
        STATE[key] = LinearOperator(
            (T.n, T.n), matvec=lambda u: u + Zs @ lu.solve(ZsH @ u), dtype=complex
        )
    return STATE[key]


def transform(Q, U, lmax, nside, mode, tol=1e-2, rtol=1e-7, eps=1e-8):
    orig = _nu.cg

    def wrapped(A, b, **kw):
        if mode == "two-level":
            kw["M"] = op_for(nside, tol, eps)
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
            gE, gB = forward_spin(Q, U, lmax, alias_tol=tol, rtol=rtol)
        dt = time.perf_counter() - t
    finally:
        _nu.cg = orig
    return np.ascontiguousarray(gE), np.ascontiguousarray(gB), dt, ITERS[0]


def sky(nside, seed, bmode=True):
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
    return aE, aB, Q, U, lmax


def main(nside, seeds=(0, 1)):
    print(f"nside {nside}")
    for seed in seeds:
        aE, aB, Q, U, lmax = sky(nside, seed)
        gEp, gBp, tp, itp = transform(Q, U, lmax, nside, "plain")
        gEt, gBt, tt, itt = transform(Q, U, lmax, nside, "two-level")

        top = slice(3 * lmax // 4, lmax + 1)
        out = []
        for name, ain, gp, gt in (("E", aE, gEp, gEt), ("B", aB, gBp, gBt)):
            cin = hp.alm2cl(ain, lmax=lmax)
            cp = hp.alm2cl(gp, lmax=lmax)
            ct = hp.alm2cl(gt, lmax=lmax)
            ep = np.sqrt(np.mean((np.abs(cp - cin) / cin)[top] ** 2))
            et = np.sqrt(np.mean((np.abs(ct - cin) / cin)[top] ** 2))
            nz = cp > 0
            dcl = np.max(np.abs(ct - cp)[nz] / cp[nz])
            dalm = np.linalg.norm(gt - gp) / np.linalg.norm(gp)
            out.append(
                f"{name}: plain {ep:.4e} two-level {et:.4e}  "
                f"max_l dC_l/C_l {dcl:.2e}  |d alm|/|alm| {dalm:.2e}"
            )
        print(
            f"  seed {seed}: plain {itp:4d} its {tp:6.2f} s | "
            f"two-level {itt:3d} its {tt:6.2f} s ({tp / tt:.2f}x)"
        )
        for o in out:
            print(f"      {o}")

    # E -> B leakage, pure E input
    aE, aB, Q, U, lmax = sky(nside, 0, bmode=False)
    top = slice(3 * lmax // 4, lmax + 1)
    cin = hp.alm2cl(aE, lmax=lmax)
    for mode in ("plain", "two-level"):
        gE, gB, dt, it = transform(Q, U, lmax, nside, mode)
        leak = np.sqrt(np.mean((hp.alm2cl(gB, lmax=lmax) / cin)[top] ** 2))
        print(f"  E->B leakage, {mode:10s}: {leak:.4e}   ({it} its, {dt:.2f} s)")


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [64]:
        main(ns)
