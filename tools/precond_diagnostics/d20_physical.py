"""D20: the only ground truth -- the physical C_l^EE error, plain vs two-level.

At nside 128 the "tightly converged" plain reference used by d18 is itself not
converged (8000 iterations, residual 9.9e-11, hit maxiter), so "error vs reference"
penalises the two-level solve for resolving directions the reference never reached.
Compare both against the INPUT alm instead.
"""

import sys, time, contextlib, io
import numpy as np
import healpy as hp
import scipy.sparse as sp
import scipy.linalg as sla
from scipy.sparse.linalg import splu, LinearOperator

import precond_common  # noqa: F401
import src.nuFFT as _nu
from src.spin_transform import forward_spin
from precond_twolevel import TwoLevel

_T = {}
ITERS = []


def make_op(T, eps):
    if eps > 0:
        Er = (
            T.E + eps * T.E.diagonal().real.mean() * sp.eye(T.R, format="csc")
        ).tocsc()
        lu = splu(Er)
    else:
        lu = splu(T.E.tocsc())
    Zs, ZsH = T.Zs, T.ZsH
    return LinearOperator(
        (T.n, T.n), matvec=lambda u: u + Zs @ lu.solve(ZsH @ u), dtype=complex
    )


def run(nside, seed, mode, eps, tol=1e-2, rtol=1e-7):
    lmax = 2 * nside
    ell = np.arange(lmax + 1)
    np.random.seed(seed)
    aE = hp.synalm((1.0 + ell) ** -3.0, lmax=lmax, new=True)
    aB = hp.synalm((1.0 + ell) ** -3.0, lmax=lmax, new=True)
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, lmax)

    orig = _nu.cg

    def wrapped(A, b, **kw):
        if mode == "two-level":
            key = (nside, tol)
            if key not in _T:
                _T[key] = TwoLevel(nside, 2, tol, sparse=True)
            kw["M"] = _T[key].ops[eps]
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

    cl_in = hp.alm2cl(aE, lmax=lmax)
    cl_out = hp.alm2cl(np.ascontiguousarray(gE), lmax=lmax)
    rel = np.abs(cl_out - cl_in) / np.maximum(cl_in, 1e-300)
    return (
        dt,
        ITERS[0],
        np.sqrt(np.mean(rel[3 * lmax // 4 :] ** 2)),
        np.sqrt(np.mean(rel[lmax - 4 :] ** 2)),
    )


def main(nside, seeds=(0,), epslist=(0.0, 1e-8, 1e-6, 1e-4)):
    print(
        f"nside {nside}  (RMS relative C_l^EE error: top quarter / last 5 multipoles)"
    )
    r = [run(nside, s, "plain", 0.0) for s in seeds]
    print(
        f"  plain                {int(np.median([x[1] for x in r])):6d} its "
        f"{np.median([x[0] for x in r]):7.2f} s   "
        f"top {np.median([x[2] for x in r]):.4e}  edge {np.median([x[3] for x in r]):.4e}"
    )
    T = TwoLevel(nside, 2, 1e-2, sparse=True)
    _T[(nside, 1e-2)] = T
    T.ops = {}
    for eps in epslist:
        t = time.perf_counter()
        T.ops[eps] = make_op(T, eps)
        tf = time.perf_counter() - t
        r = [run(nside, s, "two-level", eps) for s in seeds]
        print(
            f"  two-level eps {eps:.0e}  "
            f"{int(np.median([x[1] for x in r])):6d} its "
            f"{np.median([x[0] for x in r]):7.2f} s   "
            f"top {np.median([x[2] for x in r]):.4e}  "
            f"edge {np.median([x[3] for x in r]):.4e}   (factorise {tf:.1f} s)"
        )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [64]:
        main(ns)
