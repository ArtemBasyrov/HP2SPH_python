"""D12: the practical payoff -- a tighter alias_tol at unchanged cost.

CLAUDE.md records alias_tol=1e-3 / rtol=1e-8 as 3x more accurate than the defaults and
6.7x dearer (nside 64: 3.85e-6 at 9.85 s against 1.18e-5 at 1.46 s).  The preconditioner
should remove most of that cost.
"""

import sys, time, contextlib, io
import numpy as np
import healpy as hp
import precond_common  # noqa: F401

import src.nuFFT as _nu
from src.spin_transform import forward_spin
from precond_twolevel import TwoLevel

_T = {}
ITERS = []


def run(nside, seed, tol, rtol, enable):
    lmax = 2 * nside
    ell = np.arange(lmax + 1)
    np.random.seed(seed)
    aE = hp.synalm((1.0 + ell) ** -3.0, lmax=lmax, new=True)
    aB = hp.synalm((1.0 + ell) ** -3.0, lmax=lmax, new=True)
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, lmax)

    orig = _nu.cg

    def wrapped(A, b, **kw):
        if enable:
            key = (nside, tol)
            if key not in _T:
                _T[key] = TwoLevel(nside, 2, tol)
            T = _T[key]
            import scipy.linalg as sla
            from scipy.sparse.linalg import LinearOperator

            c, low = T.chol
            c64 = c.astype(np.complex64)

            def apply(v):
                return v + T.Zs @ sla.cho_solve(
                    (c64, low), (T.ZsH @ v).astype(np.complex64)
                ).astype(complex)

            kw["M"] = LinearOperator((T.n, T.n), matvec=apply, dtype=complex)
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


def main(nside):
    print(
        f"nside {nside}   (RMS relative C_l^EE error: top quarter / last 5 multipoles)"
    )
    for tol, rtol in ((1e-2, 1e-7), (1e-3, 1e-8)):
        for enable in (False, True):
            r = [run(nside, s, tol, rtol, enable) for s in (0, 1, 2)]
            print(
                f"  alias_tol {tol:.0e} rtol {rtol:.0e}  "
                f"{'two-level' if enable else 'plain    '}  "
                f"{int(np.median([x[1] for x in r])):6d} its  "
                f"{np.median([x[0] for x in r]):7.2f} s   "
                f"top {np.median([x[2] for x in r]):.3e}   "
                f"edge {np.median([x[3] for x in r]):.3e}"
            )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [32]:
        main(ns)
