"""D10: end-to-end physical accuracy with and without the two-level preconditioner.

The preconditioned solve converges to the SAME least-squares solution, but the plain
solve stops far short of it (CG's residual tolerance does not control the error in the
plunge directions).  fix_pass_2 records that tightening the solve has previously made
the physical answer WORSE, so this has to be measured, not assumed.
"""

import sys, time, contextlib, io
import numpy as np
import healpy as hp
import precond_common  # noqa: F401  (sets sys.path)

import src.nuFFT as _nu
from src.spin_transform import forward_spin
from precond_twolevel import TwoLevel

_CACHE = {}


def patch(spin=2, tol=1e-2, enable=True):
    orig = _nu.cg

    def wrapped(A, b, **kw):
        if enable:
            n = A.shape[0]
            nside = _CACHE.get(("ns", n))
            if nside is None:
                for cand in (8, 16, 32, 64, 128, 256):
                    if (4 * cand) * (4 * cand + 1) == n:
                        nside = cand
                        break
                _CACHE[("ns", n)] = nside
            if nside is not None:
                T = _CACHE.get(nside)
                if T is None:
                    T = _CACHE[nside] = TwoLevel(nside, spin, tol)
                kw["M"] = T.operator()
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


ITERS = []


def run(nside, seed, rtol, enable, spin=2, tol=1e-2):
    lmax = 2 * nside
    ell = np.arange(lmax + 1)
    cl = (1.0 + ell) ** (-3.0)
    np.random.seed(seed)
    aE = hp.synalm(cl, lmax=lmax, new=True)
    aB = hp.synalm(cl, lmax=lmax, new=True)
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, lmax)

    orig, wrapped = patch(spin, tol, enable)
    _nu.cg = wrapped
    ITERS.clear()
    try:
        t = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            gE, gB = forward_spin(Q, U, lmax, alias_tol=tol, rtol=rtol)
        dt = time.perf_counter() - t
    finally:
        _nu.cg = orig

    cl_in = hp.alm2cl(aE, lmax=lmax)
    cl_out = hp.alm2cl(np.ascontiguousarray(gE), lmax=lmax)
    rel = np.abs(cl_out - cl_in) / np.maximum(cl_in, 1e-300)
    top = slice(3 * lmax // 4, lmax + 1)
    edge = slice(lmax - 4, lmax + 1)
    return (
        dt,
        ITERS[0] if ITERS else -1,
        np.sqrt(np.mean(rel[top] ** 2)),
        np.sqrt(np.mean(rel[edge] ** 2)),
    )


def main(nside):
    print(
        f"nside {nside}  (C_l^EE relative error, RMS over the top quarter / the last 5 l)"
    )
    for rtol in (1e-7, 1e-9):
        for enable in (False, True):
            r = [run(nside, s, rtol, enable) for s in (0, 1, 2)]
            dt = np.median([x[0] for x in r])
            it = int(np.median([x[1] for x in r]))
            top = np.median([x[2] for x in r])
            edge = np.median([x[3] for x in r])
            tag = "two-level" if enable else "plain    "
            print(
                f"  rtol {rtol:.0e}  {tag}  {it:5d} its  {dt:6.2f} s   "
                f"top {top:.3e}   band edge {edge:.3e}"
            )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [32]:
        main(ns)
