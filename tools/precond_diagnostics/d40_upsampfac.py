"""D40: finufft upsampling factor -- a free knob at fixed accuracy.

finufft honours ``eps`` whatever ``upsampfac`` is; the factor only trades the size of the
internal FFT against the width of the spreading kernel.  The default 2.0 is tuned for
M >> N.  Here M and N are comparable (M = 4*nside+1 points against N = 4*nside+1 modes on
the half domain), which is the regime where 1.25 is normally recommended.

Also sweeps eps, since fix_pass_2 section 8 only tested the extremes.
"""

import sys, time
import numpy as np
import finufft

import precond_common  # noqa: F401
from src.nuFFT import _upsampled_latitudes


def timeit(fn, n=5):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts)


def bench(nside, half, eps, upsampfac):
    C = 4 * nside
    N_modes = 4 * nside + 1
    x = _upsampled_latitudes(nside)
    if half:
        x = np.ascontiguousarray(x[: 4 * nside + 1])
    M = len(x)
    kw = {} if upsampfac is None else {"upsampfac": float(upsampfac)}
    pf = finufft.Plan(
        2, (N_modes,), n_trans=C, isign=1, dtype=np.complex128, eps=eps, **kw
    )
    pa = finufft.Plan(
        1, (N_modes,), n_trans=C, isign=-1, dtype=np.complex128, eps=eps, **kw
    )
    pf.setpts(x)
    pa.setpts(x)
    rng = np.random.default_rng(0)
    v = rng.standard_normal((C, N_modes)) + 1j * rng.standard_normal((C, N_modes))
    g = np.zeros((C, M), dtype=np.complex128)
    out = np.zeros((C, N_modes), dtype=np.complex128)
    t_f = timeit(lambda: pf.execute(v, g))
    pf.execute(v, g)
    t_a = timeit(lambda: pa.execute(g, out))
    return t_f, t_a, M


def main(nside):
    print(f"nside {nside}")
    for half in (False, True):
        base = None
        for eps in (1e-12, 1e-9):
            for uf in (None, 1.25):
                t_f, t_a, M = bench(nside, half, eps, uf)
                tot = t_f + t_a
                if base is None:
                    base = tot
                tag = (
                    f"{'half' if half else 'full'} M={M:5d} eps={eps:.0e} "
                    f"upsampfac={'default' if uf is None else uf}"
                )
                print(
                    f"   {tag:44s} fwd {1e3 * t_f:7.2f} + adj {1e3 * t_a:7.2f} = "
                    f"{1e3 * tot:7.2f} ms   {base / tot:5.2f}x"
                )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [128]:
        main(ns)
