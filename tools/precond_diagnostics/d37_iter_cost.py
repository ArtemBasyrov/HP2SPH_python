"""D37: where does one CG iteration actually go?

The iteration COUNT is flat at 80-90 above nside 64 (measured 151 / 115 / 92 / 81 at
nside 32 / 64 / 128 / 256, i.e. it peaks at nside 32 and falls), so at
high nside the only thing left to attack is the cost of one matrix-vector product.
Break it into its five pieces before optimising any of them:

    forward NUFFT (type 2)  ->  fold_apply  ->  weights  ->  fold_adjoint
                                                          ->  adjoint NUFFT (type 1)
"""

import sys, time, contextlib, io
import numpy as np
import healpy as hp
import finufft

import precond_common  # noqa: F401
from src.nuFFT import compute_voronoi_weights_1d, _fold_ops, _upsampled_latitudes
from src.double_fourier_sphere import dfs_fold_plan


def timeit(fn, n=5):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts)


def main(nside, spin=2, tol=1e-2, eps=1e-12):
    n_trans = 4 * nside
    M = 8 * nside
    N_modes = 4 * nside + 1
    x = _upsampled_latitudes(nside)
    assert len(x) == M, (len(x), M)

    target, phase, keep = dfs_fold_plan(nside, spin, tol)
    fold_apply, fold_adjoint = _fold_ops((target, phase), n_trans, M)
    w = compute_voronoi_weights_1d(x)
    weights = w[None, :] * keep.T.astype(float)

    plan_f = finufft.Plan(
        2, (N_modes,), n_trans=n_trans, isign=1, dtype=np.complex128, eps=eps
    )
    plan_a = finufft.Plan(
        1, (N_modes,), n_trans=n_trans, isign=-1, dtype=np.complex128, eps=eps
    )
    plan_f.setpts(x)
    plan_a.setpts(x)

    rng = np.random.default_rng(0)
    vec = rng.standard_normal((n_trans, N_modes)) + 1j * rng.standard_normal(
        (n_trans, N_modes)
    )
    g = np.zeros((n_trans, M), dtype=np.complex128)
    out = np.zeros((n_trans, N_modes), dtype=np.complex128)
    plan_f.execute(vec, g)
    y = fold_apply(g)

    t_fwd = timeit(lambda: plan_f.execute(vec, g))
    t_fold = timeit(lambda: fold_apply(g))
    t_w = timeit(lambda: g * weights)
    t_adjf = timeit(lambda: fold_adjoint(y))
    t_adj = timeit(lambda: plan_a.execute(np.ascontiguousarray(y), out))
    t_ct = timeit(lambda: np.ascontiguousarray(y))

    total = t_fwd + t_fold + t_w + t_adjf + t_adj + t_ct
    parts = (
        ("forward NUFFT ", t_fwd),
        ("fold_apply    ", t_fold),
        ("weights       ", t_w),
        ("fold_adjoint  ", t_adjf),
        ("ascontiguous  ", t_ct),
        ("adjoint NUFFT ", t_adj),
    )
    print(
        f"nside {nside:4d}   n_trans {n_trans}  M {M}  N_modes {N_modes}   "
        f"one matvec ~ {1e3 * total:7.2f} ms"
    )
    for name, t in parts:
        print(f"     {name} {1e3 * t:8.2f} ms  {100 * t / total:5.1f}%")
    nufft = t_fwd + t_adj
    print(
        f"     -> NUFFT {100 * nufft / total:4.1f}%,  "
        f"fold+glue {100 * (total - nufft) / total:4.1f}%"
    )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [128]:
        main(ns)
