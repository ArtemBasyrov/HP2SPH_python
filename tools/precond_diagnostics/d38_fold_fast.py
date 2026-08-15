"""D38: make the alias fold cheap.  It is 34.6% of a matvec at nside 256.

The current ``_fold_ops`` works in the (M, n_trans) layout while the solver holds
(n_trans, M), so every call transposes in and out.  It also bincounts ``t.real`` and
``t.imag``, which are stride-16 views of a complex array, and returns a transposed VIEW,
which forces the caller's ``np.ascontiguousarray`` to copy 33 MB.

Fixed version:
  * precompute the scatter index in the (n_trans, M) layout, so there are no transposes;
  * view the complex buffer as interleaved float64 and scatter with ONE bincount over
    2 * size contiguous values instead of two over strided ones;
  * return a C-contiguous array, which makes the caller's copy disappear.

Verified against the original to machine precision, including the adjoint identity.
"""

import sys, time
import numpy as np

import precond_common  # noqa: F401
from src.nuFFT import _fold_ops, _upsampled_latitudes
from src.double_fourier_sphere import dfs_fold_plan


def fold_ops_fast(fold, n_trans, M_samples):
    """(apply, adjoint) in the (n_trans, M) layout, no transposes, one bincount."""
    if fold is None:
        return None, None
    target = np.asarray(fold[0])
    phase = np.asarray(fold[1])
    if target.shape != (M_samples, n_trans):
        raise ValueError(f"fold arrays must be ({M_samples}, {n_trans})")
    size = M_samples * n_trans

    # index of (column c, row r) in a (n_trans, M) C-contiguous buffer is c * M + r;
    # it scatters onto (target[r, c], r), i.e. target[r, c] * M + r
    r_idx = np.arange(M_samples)[None, :]
    dest = (target.T * M_samples + r_idx).astype(np.int64)  # (n_trans, M)
    flat = dest.ravel()
    # interleaved real/imaginary destinations, so one bincount does both halves
    pair = np.empty(2 * size, dtype=np.int64)
    pair[0::2] = 2 * flat
    pair[1::2] = 2 * flat + 1

    phase_t = np.ascontiguousarray(phase.T)  # (n_trans, M)
    conj_phase_t = np.ascontiguousarray(np.conj(phase.T))

    def apply(model):
        t = model * phase_t
        w = t.view(np.float64).ravel()
        acc = np.bincount(pair, weights=w, minlength=2 * size)
        return acc.view(np.complex128).reshape(n_trans, M_samples)

    def adjoint(residual):
        gathered = residual.ravel()[flat].reshape(n_trans, M_samples)
        return gathered * conj_phase_t

    return apply, adjoint


def timeit(fn, n=5):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts)


def main(nside, spin=2, tol=1e-2):
    n_trans, M = 4 * nside, 8 * nside
    target, phase, keep = dfs_fold_plan(nside, spin, tol)
    a0, j0 = _fold_ops((target, phase), n_trans, M)
    a1, j1 = fold_ops_fast((target, phase), n_trans, M)

    rng = np.random.default_rng(0)
    g = rng.standard_normal((n_trans, M)) + 1j * rng.standard_normal((n_trans, M))
    y = rng.standard_normal((n_trans, M)) + 1j * rng.standard_normal((n_trans, M))

    e_a = np.abs(a1(g) - a0(g)).max() / np.abs(a0(g)).max()
    e_j = np.abs(j1(y) - j0(y)).max() / np.abs(j0(y)).max()
    # adjoint identity <apply(g), y> == <g, adjoint(y)>
    lhs = np.vdot(a1(g), y)
    rhs = np.vdot(g, j1(y))
    print(
        f"nside {nside:4d}  apply err {e_a:.2e}  adjoint err {e_j:.2e}  "
        f"adjointness {abs(lhs - rhs) / abs(lhs):.2e}"
    )

    t_a0 = timeit(lambda: a0(g))
    t_a1 = timeit(lambda: a1(g))
    t_j0 = timeit(lambda: j0(y))
    t_j1 = timeit(lambda: j1(y))
    t_c0 = timeit(lambda: np.ascontiguousarray(a0(g)))
    t_c1 = timeit(lambda: np.ascontiguousarray(a1(g)))
    print(
        f"    fold_apply    {1e3 * t_a0:8.2f} -> {1e3 * t_a1:8.2f} ms  ({t_a0 / t_a1:5.2f}x)"
    )
    print(
        f"    fold_adjoint  {1e3 * t_j0:8.2f} -> {1e3 * t_j1:8.2f} ms  ({t_j0 / t_j1:5.2f}x)"
    )
    print(
        f"    apply + copy  {1e3 * t_c0:8.2f} -> {1e3 * t_c1:8.2f} ms  ({t_c0 / t_c1:5.2f}x)"
    )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [128]:
        main(ns)
