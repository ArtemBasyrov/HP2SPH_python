"""D24: regularise instead of deflate -- O(1) memory, O(0) setup, O(0) per iteration.

The coarse-space route resolves the near-null directions.  d22 measured that those
directions are PHYSICALLY INERT: at nside 128 the plain and two-level latitude solutions
differ by 720% as vectors while the alm agree to 2.9e-7 and the E->B leakage is identical
to five digits.  If the content is inert, there is no reason to resolve it -- damping it
is just as valid and costs nothing.

So put the dropped zero-assertion back with a small weight alpha instead of deleting it:

    N_alpha = A0^H ( F^H Wt F  +  alpha * W * Lambda ) A0 / norm

with Lambda the indicator of the relaxed (row, column) entries.  This is positive
semi-definite, it shares the SAME A0 application as the data term, so a matrix-vector
product costs exactly what it costs now, and it needs no matrix at all.

alpha = 0 is the current fold.  alpha = 1 restores the full zero-assertion, i.e. the
zero-padding the fold was introduced to replace.  The question is how small alpha can be
and still lift the plunge region, and whether the physics moves at that alpha.

Two variants of Lambda:
  A  ring rows only (entries whose target differs from their own column)
  B  every entry whose equation was dropped, pole rows included
"""

import sys, time, contextlib, io
import numpy as np
import healpy as hp
import finufft
from scipy.sparse.linalg import cg, LinearOperator

import precond_common  # noqa: F401
import src.nuFFT as _nu
from src.nuFFT import compute_voronoi_weights_1d, _fold_ops
from src.spin_transform import forward_spin
from src.double_fourier_sphere import dfs_fold_plan

ITERS = []


def cg_nufft_forward_tik(
    x,
    f_samples,
    N_modes=None,
    rtol=1e-9,
    maxiter=None,
    eps=1e-12,
    sample_mask=None,
    fold=None,
    alpha=0.0,
    penalty=None,
):
    """cg_nufft_forward with the alpha * W * Lambda penalty added to the operator."""
    n_trans, M_samples = f_samples.shape
    plan_f = finufft.Plan(
        2, (N_modes,), n_trans=n_trans, isign=1, dtype=np.complex128, eps=eps
    )
    plan_a = finufft.Plan(
        1, (N_modes,), n_trans=n_trans, isign=-1, dtype=np.complex128, eps=eps
    )
    plan_f.setpts(x)
    plan_a.setpts(x)

    w = compute_voronoi_weights_1d(x)
    mask = np.asarray(sample_mask, dtype=float)
    if mask.shape == (M_samples, n_trans):
        mask = mask.T
    weights = w[None, :] * mask
    norm = weights.sum(axis=1, keepdims=True).mean()

    pen = None
    if alpha > 0 and penalty is not None:
        p = np.asarray(penalty, dtype=float)
        if p.shape == (M_samples, n_trans):
            p = p.T
        pen = alpha * w[None, :] * p

    fold_apply, fold_adjoint = _fold_ops(fold, n_trans, M_samples)

    def AHA(vec):
        g = np.zeros((n_trans, M_samples), dtype=np.complex128)
        plan_f.execute(vec.reshape(n_trans, N_modes), g)
        y = fold_apply(g) if fold_apply is not None else g
        y = y * weights
        y = fold_adjoint(y) if fold_adjoint is not None else y
        if pen is not None:
            y = y + pen * g
        out = np.zeros((n_trans, N_modes), dtype=np.complex128)
        plan_a.execute(np.ascontiguousarray(y), out)
        return (out / norm).ravel()

    rhs_s = f_samples * weights
    if fold_adjoint is not None:
        rhs_s = fold_adjoint(rhs_s)
    rhs = np.zeros((n_trans, N_modes), dtype=np.complex128)
    plan_a.execute(np.ascontiguousarray(rhs_s), rhs)
    rhs = (rhs / norm).ravel()

    A = LinearOperator((n_trans * N_modes,) * 2, matvec=AHA, dtype=np.complex128)
    cnt = [0]
    out, info = cg(
        A,
        rhs,
        x0=np.zeros_like(rhs),
        rtol=rtol,
        maxiter=maxiter,
        callback=lambda z: cnt.__setitem__(0, cnt[0] + 1),
    )
    ITERS.append(cnt[0])
    return out.reshape(n_trans, N_modes).T, info


def penalty_masks(nside, spin=2, tol=1e-2):
    target, phase, keep = dfs_fold_plan(nside, spin, tol)
    ident = np.arange(target.shape[1])[None, :]
    A = (~keep) & (target != ident)  # ring rows only
    B = ~keep  # everything whose equation was dropped
    return A.astype(float), B.astype(float)


def run(nside, seed, alpha, penalty, tol=1e-2, rtol=1e-7):
    lmax = 2 * nside
    ell = np.arange(lmax + 1)
    np.random.seed(seed)
    aE = hp.synalm((1.0 + ell) ** -3.0, lmax=lmax, new=True)
    aB = hp.synalm((1.0 + ell) ** -3.0, lmax=lmax, new=True)
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, lmax)

    orig = _nu.cg_nufft_forward
    _nu.cg_nufft_forward = lambda *a, **k: cg_nufft_forward_tik(
        *a, alpha=alpha, penalty=penalty, **k
    )
    ITERS.clear()
    try:
        t = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            gE, gB = forward_spin(Q, U, lmax, alias_tol=tol, rtol=rtol)
        dt = time.perf_counter() - t
    finally:
        _nu.cg_nufft_forward = orig

    cin = hp.alm2cl(aE, lmax=lmax)
    cout = hp.alm2cl(np.ascontiguousarray(gE), lmax=lmax)
    rel = np.abs(cout - cin) / np.maximum(cin, 1e-300)
    top = slice(3 * lmax // 4, lmax + 1)
    return (
        dt,
        ITERS[0],
        np.sqrt(np.mean(rel[top] ** 2)),
        np.sqrt(np.mean(rel[lmax - 4 :] ** 2)),
    )


def main(nside, seeds=(0,), alphas=(0.0, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)):
    pA, pB = penalty_masks(nside)
    print(
        f"nside {nside}  (RMS relative C_l^EE error: top quarter / last 5 l)   "
        f"penalised entries: A {int(pA.sum())}  B {int(pB.sum())}"
    )
    for label, pen in (("A ring rows", pA), ("B incl. poles", pB)):
        for alpha in alphas:
            if alpha == 0.0 and label != "A ring rows":
                continue
            r = [run(nside, s, alpha, pen) for s in seeds]
            tag = "fold (alpha=0)" if alpha == 0 else f"{label} alpha {alpha:.0e}"
            print(
                f"  {tag:26s} {int(np.median([v[1] for v in r])):6d} its "
                f"{np.median([v[0] for v in r]):7.2f} s   "
                f"top {np.median([v[2] for v in r]):.4e}  "
                f"edge {np.median([v[3] for v in r]):.4e}"
            )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [32]:
        main(ns)
