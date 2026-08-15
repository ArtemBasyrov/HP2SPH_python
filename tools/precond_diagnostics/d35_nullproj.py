"""D35: regularise EXACTLY the undetermined directions -- no matrix, no extra NUFFT.

d24 put the dropped zero-assertion back with weight alpha on each relaxed entry and got
2x for free but had to pay accuracy beyond that.  The reason is now clear.  On a polar
row the alias family {slot b, relaxed c_1..c_q} contributes  D = w v v^H  with
v = conj(p): the data DO determine the family sum, and only the SPLIT between members is
undetermined.  Penalising each member individually (d24) biases the determined sum too.

Penalise the split instead:

    N_alpha = A0^H ( F^H Wt F  +  alpha * W * Pi ) A0 / norm,     Pi = I - v v^H / |v|^2

Pi is the orthogonal projector onto null(D) inside each family, so the penalty acts ONLY
where the data say nothing.  Every eigenvalue of the family block becomes w|v|^2 (once)
and w*alpha (q times), so alpha ~ 1 makes the block as well conditioned as the fold-free
operator -- while leaving the determined component untouched.

Pi costs one scatter and one gather, which is exactly what _fold_ops already does:

    Pi y = y - keep * conj(p) * (family sum of p * y) / family size

so a matrix-vector product still costs one forward and one adjoint NUFFT, and the whole
scheme stores nothing beyond a (M, n_trans) integer array.
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


def family_size(target, keep):
    """size[r, c] = how many columns share c's slot on row r; kept[r, c] = slot kept."""
    M, C = target.shape
    size = np.empty((M, C), dtype=float)
    kept = np.empty((M, C), dtype=float)
    ar = np.arange(C)
    for r in range(M):
        counts = np.bincount(target[r], minlength=C)
        size[r] = counts[target[r]]
        kept[r] = keep[r, target[r]]
    return size, kept


def cg_nufft_forward_proj(
    x,
    f_samples,
    N_modes=None,
    rtol=1e-9,
    maxiter=None,
    eps=1e-12,
    sample_mask=None,
    fold=None,
    alpha=0.0,
    proj=None,
):
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

    fold_apply, fold_adjoint = _fold_ops(fold, n_trans, M_samples)

    pen = None
    if alpha > 0 and proj is not None:
        size, kept = proj  # both (M_samples, n_trans)
        inv_size = (kept / size).T  # (n_trans, M_samples)
        pen = alpha * w[None, :]

    def AHA(vec):
        g = np.zeros((n_trans, M_samples), dtype=np.complex128)
        plan_f.execute(vec.reshape(n_trans, N_modes), g)
        y = fold_apply(g)
        y = y * weights
        y = fold_adjoint(y)
        if pen is not None:
            # Pi g = g - inv_size * fold_adjoint(fold_apply(g))
            y = y + pen * (g - inv_size * fold_adjoint(fold_apply(g)))
        out = np.zeros((n_trans, N_modes), dtype=np.complex128)
        plan_a.execute(np.ascontiguousarray(y), out)
        return (out / norm).ravel()

    rhs_s = fold_adjoint(f_samples * weights)
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


def run(nside, seed, alpha, proj, tol=1e-2, rtol=1e-7):
    lmax = 2 * nside
    ell = np.arange(lmax + 1)
    np.random.seed(seed)
    aE = hp.synalm((1.0 + ell) ** -3.0, lmax=lmax, new=True)
    aB = hp.synalm((1.0 + ell) ** -3.0, lmax=lmax, new=True)
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, lmax)

    orig = _nu.cg_nufft_forward
    _nu.cg_nufft_forward = lambda *a, **k: cg_nufft_forward_proj(
        *a, alpha=alpha, proj=proj, **k
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
        np.ascontiguousarray(gE),
    )


def main(nside, seeds=(0, 1), alphas=(0.0, 1e-2, 1e-1, 3e-1, 1.0, 3.0), tol=1e-2):
    target, phase, keep = dfs_fold_plan(nside, 2, tol)
    proj = family_size(target, keep)
    print(f"nside {nside}  (RMS relative C_l^EE error: top quarter / last 5 l)")
    base = {}
    for alpha in alphas:
        r = [run(nside, s, alpha, proj, tol=tol) for s in seeds]
        it = int(np.median([v[1] for v in r]))
        dt = np.median([v[0] for v in r])
        top = np.median([v[2] for v in r])
        edge = np.median([v[3] for v in r])
        if alpha == 0.0:
            base = {s: v[4] for s, v in zip(seeds, r)}
            drift = 0.0
        else:
            drift = np.median(
                [
                    np.linalg.norm(v[4] - base[s]) / np.linalg.norm(base[s])
                    for s, v in zip(seeds, r)
                ]
            )
        tag = "fold (alpha=0)" if alpha == 0 else f"alpha {alpha:.0e}"
        print(
            f"  {tag:16s} {it:6d} its {dt:7.2f} s   top {top:.4e}  "
            f"edge {edge:.4e}   |d alm|/|alm| {drift:.2e}"
        )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [32]:
        main(ns)
