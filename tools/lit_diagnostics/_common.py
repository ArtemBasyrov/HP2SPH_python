"""Shared setup for the literature-driven latitude-solve diagnostics.

These scripts test methods proposed in the works catalogued in
``literature_review.md`` against the HP2SPH spin-2 latitude solve. Each one
builds the real solver operator rather than a model of it, so the operator here
mirrors ``nuFFT._cg_nufft_forward_half``.

Run from the repository root with ``PYTHONPATH=.``.
"""

import numpy as np
import healpy as hp

from src.data_interpolation import transform_healpix_to_grid
from src.double_fourier_sphere import DFS, dfs_fold_sparse, pole_stencil_rows
from src.nuFFT import (
    compute_voronoi_weights_1d,
    _mirror_plan,
    _upsampled_latitudes,
    _BatchPlan,
    _fold_ops,
)

DEFAULT_SLOPE = 3.0  # C_l ~ (1 + l)^-3, i.e. the `cosmology` benchmark spectrum


def make_sky(nside, spin=2, seed=0, alias_tol=1e-2):
    """A band-limited spin-2 sky, returned as (half-domain DFS array, fold plan)."""
    lmax = 2 * nside
    ell = np.arange(lmax + 1)
    cl = (1.0 + ell) ** -DEFAULT_SLOPE
    np.random.seed(seed)
    aE = hp.synalm(cl, lmax=lmax, new=True)
    aB = hp.synalm(cl, lmax=lmax, new=True)
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, lmax)
    upsampled, fft_coeff = transform_healpix_to_grid(
        Q + 1j * U, map_rows=pole_stencil_rows(nside)
    )
    _, dfs = DFS(upsampled, fft_coeff, spin=spin, half=True)
    return dfs, dfs_fold_sparse(nside, spin, alias_tol)


def build_operator(
    nside, dfs, plan, use_fold=True, spin=2, eps=1e-9, uniform=False, weights="voronoi"
):
    """(apply_AHA, rhs, n) for the half-domain normal equations.

    ``use_fold=False`` drops the alias fold, which is the well-conditioned system
    the scalar path solves. ``uniform=True`` puts the latitude nodes on an equispaced
    grid, which is the approximation tested in ``l6_uniform_nodes.py``.

    ``weights`` selects the quadrature weight, for ``l9_weights_precond.py``:

    * ``"voronoi"`` -- the shipped ``compute_voronoi_weights_1d``, i.e. Feichtinger
      et al.'s adaptive weights ``w_i = (t_{i+1} - t_{i-1}) / 2``.
    * ``"uniform"`` -- every sample weighted equally, rescaled to the same total
      mass so the normal operator keeps its scale.

    Note the uniform case changes the OBJECTIVE, not only the conditioning: the
    weight here is a quadrature weight approximating an integral, so an unweighted
    solve fits a different functional. It is a diagnostic, never an option.
    """
    N_modes = 4 * nside + 1
    x = _upsampled_latitudes(nside)
    f_samples = np.ascontiguousarray(dfs.T)
    n_trans, _ = f_samples.shape
    mu, rows, mult, parity, scale, even = _mirror_plan(x, spin, n_trans, N_modes)
    Mh = len(rows)
    K = (N_modes - 1) // 2
    xh = np.ascontiguousarray(x[rows])
    if uniform:
        xh = np.ascontiguousarray(np.linspace(xh[0], xh[-1], Mh))
    w = compute_voronoi_weights_1d(x)[rows] * mult
    if weights == "uniform":
        w = np.full_like(w, w.sum() / mult.sum()) * mult
    elif weights != "voronoi":
        raise ValueError(f"weights must be 'voronoi' or 'uniform', got {weights!r}")

    plan_fwd = _BatchPlan(2, N_modes, n_trans, 1, eps, 1)
    plan_adj = _BatchPlan(1, N_modes, n_trans, -1, eps, 1)
    plan_fwd.setpts(xh)
    plan_adj.setpts(xh)

    if use_fold:
        drop = plan.drop
        lost = np.bincount(drop // Mh, weights=w[drop % Mh], minlength=n_trans)
        norm = float((w.sum() - lost).mean())
        fold_apply, fold_adjoint = _fold_ops(plan, n_trans, Mh)
    else:
        drop = None
        norm = float(w.sum())
        fold_apply = fold_adjoint = None

    coef = np.zeros((n_trans, N_modes), dtype=np.complex128)
    gbuf = np.zeros((n_trans, Mh), dtype=np.complex128)
    rbuf = np.empty((n_trans, K + 1), dtype=np.complex128)
    par = parity[:, None]

    def expand(ch):
        np.multiply(ch, scale, out=coef[:, K:])
        np.multiply(par, coef[:, 2 * K : K : -1], out=coef[:, :K])
        coef[:, K] *= even
        return coef

    def restrict(fl):
        np.multiply(par, fl[:, K - 1 :: -1], out=rbuf[:, 1:])
        rbuf[:, 1:] += fl[:, K + 1 :]
        rbuf[:, 0] = fl[:, K]
        np.multiply(rbuf, scale, out=rbuf)
        rbuf[:, 0] *= even
        return rbuf

    def adjoint_of(samples):
        if samples is not gbuf:
            np.copyto(gbuf, samples)
        np.multiply(gbuf, w, out=gbuf)
        if drop is not None:
            gbuf.reshape(-1)[drop] = 0.0
        if fold_adjoint is not None:
            fold_adjoint(gbuf, out=gbuf)
        plan_adj.execute(gbuf, coef)
        restrict(coef)
        np.divide(rbuf, norm, out=rbuf)
        return rbuf.reshape(-1)

    def apply_AHA(vec):
        plan_fwd.execute(expand(vec.reshape(n_trans, K + 1)), gbuf)
        if fold_apply is not None:
            fold_apply(gbuf, out=gbuf)
        return adjoint_of(gbuf).copy()

    rhs = adjoint_of(np.ascontiguousarray(f_samples[:, rows])).copy()
    return apply_AHA, rhs, n_trans * (K + 1), parity
