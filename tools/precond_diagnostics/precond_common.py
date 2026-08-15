"""Shared harness: capture the folded latitude normal operator as a dense matrix."""

import sys, contextlib, io
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tools"))

import numpy as np
import healpy as hp

import src.nuFFT as _nu
from src.data_interpolation import transform_healpix_to_grid
from src.double_fourier_sphere import DFS, dfs_fold_plan


def make_spin_dfs(nside, seed=0, slope=1.5, spin=2):
    """A cosmology-like spin-2 sky, taken up to the DFS array the latitude solve sees."""
    lmax = 2 * nside
    ell = np.arange(lmax + 1)
    cl = (1.0 + ell) ** (-2.0 * slope)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    aE = hp.synalm(cl, lmax=lmax, new=True)
    aB = hp.synalm(cl, lmax=lmax, new=True)
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, lmax)
    z = Q + 1j * U
    with contextlib.redirect_stdout(io.StringIO()):
        up, fc = transform_healpix_to_grid(z)
        _, dfs = DFS(up, fc, spin=spin)
    return dfs, (aE, aB)


def capture(dfs, nside, fold_plan, rtol=1e-7):
    """Return (LinearOperator, rhs) that cg_nufft_forward would hand scipy."""
    box = {}
    orig = _nu.cg

    def spy(A, b, **kw):
        box["A"], box["b"] = A, b
        return np.zeros_like(b), 0

    _nu.cg = spy
    try:
        kwargs = dict(solver="cg", rtol=rtol, maxiter=20000)
        if fold_plan is not None:
            target, phase, keep = fold_plan
            kwargs.update(sample_mask=keep, fold=(target, phase))
        _nu.apply_nuFFT(dfs, **kwargs)
    finally:
        _nu.cg = orig
    return box["A"], box["b"]


def densify(A):
    n = A.shape[0]
    out = np.empty((n, n), dtype=complex)
    e = np.zeros(n, dtype=complex)
    for i in range(n):
        e[i] = 1.0
        out[:, i] = A.matvec(e)
        e[i] = 0.0
    return out


def cg_count(A, b, M=None, rtol=1e-7, maxiter=5000, x0=None):
    from scipy.sparse.linalg import cg

    n = [0]

    def cb(xk):
        n[0] += 1

    x, info = cg(A, b, M=M, rtol=rtol, maxiter=maxiter, callback=cb, x0=x0)
    return x, n[0], info
