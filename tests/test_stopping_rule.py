"""The stopping rule where it is used: the spin latitude solve.

``tests/test_cg.py`` covers the solver itself on dense systems. This file covers what
only the real operator can show.

The solver must agree with SciPy's on the pipeline's own system, since the point of the
change is to alter WHEN the iteration stops and nothing about what it converges to.

The stagnation stop must fire on that system, cut the iteration count by a large factor,
and leave both the recovered spectrum and the E->B leakage alone.

The rule must be inert on the unfolded scalar solve, which is well posed.

And the premise of the whole rule -- that the data residual stalls at a floor set by the
alias fold's modelling error -- must hold for the real operator, not just in principle.
"""

import contextlib
import io

import healpy as hp
import numpy as np
import pytest

from src.data_interpolation import transform_healpix_to_grid
from src.double_fourier_sphere import DFS, dfs_fold_plan
from src.spin_transform import ALIAS_ETA, forward_spin


def _spin_sky(nside, seed=0, pure_E=False):
    lmax = 2 * nside
    ell = np.arange(lmax + 1)
    np.random.seed(seed)
    aE = hp.synalm((1.0 + ell) ** -3.0, lmax=lmax, new=True)
    aB = (
        np.zeros_like(aE)
        if pure_E
        else hp.synalm((1.0 + ell) ** -3.0, lmax=lmax, new=True)
    )
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, lmax)
    return aE, aB, Q, U, lmax


def _run(Q, U, lmax, **kw):
    """``forward_spin`` plus the iteration count the solver reached."""
    last = [0]
    with contextlib.redirect_stdout(io.StringIO()):
        out = forward_spin(
            Q, U, lmax, monitor=lambda k, rho, rrel, g: last.__setitem__(0, k), **kw
        )
    return out, last[0]


def test_monitored_cg_reproduces_scipy():
    """Instrumenting the iteration must not change the answer it converges to."""
    _, _, Q, U, lmax = _spin_sky(16)
    with contextlib.redirect_stdout(io.StringIO()):
        ref = forward_spin(Q, U, lmax, eta=None, rtol=1e-9)
    got, _ = _run(Q, U, lmax, eta=None, rtol=1e-9)
    for a, b in zip(ref, got):
        assert np.linalg.norm(a - b) <= 1e-12 * np.linalg.norm(a)


@pytest.mark.parametrize("nside", [16, 32])
def test_stagnation_stop_cuts_the_iteration_count_without_moving_the_answer(nside):
    """The shipped rule against a fully converged solve, on C_l^EE and on leakage."""
    aE, _, Q, U, lmax = _spin_sky(nside, pure_E=True)
    cin = hp.alm2cl(aE, lmax=lmax)
    top = slice(3 * lmax // 4, lmax + 1)

    def metrics(out):
        gE, gB = out
        rel = np.abs(hp.alm2cl(np.ascontiguousarray(gE), lmax=lmax) - cin) / cin
        leak = hp.alm2cl(np.ascontiguousarray(gB), lmax=lmax) / cin
        return (
            np.sqrt(np.mean(rel[top] ** 2)),
            np.sqrt(np.mean(leak[top] ** 2)),
        )

    ref, its_ref = _run(Q, U, lmax, eta=None, rtol=1e-9, maxiter=4000)
    got, its = _run(Q, U, lmax, eta=ALIAS_ETA, rtol=1e-9, maxiter=4000)

    err_ref, leak_ref = metrics(ref)
    err, leak = metrics(got)

    assert its < its_ref, "the stagnation rule never fired"
    assert its_ref / its > 4.0, f"only {its_ref / its:.1f}x fewer iterations"
    assert err <= 1.10 * err_ref, f"C_l^EE error {err:.3e} against {err_ref:.3e}"
    assert leak <= 1.10 * leak_ref, f"E->B leakage {leak:.3e} against {leak_ref:.3e}"


def test_stagnation_stop_is_inert_on_the_unfolded_scalar_solve():
    """Without the fold the system is well posed, so the rule must not stop it early."""
    from src.nuFFT import apply_nuFFT

    nside, lmax = 32, 64
    ell = np.arange(lmax + 1)
    np.random.seed(0)
    alm = hp.synalm((1.0 + ell) ** -3.0, lmax=lmax, new=True)
    mp = hp.alm2map(alm, nside, lmax=lmax)
    with contextlib.redirect_stdout(io.StringIO()):
        up, fc = transform_healpix_to_grid(mp)
        _, dfs = DFS(up, fc, spin=0)
        ref = apply_nuFFT(dfs, rtol=1e-9, spin=0)
        got = apply_nuFFT(dfs, rtol=1e-9, eta=ALIAS_ETA, spin=0)
    assert np.linalg.norm(ref - got) <= 1e-10 * np.linalg.norm(ref)


def test_the_alias_fold_plan_is_what_makes_the_residual_stagnate():
    """The data residual must stall well above zero; that floor is the model error."""
    nside = 16
    _, _, Q, U, lmax = _spin_sky(nside)
    z = Q + 1j * U
    with contextlib.redirect_stdout(io.StringIO()):
        up, fc = transform_healpix_to_grid(z)
        _, dfs = DFS(up, fc, spin=2)
    target, phase, keep = dfs_fold_plan(nside, 2, 1e-2)
    from src.nuFFT import apply_nuFFT

    rho = []
    with contextlib.redirect_stdout(io.StringIO()):
        apply_nuFFT(
            dfs,
            solver="cg",
            sample_mask=keep,
            fold=(target, phase),
            rtol=1e-12,
            maxiter=400,
            spin=2,
            monitor=lambda k, r, rr, g: rho.append(r),
        )
    rho = np.array(rho)
    assert np.all(np.diff(rho) <= 1e-12), "the data residual is not monotone"
    assert rho[-1] > 1e-4, f"expected a modelling-error floor, got {rho[-1]:.3e}"
    # and it is reached long before the normal-equation residual is
    assert rho[len(rho) // 4] <= 1.001 * rho[-1]
