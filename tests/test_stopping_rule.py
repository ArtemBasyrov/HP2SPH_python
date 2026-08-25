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

from hp2sph.data_interpolation import transform_healpix_to_grid
from hp2sph.double_fourier_sphere import DFS, dfs_fold_plan
from hp2sph.spin_transform import ALIAS_ETA, forward_spin


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
    from hp2sph.nuFFT import apply_nuFFT

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


def _dfs_of(Q, U, spin=2):
    with contextlib.redirect_stdout(io.StringIO()):
        up, fc = transform_healpix_to_grid(Q + 1j * U)
        _, dfs = DFS(up, fc, spin=spin)
    return dfs


def _residual_floor(Q, U, lmax, nside, alias_tol=1e-2):
    """The converged data residual, in absolute (un-normalised) units."""
    from hp2sph.nuFFT import compute_voronoi_weights_1d, _upsampled_latitudes

    dfs = _dfs_of(Q, U)
    _, _, keep = dfs_fold_plan(nside, 2, alias_tol)
    w = compute_voronoi_weights_1d(_upsampled_latitudes(nside))
    bw = np.sqrt(
        np.sum(np.where(np.asarray(keep), w[:, None] * np.abs(np.asarray(dfs)) ** 2, 0))
    )
    last = [None]
    with contextlib.redirect_stdout(io.StringIO()):
        forward_spin(
            Q,
            U,
            lmax,
            alias_tol=alias_tol,
            rtol=1e-12,
            maxiter=800,
            eta=None,
            monitor=lambda k, rho, rr, g: last.__setitem__(0, rho),
        )
    return last[0] * bw


@pytest.mark.parametrize("nside", [16, 32])
def test_nyquist_discrepancy_predicts_the_residual_floor(nside):
    """The a-priori level must match the floor it is meant to estimate."""
    from hp2sph.nuFFT import nyquist_discrepancy

    aE, _, Q, U, lmax = _spin_sky(nside)
    got = nyquist_discrepancy(_dfs_of(Q, U))
    want = _residual_floor(Q, U, lmax, nside)
    assert got == pytest.approx(want, rel=0.10), f"{got:.4e} against a floor {want:.4e}"


def test_nyquist_discrepancy_is_blind_to_a_sky_without_grid_nyquist_power():
    """The documented limitation: no |m| = 2*nside content, no estimate.

    This is why the level cannot be the only stopping rule -- it silently never fires.
    """
    from hp2sph.nuFFT import nyquist_discrepancy

    nside = 32
    slmax = 2 * nside - 1  # one below the grid Nyquist
    ell = np.arange(slmax + 1)
    np.random.seed(0)
    aE = hp.synalm((1.0 + ell) ** -3.0, lmax=slmax, new=True)
    aB = hp.synalm((1.0 + ell) ** -3.0, lmax=slmax, new=True)
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, slmax)
    got = nyquist_discrepancy(_dfs_of(Q, U))
    floor = _residual_floor(Q, U, 2 * nside, nside)
    assert got < 1e-12 * floor, "expected the estimate to collapse, not to track"
    assert floor > 1e-6, "the true floor is not zero, so a fallback rule is required"


def test_theta_falls_through_to_the_stagnation_rule_when_the_level_is_unreachable():
    """Together they cover the case neither covers alone."""
    nside, slmax = 32, 2 * 32 - 1
    ell = np.arange(slmax + 1)
    np.random.seed(0)
    aE = hp.synalm((1.0 + ell) ** -3.0, lmax=slmax, new=True)
    aB = hp.synalm((1.0 + ell) ** -3.0, lmax=slmax, new=True)
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, slmax)
    _, its_level_only = _run(
        Q, U, 2 * nside, theta=1.02, eta=None, rtol=1e-9, maxiter=900
    )
    _, its_both = _run(
        Q, U, 2 * nside, theta=1.02, eta=ALIAS_ETA, rtol=1e-9, maxiter=900
    )
    assert its_both < its_level_only, (
        "the stagnation rule must still stop the run when the a-priori level is "
        f"unreachable ({its_both} against {its_level_only})"
    )


def _floor_at(nside, alias_tol, signal_lmax, seed=0, maxiter=6000):
    ell = np.arange(signal_lmax + 1)
    np.random.seed(seed)
    aE = hp.synalm((1.0 + ell) ** -3.0, lmax=signal_lmax, new=True)
    aB = hp.synalm((1.0 + ell) ** -3.0, lmax=signal_lmax, new=True)
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, signal_lmax)
    return _residual_floor(Q, U, 2 * nside, nside, alias_tol=alias_tol)


def test_the_second_error_term_is_the_alias_assertion_and_scales_with_alias_tol():
    """With the Nyquist column gone, the floor tracks ``alias_tol``.

    It does NOT track it while the column is present, which is why the proportionality
    looked falsified: the Nyquist term does not depend on ``alias_tol`` at all and is
    about 70x larger.
    """
    nside = 16
    below = 2 * nside - 1  # no power at the grid Nyquist
    at = 2 * nside
    loose = _floor_at(nside, 1e-1, below)
    tight = _floor_at(nside, 1e-3, below)
    assert loose / tight > 100, (
        f"expected the floor to fall with alias_tol, got {loose:.3e} -> {tight:.3e}"
    )
    # and the same comparison with the Nyquist column present shows no such dependence
    loose_at = _floor_at(nside, 1e-1, at)
    tight_at = _floor_at(nside, 1e-3, at)
    assert 0.5 < tight_at / loose_at < 2.0, (
        "with the Nyquist column present the floor must be insensitive to alias_tol, "
        f"got {loose_at:.3e} -> {tight_at:.3e}"
    )


def test_the_pole_fill_is_not_the_second_error_term():
    """Falsification test: the Lagrange pole order does not move the floor.

    Recorded because the pole fill was the obvious suspect, and because a pole error the
    model can absorb leaves no residual at all -- so a null result here does not mean the
    pole fill is harmless, only that the discrepancy principle cannot see it.
    """
    import hp2sph.double_fourier_sphere as dfsmod

    nside = 16
    original = dfsmod.POLE_INTERP_NPTS
    try:
        dfsmod.POLE_INTERP_NPTS = 2
        coarse = _floor_at(nside, 1e-4, 2 * nside - 1)
        dfsmod.POLE_INTERP_NPTS = 10
        fine = _floor_at(nside, 1e-4, 2 * nside - 1)
    finally:
        dfsmod.POLE_INTERP_NPTS = original
    assert abs(fine - coarse) <= 0.2 * coarse, (
        f"pole order changed the floor: {coarse:.3e} -> {fine:.3e}"
    )


def test_the_alias_fold_plan_is_what_makes_the_residual_stagnate():
    """The data residual must stall well above zero; that floor is the model error."""
    nside = 16
    _, _, Q, U, lmax = _spin_sky(nside)
    z = Q + 1j * U
    with contextlib.redirect_stdout(io.StringIO()):
        up, fc = transform_healpix_to_grid(z)
        _, dfs = DFS(up, fc, spin=2)
    target, phase, keep = dfs_fold_plan(nside, 2, 1e-2)
    from hp2sph.nuFFT import apply_nuFFT

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
