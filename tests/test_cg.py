"""``src.cg`` as a linear-algebra routine, independent of the pipeline.

Everything here is a small dense system built in the test, so a reference answer is
available in closed form and nothing depends on HEALPix, finufft or the alias fold.
The pipeline-level behaviour of the stopping rules lives in ``test_stopping_rule.py``.

The three groups are: the iteration itself (does it solve the system, does it agree
with SciPy), the data-residual bookkeeping (is ``rho`` the quantity it claims to be),
and the stopping rules (do they fire, in the documented order, with the documented
``info``).
"""

import numpy as np
import pytest
from scipy.sparse.linalg import cg as scipy_cg

from src.cg import cg_normal_equations, weighted_norm2


def _least_squares_problem(m=40, n=12, seed=0, complex_=True, norm=1.0):
    """A full-rank weighted least-squares fit and everything needed to check it.

    Returns ``(A, b, w, N, rhs, x_ls, res_ls)`` where ``N`` and ``rhs`` are what the
    solver is handed and ``x_ls`` minimises ``||b - A x||_W``.
    """
    rng = np.random.default_rng(seed)
    if complex_:
        A = rng.normal(size=(m, n)) + 1j * rng.normal(size=(m, n))
        b = rng.normal(size=m) + 1j * rng.normal(size=m)
    else:
        A = rng.normal(size=(m, n))
        b = rng.normal(size=m)
    w = rng.uniform(0.5, 1.5, size=m)
    AHW = A.conj().T * w
    N = AHW @ A / norm
    rhs = AHW @ b / norm
    x_ls = np.linalg.solve(AHW @ A, AHW @ b)
    res_ls = np.sqrt(np.sum(w * np.abs(b - A @ x_ls) ** 2))
    return A, b, w, N, rhs, x_ls, res_ls


# --------------------------------------------------------------------------- #
# the iteration
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("complex_", [True, False])
def test_solves_the_normal_equations(complex_):
    """CG must reach the least-squares minimiser, not merely reduce the residual."""
    A, b, w, N, rhs, x_ls, _ = _least_squares_problem(complex_=complex_)
    x, info = cg_normal_equations(
        lambda v: N @ v, rhs, weighted_norm2(b, w), rtol=1e-14, maxiter=200
    )
    assert info == 0
    assert np.linalg.norm(x - x_ls) <= 1e-9 * np.linalg.norm(x_ls)


def test_terminates_within_n_iterations_in_exact_arithmetic():
    """CG on an n-dimensional system has finite termination; allow a small margin."""
    _, b, w, N, rhs, _, _ = _least_squares_problem(m=40, n=12)
    ks = []
    cg_normal_equations(
        lambda v: N @ v,
        rhs,
        weighted_norm2(b, w),
        rtol=1e-13,
        maxiter=200,
        monitor=lambda k, rho, rrel, x: ks.append(k),
    )
    assert ks[-1] <= 16, f"took {ks[-1]} iterations on a 12-dimensional system"


def test_matches_scipy_cg():
    """Same iteration, so the same answer to rounding."""
    _, b, w, N, rhs, _, _ = _least_squares_problem()
    x, _ = cg_normal_equations(
        lambda v: N @ v, rhs, weighted_norm2(b, w), rtol=1e-12, maxiter=500
    )
    ref, _ = scipy_cg(N, rhs, rtol=1e-12, maxiter=500)
    assert np.linalg.norm(x - ref) <= 1e-9 * np.linalg.norm(ref)


def test_norm_scaling_leaves_the_answer_and_the_data_residual_unchanged():
    """``norm`` only rescales the system; it must not leak into either output."""
    A, b, w, _, _, x_ls, res_ls = _least_squares_problem(norm=1.0)
    out = {}
    for norm in (1.0, 7.3):
        AHW = A.conj().T * w
        x, _ = cg_normal_equations(
            lambda v, M=AHW @ A / norm: M @ v,
            AHW @ b / norm,
            weighted_norm2(b, w),
            norm=norm,
            rtol=1e-13,
            maxiter=300,
            monitor=lambda k, rho, rrel, xx: out.setdefault(norm, []).append(rho),
        )
        assert np.linalg.norm(x - x_ls) <= 1e-9 * np.linalg.norm(x_ls)
    a, c = out[1.0], out[7.3]
    n = min(len(a), len(c))
    assert np.allclose(a[:n], c[:n], rtol=1e-8, atol=1e-12)


def test_zero_right_hand_side_returns_zero_immediately():
    _, b, w, N, rhs, _, _ = _least_squares_problem()
    calls = []
    x, info = cg_normal_equations(
        lambda v: N @ v,
        np.zeros_like(rhs),
        weighted_norm2(b, w),
        monitor=lambda *a: calls.append(a),
    )
    assert info == 0
    assert not np.any(x)
    assert calls == [], "no iteration should run on a zero right-hand side"


def test_indefinite_operator_stops_instead_of_diverging():
    """A negative curvature direction ends the iteration; it must not blow up."""
    n = 8
    d = np.ones(n)
    d[3] = -2.0  # not positive definite
    N = np.diag(d)
    rhs = np.ones(n)
    x, info = cg_normal_equations(lambda v: N @ v, rhs, 1.0, rtol=1e-12, maxiter=50)
    assert np.all(np.isfinite(x))
    assert info != 0, "an indefinite system must not report convergence"


# --------------------------------------------------------------------------- #
# the data residual
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("norm", [1.0, 3.7])
def test_data_residual_matches_a_direct_evaluation(norm):
    """``rho`` must equal ``||b - A x_k||_W / ||b||_W`` at every iterate."""
    A, b, w, _, _, _, _ = _least_squares_problem(norm=norm)
    AHW = A.conj().T * w
    bw2 = weighted_norm2(b, w)
    seen = []

    def monitor(k, rho, rrel, x):
        direct = np.sqrt(np.sum(w * np.abs(b - A @ x) ** 2))
        seen.append((rho * np.sqrt(bw2), direct))

    cg_normal_equations(
        lambda v: (AHW @ A / norm) @ v,
        AHW @ b / norm,
        bw2,
        norm=norm,
        rtol=1e-14,
        maxiter=40,
        monitor=monitor,
    )
    assert len(seen) > 3
    for tracked, direct in seen:
        assert tracked == pytest.approx(direct, rel=1e-8, abs=1e-12)


def test_data_residual_decreases_monotonically():
    """CG minimises the W-norm of the data residual over a growing Krylov space."""
    _, b, w, N, rhs, _, _ = _least_squares_problem()
    rho = []
    cg_normal_equations(
        lambda v: N @ v,
        rhs,
        weighted_norm2(b, w),
        rtol=1e-14,
        maxiter=60,
        monitor=lambda k, r, rr, x: rho.append(r),
    )
    rho = np.array(rho)
    assert np.all(np.diff(rho) <= 1e-12 * rho[0]), "the data residual is not monotone"


def test_data_residual_stagnates_at_the_least_squares_floor():
    """An overdetermined fit cannot reach zero; the floor is the LS residual."""
    A, b, w, N, rhs, _, res_ls = _least_squares_problem(m=40, n=12)
    rho = []
    cg_normal_equations(
        lambda v: N @ v,
        rhs,
        weighted_norm2(b, w),
        rtol=1e-14,
        maxiter=100,
        monitor=lambda k, r, rr, x: rho.append(r),
    )
    floor = rho[-1] * np.sqrt(weighted_norm2(b, w))
    assert res_ls > 0
    assert floor == pytest.approx(res_ls, rel=1e-7)


def test_data_residual_reaches_zero_for_a_consistent_square_system():
    """With b in the range of A the fit is exact, so the floor is 0."""
    rng = np.random.default_rng(3)
    n = 10
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    w = np.ones(n)
    b = A @ (rng.normal(size=n) + 1j * rng.normal(size=n))
    AHW = A.conj().T * w
    rho = []
    cg_normal_equations(
        lambda v: (AHW @ A) @ v,
        AHW @ b,
        weighted_norm2(b, w),
        rtol=1e-14,
        maxiter=200,
        monitor=lambda k, r, rr, x: rho.append(r),
    )
    assert rho[-1] < 1e-6


def test_monitor_receives_consecutive_indices_and_the_live_iterate():
    _, b, w, N, rhs, _, _ = _least_squares_problem()
    seen = []
    x_final, _ = cg_normal_equations(
        lambda v: N @ v,
        rhs,
        weighted_norm2(b, w),
        rtol=1e-13,
        maxiter=40,
        monitor=lambda k, rho, rrel, x: seen.append((k, x)),
    )
    ks = [k for k, _ in seen]
    assert ks == list(range(1, len(ks) + 1))
    # the documented contract: x is the live buffer, so the retained references all
    # alias the returned array rather than snapshotting the iterate
    assert all(x is x_final for _, x in seen)


# --------------------------------------------------------------------------- #
# the stopping rules
# --------------------------------------------------------------------------- #


def test_rtol_is_the_scipy_convention_and_is_honoured():
    _, b, w, N, rhs, _, _ = _least_squares_problem()
    for rtol in (1e-2, 1e-6, 1e-10):
        x, info = cg_normal_equations(
            lambda v: N @ v, rhs, weighted_norm2(b, w), rtol=rtol, maxiter=500
        )
        assert info == 0
        got = np.linalg.norm(rhs - N @ x) / np.linalg.norm(rhs)
        assert got <= rtol


def test_maxiter_exhaustion_reports_maxiter():
    _, b, w, N, rhs, _, _ = _least_squares_problem()
    x, info = cg_normal_equations(
        lambda v: N @ v, rhs, weighted_norm2(b, w), rtol=1e-16, maxiter=2
    )
    assert info == 2
    assert np.all(np.isfinite(x))


def test_level_stops_at_the_requested_data_residual():
    """The discrepancy stop must fire on the data residual, not the normal one."""
    A, b, w, N, rhs, _, res_ls = _least_squares_problem()
    bnorm = np.sqrt(weighted_norm2(b, w))
    level = 1.5 * res_ls  # reachable, and well above the floor
    rho = []
    x, info = cg_normal_equations(
        lambda v: N @ v,
        rhs,
        weighted_norm2(b, w),
        rtol=1e-16,
        maxiter=200,
        level=level,
        monitor=lambda k, r, rr, xx: rho.append(r),
    )
    assert info == 0
    assert rho[-1] * bnorm <= level
    assert all(r * bnorm > level for r in rho[:-1]), "stopped later than it had to"


def test_unreachable_level_falls_back_to_rtol():
    """Below the least-squares floor the level is never met; rtol must still stop it."""
    _, b, w, N, rhs, _, res_ls = _least_squares_problem()
    x, info = cg_normal_equations(
        lambda v: N @ v,
        rhs,
        weighted_norm2(b, w),
        rtol=1e-10,
        maxiter=500,
        level=0.01 * res_ls,
    )
    assert info == 0
    assert np.linalg.norm(rhs - N @ x) / np.linalg.norm(rhs) <= 1e-10


def test_eta_fires_and_is_monotone_in_eta():
    """A looser stagnation threshold must stop no later than a tighter one."""
    _, b, w, N, rhs, _, _ = _least_squares_problem(m=200, n=60, seed=5)
    bw2 = weighted_norm2(b, w)
    stops = {}
    for eta in (1e-1, 1e-2, 1e-3, 1e-5):
        ks = []
        _, info = cg_normal_equations(
            lambda v: N @ v,
            rhs,
            bw2,
            rtol=1e-16,
            maxiter=400,
            eta=eta,
            monitor=lambda k, r, rr, x: ks.append(k),
        )
        assert info == 0, f"eta={eta:.0e} never fired"
        stops[eta] = ks[-1]
    ordered = [stops[e] for e in (1e-1, 1e-2, 1e-3, 1e-5)]
    assert ordered == sorted(ordered), f"not monotone in eta: {stops}"


def test_eta_needs_the_full_delay_window_before_it_can_fire():
    _, b, w, N, rhs, _, _ = _least_squares_problem(m=200, n=60, seed=5)
    for delay in (2, 5, 10):
        ks = []
        cg_normal_equations(
            lambda v: N @ v,
            rhs,
            weighted_norm2(b, w),
            rtol=1e-16,
            maxiter=400,
            eta=1.0,  # fires as soon as it is allowed to
            delay=delay,
            monitor=lambda k, r, rr, x: ks.append(k),
        )
        assert ks[-1] == delay


def test_eta_is_inert_when_the_residual_is_still_falling_fast():
    """On a well-conditioned system rtol should win the race, not eta."""
    n = 20
    N = np.diag(np.linspace(1.0, 1.5, n))  # cond 1.5
    rhs = np.ones(n)
    x_ref = np.linalg.solve(N, rhs)
    x, info = cg_normal_equations(
        lambda v: N @ v, rhs, 1.0, rtol=1e-10, maxiter=200, eta=1e-3
    )
    assert info == 0
    assert np.linalg.norm(x - x_ref) <= 1e-8 * np.linalg.norm(x_ref)


def test_level_takes_precedence_over_eta():
    """Documented order: level is tested first."""
    _, b, w, N, rhs, _, res_ls = _least_squares_problem(m=200, n=60, seed=5)
    bw2 = weighted_norm2(b, w)
    bnorm = np.sqrt(bw2)
    # a level that is met early, paired with an eta that would run much longer
    rho_curve = []
    cg_normal_equations(
        lambda v: N @ v,
        rhs,
        bw2,
        rtol=1e-16,
        maxiter=400,
        monitor=lambda k, r, rr, x: rho_curve.append(r),
    )
    level = rho_curve[2] * bnorm  # reached at iteration 3
    ks = []
    _, info = cg_normal_equations(
        lambda v: N @ v,
        rhs,
        bw2,
        rtol=1e-16,
        maxiter=400,
        level=level,
        eta=1e-8,
        monitor=lambda k, r, rr, x: ks.append(k),
    )
    assert info == 0
    assert ks[-1] == 3
