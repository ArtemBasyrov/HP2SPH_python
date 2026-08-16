"""The DFS mirror symmetry (half-domain latitude solve) and the fold in solver layout.

Both are pure speed changes that must not move a single digit of the answer, so every
test here is an equality test against the full-domain path or against a reference
implementation, not a tolerance on physics.
"""

import numpy as np
import pytest

from src.data_interpolation import transform_healpix_to_grid
from src.double_fourier_sphere import DFS, dfs_fold_plan
from src.nuFFT import (
    _fold_ops,
    _is_mirror_symmetric,
    _mirror_plan,
    _upsampled_latitudes,
    apply_nuFFT,
)


def _reference_fold_ops(fold, n_trans, M_samples):
    """The pre-optimisation implementation, kept as the equality reference."""
    target = np.asarray(fold[0])
    phase = np.asarray(fold[1])
    flat = (target + n_trans * np.arange(M_samples)[:, None]).ravel()
    conj_phase = np.conj(phase)
    size = M_samples * n_trans

    def apply(model):
        t = (model.T * phase).ravel()
        folded = np.bincount(flat, weights=t.real, minlength=size).astype(complex)
        folded += 1j * np.bincount(flat, weights=t.imag, minlength=size)
        return folded.reshape(M_samples, n_trans).T

    def adjoint(residual):
        gathered = residual.T.ravel()[flat].reshape(M_samples, n_trans)
        return (gathered * conj_phase).T

    return apply, adjoint


def _dfs(healpix_map, spin=0):
    upsampled, fft_coeff = transform_healpix_to_grid(healpix_map)
    _, double_fft = DFS(upsampled, fft_coeff, spin=spin)
    return double_fft


# --- the fold ---------------------------------------------------------------------


@pytest.mark.parametrize("spin", [0, 2])
def test_fold_ops_match_the_dense_reference(nside, spin):
    """The sparse fold against a dense scatter of every entry.

    The adjoint is a pure gather and stays BIT-identical. ``apply`` is not, and must
    not be asserted to be: the sparse form adds each slot's own (identity) contribution
    first and the folded-in ones afterwards, where the dense reference sums the whole
    alias family in index order. That is a different summation order for the same sum,
    so the two agree to rounding rather than exactly. Measured at 1 ulp.
    """
    n_trans, M = 4 * nside, 8 * nside
    target, phase, _ = dfs_fold_plan(nside, spin, 1e-2)
    fast_a, fast_j = _fold_ops((target, phase), n_trans, M)
    ref_a, ref_j = _reference_fold_ops((target, phase), n_trans, M)
    rng = np.random.default_rng(0)
    g = rng.standard_normal((n_trans, M)) + 1j * rng.standard_normal((n_trans, M))
    y = rng.standard_normal((n_trans, M)) + 1j * rng.standard_normal((n_trans, M))
    got, want = fast_a(g), ref_a(g)
    assert np.abs(got - want).max() <= 8 * np.finfo(float).eps * np.abs(want).max()
    assert np.array_equal(fast_j(y), ref_j(y))


@pytest.mark.parametrize("spin", [0, 2])
def test_fold_ops_in_place_matches_the_out_of_place_form(nside, spin):
    """Passing the input as ``out`` is what keeps the solver's peak memory down.

    It is only sound because a relaxed entry always targets a slot that itself stays
    put, so sources and destinations are disjoint -- zeroing a source cannot destroy a
    destination, and writing a gathered source cannot disturb a value still to be read.
    That disjointness is asserted here too, since the in-place form is wrong without it.
    """
    n_trans, M = 4 * nside, 8 * nside
    target, phase, _ = dfs_fold_plan(nside, spin, 1e-2)
    moved = target != np.arange(n_trans)[None, :]
    r, c = np.nonzero(moved)
    assert not moved[r, target[r, c]].any(), "a destination is itself a source"
    assert np.all(phase[~moved] == 1.0), "an entry that stays put must carry phase 1"

    apply, adjoint = _fold_ops((target, phase), n_trans, M)
    rng = np.random.default_rng(0)
    g = rng.standard_normal((n_trans, M)) + 1j * rng.standard_normal((n_trans, M))
    for op in (apply, adjoint):
        ref = op(g).copy()
        buf = g.copy()
        assert np.array_equal(op(buf, out=buf), ref)


def test_fold_ops_return_contiguous_solver_layout(nside):
    """The callers' ``np.ascontiguousarray`` must be a no-op, or the copy comes back."""
    n_trans, M = 4 * nside, 8 * nside
    target, phase, _ = dfs_fold_plan(nside, 2, 1e-2)
    apply, adjoint = _fold_ops((target, phase), n_trans, M)
    rng = np.random.default_rng(0)
    g = rng.standard_normal((n_trans, M)) + 1j * rng.standard_normal((n_trans, M))
    for out in (apply(g), adjoint(g)):
        assert out.shape == (n_trans, M)
        assert out.flags["C_CONTIGUOUS"]


def test_fold_ops_are_exact_adjoints(nside):
    n_trans, M = 4 * nside, 8 * nside
    target, phase, _ = dfs_fold_plan(nside, 2, 1e-2)
    apply, adjoint = _fold_ops((target, phase), n_trans, M)
    rng = np.random.default_rng(1)
    g = rng.standard_normal((n_trans, M)) + 1j * rng.standard_normal((n_trans, M))
    y = rng.standard_normal((n_trans, M)) + 1j * rng.standard_normal((n_trans, M))
    lhs = np.vdot(apply(g), y)
    rhs = np.vdot(g, adjoint(y))
    assert abs(lhs - rhs) <= 1e-12 * abs(lhs)


# --- the mirror plan --------------------------------------------------------------


@pytest.mark.parametrize("spin", [0, 2])
def test_mirror_plan_halves_the_rows(nside, spin):
    x = _upsampled_latitudes(nside)
    plan = _mirror_plan(x, spin, 4 * nside, 4 * nside + 1)
    assert plan is not None
    mu, rows, mult, parity, scale, even = plan
    # the two poles are the fixed points, everything else pairs up
    assert len(rows) == 4 * nside + 1
    assert np.sum(mu == np.arange(len(x))) == 2
    assert np.array_equal(mu[mu], np.arange(len(x)))
    # x[mu] == -x on the circle; np.mod puts an exact match at either 0 or 2*pi, so
    # compare the circular distance rather than the residue itself
    d = np.mod(x[mu] + x, 2 * np.pi)
    assert np.allclose(np.minimum(d, 2 * np.pi - d), 0.0, atol=1e-9)
    # a fixed point counts once, a pair counts twice
    assert np.array_equal(mult, np.where(mu[rows] == rows, 1.0, 2.0))


def test_mirror_plan_embedding_is_an_isometry(nside):
    """c_k with k >= 1 feeds both +k and -k, so it must carry 1/sqrt(2).

    Without this the half operator has a different spectrum from the full one on the
    symmetric subspace and CG needs more iterations, not fewer.
    """
    plan = _mirror_plan(_upsampled_latitudes(nside), 2, 4 * nside, 4 * nside + 1)
    _, _, _, parity, scale, even = plan
    assert scale[0] == 1.0
    assert np.allclose(scale[1:], 1.0 / np.sqrt(2.0))
    # c_{-k} = -c_k forces c_0 = 0 on the odd-parity columns
    assert np.array_equal(even, (parity > 0).astype(float))
    assert even.sum() == len(even) // 2


def test_mirror_plan_declines_without_a_spin(nside):
    x = _upsampled_latitudes(nside)
    assert _mirror_plan(x, None, 4 * nside, 4 * nside + 1) is None


# --- the half solve ---------------------------------------------------------------


@pytest.mark.parametrize("spin", [0, 2])
def test_dfs_array_is_mirror_symmetric(nside, healpix_map, spin):
    dfs = _dfs(healpix_map, spin)
    x = _upsampled_latitudes(nside)
    mu, _, _, parity, _, _ = _mirror_plan(x, spin, 4 * nside, 4 * nside + 1)
    assert _is_mirror_symmetric(np.asarray(dfs).T, mu, parity)


def test_half_solve_equals_the_full_solve(nside, healpix_map):
    """The restriction is exact, so the two must agree to machine precision."""
    dfs = _dfs(healpix_map)
    full = apply_nuFFT(dfs, solver="cg", rtol=1e-10)
    half = apply_nuFFT(dfs, solver="cg", rtol=1e-10, spin=0)
    assert half.shape == full.shape
    assert np.linalg.norm(half - full) <= 1e-10 * np.linalg.norm(full)


def test_half_solve_equals_the_full_solve_with_the_alias_fold(nside, healpix_map):
    target, phase, keep = dfs_fold_plan(nside, 2, 1e-2)
    dfs = _dfs(healpix_map, spin=2)
    kw = dict(solver="cg", rtol=1e-9, sample_mask=keep, fold=(target, phase))
    full = apply_nuFFT(dfs, **kw)
    half = apply_nuFFT(dfs, spin=2, **kw)
    assert np.linalg.norm(half - full) <= 1e-8 * np.linalg.norm(full)


def test_asymmetric_input_falls_back_to_the_full_domain(nside, healpix_map):
    """A bare array need not be a DFS array, so the symmetry is checked, not assumed."""
    dfs = np.asarray(_dfs(healpix_map)).copy()
    x = _upsampled_latitudes(nside)
    mu, _, _, parity, _, _ = _mirror_plan(x, 0, 4 * nside, 4 * nside + 1)
    dfs[3, 5] += 10.0 * np.abs(dfs).max()  # break the symmetry
    assert not _is_mirror_symmetric(dfs.T, mu, parity)
    # the solve still runs, on the full domain, and matches the full-domain answer
    got = apply_nuFFT(dfs, solver="cg", rtol=1e-10, spin=0)
    ref = apply_nuFFT(dfs, solver="cg", rtol=1e-10)
    assert np.linalg.norm(got - ref) <= 1e-10 * np.linalg.norm(ref)


# --- the half-domain DFS ------------------------------------------------------------


@pytest.mark.parametrize("spin", [0, 2])
def test_half_dfs_matches_the_first_half_of_the_full_one(nside, spin):
    """``half=True`` must reproduce the full layout's leading rows exactly.

    The mirrored rings are an exact reflection of the originals, and the latitude solve
    restricts to the fundamental rows anyway, so building them is pure cost. The pole
    stencils still need the mirror, but only of the few rings nearest each pole, which
    ``_pole_stencils`` forms directly.
    """
    rng = np.random.default_rng(0)
    npix = 12 * nside * nside
    mp = rng.standard_normal(npix) + (1j * rng.standard_normal(npix) if spin else 0)
    upsampled, fft_coeff = transform_healpix_to_grid(mp)
    full_map, full_fft = DFS(upsampled, fft_coeff, spin=spin)
    half_map, half_fft = DFS(upsampled, fft_coeff, spin=spin, half=True)
    n = 4 * nside + 1
    assert half_fft.shape == (n, 4 * nside)
    assert np.array_equal(full_fft[:n], half_fft)
    assert half_map is None, "the map half is not built; no caller uses it"
    assert full_map.shape[0] == 8 * nside


@pytest.mark.parametrize("spin", [0, 2])
def test_half_fold_plan_matches_the_first_half_of_the_full_one(nside, spin):
    n = 4 * nside + 1
    full = dfs_fold_plan(nside, spin, 1e-2)
    half = dfs_fold_plan(nside, spin, 1e-2, half=True)
    for f, h in zip(full, half):
        assert np.array_equal(f[:n], h)


def test_half_domain_input_gives_the_identical_solve(nside):
    """Feeding the half arrays must change nothing about the answer."""
    rng = np.random.default_rng(0)
    npix = 12 * nside * nside
    z = rng.standard_normal(npix) + 1j * rng.standard_normal(npix)
    upsampled, fft_coeff = transform_healpix_to_grid(z)
    _, full = DFS(upsampled, fft_coeff, spin=2)
    _, half = DFS(upsampled, fft_coeff, spin=2, half=True)
    ft, fp, fk = dfs_fold_plan(nside, 2, 1e-2)
    ht, hp_, hk = dfs_fold_plan(nside, 2, 1e-2, half=True)
    kw = dict(solver="cg", rtol=1e-12, maxiter=600, eta=None, spin=2)
    a = apply_nuFFT(full, sample_mask=fk, fold=(ft, fp), **kw)
    b = apply_nuFFT(half, sample_mask=hk, fold=(ht, hp_), half_domain=True, **kw)
    assert np.array_equal(a, b)


def test_half_domain_rejects_a_full_height_array(nside):
    """The row count is the only thing distinguishing the two layouts, so check it."""
    rng = np.random.default_rng(0)
    npix = 12 * nside * nside
    z = rng.standard_normal(npix) + 1j * rng.standard_normal(npix)
    upsampled, fft_coeff = transform_healpix_to_grid(z)
    _, full = DFS(upsampled, fft_coeff, spin=2)
    with pytest.raises(ValueError, match="half_domain expects"):
        apply_nuFFT(full, solver="cg", spin=2, half_domain=True)


def test_map_rows_returns_exactly_the_pole_stencil_slices(nside):
    """The short map form must be bit-identical to the slices it replaces.

    The pole fill is the only consumer of the map, and it reads
    ``pole_stencil_rows(nside)`` rings from each end. Transforming just those is
    bit-identical to slicing the full inverse FFT, so this is a pure saving.
    """
    from src.double_fourier_sphere import pole_stencil_rows

    rng = np.random.default_rng(0)
    npix = 12 * nside * nside
    for mp in (
        rng.standard_normal(npix),
        rng.standard_normal(npix) + 1j * rng.standard_normal(npix),
    ):
        k = pole_stencil_rows(nside)
        full, fc_full = transform_healpix_to_grid(mp)
        edge, fc_edge = transform_healpix_to_grid(mp, map_rows=k)
        assert np.array_equal(fc_full, fc_edge)
        assert edge.shape == (2 * k, 4 * nside)
        assert np.array_equal(edge[:k], full[:k])
        assert np.array_equal(edge[k:], full[-k:])


@pytest.mark.parametrize("spin", [0, 2])
def test_dfs_half_accepts_the_short_map(nside, spin):
    """``DFS(half=True)`` must give the same answer from the short map as the full one."""
    from src.double_fourier_sphere import pole_stencil_rows

    rng = np.random.default_rng(0)
    npix = 12 * nside * nside
    mp = rng.standard_normal(npix) + (1j * rng.standard_normal(npix) if spin else 0)
    k = pole_stencil_rows(nside)
    full, fc = transform_healpix_to_grid(mp)
    edge, _ = transform_healpix_to_grid(mp, map_rows=k)
    _, from_full = DFS(full, fc, spin=spin, half=True)
    _, from_edge = DFS(edge, fc, spin=spin, half=True)
    assert np.array_equal(from_full, from_edge)


def test_pole_stencils_rejects_too_few_rows(nside):
    from src.double_fourier_sphere import _pole_stencils, pole_stencil_rows

    k = pole_stencil_rows(nside)
    too_short = np.zeros((2 * k - 1, 4 * nside))
    with pytest.raises(ValueError, match="rings from each end"):
        _pole_stencils(too_short, 0)
