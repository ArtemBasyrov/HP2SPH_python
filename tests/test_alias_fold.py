"""The polar-ring longitude alias, folded into the latitude forward operator.

A HEALPix ring of ``npix`` pixels does not measure longitude mode ``m``; it measures the
whole alias family of ``m``. ``transform_healpix_to_grid`` zero-pads each polar ring's
spectrum, which asserts both that the ring measured each mode separately AND that every
unresolved mode vanishes there. The first assertion is simply false; the second is true
to many digits for all but a handful of (ring, mode) pairs.

``ring_alias_target`` / ``ring_fold_plan`` / ``dfs_fold_plan`` state that exactly, and
``apply_nuFFT(fold=...)`` solves against it.
"""

import numpy as np
import healpy as hp
import pytest

from hp2sph.data_interpolation import (
    mode_pole_envelope,
    ring_alias_target,
    ring_fold_plan,
    ring_pixel_counts,
    transform_healpix_to_grid,
)
from hp2sph.double_fourier_sphere import DFS, dfs_fold_plan
from hp2sph.nuFFT import _fold_ops, apply_nuFFT
from hp2sph.FSHT import inverse_FSHT_spin
from hp2sph.spin_transform import _build_spin_F, _signed_spin_alm, forward_spin, SPIN


def _fold(array, target, phase):
    """Apply the fold to a (rows, n_lon) coefficient array."""
    array = np.asarray(array)
    rows, n_lon = array.shape
    flat = (target + n_lon * np.arange(rows)[:, None]).ravel()
    t = (array * phase).ravel()
    out = np.bincount(flat, weights=t.real, minlength=rows * n_lon).astype(complex)
    out += 1j * np.bincount(flat, weights=t.imag, minlength=rows * n_lon)
    return out.reshape(rows, n_lon)


def _spin_probe(nside, lmax, ell, m, amp=1.0 + 0.5j):
    aE = np.zeros(hp.Alm.getsize(lmax), dtype=complex)
    aB = np.zeros_like(aE)
    aE[hp.Alm.getidx(lmax, ell, m)] = amp
    Q, U = hp.alm2map_spin([aE, aB], nside, SPIN, lmax)
    return aE, aB, np.asarray(Q), np.asarray(U)


# --------------------------------------------------------------------------- #
# The alias model itself                                                       #
# --------------------------------------------------------------------------- #
def test_alias_model_reproduces_the_measured_ring_spectrum(nside):
    """``fold(true coefficients) == what transform_healpix_to_grid measured``.

    The exact ``c_m(theta_r)`` come from the (machine-exact) backward pipeline, so this
    pins the whole model -- residue folding AND the ``exp(i (m-b) phi0)`` the per-ring
    longitude referencing contributes -- against the live forward code.
    """
    lmax = 2 * nside
    aE, aB, Q, U = _spin_probe(nside, lmax, lmax - 1, min(3, lmax - 1))

    L = 2 * nside
    F = _build_spin_F(
        _signed_spin_alm(aE, aB, lmax, +SPIN), lmax, L + 1, 2 * L + 1, +SPIN
    )
    from hp2sph.nuFFT import inverse_nuFFT

    _, bivar = inverse_FSHT_spin(F, nside, +SPIN)
    exact = np.asarray(inverse_nuFFT(np.asarray(bivar)))

    _, fft_coeff = transform_healpix_to_grid(Q + 1j * U)
    _, measured = DFS(
        np.fft.ifft(fft_coeff, axis=-1, norm="forward"), fft_coeff, spin=+SPIN
    )
    measured = np.asarray(measured)

    target, phase, keep = dfs_fold_plan(nside, +SPIN, tol=0.0)  # nothing trusted away
    folded = _fold(exact, target, phase)

    n_rings = 4 * nside - 1
    rings = np.r_[1 : n_rings + 1, n_rings + 2 : 2 * n_rings + 2]
    scale = np.abs(exact).max()
    kept = keep[rings]
    assert np.abs(folded - measured)[rings][kept].max() < 1e-10 * scale
    # ... and the padded slots the ring never produced really are empty
    assert np.abs(measured)[rings][~kept].max() == 0.0


def test_fold_targets_are_the_ring_residues(nside):
    """Mode ``m`` lands in bin ``m mod npix``, and the Nyquist bin counts as produced."""
    target, phase, resolved = ring_alias_target(nside)
    sizes = ring_pixel_counts(nside)
    n_lon = 4 * nside
    m = np.arange(n_lon) - n_lon // 2
    for r, npix in enumerate(sizes):
        b = target[r] - n_lon // 2
        np.testing.assert_array_equal((m - b) % npix, 0)
        assert (np.abs(b) <= npix // 2).all()
        assert resolved[r].sum() == npix
        # the equatorial belt is untouched: identity target, unit phase, all resolved
        if npix == n_lon:
            np.testing.assert_array_equal(target[r], np.arange(n_lon))
            np.testing.assert_allclose(phase[r], 1.0, atol=1e-15)
            assert resolved[r].all()


def test_fold_operator_is_a_true_adjoint(nside):
    """LSMR and CG are only valid if ``apply``/``adjoint`` really are adjoint."""
    n_lon = 4 * nside
    M = 8 * nside
    target, phase, _ = dfs_fold_plan(nside, +SPIN, tol=1e-2)
    apply, adjoint = _fold_ops((target, phase), n_lon, M)
    rng = np.random.default_rng(3)
    x = rng.standard_normal((n_lon, M)) + 1j * rng.standard_normal((n_lon, M))
    y = rng.standard_normal((n_lon, M)) + 1j * rng.standard_normal((n_lon, M))
    lhs = np.vdot(y, apply(x))
    rhs = np.vdot(adjoint(y), x)
    assert abs(lhs - rhs) <= 1e-12 * abs(lhs)


def test_envelope_bounds_the_true_latitude_profile(nside):
    """``mode_pole_envelope`` must not UNDERSTATE what a band-limited field carries.

    It is the rule that decides which "unresolved mode = 0" assertions are safe to keep,
    so understating is the dangerous direction -- that is exactly the bug the plain
    ``sin^a(theta/2)`` shape had (it misses the Bessel ``l*theta`` scale and understates
    by six orders at the band edge).
    """
    lmax = 2 * nside
    from hp2sph.nuFFT import inverse_nuFFT

    for m in (2, 3):
        if m > lmax - 1:
            continue
        aE, aB, _, _ = _spin_probe(nside, lmax, lmax - 1, m)
        L = 2 * nside
        F = _build_spin_F(
            _signed_spin_alm(aE, aB, lmax, +SPIN), lmax, L + 1, 2 * L + 1, +SPIN
        )
        _, bivar = inverse_FSHT_spin(F, nside, +SPIN)
        exact = np.asarray(inverse_nuFFT(np.asarray(bivar)))
        rho = mode_pole_envelope(nside, +SPIN)
        n_rings = 4 * nside - 1
        rings = exact[1 : n_rings + 1]
        peak = np.abs(rings).max(axis=0)
        nz = peak > 0
        true_frac = np.abs(rings)[:, nz] / peak[nz]
        # a factor 4 of headroom on a rule that spans 16 orders of magnitude
        assert (true_frac <= 4 * rho[:, nz] + 1e-12).all()


# --------------------------------------------------------------------------- #
# Reduction to the existing behaviour                                          #
# --------------------------------------------------------------------------- #
def test_fold_with_nothing_relaxed_is_the_current_path(nside, healpix_map, relerr):
    """The fold is a strict generalization: relax nothing and it is the identity.

    ``mode_pole_envelope`` is a fraction of a mode's own peak, so ``tol > 1`` trusts
    every unresolved mode and the plan collapses to "assert zero in place" -- which is
    exactly the zero-padding the pipeline already does.
    """
    target, phase, keep = ring_fold_plan(nside, spin=0, tol=2.0)
    assert keep.all()
    identity = np.broadcast_to(np.arange(target.shape[1]), target.shape)
    np.testing.assert_array_equal(target, identity)
    np.testing.assert_allclose(phase, 1.0, atol=0)

    up, fft_coeff = transform_healpix_to_grid(np.asarray(healpix_map))
    _, dfs = DFS(up, fft_coeff, spin=0)
    dtarget, dphase, dkeep = dfs_fold_plan(nside, spin=0, tol=2.0)
    plain = apply_nuFFT(dfs, solver="cg")
    folded = apply_nuFFT(dfs, solver="cg", fold=(dtarget, dphase), sample_mask=dkeep)
    assert relerr(folded, plain) < 1e-14


def test_svd_solver_rejects_the_fold(nside, healpix_map):
    """The SVD path shares one per-column factorization; the fold couples the columns."""
    up, fft_coeff = transform_healpix_to_grid(np.asarray(healpix_map))
    _, dfs = DFS(up, fft_coeff, spin=0)
    target, phase, _ = dfs_fold_plan(nside, spin=0, tol=1e-2)
    with pytest.raises(NotImplementedError):
        apply_nuFFT(dfs, solver="svd", fold=(target, phase))


# --------------------------------------------------------------------------- #
# What it buys                                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("m", [3, 5])
def test_fold_kills_the_alias_leakage_of_a_single_harmonic(m):
    """The symptom: a pure ``a^E_(l,m)`` leaked ~2-5% into other cells for ``m >= 3``.

    ``m = 3`` aliases onto ``m = -1`` on the innermost 4-pixel ring, where a spin-2 field
    is O(theta) rather than O(theta^3), so the ring's own bin carried it and the pipeline
    attributed it to the wrong mode.

    The threshold is absolute, not a ratio against the superseded masked route, which no
    longer exists. For reference that route measured 2.5e-2 (m=3) and 1.9e-2 (m=5) here.
    """
    nside, lmax = 16, 32
    ell = lmax - 1
    aE, _, Q, U = _spin_probe(nside, lmax, ell, m)
    i0 = hp.Alm.getidx(lmax, ell, m)

    E, _ = forward_spin(Q, U, lmax)
    leak = np.sqrt(np.sum(np.abs(E - aE) ** 2)) / abs(aE[i0])
    assert leak < 2e-3
    assert abs(E[i0] / aE[i0] - 1.0) < 1e-3


def test_fold_beats_ring_weights_at_the_band_edge():
    """The point of item A: the top of the band, which is where the error lived.

    healpy's weighted IQU path is the strongest single-pass polarization route it has
    (``map2alm_spin`` takes neither weights nor iteration), and the alias was what kept
    HP2SPH behind it here.
    """
    nside, lmax = 16, 32
    from benchmarks.common import random_EB

    aE, aB = random_EB(lmax, seed=0, slope=1.5, mmax_cap=2 * nside - 1)
    Q, U = hp.alm2map_spin([aE, aB], nside, SPIN, lmax)

    def cl_err(a):
        c = hp.alm2cl(a, lmax=lmax)
        cr = hp.alm2cl(aE, lmax=lmax)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.abs(c - cr) / np.abs(cr)

    edge = slice(lmax - 4, lmax + 1)
    zero = np.zeros_like(np.asarray(Q))
    _, ring, _ = hp.map2alm([zero, Q, U], lmax=lmax, pol=True, use_weights=True, iter=0)

    def rms(a):
        return np.sqrt(np.mean(cl_err(a)[edge] ** 2))

    assert rms(forward_spin(Q, U, lmax)[0]) < rms(ring) / 5


def test_fold_survives_above_band_power():
    """The regime where the superseded masked route failed worst.

    Its accuracy there rested on LSMR's early stopping rather than on stability, and CG
    has no equivalent accident, so this had to be checked before dropping that route.
    Measured at nside 8, ``signal_lmax = 4*nside``: the masked route gave a band-edge
    relative C_l error of 1.7e+1, the fold gives ~8e-2.
    """
    nside, lmax = 8, 16
    signal_lmax = 4 * nside
    from benchmarks.common import random_EB

    aE, aB = random_EB(signal_lmax, seed=0, slope=1.5, mmax_cap=2 * nside - 1)
    Q, U = hp.alm2map_spin([aE, aB], nside, SPIN, signal_lmax)
    ref = hp.resize_alm(aE, signal_lmax, signal_lmax, lmax, lmax)

    E, _ = forward_spin(Q, U, lmax)
    c, cr = hp.alm2cl(E, lmax=lmax), hp.alm2cl(ref, lmax=lmax)
    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.abs(c - cr) / np.abs(cr)
    assert np.sqrt(np.mean(err[lmax - 4 : lmax + 1] ** 2)) < 1.0
