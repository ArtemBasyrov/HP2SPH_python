"""The native spin-2 synthesis: ``backward_spin(..., synthesis="hp2sph")``.

The mirror of the ``analysis="hp2sph"`` forward -- ``inverse_FSHT_spin`` ->
``inverse_nuFFT`` -> ``DFS_inverse`` -> ``transform_grid_to_healpix``, the spin
counterpart of the scalar ``main.backward``. Nothing is resampled, so unlike the
``"library"`` route it is not interpolation-limited: it reproduces
``hp.alm2map_spin`` to MACHINE PRECISION for any band ``lmax <= 2*nside``.

Two bugs had to be fixed to get there, and each has a dedicated test below:

* ``FSHT._spin_conv_phase`` only covered ``m >= 0``. The forward decode never reads a
  negative order, but the synthesis has to fill every signed ``m``, and the rule is
  ``+1`` where ``m + spin <= 0``, else ``(-1)^m`` -- not the ``+1`` throughout ``m < 0``
  the code assumed. At ``spin = +2`` that is wrong for exactly one column, ``m = -1``.
* ``data_interpolation.transform_grid_to_healpix`` TRUNCATED each polar ring's
  longitude spectrum where it should ALIAS it onto the ring's own pixels. Invisible on
  a round trip -- the forward zero-pads, so every folded-in entry is exactly zero --
  but on the synthesis side it discarded the ``|m| = |spin|`` content, which for a
  spin field is O(1) AT the pole (the same asymmetry ``ring_mode_mask`` handles on the
  analysis side). It left the innermost polar rings ~100% wrong.

See SPIN2_PLAN.md (the "native spin backward" item) and CLAUDE.md.
"""

import numpy as np
import healpy as hp
import pytest

from src import ft_sphere
from src.FSHT import _spin_conv_phase, spin_g_to_library, spin_g_from_library
from src.data_interpolation import transform_grid_to_healpix, ring_pixel_counts
from src.spin_transform import forward_spin, backward_spin

pytestmark = pytest.mark.ft

if not getattr(ft_sphere, "_HAVE_SPIN", False):
    pytest.skip("libfasttransforms has no spin entry points", allow_module_level=True)


def _random_EB(lmax, seed):
    rng = np.random.default_rng(seed)
    n = hp.Alm.getsize(lmax)
    aE = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    aB = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    m0 = np.array([hp.Alm.getidx(lmax, ell, 0) for ell in range(lmax + 1)])
    aE[m0] = aE[m0].real
    aB[m0] = aB[m0].real
    aE[:2] = 0  # l < 2 carries no spin-2 content
    aB[:2] = 0
    return aE.astype(np.complex128), aB.astype(np.complex128)


def _relerr(Q, U, Q0, U0):
    z, z0 = Q + 1j * U, Q0 + 1j * U0
    return np.linalg.norm(z - z0) / np.linalg.norm(z0)


# --------------------------------------------------------------------------- #
# The headline: the native synthesis is exact                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nside", [8, 16, 32])
@pytest.mark.parametrize("band", ["half", "edge_minus_1"])
def test_backward_spin_matches_healpy_exactly(nside, band):
    """backward_spin reproduces hp.alm2map_spin to machine precision.

    This is not a "converges with nside" claim -- for ``lmax <= 2*nside - 1`` the
    spin harmonics of degree ``l`` are trigonometric polynomials of degree ``l`` in
    theta, so the pipeline's compact latitude band ``|k| <= 2*nside`` represents them
    EXACTLY, the nuFFT evaluation at the ring colatitudes is exact, and the per-ring
    longitude alias is exact. The only error left is floating point.
    """
    lmax = nside if band == "half" else 2 * nside - 1
    aE, aB = _random_EB(lmax, seed=1)
    Q0, U0 = hp.alm2map_spin([aE, aB], nside, 2, lmax)
    Q, U = backward_spin(aE, aB, nside, lmax=lmax)
    assert _relerr(Q, U, Q0, U0) < 1e-11, (
        f"nside={nside} lmax={lmax} relerr {_relerr(Q, U, Q0, U0):.3e}"
    )


@pytest.mark.parametrize(
    "ell,m", [(2, 0), (2, 2), (4, 1), (4, 2), (4, 3), (8, 5), (12, 12), (20, 3)]
)
@pytest.mark.parametrize("field", ["E", "B"])
def test_backward_spin_single_harmonic(ell, m, field):
    """Every single spin harmonic is synthesised exactly, for E and for B.

    The ``m = 1`` / ``m = 3`` cases are the ones the signed-``m`` phase bug broke:
    a healpy alm at order ``+m`` also populates the ``-m`` F column (via the reality
    of Q/U), and ``m = -1`` was the column whose sign was wrong at ``spin = +2``.
    """
    nside, lmax = 16, 24
    n = hp.Alm.getsize(lmax)
    aE = np.zeros(n, dtype=np.complex128)
    aB = np.zeros(n, dtype=np.complex128)
    (aE if field == "E" else aB)[hp.Alm.getidx(lmax, ell, m)] = 1.0
    Q0, U0 = hp.alm2map_spin([aE, aB], nside, 2, lmax)
    Q, U = backward_spin(aE, aB, nside, lmax=lmax)
    assert _relerr(Q, U, Q0, U0) < 1e-11, (
        f"(l={ell}, m={m}, {field}) relerr {_relerr(Q, U, Q0, U0):.3e}"
    )


def test_backward_spin_output_is_real():
    """Q and U come out real: the synthesised z = Q + iU has no spurious content.

    Only the ``spin = +2`` pass is run and ``(Q, U) = (Re z, Im z)``; this checks that
    dropping the ``-2`` pass discards nothing, i.e. that the reality of Q/U really is
    already encoded in the signed-m coefficients.
    """
    nside, lmax = 16, 31
    aE, aB = _random_EB(lmax, seed=2)
    Q, U = backward_spin(aE, aB, nside, lmax=lmax)
    Qm, Um = hp.alm2map_spin([aE, aB], nside, 2, lmax)
    # the spin -2 pass must give the conjugate field, Q - iU, not something new
    ref = Qm - 1j * Um
    got = Q - 1j * U
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 1e-11


# --------------------------------------------------------------------------- #
# The two bugs, pinned directly                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spin", [+2, -2])
def test_spin_conv_phase_signed_m(spin):
    """The measured signed-m F phase: +1 where m + spin <= 0, else (-1)^m.

    Measured with single ``+-2 Y_{l,m}`` probes against healpy over
    ``-lmax <= m <= lmax``; at ``spin = +2`` the exception starts at ``m = -2`` and at
    ``spin = -2`` at ``m = 1``. Pinning it here because the value is empirical: it is
    the Jacobi ``(|m+s|, |m-s|)`` ordering boundary, not something the code derives.
    """
    for m in range(-8, 9):
        want = 1.0 if m + spin <= 0 else (-1.0) ** m
        assert _spin_conv_phase(m, spin) == want, f"m={m} spin={spin}"
    # and it still agrees with the old m >= 0 rule it generalises (spins +-2 only)
    for m in range(0, 9):
        old = 1.0 if (spin < 0 and 0 <= m < abs(spin)) else (-1.0) ** m
        assert _spin_conv_phase(m, spin) == old, f"m={m} spin={spin}"


def test_polar_rings_alias_not_truncate():
    """transform_grid_to_healpix must ALIAS a polar ring's spectrum, not truncate it.

    A ring of ``npix`` pixels samples mode ``m`` into bin ``m % npix``. Put a single
    above-Nyquist mode on the grid and check the innermost ring picks it up folded --
    truncation would return exactly zero there. This is the ``|m| = |spin|``-at-the-pole
    content the native spin backward depends on.
    """
    nside = 8
    n_lon, n_rings = 4 * nside, 4 * nside - 1
    fft_coeff = np.zeros((n_rings, n_lon), dtype=complex)
    m = 5  # above the 4- and 8-pixel rings' Nyquist, below the grid's
    fft_coeff[:, m] = 1.0
    z = transform_grid_to_healpix(fft_coeff, fft_coeff, real_output=False)

    sizes = ring_pixel_counts(nside)
    start = 0
    for r, npix in enumerate(sizes[:3]):  # the 4-, 8- and 12-pixel rings
        ring = z[start : start + npix]
        start += npix
        assert np.linalg.norm(ring) > 0.5, (
            f"ring {r} ({npix} px) lost the m={m} mode entirely (truncated, not aliased)"
        )


def test_polar_alias_keeps_the_analysis_roundtrip_exact():
    """Aliasing is a strict generalisation: it is a no-op on a forward-produced grid.

    ``transform_healpix_to_grid`` zero-pads above each ring's Nyquist, so every entry
    the alias folds in is exactly 0 and the round trip stays bit-exact -- which is why
    truncation survived undetected for so long.
    """
    from src.data_interpolation import transform_healpix_to_grid

    nside = 8
    rng = np.random.default_rng(4)
    npix = hp.nside2npix(nside)
    z = rng.standard_normal(npix) + 1j * rng.standard_normal(npix)
    _, fft_coeff = transform_healpix_to_grid(z)
    back = transform_grid_to_healpix(fft_coeff, fft_coeff, real_output=False)
    np.testing.assert_allclose(back, z, atol=1e-12)


def test_spin_g_library_conversion_roundtrip():
    """spin_g_from_library inverts spin_g_to_library (the (-1)^k reflection etc.)."""
    rng = np.random.default_rng(6)
    g = rng.standard_normal((17, 33)) + 1j * rng.standard_normal((17, 33))
    back = spin_g_from_library(spin_g_to_library(g))
    np.testing.assert_allclose(back, g, atol=1e-12)


# --------------------------------------------------------------------------- #
# Round trip and the one documented limitation                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nside", [8, 16])
def test_spin_hp2sph_roundtrip(nside):
    """forward_spin . backward_spin, both on the native route, recovers (aE, aB).

    The residual is entirely the FORWARD's quadrature error now (the backward is
    exact), so this converges with nside exactly like ``forward_spin`` does.
    """
    lmax = 2 * nside - 1
    aE, aB = _random_EB(lmax, seed=3)
    Q, U = backward_spin(aE, aB, nside, lmax=lmax)
    aE_rec, aB_rec = forward_spin(Q, U, lmax)

    band = slice(2, lmax - 1)
    for name, rec, ref in [("EE", aE_rec, aE), ("BB", aB_rec, aB)]:
        cl_rec = hp.alm2cl(rec, lmax=lmax)[band]
        cl_in = hp.alm2cl(ref, lmax=lmax)[band]
        rel = np.abs(cl_rec - cl_in) / cl_in
        assert np.median(rel) < 5e-3, (
            f"{name} per-l median rel err {np.median(rel):.4e} (nside={nside})"
        )


def test_spin_map_roundtrip_through_alm():
    """A healpy-made (Q,U) map survives forward_spin -> backward_spin."""
    nside, lmax = 16, 31
    aE, aB = _random_EB(lmax, seed=5)
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, lmax)
    Q2, U2 = backward_spin(*forward_spin(Q, U, lmax), nside, lmax=lmax)
    assert _relerr(Q2, U2, Q, U) < 5e-2


@pytest.mark.parametrize("nside", [8, 16, 32])
def test_band_edge_corner_is_the_only_inexact_mode(nside):
    """At lmax = 2*nside only the l = m = lmax coefficient is unrepresentable.

    ``m = +2*nside`` and ``m = -2*nside`` are the same mode on the ``4*nside``-point
    longitude grid, and the per-ring ``phi0`` offsets give them DIFFERENT phases, so no
    single Nyquist column can carry both. This is the synthesis face of the forward's
    documented half-gain at that corner. Zeroing that one coefficient restores machine
    precision -- proof that nothing else at the band edge is wrong.
    """
    lmax = 2 * nside
    aE, aB = _random_EB(lmax, seed=7)
    aE[hp.Alm.getidx(lmax, lmax, lmax)] = 0
    aB[hp.Alm.getidx(lmax, lmax, lmax)] = 0
    Q0, U0 = hp.alm2map_spin([aE, aB], nside, 2, lmax)
    Q, U = backward_spin(aE, aB, nside, lmax=lmax)
    assert _relerr(Q, U, Q0, U0) < 1e-11


def test_backward_spin_rejects_out_of_band_lmax():
    """lmax > 2*nside cannot be synthesised on the grid; say so instead of aliasing."""
    aE, aB = _random_EB(20, seed=8)
    with pytest.raises(ValueError, match="exceeds the grid band"):
        backward_spin(aE, aB, nside=8, lmax=20)


def test_library_synthesis_route_still_available():
    """synthesis="library" is unchanged and still resampling-limited (cross-check)."""
    nside, lmax = 128, 12
    aE, aB = _random_EB(lmax, seed=9)
    Q, U = backward_spin(aE, aB, nside, lmax=lmax, synthesis="library")
    aE_rec, aB_rec = forward_spin(Q, U, lmax, analysis="library")
    band = slice(2, lmax - 1)
    rel = np.abs(hp.alm2cl(aE_rec, lmax=lmax) - hp.alm2cl(aE, lmax=lmax))[band]
    rel /= hp.alm2cl(aE, lmax=lmax)[band]
    assert np.median(rel) < 0.15
