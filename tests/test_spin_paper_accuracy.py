"""Phase 5: end-to-end spin-2 (Q/U <-> E/B) transforms vs healpy.

``src.spin_transform.forward_spin`` / ``backward_spin`` go HEALPix (Q,U) <-> (aE,aB)
through the spin stages. Two routes are covered:

* ``analysis="hp2sph"`` (the default, the true method): the hand-rolled DFS + latitude
  nuFFT with no resampling. ``test_spin_hp2sph_*`` pin it -- single harmonics come back
  with gain 1 and no l-spread, and a full sky beats healpy ``map2alm_spin``.
* ``analysis="library"``: equiangular resample + the library's exact spin analysis +
  the validated E/B decode. Its floor is the HEALPix<->equiangular RESAMPLING, so those
  tests run oversampled (nside well above lmax) where that floor is small. They are
  pinned to the library route explicitly rather than relying on the default.

``backward_spin`` has both routes too; the round-trip test here keeps the LIBRARY
route on both ends so it stays a test of that route. The native synthesis
(``synthesis="hp2sph"``, exact) has its own file, ``tests/test_spin_backward.py``.

See SPIN2_PLAN.md (Phase 5).
"""

import numpy as np
import healpy as hp
import pytest

from src import ft_sphere
from src.spin_transform import forward_spin, backward_spin

pytestmark = pytest.mark.ft

if not getattr(ft_sphere, "_HAVE_SPIN_FFTW", False):
    pytest.skip(
        "libfasttransforms has no spinsph FFTW analysis", allow_module_level=True
    )


def _random_EB(lmax, seed):
    rng = np.random.default_rng(seed)
    n = hp.Alm.getsize(lmax)
    aE = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    aB = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    m0 = np.array([hp.Alm.getidx(lmax, ell, 0) for ell in range(lmax + 1)])
    aE[m0] = aE[m0].real
    aB[m0] = aB[m0].real
    return aE.astype(np.complex128), aB.astype(np.complex128)


def test_spin_forward_vs_healpy():
    """forward_spin recovers (aE, aB): per-l C_l^EE / C_l^BB match healpy in the bulk."""
    nside, lmax = 128, 16  # oversampled so the bilinear resampling floor is small
    aE, aB = _random_EB(lmax, seed=1)
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, lmax)

    aE_rec, aB_rec = forward_spin(Q, U, lmax, analysis="library")

    band = slice(2, lmax - 1)  # exclude the equiangular-grid Nyquist edge
    for name, rec, ref in [("EE", aE_rec, aE), ("BB", aB_rec, aB)]:
        cl_rec = hp.alm2cl(rec, lmax=lmax)[band]
        cl_in = hp.alm2cl(ref, lmax=lmax)[band]
        rel = np.abs(cl_rec - cl_in) / cl_in
        assert np.median(rel) < 0.05, (
            f"{name} per-l median rel err {np.median(rel):.3f}"
        )


def test_spin_pure_E_stays_E():
    """A pure-E sky decodes to negligible B (no E->B leakage beyond the floor)."""
    nside, lmax = 128, 16
    aE, _ = _random_EB(lmax, seed=2)
    aB = np.zeros_like(aE)
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, lmax)

    aE_rec, aB_rec = forward_spin(Q, U, lmax, analysis="library")
    pE = np.sum(np.abs(aE_rec) ** 2)
    pB = np.sum(np.abs(aB_rec) ** 2)
    assert pB / pE < 1e-2, f"B/E power leak {pB / pE:.2e}"


def test_spin_roundtrip():
    """backward_spin(forward_spin(Q,U)) reproduces (Q,U) in the bulk (resample-limited).

    Both directions use the LIBRARY route on purpose: it synthesizes on the
    FastTransforms equiangular grid and bilinearly resamples to HEALPix, so it is
    resampling-limited, and pairing it with the matching library forward keeps the test
    about the transform rather than about the interpolation. The NATIVE round trip
    (``hp2sph`` both ways, far more accurate) is in ``tests/test_spin_backward.py``.
    """
    nside, lmax = 128, 16
    aE, aB = _random_EB(lmax, seed=3)
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, lmax)

    aE_rec, aB_rec = forward_spin(Q, U, lmax, analysis="library")
    Q_rt, U_rt = backward_spin(aE_rec, aB_rec, nside, lmax=lmax, synthesis="library")

    # compare back in harmonic space (robust to the pixel resampling of unsampled
    # high-frequency content): C_l^EE of a second forward should match the first
    aE2, aB2 = forward_spin(Q_rt, U_rt, lmax, analysis="library")
    band = slice(2, lmax - 1)
    rel = np.abs(hp.alm2cl(aE2, lmax=lmax)[band] - hp.alm2cl(aE_rec, lmax=lmax)[band])
    rel /= hp.alm2cl(aE_rec, lmax=lmax)[band]
    assert np.median(rel) < 0.1, f"round-trip EE median rel err {np.median(rel):.3f}"


def test_spin_hp2sph_m0_matches_healpy():
    """The hand-rolled (no-resample) route is correct for the m=0 (zonal) modes."""
    nside, lmax = 16, 32
    aE, aB = _random_EB(lmax, seed=4)
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, lmax)

    aE_rec, aB_rec = forward_spin(Q, U, lmax, analysis="hp2sph")

    # m=0 (zonal) coefficients only, mid band (avoid the very top l)
    idx0 = [hp.Alm.getidx(lmax, ell, 0) for ell in range(2, lmax - lmax // 4)]
    idx0 = np.array(idx0)
    relE = np.abs(aE_rec[idx0] - aE[idx0]) / (np.abs(aE[idx0]) + 1e-12)
    assert np.median(relE) < 0.1, f"hp2sph m=0 EE median rel err {np.median(relE):.3f}"


@pytest.mark.parametrize(
    "ell,m", [(4, 0), (6, 0), (4, 1), (4, 2), (8, 2), (4, 3), (4, 4), (8, 5), (10, 10)]
)
@pytest.mark.parametrize("field", ["E", "B"])
def test_spin_hp2sph_single_harmonic_gain(ell, m, field):
    """The true HP2SPH spin route recovers a single spin harmonic with gain 1.

    This is the test the ``m != 0`` fix is about. Three things had to be right:

    * the polar-ring alias (``ring_fold_plan``) -- the innermost polar rings (4 pixels)
      do not resolve ``|m| = 2``, which for a spin-2 field is exactly the mode that is
      O(1) AT the pole. Zero-padding them asserted that content was zero; the fold
      models what the ring actually measured, the alias sum over the mode's residue
      family.
    * the south-pole row of the DFS (``test_dfs_south_pole_row_is_the_pole``);
    * ``FSHT.spin_g_to_library`` -- the ``x = pi - theta`` reflection has to be undone
      in the BIVARIATE domain, because the reflection flips the spin
      (``{}_s Y_lm(pi-theta) = (-1)^(l+m) {}_{-s} Y_lm(theta)``) and so cannot be
      undone by any phase on the output coefficients.

    Before the fix the ``m != 0`` gains were 0.1-0.68 with power spread over every l of
    the same parity; ``m = 0`` was already exact.
    """
    nside, lmax = 16, 24
    n = hp.Alm.getsize(lmax)
    aE = np.zeros(n, dtype=np.complex128)
    aB = np.zeros(n, dtype=np.complex128)
    (aE if field == "E" else aB)[hp.Alm.getidx(lmax, ell, m)] = 1.0
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, lmax)

    aE_rec, aB_rec = forward_spin(Q, U, lmax, analysis="hp2sph")
    rec, other = (aE_rec, aB_rec) if field == "E" else (aB_rec, aE_rec)

    i = hp.Alm.getidx(lmax, ell, m)
    assert abs(rec[i] - 1.0) < 5e-3, f"gain {rec[i]:.6f} != 1 at (l={ell}, m={m})"
    assert abs(other[i]) < 5e-3, f"E<->B leak {abs(other[i]):.2e}"

    # essentially all the power stays in the input l (no spreading across l)
    col = np.array([rec[hp.Alm.getidx(lmax, L, m)] for L in range(max(m, 2), lmax + 1)])
    frac = abs(rec[i]) ** 2 / np.sum(np.abs(col) ** 2)
    assert frac > 0.999, f"only {frac:.4f} of the power landed at l={ell}"


def test_spin_hp2sph_matches_healpy_full_sky():
    """A full random (aE, aB) sky through the true HP2SPH route matches healpy.

    No resampling anywhere -- this is the route whose ``m != 0`` output the module
    docstring used to warn against.
    """
    nside, lmax = 16, 24
    aE, aB = _random_EB(lmax, seed=11)
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, lmax)

    aE_rec, aB_rec = forward_spin(Q, U, lmax, analysis="hp2sph")

    band = slice(2, lmax - 1)
    for name, rec, ref in [("EE", aE_rec, aE), ("BB", aB_rec, aB)]:
        cl_rec = hp.alm2cl(rec, lmax=lmax)[band]
        cl_in = hp.alm2cl(ref, lmax=lmax)[band]
        rel = np.abs(cl_rec - cl_in) / cl_in
        assert np.median(rel) < 0.02, (
            f"{name} per-l median rel err {np.median(rel):.4f}"
        )


def test_spin_hp2sph_pure_E_stays_E():
    """A pure-E sky stays B-free through the hand-rolled route."""
    nside, lmax = 16, 24
    aE, _ = _random_EB(lmax, seed=12)
    aB = np.zeros_like(aE)
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, lmax)

    aE_rec, aB_rec = forward_spin(Q, U, lmax, analysis="hp2sph")
    pE = np.sum(np.abs(aE_rec) ** 2)
    pB = np.sum(np.abs(aB_rec) ** 2)
    assert pB / pE < 1e-3, f"B/E power leak {pB / pE:.2e}"
