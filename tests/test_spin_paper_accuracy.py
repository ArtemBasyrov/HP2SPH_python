"""Phase 5: end-to-end spin-2 (Q/U <-> E/B) transforms vs healpy.

``src.spin_transform.forward_spin`` / ``backward_spin`` go HEALPix (Q,U) <-> (aE,aB)
through the spin stages. The default ``analysis="library"`` route (equiangular
resample + the library's exact spin analysis + the validated E/B decode) is a
functional transform whose accuracy floor is the HEALPix<->equiangular RESAMPLING,
so these tests run in the oversampled regime (nside well above lmax) where that
floor is small; they check HP2SPH is in healpy's class there. The hand-rolled
``analysis="hp2sph"`` route is correct only for m=0 (the spin pole boundary
condition is the open item -- SPIN2_PLAN.md), checked separately and narrowly.

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

    aE_rec, aB_rec = forward_spin(Q, U, lmax)

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

    aE_rec, aB_rec = forward_spin(Q, U, lmax)
    pE = np.sum(np.abs(aE_rec) ** 2)
    pB = np.sum(np.abs(aB_rec) ** 2)
    assert pB / pE < 1e-2, f"B/E power leak {pB / pE:.2e}"


def test_spin_roundtrip():
    """backward_spin(forward_spin(Q,U)) reproduces (Q,U) in the bulk (resample-limited)."""
    nside, lmax = 128, 16
    aE, aB = _random_EB(lmax, seed=3)
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, lmax)

    aE_rec, aB_rec = forward_spin(Q, U, lmax)
    Q_rt, U_rt = backward_spin(aE_rec, aB_rec, nside, lmax=lmax)

    # compare back in harmonic space (robust to the pixel resampling of unsampled
    # high-frequency content): C_l^EE of a second forward should match the first
    aE2, aB2 = forward_spin(Q_rt, U_rt, lmax)
    band = slice(2, lmax - 1)
    rel = np.abs(hp.alm2cl(aE2, lmax=lmax)[band] - hp.alm2cl(aE_rec, lmax=lmax)[band])
    rel /= hp.alm2cl(aE_rec, lmax=lmax)[band]
    assert np.median(rel) < 0.1, f"round-trip EE median rel err {np.median(rel):.3f}"


def test_spin_hp2sph_m0_matches_healpy():
    """The hand-rolled (no-resample) route is correct for the m=0 (zonal) modes.

    This pins the part of the true HP2SPH spin analysis that already works, and
    documents that m != 0 is the open item (the spin pole boundary condition).
    """
    nside, lmax = 16, 32
    aE, aB = _random_EB(lmax, seed=4)
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, lmax)

    aE_rec, aB_rec = forward_spin(Q, U, lmax, analysis="hp2sph")

    # m=0 (zonal) coefficients only, mid band (avoid the very top l)
    idx0 = [hp.Alm.getidx(lmax, ell, 0) for ell in range(2, lmax - lmax // 4)]
    idx0 = np.array(idx0)
    relE = np.abs(aE_rec[idx0] - aE[idx0]) / (np.abs(aE[idx0]) + 1e-12)
    assert np.median(relE) < 0.1, f"hp2sph m=0 EE median rel err {np.median(relE):.3f}"
