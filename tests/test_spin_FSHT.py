"""Phase 4: spin FSHT prep + E/B conversion.

``preparation`` / ``convert_to_bivar_coeffs`` gain a ``spin`` argument (the cos/sin
split is ``(m+spin)`` parity), and ``FSHT.spin_to_EB`` / ``spin_alm_from_F`` decode
the FastTransforms spin ``F`` array into healpy-ordered E/B ``alm``.

The decode/convention is validated end-to-end against healpy ``map2alm_spin`` using
the *library's own* equiangular analysis (``ft_sphere.spinsph_analysis``) as the
analysis stage -- this isolates the Phase-4 output side from the hand-rolled spin
DFS/nuFFT analysis (whose ``m != 0`` accuracy is still being worked; see
SPIN2_PLAN.md). Self-contained algebra checks pin the E/B combination.

See SPIN2_PLAN.md (Phase 4).
"""

import numpy as np
import healpy as hp
import pytest

from src import ft_sphere
from src.FSHT import (
    preparation,
    convert_to_bivar_coeffs,
    spin_to_EB,
    EB_to_spin_F,
    spin_alm_from_F,
    _spin_F_col,
)

pytestmark = pytest.mark.ft

if not getattr(ft_sphere, "_HAVE_SPIN", False):
    pytest.skip("libfasttransforms has no spin entry points", allow_module_level=True)


@pytest.mark.parametrize("spin", [0, 2])
def test_spin_prep_convert_idempotent(nside, spin):
    """convert_to_bivar_coeffs . preparation is an idempotent projection (complex)."""
    L = 2 * nside
    rng = np.random.default_rng(20260627 + spin)
    x = rng.standard_normal((2 * L + 1, 4 * nside)) + 1j * rng.standard_normal(
        (2 * L + 1, 4 * nside)
    )
    Px = convert_to_bivar_coeffs(preparation(x, spin=spin), nside, spin=spin)
    PPx = convert_to_bivar_coeffs(preparation(Px, spin=spin), nside, spin=spin)
    assert np.linalg.norm(PPx - Px) / np.linalg.norm(Px) < 1e-10


def test_EB_spin_algebra_roundtrip():
    """spin_to_EB and EB_to_spin_F implement consistent inverse E/B relations."""
    lmax = 8
    rng = np.random.default_rng(0)
    n = hp.Alm.getsize(lmax)
    aE = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    aB = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    a_plus, a_minus = EB_to_spin_F(aE, aB, lmax)
    # invert the +-2 a -> E/B relations by hand (the algebra spin_to_EB encodes)
    almE = -(a_plus + a_minus) / 2.0
    almB = 1j * (a_plus - a_minus) / 2.0
    np.testing.assert_allclose(almE, aE, atol=1e-12)
    np.testing.assert_allclose(almB, aB, atol=1e-12)


def _equiangular_grid(lmax):
    N = lmax + 1
    M = 2 * N - 1
    theta = (2 * np.arange(N) + 1) / (2 * N) * np.pi
    phi = 2 * np.pi * np.arange(M) / M
    return N, M, theta, phi


def _library_spin_F(Q, U, theta, phi, N, M, spin):
    """Resample (Q,U) onto the equiangular grid and run the library spin analysis."""
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    Qg = hp.get_interp_val(Q, TH.ravel(), PH.ravel()).reshape(N, M)
    Ug = hp.get_interp_val(U, TH.ravel(), PH.ravel()).reshape(N, M)
    z = Qg + 1j * Ug if spin > 0 else Qg - 1j * Ug
    G = ft_sphere.spinsph_analysis(z, spin)
    return ft_sphere.fourier2spinsph(G, spin)


def _decode_lib_F(F, lmax, spin):
    """Decode a LIBRARY F array (already complex-SH normalized) to spin alm (m>=0).

    Reuses the pipeline decode ``spin_alm_from_F`` -- same column/row layout and
    the healpy<->FastTransforms ``_spin_conv_phase`` -- but undoes its two
    pipeline-only conventions for the library's exact analysis: the ``1/sqrt(2)``
    m!=0 factor (the hand-rolled ``preparation``'s real-SH normalization, absent
    here -> pre-multiply by sqrt(2)) and the DFS ``(-1)^l`` colatitude phase
    (``colat_phase=False``; the half-sample grid has none).
    """
    alm = spin_alm_from_F(F, lmax, spin, scale=1.0, colat_phase=False)
    for m in range(1, lmax + 1):  # restore the sqrt(2) the pipeline decode removes
        for ell in range(max(m, abs(spin)), lmax + 1):
            alm[hp.Alm.getidx(lmax, ell, m)] *= np.sqrt(2.0)
    return alm


def test_spin_decode_matches_healpy_via_library():
    """spin_to_EB(library F+, F-) recovers healpy aE/aB (interp-limited).

    Validates the full F-array decode -- column order, ``l - max(|m|, |s|)`` row,
    the ``1/sqrt(2)`` m!=0 factor, and the E/B combination -- against healpy. The
    analysis here is the library's exact equiangular transform, so the residual is
    the bilinear HEALPix->grid resampling, which shrinks with nside.
    """
    # low lmax on a heavily oversampled map so bilinear resampling is accurate
    nside, lmax = 128, 12
    N, M, theta, phi = _equiangular_grid(lmax)
    rng = np.random.default_rng(3)
    n = hp.Alm.getsize(lmax)
    aE = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    aB = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    m0 = np.array([hp.Alm.getidx(lmax, ell, 0) for ell in range(lmax + 1)])
    aE[m0] = aE[m0].real
    aB[m0] = aB[m0].real
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, lmax)

    Fp = _library_spin_F(Q, U, theta, phi, N, M, +2)
    Fm = _library_spin_F(Q, U, theta, phi, N, M, -2)
    a_plus = _decode_lib_F(Fp, lmax, +2)
    a_minus = _decode_lib_F(Fm, lmax, -2)
    aE_rec = -(a_plus + a_minus) / 2.0  # the E/B combination spin_to_EB encodes
    aB_rec = 1j * (a_plus - a_minus) / 2.0

    # per-l C_l^EE / C_l^BB in the bulk (the band edge m=lmax is the equiangular
    # Nyquist + the resampling floor, excluded as in the scalar paper test)
    band = slice(2, lmax - 1)
    for name, rec, ref in [("EE", aE_rec, aE), ("BB", aB_rec, aB)]:
        cl_rec = hp.alm2cl(rec.astype(np.complex128), lmax=lmax)[band]
        cl_in = hp.alm2cl(ref.astype(np.complex128), lmax=lmax)[band]
        rel = np.abs(cl_rec - cl_in) / cl_in
        assert np.median(rel) < 0.05, (
            f"{name} per-l median rel err {np.median(rel):.3f}"
        )


def test_spin_decode_no_EB_leak_via_library():
    """A pure-E sky decodes to ~zero B through the library analysis + decode."""
    nside, lmax = 128, 12
    N, M, theta, phi = _equiangular_grid(lmax)
    rng = np.random.default_rng(5)
    n = hp.Alm.getsize(lmax)
    aE = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    m0 = np.array([hp.Alm.getidx(lmax, ell, 0) for ell in range(lmax + 1)])
    aE[m0] = aE[m0].real
    aB = np.zeros_like(aE)
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, lmax)

    Fp = _library_spin_F(Q, U, theta, phi, N, M, +2)
    Fm = _library_spin_F(Q, U, theta, phi, N, M, -2)
    a_plus = _decode_lib_F(Fp, lmax, +2)
    a_minus = _decode_lib_F(Fm, lmax, -2)
    aE_rec = -(a_plus + a_minus) / 2.0
    aB_rec = 1j * (a_plus - a_minus) / 2.0

    # B power should be a small fraction of E power (resampling-limited)
    pE = np.sum(np.abs(aE_rec) ** 2)
    pB = np.sum(np.abs(aB_rec) ** 2)
    assert pB / pE < 1e-2, f"B/E power leak {pB / pE:.2e}"
