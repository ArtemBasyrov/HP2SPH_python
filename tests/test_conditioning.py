"""The latitude-solve conditioning wall, and the dense-SVD solver that scales it.

The HP2SPH pipeline's only ill-conditioned stage is the latitude nuFFT: fitting
Fourier modes to the clustered HEALPix colatitudes (after the DFS doubling). Two
facts, established by dense linear algebra, drive the accuracy/invertibility story
and are pinned here so they don't silently regress:

1. The FSHT ``preparation`` is calibrated for the latitude band L = 4*nside, i.e.
   N = 8*nside+1 latitude modes -- the SQUARE interpolation (one mode per sample).
   That square Vandermonde at the clustered latitudes is severely ill-conditioned
   and gets worse with nside (cond ~ 2e2 @ ns16 -> 8e10 @ ns64 -> ~1/eps @ ns128),
   while the *truncated* band |k| <= 2*nside is perfectly conditioned (cond ~ 1.15).
   We are forced onto the square (ill-conditioned) count by the FSHT, NOT by the
   latitude content -- so the real fix for nside >= 128 is to rework ``preparation``
   at L = lmax = 2*nside.

2. Because of (1), HOW you solve the square system matters. CG on the normal
   equations works at the SQUARED condition number and finufft's transform error is
   amplified by the condition number, so the finufft+CG path floors the round trip
   at ~1e-2 by nside=64. A dense truncated-SVD solve on the exact (cached, shared)
   Vandermonde avoids both and restores near machine-precision invertibility through
   nside=64 -- at the same forward accuracy. This is the ``solver="svd"`` default.

These tests use only numpy/healpy for the conditioning facts (fast, no FSHT), and
the in-process libfasttransforms backend for the end-to-end invertibility.
"""

import numpy as np
import healpy as hp
import pytest

from src.nuFFT import _upsampled_latitudes


def _vandermonde_cond(nside, half_band):
    """Condition number of the latitude Vandermonde with modes |k| <= half_band."""
    x = _upsampled_latitudes(nside)
    k = np.arange(-half_band, half_band + 1)
    A = np.exp(1j * np.outer(x, k))
    s = np.linalg.svd(A, compute_uv=False)
    return s[0] / s[-1]


@pytest.mark.parametrize("nside", [16, 32, 64])
def test_truncated_band_is_well_conditioned(nside):
    """The latitude band |k| <= 2*nside is perfectly conditioned at every nside.

    This is the band a band-limited (lmax = 2*nside) signal lives in; the clustered
    grid resolves it cleanly. The conditioning problem is entirely about the modes
    ABOVE it that the square (FSHT-required) count reaches for.
    """
    assert _vandermonde_cond(nside, 2 * nside) < 2.0


def test_square_band_conditioning_blows_up_with_nside():
    """The square band |k| <= 4*nside (what the FSHT forces) is ill-conditioned and
    worsens with nside -- the root cause of the high-nside invertibility wall."""
    c16 = _vandermonde_cond(16, 4 * 16)
    c32 = _vandermonde_cond(32, 4 * 32)
    c64 = _vandermonde_cond(64, 4 * 64)
    assert c16 < c32 < c64  # monotonic growth
    assert c16 < 1e4  # still benign at ns16
    assert c64 > 1e8  # already catastrophic by ns64 (CG/finufft floor ~ eps*cond)


# --------------------------------------------------------------------------- #
# End-to-end: the two regimes                                                  #
# --------------------------------------------------------------------------- #
def _roundtrip(nside, **nufft_kw):
    from src.data_interpolation import transform_healpix_to_grid
    from src.double_fourier_sphere import DFS
    from src.nuFFT import apply_nuFFT
    from src.FSHT import FSHT
    from tests.pipeline_helpers import backward_map

    lmax = 2 * nside
    rng = np.random.default_rng(20260620)
    ncoeff = hp.Alm.getsize(lmax)
    alm = rng.standard_normal(ncoeff) + 1j * rng.standard_normal(ncoeff)
    m0 = np.array([hp.Alm.getidx(lmax, ell, 0) for ell in range(lmax + 1)])
    alm[m0] = alm[m0].real
    mp = hp.alm2map(alm.astype(np.complex128), nside=nside, lmax=lmax)

    ups, fc = transform_healpix_to_grid(mp)
    _, dfs = DFS(ups, fc)
    C = FSHT(apply_nuFFT(np.asarray(dfs), **nufft_kw))
    rec = backward_map(C, nside)
    return np.linalg.norm(rec - mp) / np.linalg.norm(mp)


@pytest.mark.ft
def test_square_svd_invertible_at_nside64_where_cg_floors():
    """SQUARE band: dense-SVD reaches ~1e-5 invertibility where CG-on-normal floors.

    With the (ill-conditioned) square ``solve_modes=8*nside+1`` band, solving via an
    exact SVD instead of finufft+CG-on-normal-equations recovers near
    machine-precision invertibility at nside=64, where CG floors at
    ~finufft_eps*cond.
    """
    sq = dict(solve_modes=8 * 64 + 1)
    rt_svd = _roundtrip(64, solver="svd", **sq)
    rt_cg = _roundtrip(64, solver="cg", **sq)
    assert rt_svd < 1e-4, f"SVD round trip {rt_svd:.2e} not invertible at ns64"
    assert rt_cg > 1e-3, (
        f"CG round trip {rt_cg:.2e} unexpectedly good (regime changed?)"
    )
    assert rt_svd < rt_cg / 100  # SVD is orders of magnitude better


@pytest.mark.ft
@pytest.mark.parametrize("nside", [8, 16])
def test_square_svd_roundtrip_is_machine_precision(nside):
    """SQUARE band + SVD round trips to ~machine precision in the low-nside range."""
    assert _roundtrip(nside, solver="svd", solve_modes=8 * nside + 1) < 1e-10


@pytest.mark.ft
@pytest.mark.parametrize("nside", [64, 128, 256])
def test_scalable_default_path_works_at_high_nside(nside):
    """The default well-conditioned band scales past where the square band walls.

    nside=128 in particular was broken (~20%) by a float-parity bug in
    ``FSHT.preparation`` (mode indices misclassified even<->odd at L=512); with that
    fixed and the well-conditioned 4*nside+1 solve, the round trip is ~1e-2 and
    converging at all powers of two -- the scalable regime for nside 256/512.
    """
    assert _roundtrip(nside) < 2e-2  # default solver="cg", solve_modes=4*nside+1
