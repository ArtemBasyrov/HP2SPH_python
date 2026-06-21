"""End-to-end pipeline tests (all require Julia / FastTransforms.jl)."""

import numpy as np
import healpy as hp
import pytest

from tests.pipeline_helpers import (
    forward_C,
    forward_alm,
    backward_map,
    calibrate_scale,
)


def _sub_band(alm, lmax, cut=1):
    """Mask selecting coefficients with l <= lmax - cut.

    The top band l = lmax = 2*nside is the grid's longitude Nyquist edge (m up to
    2*nside has only one stored coefficient), so no transform on this grid can
    determine it accurately. Correctness is therefore asserted below that edge.
    """
    ls, _ = hp.Alm.getlm(lmax, np.arange(len(alm)))
    return ls <= lmax - cut


@pytest.mark.julia
def test_forward_backward_map_roundtrip_exact_mode(nside, healpix_map, relerr):
    """SQUARE-interpolation mode round-trips the map to near machine precision.

    With ``solve_modes = 8*nside+1`` (one mode per DFS sample) + the dense SVD
    solver, the latitude system interpolates every sample, so map -> C -> map is
    bit-exact (independent of the absolute alm normalization). This is the
    invertibility-first regime; it is only well-conditioned up to nside ~64.
    """
    kw = dict(solver="svd", solve_modes=8 * nside + 1)
    C = forward_C(healpix_map, **kw)
    recovered = backward_map(C, nside)
    assert relerr(recovered, healpix_map) < 1e-5


@pytest.mark.julia
def test_forward_backward_map_roundtrip_default(nside, healpix_map, relerr):
    """The default (well-conditioned, scalable) path round-trips to a few percent.

    The default ``solve_modes = 4*nside+1`` band is a projection -- it drops the
    above-band polar-aliasing content -- so the round trip is accurate but not
    bit-exact. The residual is large at very coarse nside (~10% at nside=4, lmax=8,
    where the dropped band is a big fraction) and shrinks quickly with nside
    (~2e-2 by nside=16, ~5e-4 by nside=512 -- see the high-nside conditioning test).
    """
    C = forward_C(healpix_map)
    recovered = backward_map(C, nside)
    assert relerr(recovered, healpix_map) < 1.5e-1


@pytest.mark.julia
def test_forward_alm_matches_input(nside, lmax, healpix_map, random_alm, relerr):
    """Forward alm must recover the alm that synthesised the map (below Nyquist).

    ``healpix_map`` was built by ``hp.alm2map(random_alm)``, so a correct forward
    transform must return ``random_alm`` (up to the per-nside global ``scale``).

    The three convention fixes -- per-ring longitude referencing in
    ``data_interpolation``, the T_0-row factor in ``preparation``, and
    ``mono_factor=1`` in ``to_healpy_alm`` -- bring the diagonal gains to 1 with
    no monopole leakage or longitude phase. The remaining error is the genuine
    latitude quadrature accuracy and it is concentrated entirely in the top band
    l = lmax (the Nyquist edge, see ``_sub_band``); below it the agreement is
    ~1e-2 and improves with nside (1.9% @ ns4, 0.9% @ ns8, 0.7% @ ns16).
    """
    scale = calibrate_scale(nside, lmax)
    alm = forward_alm(healpix_map, lmax=lmax, scale=scale)
    sel = _sub_band(alm, lmax)
    err = relerr(alm[sel], random_alm[sel])
    # Default (well-conditioned 4*nside+1) band: ~3e-2 at the coarse nside=4
    # (lmax=8), improving to ~1e-2 by nside=16 (see the convergence test).
    assert err < 3.5e-2, f"forward alm rel error {err:.4f} (nside={nside}, l<=lmax-1)"


@pytest.mark.julia
def test_forward_alm_matches_healpy(nside, lmax, healpix_map, relerr):
    """Forward alm must agree with hp.map2alm below the Nyquist band.

    Compares against healpy's own analysis of the same map (same lmax), the
    achievable reference on this grid. Agreement is ~1e-2 below l = lmax and
    improves with nside.
    """
    scale = calibrate_scale(nside, lmax)
    alm = forward_alm(healpix_map, lmax=lmax, scale=scale)
    hp_alm = hp.map2alm(healpix_map, lmax=lmax, use_weights=True)
    sel = _sub_band(alm, lmax)
    err = relerr(alm[sel], hp_alm[sel])
    assert err < 3.5e-2, f"forward alm vs map2alm rel error {err:.4f} (nside={nside})"


@pytest.mark.julia
def test_forward_alm_converges_with_nside(relerr):
    """Sub-band forward error must shrink as nside grows (genuine convergence).

    A systematic convention bug would leave a constant floor; a quadrature-limited
    transform converges. This guards against regressions that reintroduce a
    constant error.
    """
    errs = []
    rng = np.random.default_rng(20260620)
    for nside in (4, 16):
        lmax = 2 * nside
        ncoeff = hp.Alm.getsize(lmax)
        alm_in = rng.standard_normal(ncoeff) + 1j * rng.standard_normal(ncoeff)
        m0 = np.array([hp.Alm.getidx(lmax, ell, 0) for ell in range(lmax + 1)])
        alm_in[m0] = alm_in[m0].real
        mp = hp.alm2map(alm_in, nside=nside, lmax=lmax)
        scale = calibrate_scale(nside, lmax)
        alm = forward_alm(mp, lmax=lmax, scale=scale)
        sel = _sub_band(alm, lmax)
        errs.append(relerr(alm[sel], alm_in[sel]))
    assert errs[1] < errs[0], f"no convergence: ns4={errs[0]:.4f} ns16={errs[1]:.4f}"
