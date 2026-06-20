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


@pytest.mark.julia
def test_forward_backward_map_roundtrip(nside, healpix_map, relerr):
    """map -> C -> map is self-consistent to near machine precision.

    This checks invertibility of the whole chain; it says nothing about the
    absolute alm normalization (see the forward-vs-healpy test for that).
    """
    C = forward_C(healpix_map)
    recovered = backward_map(C, nside)
    assert relerr(recovered, healpix_map) < 1e-5


@pytest.mark.julia
def test_forward_alm_matches_input(nside, lmax, healpix_map, random_alm, relerr):
    """Forward alm must recover the alm that synthesised the map.

    ``healpix_map`` was built by ``hp.alm2map(random_alm)``, so a correct forward
    transform must return ``random_alm`` (up to the per-nside global ``scale``).

    After the polar-ring normalization fix in ``data_interpolation`` the error
    dropped from ~0.59 to ~0.37 and the latitude path is now exact. The residual
    ~0.37 is CONSTANT with nside (systematic, not a quadrature-accuracy issue):
    it is the FastTransforms<->healpy convention conversion -- monopole leakage
    (a pure Y_l0 leaks into a_00 via the m=0 / sqrt(2) factors in ``preparation``
    /``to_healpy_alm``) plus an m-dependent longitude phase for m>0. Tight
    tolerance on purpose: this red bar is the target of the next (convention) fix.
    """
    scale = calibrate_scale(nside, lmax)
    alm = forward_alm(healpix_map, lmax=lmax, scale=scale)
    err = relerr(alm, random_alm)
    assert err < 1e-2, f"forward alm rel error {err:.4f} (nside={nside})"


@pytest.mark.julia
def test_forward_alm_matches_healpy(nside, lmax, healpix_map, relerr):
    """Forward alm must agree with hp.map2alm at the same band limit.

    Compares against healpy's own analysis of the same map (same lmax), which is
    the achievable reference on this grid. Currently ~0.37 (constant with nside);
    the gap is the FastTransforms<->healpy convention conversion, not quadrature
    (see test_forward_alm_matches_input).
    """
    scale = calibrate_scale(nside, lmax)
    alm = forward_alm(healpix_map, lmax=lmax, scale=scale)
    hp_alm = hp.map2alm(healpix_map, lmax=lmax, use_weights=True)
    err = relerr(alm, hp_alm)
    assert err < 1e-2, f"forward alm vs map2alm rel error {err:.4f} (nside={nside})"
