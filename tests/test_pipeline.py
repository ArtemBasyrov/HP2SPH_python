"""End-to-end pipeline tests (all require the libfasttransforms C library)."""

import numpy as np
import healpy as hp
import pytest

# The pipeline helpers load the C library on import; skip cleanly if it is missing.
pytest.importorskip("src.ft_sphere")

from src.FSHT import from_healpy_alm  # noqa: E402
from tests.pipeline_helpers import (  # noqa: E402
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


@pytest.mark.ft
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


@pytest.mark.ft
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


@pytest.mark.ft
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


@pytest.mark.ft
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


@pytest.mark.ft
def test_backward_from_healpy_alm_matches_healpy(nside, lmax, random_alm, relerr):
    """``from_healpy_alm`` + the inverse pipeline reproduces ``hp.alm2map`` exactly.

    The scalar synthesis is EXACT, not merely convergent, in healpy's own
    coefficient convention: measured 2.8e-13 / 2.9e-13 / 3.4e-13 at nside
    8 / 16 / 32, matching what the native spin backward reaches. That is expected
    -- a degree-l harmonic is a trigonometric polynomial of degree l in theta, so
    the compact latitude band represents it with no truncation.

    ``l = m = lmax`` is excluded. With ``lmax = 2*nside`` on a ``4*nside``-point
    longitude grid, ``m = +lmax`` and ``m = -lmax`` are the same mode and the
    per-ring ``phi0`` offsets give them different phases, so no single column can
    carry both. That corner is a property of the HEALPix grid, and the benchmark
    suite measures it separately (HP2SPH returns gain 0.5000 there at every nside;
    healpy and ducc0 return 1.0).
    """
    alm = random_alm.copy()
    alm[hp.Alm.getidx(lmax, lmax, lmax)] = 0.0  # the grid-Nyquist corner

    C = from_healpy_alm(alm, lmax=lmax, L=2 * nside)
    recovered = backward_map(C, nside)
    reference = hp.alm2map(alm, nside=nside, lmax=lmax)
    err = relerr(recovered, reference)
    assert err < 1e-11, f"backward vs hp.alm2map rel error {err:.3e} (nside={nside})"


@pytest.mark.ft
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


# --------------------------------------------------------------------------- #
# main.py's public entry points                                                #
# --------------------------------------------------------------------------- #
# ``main.forward`` used to demand a 3-row (I, Q, U) stack and transform only I,
# which is why the benchmarks bypassed it and imported the composition out of
# ``tests/`` instead. It now takes a single map and delegates to ``src.pipeline``.
@pytest.mark.ft
def test_main_forward_is_the_pipeline_composition(nside, healpix_map, relerr):
    import main

    assert relerr(main.forward(healpix_map), forward_C(healpix_map)) == 0.0


@pytest.mark.ft
def test_main_roundtrip_matches_the_pipeline(nside, healpix_map, relerr):
    import main

    C = main.forward(healpix_map)
    assert relerr(main.backward(C), backward_map(C, nside)) == 0.0


def test_main_forward_rejects_an_iqu_stack():
    """The old signature silently ignored Q and U; the new one must not accept it."""
    import main

    with pytest.raises(ValueError, match="single HEALPix map"):
        main.forward(np.zeros((3, 12 * 4**2)))
