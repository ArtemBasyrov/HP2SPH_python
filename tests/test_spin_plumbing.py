"""Phase 2: complex-data plumbing through stages 1 (interp) and 3 (nuFFT).

A spin-2 polarization field is carried as a single complex map ``Q + i*U``. The
interpolation stage must keep that complex content (it previously took ``.real``
of every ring), and the latitude nuFFT must round-trip a complex spectrum. The
real intensity (I) path must stay bit-identical.

See SPIN2_PLAN.md (Phase 2).
"""

import numpy as np
import healpy as hp
import pytest

from src.data_interpolation import (
    transform_healpix_to_grid,
    transform_grid_to_healpix,
)
from src.nuFFT import apply_nuFFT, inverse_nuFFT


def _real_map(nside, lmax, seed):
    rng = np.random.default_rng(seed)
    ncoeff = hp.Alm.getsize(lmax)
    alm = rng.standard_normal(ncoeff) + 1j * rng.standard_normal(ncoeff)
    m0 = np.array([hp.Alm.getidx(lmax, ell, 0) for ell in range(lmax + 1)])
    alm[m0] = alm[m0].real
    return hp.alm2map(alm.astype(np.complex128), nside=nside, lmax=lmax)


def test_interp_real_unchanged(nside, healpix_map, relerr):
    """The real intensity (I) path is unchanged.

    A real input still yields a REAL-dtype upsampled grid (the ``.real`` is only
    deferred, never dropped, for real inputs), the Fourier coefficients are
    identical whether the values are passed as real or complex (the FFT carries
    the same content either way), and the real-path inverse still round-trips the
    map exactly. NB: the upsampled grid's imaginary part is NOT zero when carried
    as complex -- the phi=0 phase referencing puts a genuine imaginary component
    on the Nyquist longitude mode, which ``.real`` discards for the I path; that
    is why the spin path keeps the full complex grid (see Phase 3).
    """
    up_real, fc_real = transform_healpix_to_grid(healpix_map)
    assert not np.iscomplexobj(np.asarray(up_real))

    _, fc_cplx = transform_healpix_to_grid(healpix_map.astype(np.complex128))
    np.testing.assert_allclose(np.asarray(fc_cplx), np.asarray(fc_real), atol=1e-14)

    recovered = transform_grid_to_healpix(fc_real, fc_real)  # real_output=True default
    assert not np.iscomplexobj(recovered)
    assert relerr(recovered, healpix_map) < 1e-10


def test_interp_complex_roundtrip(nside, lmax, relerr):
    """interp(z) -> de-interp(z) recovers a complex z = mapQ + i*mapU."""
    mapQ = _real_map(nside, lmax, 11)
    mapU = _real_map(nside, lmax, 22)
    z = mapQ + 1j * mapU

    _, fft_coeff = transform_healpix_to_grid(z)
    recovered = transform_grid_to_healpix(fft_coeff, fft_coeff, real_output=False)
    assert np.iscomplexobj(recovered)
    assert relerr(recovered, z) < 1e-10


def test_nuFFT_complex_roundtrip(nside, relerr):
    """The latitude analysis/synthesis is a band-limited projector on complex data.

    ``inverse_nuFFT . apply_nuFFT`` projects the DFS latitude samples onto the
    band the solve fits; a projector is idempotent, so a second pass reproduces
    the first to the CG floor. A plain FFT does not apply at the non-uniform
    HEALPix latitudes, which is why this stage exists; here we only check it
    carries genuinely complex (Q + iU) content, not just real I.
    """
    n_samples = 8 * nside
    n_lon = 4 * nside
    rng = np.random.default_rng(55)
    s = rng.standard_normal((n_samples, n_lon)) + 1j * rng.standard_normal(
        (n_samples, n_lon)
    )

    proj = inverse_nuFFT(apply_nuFFT(s))  # P(s)
    proj2 = inverse_nuFFT(apply_nuFFT(proj))  # P(P(s))
    assert np.iscomplexobj(proj)
    # the projector must actually act (not the identity) and be idempotent
    assert relerr(proj, s) > 1e-3
    assert relerr(proj2, proj) < 1e-6
