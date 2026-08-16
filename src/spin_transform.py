"""Spin-2 (polarization Q/U <-> E/B) transforms.

This wires the spin stages into an end-to-end ``forward_spin`` / ``backward_spin``
and follows SPIN2_PLAN.md. Both directions are native (no resampling) by default;
``backward_spin(..., synthesis="hp2sph")`` reproduces ``hp.alm2map_spin`` to machine
precision. Two routes exist for each direction:

* ``"hp2sph"`` (default): the hand-rolled DFS + latitude nuFFT, no resampling anywhere
  -- the true HP2SPH method. Since the alias fold (see ``_spin_F_hp2sph``) the forward
  beats healpy's WEIGHTED polarization route by 13-30x on the top-band C_l^EE error at
  every nside tested (8 to 128) and converges at 2.3-2.8x per nside doubling; the
  backward is exact to ~3e-13 for any ``lmax <= 2*nside``, save the single
  ``l = m = lmax`` grid-Nyquist corner at ``lmax = 2*nside``
  (``tests/test_alias_fold.py``, ``tests/test_spin_paper_accuracy.py``,
  ``tests/test_spin_backward.py``).

* ``"library"``: resample between HEALPix and the FastTransforms equiangular grid and
  use the library's own ``spinsph_analysis`` / ``spinsph_synthesis``. Exact in the
  grid<->coefficients step, but its accuracy floor is the HEALPix<->equiangular
  RESAMPLING (bilinear ``hp.get_interp_val``), so it is only accurate when the map is
  well oversampled (``nside`` well above ``lmax``). Kept as an independent cross-check
  of the decode, which it validates against healpy in ``tests/test_spin_FSHT.py``.

The ``m != 0`` failure the plan lists as its open gating item is FIXED. It was never
the pole boundary condition the plan hypothesized (a spin field's latitude profile is
``sin^|m+s|(theta/2) cos^|m-s|(theta/2) P(cos theta)``, whose DFS mirror extension is
analytic -- there is no kink). Three separate bugs were responsible:

1. The innermost polar rings have 4, 8, 12 pixels and cannot resolve ``|m| = |s|``,
   which is exactly the mode a spin-s field carries at O(1) INTO the pole. The
   zero-padding asserted that content was zero. Fixed by the alias fold
   (``data_interpolation.ring_fold_plan``), which models what the coarse ring actually
   measured. An earlier fix dropped those entries instead (``ring_mode_mask`` plus an
   LSMR minimum-norm solve); it was 12-36x less accurate and about 2x slower, and has
   been removed.
2. ``double_fourier_sphere.DFS`` read the last original ring instead of the filled
   south pole row (an off-by-one), so half the polynomial pole fill was discarded.
3. ``FSHT.spin_g_to_library`` -- the pipeline's ``x = pi - theta`` reflection has to be
   undone in the bivariate domain, because the reflection FLIPS THE SPIN and so cannot
   be undone by a phase on the output coefficients.

The native backward needed two more:

4. ``FSHT._spin_conv_phase`` only covered ``m >= 0``; the synthesis needs the signed-m
   rule (``+1`` where ``m + spin <= 0``, else ``(-1)^m``). Assuming ``+1`` throughout
   ``m < 0`` flipped the sign of the ``m = -1`` column at ``spin = +2``.
5. ``data_interpolation.transform_grid_to_healpix`` TRUNCATED each polar ring's
   longitude spectrum instead of ALIASING it onto the ring's own pixels -- invisible on
   a round trip (the forward zero-pads, so the folded-in entries are exactly 0) but on
   the synthesis side it discarded the ``|m| = |spin|`` content that is O(1) at the pole.

Ground truth throughout is healpy ``map2alm_spin`` / ``alm2map_spin`` with spin 2.
"""

import numpy as np
import healpy as hp

from .ft_sphere import (
    spinsph_analysis,
    fourier2spinsph,
    spinsph2fourier,
    spinsph_synthesis,
)
from .FSHT import (
    FSHT_spin,
    inverse_FSHT_spin,
    spin_to_EB,
    spin_to_EB_real,
    _spin_F_col,
    _spin_conv_phase,
)
from .data_interpolation import transform_healpix_to_grid, transform_grid_to_healpix
from .double_fourier_sphere import DFS, DFS_inverse, dfs_fold_plan
from .nuFFT import apply_nuFFT, inverse_nuFFT, nyquist_discrepancy

SPIN = 2  # the polarization spin; the native routes run the +SPIN pass only


def _equiangular_grid(lmax):
    """The FastTransforms equiangular grid for a degree-``lmax`` spin field."""
    N = lmax + 1
    M = 2 * N - 1
    theta = (2 * np.arange(N) + 1) / (2 * N) * np.pi
    phi = 2 * np.pi * np.arange(M) / M
    return N, M, theta, phi


def _resample_to_grid(hmap, theta, phi):
    """Bilinearly sample a HEALPix map at the (theta, phi) tensor-product grid."""
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    return hp.get_interp_val(np.asarray(hmap), TH.ravel(), PH.ravel()).reshape(TH.shape)


def _spin_F_library(Q, U, theta, phi, spin):
    """Equiangular (Q,U) resample -> library spin analysis -> spin-SH ``F`` array."""
    Qg = _resample_to_grid(Q, theta, phi)
    Ug = _resample_to_grid(U, theta, phi)
    z = Qg + 1j * Ug if spin > 0 else Qg - 1j * Ug
    return fourier2spinsph(spinsph_analysis(z, spin), spin)


ALIAS_TOL = 1e-2  # relative amplitude below which a polar-ring mode is asserted zero
ALIAS_ETA = 1e-3  # stagnation threshold that stops the latitude CG
ALIAS_RTOL = 1e-7  # fallback stopping rule, in case the residual never stagnates
ALIAS_EPS = 1e-6  # NUFFT tolerance for the latitude solve; see ``_spin_F_hp2sph``
CG_MAXITER = 20000  # a safety cap only


def _spin_F_hp2sph(
    Q,
    U,
    spin,
    alias_tol=ALIAS_TOL,
    rtol=ALIAS_RTOL,
    maxiter=CG_MAXITER,
    eps=ALIAS_EPS,
    level=None,
    theta=None,
    eta=ALIAS_ETA,
    delay=5,
    monitor=None,
):
    """Hand-rolled HP2SPH analysis -> spin-SH ``F`` array (no resampling).

    The latitude fit uses ``dfs_fold_plan``: model each polar ring's ALIAS SUM exactly,
    and keep the "unresolved mode = 0" assertion wherever the spin envelope says the
    mode is negligible on that ring (below ``alias_tol``). That is full rank, so plain
    CG solves it.

    ``alias_tol`` sets the size of the error in the forward operator, so it sets how
    accurate an answer the data can support. Below about 1e-2 it stops mattering: the
    alias assertion is no longer the dominant model error, and tightening it only adds
    iterations.

    ``eta`` stops the iteration. The relaxed alias families make this system ill-posed,
    so CG semiconverges -- the physical answer improves, bottoms out, and then decays as
    the iteration resolves directions the data do not determine. The stop is Morozov's
    discrepancy principle with the level read off the iteration itself: the weighted
    data residual falls to the model error and stagnates there, so the run is halted
    once the squared data residual has dropped by less than a fraction ``eta`` over the
    preceding ``delay`` iterations. It is dimensionless and needs no per-resolution
    tuning. Raise it to stop sooner, lower it to converge further.

    ``theta`` selects a different stop: Morozov's discrepancy principle with the level
    computed in advance rather than read off the iteration. ``nyquist_discrepancy``
    estimates the one modelling error that dominates here -- the longitude grid cannot
    represent its own Nyquist mode -- and the run halts once the data residual reaches
    ``theta`` times it. Values a little above 1 are the useful range. It stops in about
    half the iterations ``eta`` needs, at a few percent more error, and it is only
    meaningful when the field carries power at ``|m| = 2*nside``; below that the level
    is unreachable and ``eta`` takes over. Leave it ``None`` to use ``eta`` alone.

    ``rtol`` is only a fallback for the case where the residual never stagnates, and
    ``maxiter`` a safety cap above that.
    """
    z = Q + 1j * U if spin > 0 else Q - 1j * U
    nside = hp.npix2nside(np.asarray(z).shape[0])
    upsampled, fft_coeff = transform_healpix_to_grid(z)
    # Only the mirror-fundamental rows are built: the latitude solve restricts to them
    # anyway, and the mirrored half is the largest single item in the transform's peak
    # memory.
    _, dfs = DFS(upsampled, fft_coeff, spin=spin, half=True)
    target, phase, keep = dfs_fold_plan(nside, spin, alias_tol, half=True)
    if theta is not None:
        level = theta * nyquist_discrepancy(dfs)
    fft_lat = apply_nuFFT(
        dfs,
        solver="cg",
        sample_mask=keep,
        fold=(target, phase),
        rtol=rtol,
        maxiter=maxiter,
        eps=eps,
        spin=spin,
        half_domain=True,
        level=level,
        eta=eta,
        delay=delay,
        monitor=monitor,
    )
    return FSHT_spin(fft_lat, spin)


def forward_spin(Q, U, lmax, analysis="hp2sph", **kw):
    """HEALPix ``(Q, U)`` polarization map -> healpy-ordered ``(aE, aB)``.

    ``analysis`` selects the grid->coefficients route (see the module docstring):
    ``"hp2sph"`` (default, the hand-rolled DFS+nuFFT, no resampling) or ``"library"``
    (resample + the library's exact analysis, resampling-limited). Extra keywords go to
    ``_spin_F_hp2sph`` (``alias_tol``, ``rtol``).

    ``"hp2sph"`` runs the ``+2`` pass only: Q and U are real, so the ``-2``
    coefficients follow from ``-2a_{l,m} = (-1)^m conj(+2a_{l,-m})`` and are read out
    of the same ``F`` array (``FSHT.spin_to_EB_real``). ``"library"`` keeps both passes
    as an independent cross-check of the decode.
    """
    Q = np.asarray(Q)
    U = np.asarray(U)
    if analysis == "library":
        if kw:
            raise TypeError(f"unexpected keywords for analysis='library': {sorted(kw)}")
        _, _, theta, phi = _equiangular_grid(lmax)
        Fp = _spin_F_library(Q, U, theta, phi, +SPIN)
        Fm = _spin_F_library(Q, U, theta, phi, -SPIN)
        # library F is complex-SH normalized and the half-sample grid has no DFS
        # colatitude phase: scale 1, no real-SH sqrt(2), no (-1)^l.
        return spin_to_EB(
            Fp, Fm, lmax, scale=1.0, colat_phase=False, real_sh_norm=False
        )
    elif analysis == "hp2sph":
        if np.iscomplexobj(Q) or np.iscomplexobj(U):
            raise ValueError(
                "analysis='hp2sph' needs real Q and U (Stokes parameters); the "
                "single-pass readout is derived from that reality"
            )
        Fp = _spin_F_hp2sph(Q, U, +SPIN, **kw)
        # FSHT_spin already converted the pipeline conventions away (FSHT.
        # spin_g_to_library), so this is the same decode the library route uses.
        return spin_to_EB_real(
            Fp, lmax, scale=1.0, colat_phase=False, real_sh_norm=False
        )
    raise ValueError(f"unknown analysis {analysis!r}; use 'library' or 'hp2sph'")


def _F_phase(m, spin):
    """Phase relating an F-array cell to the healpy spin coefficient: F = phase * s_a.

    ``_spin_conv_phase`` is measured for every SIGNED ``m`` (see its docstring), so
    this is just an alias kept for readability at the call site.
    """
    return _spin_conv_phase(m, spin)


def _build_spin_F(a_signed, lmax, N, M, spin):
    """Place healpy-ordered spin coefficients (all signed m) into an F array."""
    F = np.zeros((N, M), dtype=complex)
    s0 = abs(spin)
    for m in range(-lmax, lmax + 1):
        col = _spin_F_col(m)
        ph = _F_phase(m, spin)
        for ell in range(max(abs(m), s0), lmax + 1):
            F[ell - max(abs(m), s0), col] = ph * a_signed(ell, m)
    return F


def _signed_spin_alm(aE, aB, lmax, spin):
    """``(ell, m) -> s_a_{l,m}`` for all signed ``m``, from healpy-ordered E/B alm.

    E and B are REAL parity fields, so healpy stores only ``m >= 0``; the negative
    orders follow from ``a_{l,-m} = (-1)^m conj(a_{l,m})``. The spin coefficients are
    ``+2a = -(aE + i aB)`` and ``-2a = -(aE - i aB)``.
    """
    sgn = 1.0 if spin > 0 else -1.0

    def healpy_alm(arr, ell, m):
        if m >= 0:
            return arr[hp.Alm.getidx(lmax, ell, m)]
        return ((-1.0) ** m) * np.conj(arr[hp.Alm.getidx(lmax, ell, -m)])

    def a_signed(ell, m):
        return -(healpy_alm(aE, ell, m) + sgn * 1j * healpy_alm(aB, ell, m))

    return a_signed


def _backward_spin_library(aE, aB, nside, lmax):
    """Equiangular library synthesis + bilinear resample to HEALPix (see ``backward_spin``)."""
    N, M, theta, phi = _equiangular_grid(lmax)
    Fp = _build_spin_F(_signed_spin_alm(aE, aB, lmax, +SPIN), lmax, N, M, +SPIN)
    zp = spinsph_synthesis(spinsph2fourier(Fp, +SPIN), +SPIN)  # grid Q + iU

    npix = hp.nside2npix(nside)
    th, ph = hp.pix2ang(nside, np.arange(npix))
    Q = _grid_interp(zp.real, theta, phi, th, ph)
    U = _grid_interp(zp.imag, theta, phi, th, ph)
    return Q, U


def _backward_spin_hp2sph(aE, aB, nside, lmax):
    """The native HP2SPH spin synthesis -- the exact mirror of ``_spin_F_hp2sph``.

    ``inverse_FSHT_spin`` -> ``inverse_nuFFT`` -> ``DFS_inverse`` ->
    ``transform_grid_to_healpix``, i.e. the spin counterpart of the scalar
    ``main.backward``. Nothing is resampled, so unlike the library route this is not
    interpolation-limited: it reproduces ``hp.alm2map_spin`` to machine precision.

    Only the ``spin = +2`` pass is run. ``z = Q + iU`` already carries the whole real
    ``(Q, U)`` pair, and the ``-2`` coefficients are not independent (they are fixed by
    the reality of Q and U, which ``_signed_spin_alm`` already uses), so a second pass
    would recompute the same map.
    """
    L = 2 * nside  # the pipeline's internal (compact) latitude band = lmax
    a_plus = _signed_spin_alm(aE, aB, lmax, +SPIN)
    # Build F straight at the pipeline band: the F row (l - max(|m|, |s|)) and column
    # index are both independent of L, so a low-lmax input is simply zero-padded.
    F = _build_spin_F(a_plus, lmax, L + 1, 2 * L + 1, +SPIN)

    _, bivar = inverse_FSHT_spin(F, nside, +SPIN)
    fft_lat = inverse_nuFFT(bivar)
    fft_coeff = DFS_inverse(fft_lat, spin=+SPIN)
    z = transform_grid_to_healpix(fft_coeff, fft_coeff, real_output=False)
    return np.real(z), np.imag(z)


def backward_spin(aE, aB, nside, lmax=None, synthesis="hp2sph"):
    """healpy-ordered ``(aE, aB)`` -> HEALPix ``(Q, U)`` map.

    ``synthesis`` selects the coefficients->map route, mirroring ``forward_spin``'s
    ``analysis``:

    * ``"hp2sph"`` (default, the true method): the native inverse pipeline, no
      resampling. EXACT -- it reproduces ``hp.alm2map_spin`` to ~3e-13 for every band
      ``lmax <= 2*nside - 1``, at every nside tested (8-64).
      At ``lmax = 2*nside`` exactly, the single ``l = m = lmax`` coefficient is the
      one mode that cannot be synthesised: ``m = +2*nside`` and ``m = -2*nside`` are
      the same mode on the ``4*nside``-point longitude grid, and the per-ring ``phi0``
      offsets give them different phases, so no single Nyquist column can carry both.
      This is the synthesis face of the forward's documented half-gain at that corner.
      Everything else in that band stays exact; drop to ``lmax <= 2*nside - 1`` if the
      corner coefficient matters.
    * ``"library"``: synthesize on the FastTransforms equiangular grid and bilinearly
      resample to HEALPix. Resampling-limited, like ``analysis="library"``; kept as an
      independent cross-check.

    ``lmax`` defaults to the band of ``aE``.
    """
    aE = np.asarray(aE)
    aB = np.asarray(aB)
    if lmax is None:
        lmax = hp.Alm.getlmax(len(aE))
    if synthesis == "library":
        return _backward_spin_library(aE, aB, nside, lmax)
    if synthesis == "hp2sph":
        if lmax > 2 * nside:
            raise ValueError(
                f"lmax={lmax} exceeds the grid band 2*nside={2 * nside}; the "
                "HP2SPH synthesis grid cannot represent it (raise nside or lower lmax)"
            )
        return _backward_spin_hp2sph(aE, aB, nside, lmax)
    raise ValueError(f"unknown synthesis {synthesis!r}; use 'hp2sph' or 'library'")


def _grid_interp(grid, theta, phi, th, ph):
    """Bilinear interpolation of a tensor-product (theta, phi) grid at points (th, ph)."""
    Nt, Np = grid.shape
    # nearest-lower indices with linear weights (phi periodic, theta clamped)
    ti = np.clip(np.searchsorted(theta, th) - 1, 0, Nt - 2)
    t0, t1 = theta[ti], theta[ti + 1]
    wt = (th - t0) / (t1 - t0)
    pp = (ph % (2 * np.pi)) / (2 * np.pi) * Np
    pi0 = np.floor(pp).astype(int) % Np
    pi1 = (pi0 + 1) % Np
    wp = pp - np.floor(pp)
    g00 = grid[ti, pi0]
    g01 = grid[ti, pi1]
    g10 = grid[ti + 1, pi0]
    g11 = grid[ti + 1, pi1]
    return (1 - wt) * ((1 - wp) * g00 + wp * g01) + wt * ((1 - wp) * g10 + wp * g11)
