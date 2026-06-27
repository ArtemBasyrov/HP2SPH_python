import numpy as np
import jax.numpy as jnp

# The FSHT stage runs in-process through the libfasttransforms C library (see
# src/ft_sphere.py for how the library is located -- no env var needed when it is
# installed normally). If it cannot be loaded, the import below raises ImportError
# with a build/install hint; there is no other backend.
from .ft_sphere import fourier2sph as _ft_fourier2sph
from .ft_sphere import sph2fourier as _ft_sph2fourier


def preparation(bivar_coeffs: jnp.array, spin: int = 0) -> jnp.array:
    # bivar_coeffs: (2*L+1 latitude modes [centered], 4*NSIDE longitude [natural
    # centered order m = -2*NSIDE .. 2*NSIDE-1]). The internal latitude band
    # limit L is set by the number of latitude modes the nuFFT solved for, which
    # is decoupled from (and larger than) the longitude resolution. Longitude
    # only supports |m| <= 2*NSIDE, so the longitude axis is zero-padded out to
    # the 2*L+1 columns the Fourier->spherical-harmonic step expects.
    NSIDE = bivar_coeffs.shape[1] // 4
    L = (bivar_coeffs.shape[0] - 1) // 2  # internal latitude band limit

    # expand longitude to 4*NSIDE+1 (split the m = -2*NSIDE column across +-)
    X_small = np.zeros((2 * L + 1, 4 * NSIDE + 1), dtype=complex)
    neg_column = bivar_coeffs[:, 0]  # m = -2*NSIDE (index 0 in natural ordering)
    X_small[:, : 4 * NSIDE] = bivar_coeffs
    X_small[:, 0] = 0.5 * neg_column
    X_small[:, -1] = 0.5 * neg_column

    # embed the populated m = -2*NSIDE .. 2*NSIDE block centered inside the full
    # m = -L .. L grid (everything with |m| > 2*NSIDE stays zero)
    X_coeff = np.zeros((2 * L + 1, 2 * L + 1), dtype=complex)
    X_coeff[:, L - 2 * NSIDE : L + 2 * NSIDE + 1] = X_small

    # transform X into g array, size (L+1, 2*L+1)
    g = np.zeros((L + 1, 2 * L + 1), dtype=complex)

    # rearange X into [0 ,-1, 1, -2, 2, ...] order along k. These are integer mode
    # numbers, but fftfreq*(2L+1) returns floats with rounding error that grows with
    # L; rint to exact ints so the `% 2` parity test below cannot misclassify a mode.
    # (Without this, ~20-80 modes flip even<->odd at L=512/2048 -> nside 128/512 lost
    # ~20% of the transform, while nside 64/256 happened to round cleanly.)
    indx = np.rint(np.fft.fftfreq(2 * L + 1, d=1) * (2 * L + 1)).astype(int)
    indx = np.fft.fftshift(indx)
    sel = np.argsort(np.abs(indx), kind="stable")
    indx = indx[sel]
    X_sort = X_coeff[:, sel]

    X_pos_ell = X_sort[L:]  # including 0 and positive ell = [0, 1, 2, ..., L]
    X_neg_ell = X_sort[:L]  # negative ell = [-L, ..., -2, -1]
    X_neg_ell = np.flip(X_neg_ell, axis=0)  # [-1, -2, ..., -L]

    # create sel for odd and even longitude modes. The bivariate Fourier basis is
    # cos(l*theta) when (m + spin) is even and sin((l+1)*theta) when (m + spin) is
    # odd (FastTransforms spinsph2fourier convention; the scalar spin=0 case is the
    # plain "m even -> cosine, m odd -> sine"). For spin = +-2 this parity is the
    # same columns as scalar, but writing it as (m + spin) keeps it correct for any
    # spin and documents the dependence.
    sel_even = (indx[1:] + spin) % 2 == 0
    sel_odd = ~sel_even

    # first row j = 0 (the Chebyshev T_0 / latitude-DC term). For an even
    # cosine series f = sum_k c_k e^{ikθ} (c_{-k}=c_k) the Chebyshev coeffs are
    # g_0 = c_0 and g_k = 2 c_k for k>0 -- i.e. the DC row carries NO factor 2.
    # The previous factor 2 here over-weighted the latitude DC, leaking even-l
    # zonal power into the monopole and inflating even-m gains. (The odd-m cells
    # below are a sine series and are handled separately.)
    g[0, 1:] = X_pos_ell[0, 1:] * np.sqrt(1.0 / np.pi)
    g[0, 1:][sel_odd] = (
        1j * (X_pos_ell[1, 1:] - X_neg_ell[0, 1:])[sel_odd] * np.sqrt(1.0 / np.pi)
    )

    # first column k = m = 0 ; T_0 row again -> no factor 2 (see above)
    g[0, 0] = X_pos_ell[0, 0] * np.sqrt(0.5 / np.pi)
    g[1:, 0] = (X_pos_ell[1:, 0] + X_neg_ell[:, 0]) * np.sqrt(0.5 / np.pi)

    # everyhting inside the matrix except the zero row and column
    g_k_even = (X_pos_ell[1:, 1:] + X_neg_ell[:, 1:]) * np.sqrt(1.0 / np.pi)  # k even
    g_k_odd = (
        1j * (X_pos_ell[1 + 1 :, 1:] - X_neg_ell[1:, 1:]) * np.sqrt(1.0 / np.pi)
    )  # k odd

    g[1:, 1:][:, sel_even] = g_k_even[:, sel_even]
    g[1:L, 1:][:, sel_odd] = g_k_odd[:, sel_odd]  # all odd m at l = lmax are zero

    return g


def FSHT(bivar_coeffs: jnp.array) -> jnp.array:
    g = preparation(bivar_coeffs)
    return _ft_fourier2sph(g)


SCALE_2PI = 1.0 / (2.0 * np.pi)  # first-principles global gain (see to_healpy_alm)


def to_healpy_alm(
    C: np.array, lmax: int, scale: float = SCALE_2PI, mono_factor: float = 1.0
) -> np.array:
    """
    Convert the FastTransforms spherical-harmonic coefficient array ``C`` into a
    1-D complex ``alm`` in healpy ordering/normalization.

    ``C`` is the (L+1, 2L+1) triangular array from ``fourier2sph``. A degree-l,
    order-m coefficient lives at row ``l-m`` (m=0 lives at ``C[l, 0]``), and the
    two real-spherical-harmonic parts of order m sit in columns ``2m-1`` and
    ``2m``. The conversion to healpy's complex, orthonormal a_lm is:

      * a_{l,0} = (-1)^l * C[l, 0] / scale
      * a_{l,m} = (-1)^l * C[l-m, 2m-1] / (sqrt(2) * scale)   for m > 0

    The ``(-1)^l`` factor undoes the colatitude-origin phase of the DFS step
    (without it every odd-l coefficient comes out sign-flipped -- the original
    cause of the apparent "even-l" power bias). The ``sqrt(2)`` is the standard
    real<->complex spherical-harmonic factor for m != 0.

    ``scale`` is the pipeline's overall normalization constant (the gain mapping
    a unit a_{l,0} onto C[l, 0]). It is EXACTLY ``1/(2*pi)`` from first principles,
    independent of nside -- single-harmonic probes show every well-resolved
    (l, m) recovers with gain exactly 1/(2*pi) (sectoral m=l harmonics, which the
    grid samples best, hit it to ~1e-9 at every nside; the small per-mode
    deviations are latitude QUADRATURE error, not a normalization that a better
    constant could absorb -- a best-fit global scale differs from 1/(2*pi) by only
    ~5e-5 and does not reduce the per-l error). So ``scale`` defaults to
    ``SCALE_2PI`` and the old empirical zonal-probe calibration is unnecessary;
    ``tests/pipeline_helpers.calibrate_scale`` is kept only for verification.
    ``mono_factor`` defaults to 1: once ``preparation`` no longer double-weights
    the latitude-DC (T_0) row, the monopole needs no special gain.

    Only column ``2m-1`` is used: ``preparation``'s real-SH packing makes column
    ``2m`` the complex conjugate of ``2m-1``, so it carries no extra information.

    With the per-ring longitude referencing fixed in ``data_interpolation`` and
    the ``preparation`` T_0 fix, the diagonal gains are 1, the monopole leakage
    and m>0 longitude phase are gone, and the only residual is the genuine
    latitude QUADRATURE error at lmax = 2*nside, which DECREASES with nside
    (~5% at nside=8, ~3% at nside=16).
    """
    alm = np.zeros(((lmax + 1) * (lmax + 2)) // 2, dtype=complex)

    def idx(l, m):
        return m * (2 * lmax + 1 - m) // 2 + l  # healpy Alm.getidx

    alm[idx(0, 0)] = C[0, 0].real / (scale * mono_factor)
    for l in range(1, lmax + 1):
        sign = (-1.0) ** l
        alm[idx(l, 0)] = sign * C[l, 0].real / scale
        for m in range(1, l + 1):
            alm[idx(l, m)] = sign * C[l - m, 2 * m - 1] / (np.sqrt(2.0) * scale)

    return alm


def convert_to_bivar_coeffs(g: jnp.array, nside: int, spin: int = 0) -> jnp.array:
    # converting 2D array of g coefficients of Fourier-Chebyshev series
    # into 2D array of bivariate Fourier coefficients.
    #
    # Inverse of preparation(): g has shape (L+1, 2*L+1) where L is the internal
    # latitude band limit (L = g.shape[0]-1). The longitude axis is built at the
    # full 2*L+1 width and then de-expanded back to the 4*nside columns the rest
    # of the pipeline uses, keeping only |m| <= 2*nside. nside must be passed in
    # because it can no longer be inferred from the (latitude-driven) g width.
    NSIDE = nside
    L = g.shape[0] - 1
    X_coeff = np.zeros((2 * L + 1, 2 * L + 1), dtype=complex)

    # m = 0
    X_pos_ell = (g[:, 0] * np.sqrt(2 * np.pi) / 2).copy()
    # preparation() no longer puts a factor 2 on the T_0 (k=0) row, so restore it
    # here when inverting (only the k=0 element; k>0 rows already carry it).
    X_pos_ell[0] *= 2
    X_coeff[L:, L] = X_pos_ell  # including ell = 0
    X_coeff[:L, L] = np.flip(X_pos_ell[1:])

    # m != 0, columns of g are ordered [0, -1, 1, -2, 2, ...]
    g_m_neg = g[:, 1::2]  # [-1, -2, -3, ..., -L]
    g_m_pos = g[:, 2::2]  # [ 1,  2,  3, ...,  L]
    # cos/sin parity per longitude mode m is (m + spin) (see preparation); the
    # positive-m columns are m = 1..L. For spin = 0 this is the plain m parity.
    sel_even = (np.arange(1, L + 1) + spin) % 2 == 0
    sel_odd = ~sel_even

    # m > 0
    X_pos_ell = g_m_pos * np.sqrt(np.pi) / 2
    X_pos_ell[0, sel_odd] = 0  # odd m, ell = 0
    X_pos_ell[1:, sel_odd] = -1j * g_m_pos[:L, sel_odd] * np.sqrt(np.pi) / 2
    X_pos_ell[0, sel_even] *= 2  # restore the T_0 (k=0) factor 2 for even m

    X_coeff[L:, L + 1 :] = X_pos_ell  # including ell = 0
    X_coeff[:L, L + 1 :][:, sel_even] = np.flip(X_pos_ell[1:], axis=0)[:, sel_even]
    X_coeff[:L, L + 1 :][:, sel_odd] = -np.flip(X_pos_ell[1:], axis=0)[:, sel_odd]

    # m < 0
    g_m_neg = np.flip(g_m_neg, axis=1)  # [-L , ..., -3, -2, -1]
    sel_odd = np.flip(sel_odd)
    sel_even = np.flip(sel_even)

    X_pos_ell = g_m_neg * np.sqrt(np.pi) / 2
    X_pos_ell[0, sel_odd] = 0  # odd m, ell = 0
    X_pos_ell[1:, sel_odd] = -1j * g_m_neg[:L, sel_odd] * np.sqrt(np.pi) / 2
    X_pos_ell[0, sel_even] *= 2  # restore the T_0 (k=0) factor 2 for even m

    X_coeff[L:, :L] = X_pos_ell  # including ell = 0
    X_coeff[:L, :L][:, sel_even] = np.flip(X_pos_ell[1:], axis=0)[:, sel_even]
    X_coeff[:L, :L][:, sel_odd] = -np.flip(X_pos_ell[1:], axis=0)[:, sel_odd]

    # de-expand longitude to the central 4*nside columns (m = -2*nside .. 2*nside-1)
    bivar_coeff = X_coeff[:, L - 2 * NSIDE : L + 2 * NSIDE].copy()
    bivar_coeff[:, 0] = 2 * X_coeff[:, L - 2 * NSIDE]  # undo the m=-2*nside split

    return bivar_coeff


def inverse_FSHT(alm: jnp.array, nside: int) -> jnp.array:
    bivar_coeffs = _ft_sph2fourier(np.asarray(alm))
    C = convert_to_bivar_coeffs(bivar_coeffs, nside)
    return bivar_coeffs, C


# --------------------------------------------------------------------------- #
# Spin-2 (polarization) FSHT                                                   #
# --------------------------------------------------------------------------- #
# Global gain mapping a unit spin-weighted coefficient onto the raw F-array cell.
# Measured (single ``+-2 Y_{l,m}`` probes vs healpy ``map2alm_spin``) to be exactly
# the scalar ``1/(2*pi)`` -- the spin transform shares the scalar's normalization
# (see ``spin_alm_from_F`` / tests/test_spin_FSHT.py). Kept as a named constant so
# the calibration can be re-pinned if the convention is ever re-derived.
SPIN_SCALE_2PI = SCALE_2PI


def FSHT_spin(bivar_coeffs: jnp.array, spin: int) -> jnp.array:
    """Bivariate Fourier coefficients -> spin-``spin`` spherical-harmonic ``F`` array.

    Mirrors the scalar ``FSHT`` but routes through ``ft_sphere.fourier2spinsph``;
    ``preparation`` is told the spin so its cos/sin (``m+spin`` parity) split and
    the resulting ``g`` array match the FastTransforms spin convention.
    """
    from .ft_sphere import fourier2spinsph

    g = preparation(bivar_coeffs, spin=spin)
    return fourier2spinsph(g, spin)


def inverse_FSHT_spin(F: jnp.array, nside: int, spin: int) -> jnp.array:
    """Spin-``spin`` ``F`` array -> bivariate Fourier coefficients (inverse FSHT)."""
    from .ft_sphere import spinsph2fourier

    bivar_coeffs = spinsph2fourier(np.asarray(F), spin)
    C = convert_to_bivar_coeffs(bivar_coeffs, nside, spin=spin)
    return bivar_coeffs, C


def _spin_conv_phase(m: int, spin: int) -> float:
    """healpy <-> FastTransforms spin-harmonic phase for order ``m``, spin ``spin``.

    The FastTransforms spin-weighted harmonics carry a longitude phase relative to
    healpy's (the spin analog of the scalar ``(-1)^l`` colatitude phase): a unit
    healpy ``s_a_{l,m}`` comes back as ``(-1)^m`` times the F-array cell, EXCEPT in
    the ``s < 0`` interior ``0 <= m < |s|`` (where ``m+s < 0`` flips the Jacobi
    ``(|m+s|, |m-s|)`` ordering), where the phase is ``+1`` instead. Measured with
    single ``+-2 Y_{l,m}`` probes against healpy (tests/test_spin_FSHT.py); for the
    polarization spins ``s = +-2`` this is all the cases that occur.
    """
    if spin < 0 and 0 <= m < abs(spin):
        return 1.0
    return (-1.0) ** m


def _spin_F_col(m: int) -> int:
    """Column of the FastTransforms spin ``F`` array holding longitude order ``m``.

    The columns are ordered ``m = 0, -1, +1, -2, +2, ...`` (the spinsph2fourier
    convention): ``m=0`` -> col 0, ``m>0`` -> col ``2m``, ``m<0`` -> col ``2|m|-1``.
    """
    if m == 0:
        return 0
    return 2 * m if m > 0 else 2 * (-m) - 1


def spin_alm_from_F(
    F: np.array,
    lmax: int,
    spin: int,
    scale: float = SPIN_SCALE_2PI,
    colat_phase: bool = True,
    real_sh_norm: bool = True,
) -> np.array:
    """Extract the spin-weighted coefficients ``s_a_{l,m}`` (m >= 0) from an ``F`` array.

    ``F`` is the ``(L+1, 2L+1)`` triangular array from ``fourier2spinsph``: cell
    ``F[l - max(|m|, |spin|), col(m)]`` holds ``s_f_l^m`` (up to the global ``scale``
    and a ``(-1)^l`` colatitude-origin phase, exactly as the scalar ``to_healpy_alm``).
    The ``m != 0`` columns additionally carry the ``1/sqrt(2)`` real<->complex
    spherical-harmonic factor: ``preparation`` reuses the scalar (real-SH) bivariate
    Fourier normalization, in which every ``|m| > 0`` mode is a factor ``sqrt(2)``
    larger than the complex-harmonic coefficient -- the same ``sqrt(2)`` the scalar
    ``to_healpy_alm`` divides out. The returned 1-D array is in healpy ordering for
    ``m >= 0`` (E and B are real parity fields, so only ``m >= 0`` is stored; the
    combination into E/B happens in :func:`spin_to_EB`).
    """
    F = np.asarray(F)
    alm = np.zeros(((lmax + 1) * (lmax + 2)) // 2, dtype=complex)

    def idx(l, m):
        return m * (2 * lmax + 1 - m) // 2 + l  # healpy Alm.getidx

    s0 = abs(spin)
    sqrt2 = np.sqrt(2.0) if real_sh_norm else 1.0  # library F is already complex-SH
    for m in range(0, lmax + 1):
        col = _spin_F_col(m)
        norm = scale if m == 0 else scale * sqrt2
        phase_m = _spin_conv_phase(m, spin)  # healpy<->FT spin longitude phase
        for l in range(max(m, s0), lmax + 1):
            row = l - max(m, s0)
            # ``(-1)^l`` undoes the DFS colatitude-origin phase, exactly as the
            # scalar ``to_healpy_alm``. The library's half-sample equiangular grid
            # has no such phase, so callers feeding a raw library ``F`` pass
            # ``colat_phase=False``.
            sign = phase_m * ((-1.0) ** l if colat_phase else 1.0)
            alm[idx(l, m)] = sign * F[row, col] / norm
    return alm


def spin_to_EB(
    F_plus: np.array,
    F_minus: np.array,
    lmax: int,
    scale: float = SPIN_SCALE_2PI,
    colat_phase: bool = True,
    real_sh_norm: bool = True,
):
    """Combine the spin +2 and spin -2 ``F`` arrays into healpy E/B ``alm``.

    With the spin coefficients
    ``s_a = spin_alm_from_F(F_s, ...)`` the parity eigenmodes are
    ``a^E = -(+2a + -2a)/2`` and ``a^B = +i(+2a - -2a)/2`` (the CMB convention; the
    overall sign/normalization is pinned by ``scale`` and matched to healpy's
    ``map2alm_spin`` in tests/test_spin_FSHT.py). Returns ``(almE, almB)``.
    """
    a_plus = spin_alm_from_F(
        F_plus,
        lmax,
        spin=2,
        scale=scale,
        colat_phase=colat_phase,
        real_sh_norm=real_sh_norm,
    )
    a_minus = spin_alm_from_F(
        F_minus,
        lmax,
        spin=-2,
        scale=scale,
        colat_phase=colat_phase,
        real_sh_norm=real_sh_norm,
    )
    almE = -(a_plus + a_minus) / 2.0
    almB = 1j * (a_plus - a_minus) / 2.0
    return almE, almB


def EB_to_spin_F(almE: np.array, almB: np.array, lmax: int):
    """Inverse of :func:`spin_to_EB` up to scale: build healpy-ordered spin alm.

    Returns the spin coefficients ``(+2a, -2a)`` as 1-D healpy-ordered arrays
    (m >= 0): ``+2a = -(a^E + i a^B)``, ``-2a = -(a^E - i a^B)``. The backward
    pipeline turns these into the ``F`` arrays the inverse spin FSHT consumes.
    """
    a_plus = -(almE + 1j * almB)
    a_minus = -(almE - 1j * almB)
    return a_plus, a_minus
