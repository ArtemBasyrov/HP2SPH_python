import numpy as np

# The FSHT stage runs in-process through the libfasttransforms C library (see
# hp2sph/ft_sphere.py for how the library is located -- no env var needed when it is
# installed normally). If it cannot be loaded, the import below raises ImportError
# with a build/install hint; there is no other backend.
from .ft_sphere import fourier2sph as _ft_fourier2sph
from .ft_sphere import sph2fourier as _ft_sph2fourier


def _fill_columns(out, X, L, cosine, c_m):
    """Write one parity class of longitude columns from their latitude modes.

    ``X`` holds the source columns, rows ordered as the centred latitude modes
    ``k = -L .. L``. A column whose ``(m + spin)`` is even is a cosine series in
    latitude and one whose ``(m + spin)`` is odd is a sine series, which is the whole
    of the difference between the two branches here.
    """
    if X.shape[1] == 0:
        return
    if cosine:
        # Row 0 is the Chebyshev T_0 / latitude-DC term. For an even cosine series
        # f = sum_k c_k e^{ikθ} (c_{-k} = c_k) the Chebyshev coefficients are g_0 = c_0
        # and g_k = 2 c_k for k > 0 -- i.e. the DC row carries NO factor 2. A factor 2
        # here over-weighted the latitude DC, leaking even-l zonal power into the
        # monopole and inflating even-m gains.
        out[0] = X[L] * c_m
        out[1:] = (X[L + 1 :] + X[L - 1 :: -1]) * c_m
    else:
        out[0] = 1j * (X[L + 1] - X[L - 1]) * c_m
        # all odd m at l = lmax are zero, so the last row is left at 0
        out[1:L] = 1j * (X[L + 2 :] - X[L - 2 :: -1]) * c_m


def preparation(bivar_coeffs: np.ndarray, spin: int = 0) -> np.ndarray:
    # bivar_coeffs: (2*L+1 latitude modes [centered], 4*NSIDE longitude [natural
    # centered order m = -2*NSIDE .. 2*NSIDE-1]). The internal latitude band
    # limit L is set by the number of latitude modes the nuFFT solved for, which
    # is decoupled from (and larger than) the longitude resolution. Longitude
    # only supports |m| <= 2*NSIDE, so the longitude axis is zero-padded out to
    # the 2*L+1 columns the Fourier->spherical-harmonic step expects.
    NSIDE = bivar_coeffs.shape[1] // 4
    L = (bivar_coeffs.shape[0] - 1) // 2  # internal latitude band limit
    J = 2 * NSIDE  # largest |m| the longitude grid carries
    if L < J:
        raise ValueError(
            f"the latitude band L={L} cannot hold longitude modes up to |m|={J}"
        )

    # transform X into g array, size (L+1, 2*L+1)
    g = np.zeros((L + 1, 2 * L + 1), dtype=complex)

    c_m = np.sqrt(1.0 / np.pi)
    c_0 = np.sqrt(0.5 / np.pi)

    # The FSHT wants the longitude modes in the order [0, -1, 1, -2, 2, ...], so output
    # column 2j-1 carries m = -j and column 2j carries m = +j, while the input holds
    # them in natural centred order (column m + 2*NSIDE). That permutation used to be
    # built explicitly -- widen to 4*NSIDE+1 columns, embed centred in 2*L+1, then
    # argsort -- as three full (2*L+1, 2*L+1) complex arrays, 1.1 GB apiece at nside
    # 2048, of which the first two are copies into an array of the same shape whenever
    # L = 2*NSIDE, the shipped compact band. It needs none of them: reading m = -j and
    # writing 2j-1 is a reversed slice against a strided one, and columns beyond
    # |m| = 2*NSIDE are zero and simply left alone.
    #
    # A column's branch depends on the parity of (m + spin) and m = -j and m = +j have
    # the same parity, so both of a j's columns are filled together and the j of one
    # parity are a stride-2 slice of the input and a stride-4 slice of the output.
    g[0, 0] = bivar_coeffs[L, J] * c_0
    g[1:, 0] = (bivar_coeffs[L + 1 :, J] + bivar_coeffs[L - 1 :: -1, J]) * c_0

    for j0 in (1, 2):
        cosine = (j0 + spin) % 2 == 0
        n = (J - 1 - j0) // 2 + 1  # how many j of this parity below the Nyquist one
        if n <= 0:
            continue
        # m = -j: input column 2*NSIDE - j, output column 2j-1
        _fill_columns(
            g[:, 2 * j0 - 1 :: 4][:, :n],
            bivar_coeffs[:, J - j0 :: -2][:, :n],
            L,
            cosine,
            c_m,
        )
        # m = +j: input column 2*NSIDE + j, output column 2j
        _fill_columns(
            g[:, 2 * j0 :: 4][:, :n],
            bivar_coeffs[:, J + j0 :: 2][:, :n],
            L,
            cosine,
            c_m,
        )

    # |m| = 2*NSIDE is the longitude grid's Nyquist: +m and -m are the same sampled
    # mode, so the single stored column is split half onto each end. This is where the
    # documented l = m = 2*nside half gain comes from. Halve first and then sum, which
    # is the order the columns were formed in before.
    half = 0.5 * bivar_coeffs[:, 0]
    _fill_columns(
        g[:, 2 * J - 1 : 2 * J + 1],
        np.stack((half, half), axis=1),
        L,
        (J + spin) % 2 == 0,
        c_m,
    )

    return g


def FSHT(bivar_coeffs: np.ndarray) -> np.ndarray:
    g = preparation(bivar_coeffs)
    # g is preparation's own output and dies here, so let the transform reuse it
    # rather than hold a second (L+1, 2L+1) array beside it.
    return _ft_fourier2sph(g, overwrite=True)


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

    # healpy orders alm by m, so at fixed m the destination is contiguous in l and
    # the source is a contiguous slice of column 2m-1: one assignment per order.
    sign_l = np.where(np.arange(lmax + 1) % 2, -1.0, 1.0)  # (-1)^l

    alm[: lmax + 1] = sign_l * C[: lmax + 1, 0].real / scale
    alm[idx(0, 0)] = C[0, 0].real / (scale * mono_factor)
    for m in range(1, lmax + 1):
        n = lmax + 1 - m
        start = idx(m, m)
        alm[start : start + n] = sign_l[m:] * C[:n, 2 * m - 1] / (np.sqrt(2.0) * scale)

    return alm


def from_healpy_alm(
    alm: np.array, lmax: int, L: int, scale: float = SCALE_2PI, mono_factor: float = 1.0
) -> np.array:
    """Inverse of :func:`to_healpy_alm`: healpy alm -> FastTransforms ``C`` array.

    Returns the ``(L+1, 2L+1)`` triangular array that ``inverse_FSHT`` consumes,
    so a caller holding healpy-ordered coefficients can drive the scalar inverse
    pipeline. ``main.backward`` only ever receives a ``C`` straight back from the
    forward, which is why this direction did not exist before; anything starting
    from coefficients (a round trip, a synthesis of a known sky) needs it.

    Reading ``to_healpy_alm``'s mapping backwards:

      * C[l, 0]      = (-1)^l * scale * Re a_{l,0}
      * C[l-m, 2m-1] = (-1)^l * sqrt(2) * scale * a_{l,m}     for m > 0
      * C[l-m, 2m]   = conj(C[l-m, 2m-1])

    That last line is what makes the inverse well defined at all. ``to_healpy_alm``
    reads only the odd column of each real-spherical-harmonic pair because
    ``preparation``'s packing makes the even column its conjugate; verified to hold
    to 2.8e-16 on real pipeline output.

    ``L`` is the band limit of the target array and is passed explicitly because it
    is set by the latitude solve, not by ``lmax`` -- with the compact default band
    ``L = 2*nside``. Coefficients above ``min(lmax, L)`` are dropped.

    Validated against healpy: ``backward_map(from_healpy_alm(alm), nside)``
    reproduces ``hp.alm2map(alm)`` to 2.8e-13 / 2.9e-13 / 3.4e-13 at nside
    8 / 16 / 32, the same machine precision the native spin backward reaches
    (``tests/test_FSHT.py``, ``tests/test_pipeline.py``).

    This is NOT a left inverse of ``to_healpy_alm`` applied to a *forward's* output.
    The forward leaves quadrature residue in the triangular tail cells (rows
    ``> L-m``) that ``to_healpy_alm`` never reads, so that residue cannot be
    restored. Reconstructing a coefficient SET is exact; round-tripping a specific
    ``C`` is not.
    """
    C = np.zeros((L + 1, 2 * L + 1), dtype=complex)

    def idx(l, m):
        return m * (2 * lmax + 1 - m) // 2 + l  # healpy Alm.getidx

    top = min(lmax, L)
    sign_l = np.where(np.arange(top + 1) % 2, -1.0, 1.0)  # (-1)^l

    C[: top + 1, 0] = sign_l * alm[idx(0, 0) : idx(top, 0) + 1].real * scale
    C[0, 0] = alm[idx(0, 0)].real * scale * mono_factor
    for m in range(1, top + 1):
        n = top + 1 - m
        start = idx(m, m)
        v = sign_l[m:] * alm[start : start + n] * np.sqrt(2.0) * scale
        C[:n, 2 * m - 1] = v
        C[:n, 2 * m] = np.conj(v)

    return C


def convert_to_bivar_coeffs(g: np.ndarray, nside: int, spin: int = 0) -> np.ndarray:
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


def inverse_FSHT(alm: np.ndarray, nside: int) -> np.ndarray:
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


def spin_g_to_library(
    g: np.array, scale: float = SPIN_SCALE_2PI, real_sh_norm: bool = True
) -> np.array:
    """Put the pipeline's bivariate array into the FastTransforms convention.

    Measured (``tests/test_spin_dfs.py::test_spin_bivariate_matches_library``): the
    pipeline ``g`` and the library's own ``spinsph_analysis`` output ``A`` differ by

        g[k, c] = scale * (-1)^k * (sqrt(2) if c > 0 else 1) * A[k, c] ,

    so this undoes all three factors and returns an array in the library's convention.

    The ``(-1)^k`` is the important one. The pipeline's latitude variable is
    ``x = pi - theta`` (``nuFFT._upsampled_latitudes`` maps latitude +90 to x = pi), so
    its latitude spectrum is the theta-REFLECTED one, and row ``k`` picks up ``(-1)^k``
    (``cos(k(pi-theta)) = (-1)^k cos(k theta)``, and likewise for the sine rows).

    That reflection has to be undone HERE, in the bivariate domain, and not after the
    harmonic transform: under ``theta -> pi - theta`` the spin-weighted harmonics obey

        {}_s Y_lm(pi - theta, phi) = (-1)^(l+m) {}_{-s} Y_lm(theta, phi) ,

    i.e. the reflection FLIPS THE SPIN. So no phase applied to the output coefficients
    can undo it -- which is what the old ``spin_alm_from_F(colat_phase=True)`` path
    tried to do, and why every ``m != 0`` gain came out wrong (it acted like analysing
    at ``-spin``). The scalar transform has no such problem because ``s = 0`` is its own
    negative, so there the ``(-1)^l`` output phase is a valid shortcut.

    After this conversion the array decodes with the library settings
    (``scale=1, colat_phase=False, real_sh_norm=False``) -- the decode already validated
    against healpy in ``tests/test_spin_FSHT.py``.
    """
    A = np.array(g, dtype=complex)
    A *= ((-1.0) ** np.arange(A.shape[0]))[:, None]  # x = pi - theta
    A /= scale
    if real_sh_norm:
        A[:, 1:] /= np.sqrt(2.0)  # preparation's real-SH packing, m != 0 only
    return A


def spin_g_from_library(
    A: np.array, scale: float = SPIN_SCALE_2PI, real_sh_norm: bool = True
) -> np.array:
    """Inverse of :func:`spin_g_to_library`: library convention -> pipeline ``g``.

    The synthesis counterpart. ``spinsph2fourier`` hands back a bivariate array in the
    library's own convention (colatitude ``theta``, complex-SH normalized, unit gain);
    this puts it back into the array ``convert_to_bivar_coeffs`` expects -- the pipeline's
    real-SH packing, the ``1/(2*pi)`` gain, and the ``x = pi - theta`` reflection whose
    ``(-1)^k`` row phase MUST be reapplied here, in the bivariate domain, for the same
    reason ``spin_g_to_library`` removes it here (the reflection flips the spin, so no
    phase on the coefficients can stand in for it).
    """
    g = np.array(A, dtype=complex)
    if real_sh_norm:
        g[:, 1:] *= np.sqrt(2.0)
    g *= scale
    g *= ((-1.0) ** np.arange(g.shape[0]))[:, None]  # back to x = pi - theta
    return g


def FSHT_spin(
    bivar_coeffs: np.ndarray,
    spin: int,
    scale: float = SPIN_SCALE_2PI,
    real_sh_norm: bool = True,
) -> np.ndarray:
    """Bivariate Fourier coefficients -> spin-``spin`` spherical-harmonic ``F`` array.

    Mirrors the scalar ``FSHT`` but routes through ``ft_sphere.fourier2spinsph``;
    ``preparation`` is told the spin so its cos/sin (``m+spin`` parity) split and
    the resulting ``g`` array match the FastTransforms spin convention.

    The returned ``F`` is in the LIBRARY convention (see ``spin_g_to_library``), so it
    decodes with ``spin_to_EB(..., scale=1.0, colat_phase=False, real_sh_norm=False)``
    exactly like a ``spinsph_analysis`` result.
    """
    from .ft_sphere import fourier2spinsph

    g = preparation(bivar_coeffs, spin=spin)
    return fourier2spinsph(
        spin_g_to_library(g, scale=scale, real_sh_norm=real_sh_norm), spin
    )


def inverse_FSHT_spin(
    F: np.ndarray,
    nside: int,
    spin: int,
    scale: float = SPIN_SCALE_2PI,
    real_sh_norm: bool = True,
) -> np.ndarray:
    """Spin-``spin`` ``F`` array -> bivariate Fourier coefficients (inverse FSHT).

    The exact inverse of :func:`FSHT_spin`: ``spinsph2fourier`` then
    ``spin_g_from_library`` (the ``(-1)^k`` colatitude reflection, the ``1/(2*pi)``
    gain and the real-SH ``sqrt(2)``) then ``convert_to_bivar_coeffs``. Returns
    ``(library-convention g, pipeline bivariate coefficients)``.
    """
    from .ft_sphere import spinsph2fourier

    bivar_coeffs = spinsph2fourier(np.asarray(F), spin)
    g = spin_g_from_library(bivar_coeffs, scale=scale, real_sh_norm=real_sh_norm)
    C = convert_to_bivar_coeffs(g, nside, spin=spin)
    return bivar_coeffs, C


def _spin_conv_phase(m: int, spin: int) -> float:
    """healpy <-> FastTransforms spin-harmonic phase for order ``m``, spin ``spin``.

    The FastTransforms spin-weighted harmonics carry a longitude phase relative to
    healpy's (the spin analog of the scalar ``(-1)^l`` colatitude phase): a unit
    healpy ``s_a_{l,m}`` comes back as ``(-1)^m`` times the F-array cell, EXCEPT
    where ``m + spin <= 0``, which flips the Jacobi ``(|m+s|, |m-s|)`` ordering and
    absorbs the sign, leaving ``+1``.

    Valid for every SIGNED ``m`` (the synthesis direction needs ``m < 0`` too; the
    analysis only ever reads ``m >= 0``). Measured with single ``+-2 Y_{l,m}`` probes
    against healpy across ``-lmax <= m <= lmax``: for ``spin = +2`` the exception is
    ``m <= -2`` and for ``spin = -2`` it is ``m <= 1``, both exactly ``m + spin <= 0``.
    On ``m >= 0`` this is identical to the old ``spin < 0 and 0 <= m < |spin|`` rule
    for the polarization spins ``s = +-2``, which are all the cases that occur.
    """
    if m + spin <= 0:
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
    # ``(-1)^l`` undoes the DFS colatitude-origin phase, exactly as the scalar
    # ``to_healpy_alm``. The library's half-sample equiangular grid has no such
    # phase, so callers feeding a raw library ``F`` pass ``colat_phase=False``.
    sign_l = (
        np.where(np.arange(lmax + 1) % 2, -1.0, 1.0)
        if colat_phase
        else np.ones(lmax + 1)
    )
    for m in range(0, lmax + 1):
        col = _spin_F_col(m)
        norm = scale if m == 0 else scale * sqrt2
        phase_m = _spin_conv_phase(m, spin)  # healpy<->FT spin longitude phase
        lo = max(m, s0)  # degrees below |spin| carry no coefficient and stay zero
        if lo > lmax:
            continue
        n = lmax + 1 - lo
        start = idx(lo, m)
        alm[start : start + n] = (phase_m * sign_l[lo:]) * F[:n, col] / norm
    return alm


def spin_alm_from_conjugate_F(
    F: np.array,
    lmax: int,
    spin: int,
    scale: float = SPIN_SCALE_2PI,
    colat_phase: bool = True,
    real_sh_norm: bool = True,
) -> np.array:
    """The ``-spin`` coefficients of a REAL ``(Q, U)`` field, from its ``+spin`` ``F``.

    For real Q and U the two spin passes are not independent. With
    ``z_s = Q + i s/|s| U`` and ``conj(_sY_lm) = (-1)^(s+m) _{-s}Y_{l,-m}``,

        ``_{-s}a_{l,m} = (-1)^m conj(_{+s}a_{l,-m})`` ,

    so the NEGATIVE-order columns of the ``+spin`` array already hold everything a
    second ``-spin`` analysis would recompute. ``_spin_F_col`` and
    ``_spin_conv_phase`` are both defined for signed ``m`` (the synthesis needs
    them), so the readout is the same as :func:`spin_alm_from_F` with ``m -> -m``,
    conjugated, times ``(-1)^m``.

    Returns a healpy-ordered (``m >= 0``) array, like :func:`spin_alm_from_F`. The
    identity holds exactly for the true coefficients; the two pipeline estimates
    differ only by the latitude solver's own convergence noise (measured ~2e-7
    relative at nside 32, three orders below the transform's accuracy).
    """
    F = np.asarray(F)
    alm = np.zeros(((lmax + 1) * (lmax + 2)) // 2, dtype=complex)

    def idx(l, m):
        return m * (2 * lmax + 1 - m) // 2 + l  # healpy Alm.getidx

    s0 = abs(spin)
    sqrt2 = np.sqrt(2.0) if real_sh_norm else 1.0
    sign_l = (
        np.where(np.arange(lmax + 1) % 2, -1.0, 1.0)
        if colat_phase
        else np.ones(lmax + 1)
    )
    for m in range(0, lmax + 1):
        col = _spin_F_col(-m)  # the coefficient at order -m of the SUPPLIED spin
        norm = scale if m == 0 else scale * sqrt2
        # (-1)^m from the conjugation identity; the rest is spin_alm_from_F at -m.
        phase_m = ((-1.0) ** m) * _spin_conv_phase(-m, spin)
        lo = max(m, s0)  # |-m| = m, so the row layout is unchanged
        if lo > lmax:
            continue
        n = lmax + 1 - lo
        start = idx(lo, m)
        # sign and norm are real, so conjugating only F is the same thing.
        alm[start : start + n] = (phase_m * sign_l[lo:]) * np.conj(F[:n, col]) / norm
    return alm


def _EB_from_spin_alm(a_plus: np.array, a_minus: np.array):
    """The parity eigenmodes of a pair of spin coefficient arrays.

    ``a^E = -(+2a + -2a)/2`` and ``a^B = +i(+2a - -2a)/2`` (the CMB convention; the
    overall sign/normalization is matched to healpy's ``map2alm_spin`` in
    tests/test_spin_FSHT.py).
    """
    almE = -(a_plus + a_minus) / 2.0
    almB = 1j * (a_plus - a_minus) / 2.0
    return almE, almB


def spin_to_EB(
    F_plus: np.array,
    F_minus: np.array,
    lmax: int,
    scale: float = SPIN_SCALE_2PI,
    colat_phase: bool = True,
    real_sh_norm: bool = True,
):
    """Combine the spin +2 and spin -2 ``F`` arrays into healpy E/B ``alm``.

    Decodes both arrays with :func:`spin_alm_from_F` and combines them with
    :func:`_EB_from_spin_alm`. Returns ``(almE, almB)``.

    This makes no reality assumption about the two passes; for a real ``(Q, U)`` map
    :func:`spin_to_EB_real` gets the same answer from ``F_plus`` alone, at half the
    analysis cost.
    """
    kw = dict(scale=scale, colat_phase=colat_phase, real_sh_norm=real_sh_norm)
    a_plus = spin_alm_from_F(F_plus, lmax, spin=2, **kw)
    a_minus = spin_alm_from_F(F_minus, lmax, spin=-2, **kw)
    return _EB_from_spin_alm(a_plus, a_minus)


def spin_to_EB_real(
    F_plus: np.array,
    lmax: int,
    scale: float = SPIN_SCALE_2PI,
    colat_phase: bool = True,
    real_sh_norm: bool = True,
):
    """:func:`spin_to_EB` for a REAL ``(Q, U)`` map, from the ``+2`` pass alone.

    ``z = Q + iU`` already carries the whole real ``(Q, U)`` pair, so the ``-2``
    coefficients are fixed by reality rather than independent -- see
    :func:`spin_alm_from_conjugate_F` for the identity. This is the analysis
    counterpart of the single-pass native synthesis in
    ``spin_transform._backward_spin_hp2sph``, and halves the forward's cost.
    """
    kw = dict(scale=scale, colat_phase=colat_phase, real_sh_norm=real_sh_norm)
    a_plus = spin_alm_from_F(F_plus, lmax, spin=2, **kw)
    a_minus = spin_alm_from_conjugate_F(F_plus, lmax, spin=2, **kw)
    return _EB_from_spin_alm(a_plus, a_minus)


def EB_to_spin_F(almE: np.array, almB: np.array, lmax: int):
    """Inverse of :func:`spin_to_EB` up to scale: build healpy-ordered spin alm.

    Returns the spin coefficients ``(+2a, -2a)`` as 1-D healpy-ordered arrays
    (m >= 0): ``+2a = -(a^E + i a^B)``, ``-2a = -(a^E - i a^B)``. The backward
    pipeline turns these into the ``F`` arrays the inverse spin FSHT consumes.
    """
    a_plus = -(almE + 1j * almB)
    a_minus = -(almE - 1j * almB)
    return a_plus, a_minus
