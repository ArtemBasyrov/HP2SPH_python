import numpy as np
import jax.numpy as jnp
import subprocess
import json


def preparation(bivar_coeffs: jnp.array) -> jnp.array:
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

    # rearange X into [0 ,-1, 1, -2, 2, ...] order along k
    indx = np.fft.fftfreq(2 * L + 1, d=1) * (2 * L + 1)
    indx = np.fft.fftshift(indx)
    sel = np.argsort(np.abs(indx), kind="stable")
    indx = indx[sel]
    X_sort = X_coeff[:, sel]

    X_pos_ell = X_sort[L:]  # including 0 and positive ell = [0, 1, 2, ..., L]
    X_neg_ell = X_sort[:L]  # negative ell = [-L, ..., -2, -1]
    X_neg_ell = np.flip(X_neg_ell, axis=0)  # [-1, -2, ..., -L]

    # create sel for odd and even k
    sel_even = indx[1:] % 2 == 0
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


def call_Julia(g: jnp.array, scriptname: str) -> jnp.array:
    json_data = json.dumps({"real": g.real.tolist(), "imag": g.imag.tolist()})

    result = subprocess.run(
        ["julia", scriptname],
        input=json_data,  # Pass JSON as input
        text=True,
        capture_output=True,
    )

    def complex_decoder(obj):
        if "__complex__" in obj:
            return complex(obj["real"], obj["imag"])
        return obj

    output_array = json.loads(result.stdout, object_hook=complex_decoder)
    return output_array


def FSHT(bivar_coeffs: jnp.array) -> jnp.array:
    g = preparation(bivar_coeffs)

    output_array = call_Julia(g, scriptname="src/julia_sph.jl")
    C = np.array(output_array)

    return C


def to_healpy_alm(
    C: np.array, lmax: int, scale: float, mono_factor: float = 1.0
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
    a unit a_{l,0} onto C[l, 0]); it is a single number for a given nside,
    converging to 1/(2*pi), and is most simply obtained by transforming one
    zonal harmonic. ``mono_factor`` defaults to 1: once ``preparation`` no longer
    double-weights the latitude-DC (T_0) row, the monopole needs no special gain.

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


def convert_to_bivar_coeffs(g: jnp.array, nside: int) -> jnp.array:
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
    sel_even = np.arange(1, L + 1) % 2 == 0
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
    output_array = call_Julia(alm, scriptname="src/julia_sph_inverse.jl")
    bivar_coeffs = np.array(output_array)

    C = convert_to_bivar_coeffs(bivar_coeffs, nside)

    return bivar_coeffs, C
