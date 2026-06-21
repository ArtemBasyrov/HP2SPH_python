"""Latitude nuFFT -- the pipeline's only ill-conditioned stage.

After the DFS doubling there are ~8*nside latitude samples at the (clustered)
HEALPix colatitudes, and the analysis fits latitude Fourier modes to them. How many
modes you fit decides everything, because the Vandermonde's conditioning depends on
the band:

    nside                     16     32      64       128
    cond(A), |k|<=4*nside    ~2e2   ~1e5   ~8e10    ~6e15 (~1/eps)   <- SQUARE
    cond(A), |k|<=2*nside    ~1.15  ~1.15  ~1.16    ~1.16            <- band-limited

``apply_nuFFT`` exposes two regimes (``solve_modes`` + ``solver``):

* DEFAULT -- ``solve_modes = 4*nside+1`` (|k| <= 2*nside), ``solver="cg"``. This is
  the band a band-limited (lmax = 2*nside) signal lives in; the Vandermonde is
  WELL conditioned at every nside, so finufft + CG converges in a few iterations at
  machine precision and stays O(N log N). The solved spectrum is zero-padded up to
  N = 8*nside+1 (the L = 4*nside array the FSHT expects -- see ``FSHT.preparation``).
  Accurate (forward & round trip both converge, ~1e-3 by nside 256) and SCALABLE to
  nside 512+. The small round-trip residual is above-band polar aliasing the band
  can't (and arguably shouldn't) represent. This is the paper's truncation.

* EXACT round trip -- ``solve_modes = 8*nside+1`` (the SQUARE interpolation, one mode
  per sample), ``solver="svd"``. Reproduces the map bit-for-bit, but the square
  Vandermonde is severely ill-conditioned: CG on the normal equations sees the
  SQUARED condition number and finufft's error is amplified by cond (round trip
  floors ~8e-3 by nside 64), so it needs the dense truncated-SVD ``svd_nufft_forward``
  (exact matrix, cached & shared across longitude columns; O(nside^3), fine to
  ~nside 256). Good for bit-exact round trips up to nside ~64; the square band is a
  genuine 1/eps wall by nside 128 that NO solver escapes.

Synthesis (``cg_nufft_backward`` / ``inverse_nuFFT``) is a plain NUFFT evaluation
(well conditioned), so it stays O(N log N) in both regimes.
"""

import jax
import jax.numpy as jnp
import numpy as np
import finufft
from scipy.sparse.linalg import cg, LinearOperator

from .data_interpolation import create_latitude_array


def compute_voronoi_weights_1d(x, domain=(np.pi, -np.pi)):
    """
    Compute Voronoi cell-based weights for 1D nonuniform points.

    Args:
        x (np.ndarray): 1D nonuniform sample locations.
        domain (tuple): Domain boundaries (left, right).

    Returns:
        weights (np.ndarray): Density compensation weights.
    """

    # Compute midpoints between points
    midpoints = (x[1:] + x[:-1]) / 2.0

    # Add domain boundaries
    left, right = domain
    midpoints = np.concatenate([[left], midpoints, [right]])

    # Calculate Voronoi cell lengths
    weights_sorted = midpoints[1:] - midpoints[:-1]

    return -weights_sorted


def _default_N_modes(M_samples):
    # Number of latitude Fourier modes to solve for. The latitude bandwidth is
    # set by the number of latitude SAMPLES (~8*nside after the DFS doubling),
    # NOT by the longitude count. Using only 4*nside+1 modes (the old
    # `n_trans + 1`) under-resolves latitude by ~2x and aliases high-frequency
    # latitude content down into low ell. Default to the smallest ODD count
    # >= M_samples: odd so the modes are symmetric (-L..L) for the downstream
    # spherical-harmonic step, and >= M_samples so the Vandermonde system
    # interpolates the samples (the extra mode sits in the null space). NOTE the
    # FSHT `preparation` is calibrated for L = (N_modes-1)//2 = 4*nside, i.e. this
    # exact count -- it does NOT work at other latitude band limits, so changing
    # N_modes also requires reworking preparation. See module docstring / CLAUDE.md.
    return M_samples + (M_samples % 2 == 0)


# Cache of the latitude Vandermonde SVD, keyed by (sample locations, N_modes).
# The Vandermonde A is identical for every longitude column, so one factorization
# serves all n_trans transforms and is reused across forward calls at a given nside.
_SVD_CACHE = {}


def _vandermonde_svd(x, N_modes, weights):
    """SVD of the weighted latitude Vandermonde sqrt(W) A (cached).

    A[j, k] = exp(+i * k * x_j) with modes k = -L..L (L = (N_modes-1)//2), the same
    -L..L ordering finufft produces with modeord=0 (row L = DC), so the resulting
    f_hat aligns with what `FSHT.preparation` expects. ``weights`` are the
    density-compensation (Voronoi) weights; the solve is the weighted least squares
    min || sqrt(W) (A f_hat - samples) ||, the same problem the CG path solves via
    the weighted normal equations (so the two solvers agree where both are stable).
    """
    key = (x.shape[0], N_modes, hash(x.tobytes()))
    cached = _SVD_CACHE.get(key)
    if cached is None:
        L = (N_modes - 1) // 2
        k = np.arange(-L, L + 1)
        A = np.exp(1j * np.outer(x, k))  # (M_samples, N_modes), modes -L..L
        sw = np.sqrt(np.abs(weights))
        U, s, Vh = np.linalg.svd(sw[:, None] * A, full_matrices=False)
        cached = (sw, U, s, Vh)
        _SVD_CACHE[key] = cached
    return cached


def svd_nufft_forward(x, f_samples, N_modes=None, rcond=1e-13):
    """Latitude analysis via a dense truncated-SVD pseudo-inverse.

    Solves the same weighted least-squares latitude fit as ``cg_nufft_forward`` but
    with an explicit, cached SVD of the (small, shared) Vandermonde instead of CG on
    the normal equations. This matters at high nside: the latitude Vandermonde at the
    clustered HEALPix colatitudes is severely ill-conditioned (cond ~ 8e10 at
    nside=64), and CG on the *normal* equations works at the SQUARED condition number
    while finufft's transform error gets amplified by the condition number -- together
    they floor the round trip at ~1e-2 by nside=64. A direct SVD solve on the exact
    matrix avoids both, restoring near machine-precision invertibility (round trip
    ~1.8e-6 at nside=64) at the same forward accuracy. ``rcond`` truncates singular
    directions below ``rcond * sigma_max`` -- a small value keeps the (well-determined)
    modes and exposes the conditioning/accuracy/invertibility trade-off.

    Cost is O(M_samples^2 * N_modes) for the one-off factorization (cached per nside),
    i.e. O(nside^3); fine for research resolutions up to ~nside 256. For nside >= 128
    the Vandermonde is numerically rank-deficient (cond ~ 1/eps) and NO solver
    recovers invertibility -- that regime needs the FSHT reworked at L = lmax = 2*nside
    so the latitude solve can drop to the well-conditioned 4*nside+1 modes.

    Returns (f_hat of shape (N_modes, n_trans), info=0) to match ``cg_nufft_forward``.
    """
    n_trans = f_samples.shape[0]
    M_samples = f_samples.shape[1]
    if N_modes is None:
        N_modes = _default_N_modes(M_samples)

    weights = compute_voronoi_weights_1d(x)
    sw, U, s, Vh = _vandermonde_svd(x, N_modes, weights)

    data = f_samples.T  # (M_samples, n_trans)
    weighted = sw[:, None] * data
    s_inv = np.where(s > rcond * s[0], 1.0 / s, 0.0)
    f_hat = (Vh.conj().T * s_inv) @ (U.conj().T @ weighted)  # (N_modes, n_trans)
    return f_hat, 0


def cg_nufft_forward(x, f_samples, N_modes=None, rtol=1e-9, maxiter=200, eps=1e-12):
    # Get dimensions
    n_trans = f_samples.shape[0]  # = 4*nside (number of longitude transforms)
    M_samples = f_samples.shape[1]  # = 8*nside (number of latitude samples)
    if N_modes is None:
        N_modes = _default_N_modes(M_samples)

    # Precompute NUFFT plans with batch processing (n_trans transforms)
    plan_forward = finufft.Plan(
        2, (N_modes,), n_trans=n_trans, isign=1, dtype=np.complex128, eps=eps
    )
    plan_adjoint = finufft.Plan(
        1, (N_modes,), n_trans=n_trans, isign=-1, dtype=np.complex128, eps=eps
    )

    # Set nonuniform points (same for all transforms)
    plan_forward.setpts(x)
    plan_adjoint.setpts(x)

    # Calculate Voronoi weights
    weights = compute_voronoi_weights_1d(x)
    norm = np.sum(weights)  # M_samples if weights = 1

    # Reshape helpers
    def vec_to_mat_hat(vec):
        return vec.reshape(n_trans, N_modes)

    def vec_to_mat_samples(vec):
        return vec.reshape(n_trans, M_samples)

    def mat_to_vec(mat):
        return mat.ravel()

    # Define NUFFT operators with batch processing
    def forward_op(f_hat_vec):
        """A @ f_hat for all transforms (batched)"""
        f_hat_mat = vec_to_mat_hat(f_hat_vec)
        out = np.zeros((n_trans, M_samples), dtype=np.complex128)
        plan_forward.execute(f_hat_mat, out)
        return mat_to_vec(out)

    def adjoint_op(f_samples_vec):
        """A^H @ f_samples for all transforms (batched)"""
        f_samples_mat = vec_to_mat_samples(f_samples_vec)
        weighted_samples = f_samples_mat * weights
        out = np.zeros((n_trans, N_modes), dtype=np.complex128)
        plan_adjoint.execute(weighted_samples, out)
        return mat_to_vec(out / norm)

    def calc_rhs(f_sample_init):
        """A^H @ f_samples for all transforms (batched) initial"""
        weighted_samples_init = f_sample_init * weights
        out = np.zeros((n_trans, N_modes), dtype=np.complex128)
        plan_adjoint.execute(weighted_samples_init, out)
        return mat_to_vec(out / norm)

    # Solve (A^H A) f_hat = A^H f_samples using CG (batched)
    def apply_AHA(f_hat_vec):
        return adjoint_op(forward_op(f_hat_vec))

    # Reshape RHS and initial guess
    rhs = calc_rhs(f_samples)  # Flatten input data
    f_hat_guess = np.zeros(n_trans * N_modes, dtype=np.complex128)

    # Linear operator for CG
    A = LinearOperator(
        shape=(n_trans * N_modes, n_trans * N_modes),
        matvec=apply_AHA,
        dtype=np.complex128,
    )

    # Run CG. NOTE: CG runs on the NORMAL equations (A^H A), so it sees the SQUARED
    # condition number; the default rtol is tightened to 1e-9 (from 1e-6) because the
    # loose tol left the nside=32 round trip at ~1e-2 instead of ~1e-9. At nside>=64
    # the conditioning defeats CG regardless of rtol -- use solver="svd" there.
    f_hat_recon_flat, info = cg(A, rhs, x0=f_hat_guess, rtol=rtol, maxiter=maxiter)

    # Reshape back to (4*nside, N_modes)
    return f_hat_recon_flat.reshape(n_trans, N_modes).T, info


def cg_nufft_backward(x, f_hat, eps=1e-12):
    """
    Synthesis: evaluate the latitude Fourier modes f_hat at the nonuniform
    sample locations x. This is the adjoint/forward direction of the analysis
    problem and is a plain type-2 NUFFT, s = A @ f_hat -- no linear solve is
    needed.

    The previous implementation solved (A A^H) s = A f_hat with CG. That system
    is SINGULAR whenever there are more samples than modes (A A^H has rank
    <= N_modes < M_samples), so CG returned a min-norm solution that does not
    equal A f_hat and injected a large reconstruction error. Direct evaluation
    is both correct and faster.

    Synthesis is well conditioned (a plain evaluation), so finufft's ``eps`` is
    NOT amplified here -- unlike the analysis solve. Keeping it as a NUFFT (rather
    than a dense matvec) preserves the O(N log N) scaling of the inverse transform.
    """
    # Get dimensions
    n_trans = f_hat.shape[0]  # number of longitude transforms (= 4*nside)
    N_modes = f_hat.shape[1]  # number of latitude Fourier modes
    M_samples = len(x)

    plan_forward = finufft.Plan(
        2, (N_modes,), n_trans=n_trans, isign=1, dtype=np.complex128, eps=eps
    )
    plan_forward.setpts(x)

    out = np.zeros((n_trans, M_samples), dtype=np.complex128)
    plan_forward.execute(np.ascontiguousarray(f_hat), out)

    # info kept for API compatibility; direct evaluation always "converges".
    return out.T, 0


def _upsampled_latitudes(nside):
    """The DFS latitude sample locations (radians) for the given nside.

    The original HEALPix ring colatitudes, mirrored across the poles by the DFS
    step and bracketed by the two pole rings, mapped into the [-pi, pi) period the
    NUFFT/Vandermonde use.
    """
    latitudes = create_latitude_array(nside)
    lat = np.zeros(len(latitudes) * 2 + 2)
    lat[0] = 90
    lat[1 : len(latitudes) + 1] = latitudes
    lat[len(latitudes) + 1] = -90
    lat[len(latitudes) + 2 :] = -180 + latitudes
    return lat * np.pi / 180 + np.pi / 2


def _embed_centered(f_hat, N_full):
    """Zero-pad a (-Lw..Lw)-ordered latitude spectrum into a (-Lf..Lf) array.

    The solved spectrum (rows in finufft modeord=0, i.e. -Lw..Lw with DC at the
    centre) is placed centred inside the wider N_full array, zeroing |k|>Lw. This
    lets a solve at the well-conditioned band feed the FSHT, which is calibrated for
    the wider L = 4*nside array (``FSHT.preparation``).
    """
    Nw, ncols = f_hat.shape
    Lw = (Nw - 1) // 2
    Lf = (N_full - 1) // 2
    out = np.zeros((N_full, ncols), dtype=f_hat.dtype)
    out[Lf - Lw : Lf + Lw + 1] = f_hat
    return out


def apply_nuFFT(
    mp: jnp.array,
    solver: str = "cg",
    N_modes=None,
    solve_modes=None,
    rtol: float = 1e-9,
    maxiter: int = 200,
    eps: float = 1e-12,
    rcond: float = 1e-13,
) -> jnp.array:
    """Latitude analysis (the DFS grid's only ill-conditioned stage).

    Two knobs decide accuracy vs invertibility vs scalability:

    ``solve_modes`` -- how many latitude Fourier modes to actually fit:

    * ``4*nside+1`` (default, |k| <= 2*nside): the band a band-limited (lmax =
      2*nside) signal lives in. The Vandermonde here is WELL conditioned (cond ~
      1.15 at every nside), so the fit is stable and CG converges in a few
      iterations -- this is what scales to nside 256/512. The result is zero-padded
      up to ``N_modes`` for the FSHT. Forward accuracy and round trip both converge
      with nside (~1e-3 by nside 256); the small round-trip residual is above-band
      polar aliasing the band cannot (and arguably should not) represent.
    * ``8*nside+1`` (= ``N_modes``, the SQUARE interpolation): fits one mode per DFS
      sample, so the synthesis reproduces the map bit-for-bit (exact round trip) at
      low nside. But this Vandermonde is severely ill-conditioned (cond ~ 8e10 at
      nside 64, ~1/eps at nside 128), so it needs ``solver="svd"`` and walls out by
      nside ~128. Use it only for bit-exact round trips at nside <= 64.

    ``solver`` -- how to solve that fit:

    * ``"cg"`` (default): finufft + conjugate gradient on the normal equations.
      O(N log N) and scalable; ideal for the well-conditioned ``solve_modes`` above.
    * ``"svd"``: dense truncated-SVD pseudo-inverse of the (cached, shared)
      Vandermonde. O(nside^3) one-off factorisation; needed only for the
      ill-conditioned square band, where it reaches ~1e-6 round trip up to nside 64.

    ``N_modes`` is the FSHT band (default ``8*nside+1`` -> L = 4*nside); ``rcond``
    regularises the SVD; ``rtol``/``maxiter``/``eps`` tune CG and the NUFFT.
    """
    nside = mp.shape[1] // 4
    M_samples = 8 * nside  # latitude samples after the DFS doubling
    if N_modes is None:
        N_modes = _default_N_modes(M_samples)  # 8*nside+1, the band FSHT expects
    if solve_modes is None:
        solve_modes = 4 * nside + 1  # well-conditioned latitude band (|k| <= 2*nside)

    DFT_upsampled_lat = _upsampled_latitudes(nside)

    if solver == "svd":
        fft_lat, info = svd_nufft_forward(
            DFT_upsampled_lat, np.asarray(mp.T), N_modes=solve_modes, rcond=rcond
        )
    elif solver == "cg":
        fft_lat, info = cg_nufft_forward(
            DFT_upsampled_lat,
            np.asarray(mp.T).copy(),
            N_modes=solve_modes,
            rtol=rtol,
            maxiter=maxiter,
            eps=eps,
        )
        if info != 0:
            print("CG solver didn't converge!")
    else:
        raise ValueError(f"unknown solver {solver!r}; use 'svd' or 'cg'")

    if solve_modes < N_modes:
        fft_lat = _embed_centered(fft_lat, N_modes)

    return fft_lat


def inverse_nuFFT(fft_lat: jnp.array, eps: float = 1e-12) -> jnp.array:
    """
    Perform inverse NUFFT (Type-2) to reconstruct signal at non-uniform latitudes.

    Synthesis is a plain evaluation s = A @ f_hat (well conditioned), so it stays a
    NUFFT regardless of which solver the forward analysis used.

    Parameters:
    - fft_lat (jnp.array): Fourier coefficients from uniform frequency space.
    - eps (float): NUFFT precision.

    Returns:
    - jnp.array: Reconstructed signal at non-uniform latitude samples.
    """
    nside = fft_lat.shape[1] // 4
    DFT_upsampled_lat = _upsampled_latitudes(nside)

    fft_lat = np.array(fft_lat)
    mp_reconstructed, info = cg_nufft_backward(
        DFT_upsampled_lat, fft_lat.T.copy(), eps=eps
    )

    # output a warning if the solver didn't converge
    if info != 0:
        print("CG solver didn't converge!")

    return mp_reconstructed
