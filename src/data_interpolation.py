import jax.numpy as jnp
import numpy as np
import time
import jax
import jax_healpy as jhp

from functools import partial
from scipy.special import gammaln


def get_ring_indices(nside: int) -> jnp.array:
    """
    Compute the indices of the pixels in each equatorial ring.

    Parameters:
        nside (int): HEALPix resolution parameter.

    Returns:
        ring_indices (list): List of pixel indices for each ring.
    """
    num_rings = 4 * nside - 1
    i = np.arange(1, num_rings + 1)
    ring_sizes = ring_pixel_counts(nside)

    # find the start and end indices
    start_indices = jnp.cumsum(ring_sizes) - ring_sizes
    end_indices = start_indices + ring_sizes - 1
    return jnp.vstack((start_indices, end_indices, i)).T


def ring_pixel_counts(nside: int) -> np.array:
    """Number of HEALPix pixels in each RING-ordered ring (north -> south)."""
    n_rings = 4 * nside - 1
    i = np.arange(1, n_rings + 1)
    sizes = np.full(n_rings, 4 * nside)
    sizes[:nside] = 4 * i[:nside]
    sizes[3 * nside :] = 4 * (4 * nside - i[3 * nside :])
    return sizes


def ring_alias_target(nside: int) -> (np.array, np.array, np.array):
    """Where each longitude mode is actually MEASURED on each ring.

    A ring of ``npix`` pixels samples the longitude field at ``npix`` points, so it does
    not measure mode ``m``: it measures the whole alias family of ``m``. Writing
    ``b = m mod npix`` folded into ``[-npix//2, npix//2)``, what
    ``transform_healpix_to_grid`` puts in slot ``b`` of ring ``r`` is

        M[r, b] = sum over {m : m == b (mod npix)} of c_m(theta_r) exp(i (m - b) phi0_r)

    -- the exponential because the phi=0 referencing at the end of that function
    multiplies each slot by ``exp(-i b phi0)``, using the SLOT index rather than the mode
    that contributed, so an aliased contribution also arrives with the wrong longitude
    phase. Verified against the pipeline to ~1e-12 against O(10) coefficients.

    The pipeline then asserts ``M[r, b] = c_b(theta_r)``, which is false whenever another
    member of the family carries amplitude. This is the analysis-side mirror of the
    synthesis-side fold in ``transform_grid_to_healpix.process_polar_ring``.

    Returns ``(target, phase, resolved)``, each ``(n_rings, 4*nside)`` in NATURAL
    (fftshifted) longitude order -- column ``j`` is mode ``m = j - 2*nside``, the order
    ``DFS`` and ``apply_nuFFT`` use:

    * ``target[r, j]`` -- the slot mode ``j`` folds onto,
    * ``phase[r, j]``  -- the ``exp(i (m - b) phi0)`` it arrives with,
    * ``resolved[r, j]`` -- whether the ring produces slot ``j`` at all. The Nyquist slot
      ``-npix//2`` counts as produced: under the fold it is a genuine constraint on the
      ``+-npix//2`` sum, not the mis-assignment the zero-padding makes of it.
    """
    sizes = ring_pixel_counts(nside)
    n_lon = 4 * nside
    phi0 = ring_first_longitude(nside)
    m = np.arange(n_lon) - n_lon // 2
    mid = (sizes // 2)[:, None]
    b = (m[None, :] + mid) % sizes[:, None] - mid
    target = b + n_lon // 2
    phase = np.exp(1j * (m[None, :] - b) * phi0[:, None])
    j = np.arange(n_lon)[None, :]
    resolved = (j >= n_lon // 2 - mid) & (j < n_lon // 2 + mid)
    return target, phase, resolved


def mode_pole_envelope(nside: int, spin: int = 0, lmax: int = None) -> np.array:
    """Largest ``|c_m(theta_r)|`` a band-limited spin-s field can carry, as a fraction of
    that mode's own peak over latitude.

    A spin-s mode-m latitude profile is the Wigner d function ``d^l_{-s,m}``, i.e.
    ``sin^|m+s|(theta/2) cos^|m-s|(theta/2) P^(|m-s|,|m+s|)(cos theta)``. Near a pole the
    polynomial is NOT O(1): the uniform asymptotic is Bessel, ``J_a(l*theta)`` with
    ``a = |m+s|`` at the north pole and ``a = |m-s|`` at the south, so for a small
    argument the amplitude is ``(l*theta/2)^a / a!`` -- THE SCALE IS ``l*theta``, NOT
    ``theta``. Dropping the ``l`` factor underestimates by six orders at the band edge
    (measured at nside 32, l = 63, m = +2: the innermost ring carries 1.6e-2 of the
    mode's peak; ``(theta/2)^4`` predicts 2.6e-8, ``(l*theta/2)^4/4!`` predicts 1.7e-2).

    ``lmax`` defaults to the pipeline band ``2*nside``, i.e. the worst case over the band.
    """
    if lmax is None:
        lmax = 2 * nside
    theta = np.deg2rad(90.0 - create_latitude_array(nside))[:, None]
    n_lon = 4 * nside
    m = np.arange(n_lon) - n_lon // 2
    a = np.abs(m + spin).astype(float)[None, :]
    b = np.abs(m - spin).astype(float)[None, :]
    log_north = np.log(np.maximum(lmax * theta / 2, np.finfo(float).tiny))
    log_south = np.log(np.maximum(lmax * (np.pi - theta) / 2, np.finfo(float).tiny))
    north = np.exp(np.minimum(a * log_north - gammaln(a + 1), 0.0))
    south = np.exp(np.minimum(b * log_south - gammaln(b + 1), 0.0))
    return np.minimum(north, south)


def ring_fold_plan(
    nside: int, spin: int = 0, tol: float = 1e-2, lmax: int = None
) -> (np.array, np.array, np.array):
    """The forward latitude operator's longitude layout: fold + selective zero-assertion.

    The zero-padding the pipeline does today is exactly equivalent to "fold, AND assert
    ``c_m(theta_r) = 0`` for every mode the ring does not resolve" -- when the other
    members of an alias family vanish the fold degenerates to the identity. Those zero
    assertions are not a bug, they are what makes the latitude least squares WELL
    CONDITIONED. Dropping them all -- which is what folding without them does, and what
    the superseded mask fix did -- leaves the polar caps constrained only through the
    alias sums: the system stays consistent -- the true solution satisfies it to ~7e-13 --
    but becomes so ill conditioned that LSMR is still 10% off after 40000 iterations.

    Almost all of the assertions are true to many digits, because an unresolved mode is
    negligible on a polar ring unless it decays slowly into the nearby pole, which happens
    only for ``|m|`` near ``|spin|``. So keep the assertion where ``mode_pole_envelope``
    says the mode is below ``tol``, RELAX it where it is not, and FOLD so the retained
    data equations account exactly for the relaxed content aliasing into them.

    Returns ``(target, phase, keep)``. ``keep[r, j]`` False marks an entry dropped from
    the fit -- an unresolved mode with non-negligible amplitude there; its content is
    still carried onto the ring's own bin by ``target``/``phase``. A trusted unresolved
    mode is its own target, so it asserts ``c_m = 0`` in place (the data is already 0
    there, from the zero-padding).

    With an empty relax set this is the identity, i.e. the current unmasked path
    bit-for-bit.
    """
    target, phase, resolved = ring_alias_target(nside)
    relax = (~resolved) & (mode_pole_envelope(nside, spin, lmax) > tol)
    trusted = (~resolved) & ~relax
    j = np.broadcast_to(np.arange(target.shape[1]), target.shape)
    return np.where(trusted, j, target), np.where(trusted, 1.0, phase), ~relax


def ring_first_longitude(nside: int) -> np.array:
    """Longitude (radians) of the first pixel in each RING-ordered HEALPix ring.

    HEALPix rings are not aligned to phi=0: each ring's first pixel sits half a
    pixel in (phi = pi/npix) for the polar rings, while the equatorial rings
    alternate between phi = pi/(4*nside) (a half equatorial pixel) and phi = 0.
    To reference every ring's longitude Fourier coefficients to a common phi=0
    origin (the convention healpy's a_lm use), each ring's mode-m coefficient
    must be multiplied by exp(-i * m * phi_first). Getting this wrong leaves an
    m-dependent longitude phase in the output alm. Matches ``hp.pix2ang`` exactly.
    """
    n_rings = 4 * nside - 1
    phi0 = np.zeros(n_rings)
    for r in range(n_rings):
        i = r + 1  # ring number 1 .. 4*nside-1 (north -> south)
        if i < nside:  # north polar cap
            phi0[r] = np.pi / (4 * i)
        elif i <= 3 * nside:  # equatorial belt
            phi0[r] = np.pi / (4 * nside) if (i - nside) % 2 == 0 else 0.0
        else:  # south polar cap
            phi0[r] = np.pi / (4 * (4 * nside - i))
    return phi0


def transform_healpix_to_grid(healpix_map: jnp.array) -> (jnp.array, jnp.array):
    start_time = time.time()
    """
    Step 1: Transform a HEALPix map into a tensor product latitude-longitude grid,
    correctly handling shifted equatorial rings.
    
    Parameters:
        healpix_map (ndarray): HEALPix map data.
    
    Returns:
        upsampled_data (ndarray): Data mapped to the structured grid.
    """
    nside = jhp.npix2nside(healpix_map.shape[0])
    n_rings = 4 * nside - 1

    ring_info = get_ring_indices(nside)  # [start_id, end_id, ring_id]
    upsampled_data = jnp.empty((n_rings, 4 * nside))
    fft_coeff = np.zeros((n_rings, 4 * nside), dtype=complex)

    # Define function for vectorized FFT processing
    def process_equatorial_ring(ring_data):
        fft_coeffs = jnp.fft.fft(ring_data, n=4 * nside, norm="forward")
        return fft_coeffs

    def process_polar_ring(ring_data):
        num_pts = len(ring_data)
        # numpy (not jax) FFT: each polar ring has a different length so they can't
        # be batched into one vmap, and a per-ring jax dispatch costs ~25 ms of
        # tracing/dispatch overhead -- ~2*nside of them dominate the whole pipeline
        # (13 s at nside=256). numpy's FFT on these small arrays is ~microseconds.
        coeffs = np.fft.fft(np.asarray(ring_data), n=num_pts, norm="forward")

        # this padding correctly accounts for fft frequencies position in the array
        mid = num_pts // 2
        coeffs_padded = np.zeros(4 * nside, dtype=complex)
        coeffs_padded[:mid] = coeffs[:mid]  # Positive frequencies
        coeffs_padded[-mid:] = coeffs[-mid:]  # Negative frequencies

        # NB: do NOT rescale by num_pts/(4*nside). With norm='forward' the FFT
        # coefficients are already the true longitude Fourier coefficients of the
        # ring (DC = ring mean), independent of how many pixels the ring has.
        # Zero-padding to 4*nside only extends the (empty) high-frequency band,
        # so the populated coefficients must be left untouched. The old factor
        # shrank every polar-ring coefficient by i/nside, which cancelled against
        # the inverse (so round trips stayed exact) but fed the wrong Fourier
        # coefficients to DFS/nuFFT/FSHT -- the dominant forward-alm error.
        return coeffs_padded

    # A spin field (Q + iU) is carried as a complex map; an intensity (I) map is
    # real. Keep complex content when present, but drop the (zero) imaginary part
    # for a real input so the scalar path stays bit-identical to before.
    is_complex = np.iscomplexobj(np.asarray(healpix_map))

    def inverse_fft(fft_coeffs):
        return jnp.fft.ifft(fft_coeffs, n=4 * nside, norm="forward")

    # Diving data into rings
    # start_time0 = time.time()
    ring_data = [healpix_map[start : end + 1] for start, end, nring in ring_info]
    # print(f"Ring selection execution time: {time.time() - start_time0:.6f} seconds")

    # Processing of equatorial rings
    # start_time0 = time.time()
    fft_coeff[nside - 1 : 3 * nside] = jax.vmap(process_equatorial_ring)(
        jnp.array(ring_data[nside - 1 : 3 * nside])
    )
    # print(f"Equatorial ring execution time: {time.time() - start_time0:.6f} seconds")

    # Processing of polar rings
    """
    As an idea I can switch to Julia FFTW library for the FFT computation in polar rings

    In python:
    import multiprocessing

    num_cores = multiprocessing.cpu_count()
    print(f"Number of CPU cores: {num_cores}")

    And then in Julia:
    using FFTW
    FFTW.set_num_threads(num_cores)
    """
    # start_time0 = time.time()
    for i in range(nside - 1):
        fft_coeff[i] = process_polar_ring(ring_data[i])
        fft_coeff[n_rings - 1 - i] = process_polar_ring(ring_data[n_rings - 1 - i])
    # print(f"Polar ring execution time: {time.time() - start_time0:.6f} seconds")

    # Reference every ring's coefficients to a common phi=0 origin. Each ring's
    # first pixel is offset by phi_first, so its mode-m FFT coefficient carries a
    # spurious exp(+i*m*phi_first); divide it out with exp(-i*m*phi_first). m is
    # the SIGNED frequency (numpy FFT order), so use fftfreq, not arange.
    m_signed = np.fft.fftfreq(4 * nside) * (4 * nside)
    phi0 = ring_first_longitude(nside)
    fft_coeff *= np.exp(-1j * np.outer(phi0, m_signed))

    # Inverse FFT
    # start_time0 = time.time()
    upsampled_data = jax.vmap(inverse_fft)(fft_coeff)
    if not is_complex:
        upsampled_data = upsampled_data.real
    # print(f"Inverse FFT execution time: {time.time() - start_time0:.6f} seconds")

    end_time = time.time()
    print(f"data_interpolation execution time: {end_time - start_time:.6f} seconds")
    return upsampled_data, fft_coeff


# The inverse transformation
def transform_grid_to_healpix(
    grid_data: jnp.array, fft_coeff: jnp.array = None, real_output: bool = True
) -> jnp.array:
    """
    Transform a tensor product latitude-longitude grid into a HEALPix map,
    correctly handling shifted rings.

    Parameters:
        grid_data (ndarray): HEALPix map data.
        real_output (bool): take the real part of each ring (the intensity / I
            path, default). Set False to keep a complex (Q + iU) spin field --
            the inverse of carrying a complex map through the forward transform.

    Returns:
        healpix_map (ndarray): Data mapped to the structured grid.
    """

    # get general info
    nside = fft_coeff.shape[1] // 4
    n_rings = 4 * nside - 1
    ring_info = get_ring_indices(nside)  # [start_id, end_id, ring_id]

    ring_sizes = ring_pixel_counts(nside)

    map_dtype = float if real_output else complex
    healpix_map = np.empty(12 * nside**2, dtype=map_dtype)

    def _maybe_real(arr):
        return arr.real if real_output else arr

    # Define function for vectorized FFT processing
    def calc_fft(ring_data):
        fft_coeffs = jnp.fft.fft(ring_data, n=4 * nside, norm="forward")
        return fft_coeffs

    def process_polar_ring(fft_coeff, num_pts):
        # ALIAS (fold), do not truncate. A ring of num_pts pixels samples the
        # longitude field at num_pts points, so mode m lands in bin m % num_pts:
        # the correct pixel values are sum_m c_m exp(i m phi_j), which is the FOLD
        # of the 4*nside-wide spectrum onto num_pts bins. Simply DROPPING
        # |m| >= num_pts//2 (what this did) throws that content away instead.
        #
        # For a spectrum produced by ``transform_healpix_to_grid`` the two are
        # bit-identical -- the forward zero-pads, so every folded-in entry is
        # exactly 0 -- which is why the round trip never noticed. They differ when
        # the spectrum comes from the SYNTHESIS side (``main.backward`` /
        # ``spin_transform.backward_spin``), where the high-|m| entries carry real
        # signal. For a SPIN field that is the whole ball game: |m| = |spin| is O(1)
        # AT the pole and is exactly the Nyquist of the innermost 4-pixel ring (the
        # same asymmetry ``ring_fold_plan`` handles on the analysis side), and
        # truncating it left the innermost polar rings ~100% wrong.
        m_signed = np.rint(np.fft.fftfreq(len(fft_coeff)) * len(fft_coeff)).astype(int)
        corrected_coeffs_back = np.zeros(num_pts, dtype=complex)
        np.add.at(corrected_coeffs_back, m_signed % num_pts, np.asarray(fft_coeff))

        # numpy (not jax) FFT for the same per-ring-dispatch reason as the forward.
        fft_coeffs = _maybe_real(
            np.fft.ifft(corrected_coeffs_back, n=num_pts, norm="forward")
        )
        # Mirror of the forward change: no num_pts/(4*nside) rescaling. The
        # forward FFT (norm='forward') already carries the 1/num_pts, so the
        # ifft (also norm='forward') inverts it exactly with no extra factor.
        return fft_coeffs

    def process_equatorial_ring(fft_coeffs):
        return _maybe_real(jnp.fft.ifft(fft_coeffs, n=4 * nside, norm="forward"))

    if fft_coeff is None:
        fft_coeff = np.zeros((n_rings, 4 * nside), dtype=complex)
        fft_coeff[:] = jax.vmap(calc_fft)(
            grid_data
        )  # Processing all ring with process_equatorial_ring

    # Undo the phi=0 referencing applied in the forward transform: shift each
    # ring's mode-m coefficient back to its native first-pixel longitude with
    # exp(+i*m*phi_first) (the conjugate of the forward correction).
    m_signed = np.fft.fftfreq(4 * nside) * (4 * nside)
    phi0 = ring_first_longitude(nside)
    fft_coeff = np.asarray(fft_coeff) * np.exp(+1j * np.outer(phi0, m_signed))

    eq_rings = jax.vmap(process_equatorial_ring)(fft_coeff[nside - 1 : 3 * nside])
    start_id, _, _ = ring_info[nside - 1]
    _, end_id, _ = ring_info[3 * nside - 1]
    healpix_map[start_id : end_id + 1] = jnp.concatenate(eq_rings)

    # Polar rings: the phi=0 referencing was already undone above for all rows.
    for i in range(nside - 1):
        num_pts = ring_sizes[i]

        start_id, end_id, _ = ring_info[i]
        healpix_map[start_id : end_id + 1] = process_polar_ring(fft_coeff[i], num_pts)

        start_id, end_id, _ = ring_info[-1 - i]
        healpix_map[start_id : end_id + 1] = process_polar_ring(
            fft_coeff[-1 - i], num_pts
        )

    return healpix_map


# For visualisation
def create_upsampled_grid(nside: int) -> (jnp.array, jnp.array):
    """
    Creates the longitude-latitude grid corresponding to the upsampled HEALPix points,
    covering the northern polar, equatorial, and southern polar regions.

    Parameters:
    - nside: HEALPix resolution parameter.

    Returns:
    - longitudes: 1D NumPy array of uniform longitude values (in degrees).
    - latitudes: 1D NumPy array of latitude values for all rings (in degrees).
    """
    max_lon_points = 4 * nside  # Uniform number of longitude points
    longitudes = np.linspace(
        0, 360, max_lon_points, endpoint=False
    )  # Longitudes in degrees

    latitudes = create_latitude_array(nside)

    return np.meshgrid(longitudes, latitudes)


def create_latitude_array(nside: int) -> jnp.array:
    """
    Generate latitude values for HEALPix rings, covering polar and equatorial regions.

    Parameters:
    - nside: HEALPix resolution parameter.

    Returns:
    - latitudes: 1D NumPy array of latitude values (in degrees).
    """
    # HEALPix j-values for polar and equatorial regions
    j_north_south = np.arange(1, nside)  # j-values for polar region rings
    j_equatorial = np.arange(
        nside, 3 * nside + 1
    )  # j-values for equatorial region rings

    # Compute colatitudes
    colatitudes_north = np.arccos(1 - (j_north_south**2) / (3 * nside**2))
    colatitudes_equatorial = np.arccos(2 * (2 * nside - j_equatorial) / (3 * nside))

    # Fix: Compute southern latitudes correctly using the same formula
    colatitudes_south = np.arccos(-(1 - (j_north_south**2) / (3 * nside**2)))
    colatitudes_south = np.flip(colatitudes_south)

    # Convert colatitudes to latitudes (latitude = 90° - colatitude)
    latitudes_north = np.degrees(np.pi / 2 - colatitudes_north)
    latitudes_equatorial = np.degrees(np.pi / 2 - colatitudes_equatorial)
    latitudes_south = np.degrees(
        np.pi / 2 - colatitudes_south
    )  # Fix: Ensure smooth transition

    # Concatenate all latitudes in order: North → Equator → South
    latitudes = np.concatenate([latitudes_north, latitudes_equatorial, latitudes_south])

    return latitudes
