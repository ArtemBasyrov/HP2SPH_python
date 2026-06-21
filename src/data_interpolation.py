import jax.numpy as jnp
import numpy as np
import time
import jax
import jax_healpy as jhp

from functools import partial


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

    # find the ring sizes
    ring_sizes = np.full(num_rings, 4 * nside)
    ring_sizes[:nside] = 4 * i[:nside]
    ring_sizes[3 * nside :] = 4 * (4 * nside - i[3 * nside :])

    # find the start and end indices
    start_indices = jnp.cumsum(ring_sizes) - ring_sizes
    end_indices = start_indices + ring_sizes - 1
    return jnp.vstack((start_indices, end_indices, i)).T


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

    def inverse_fft(fft_coeffs):
        return jnp.fft.ifft(fft_coeffs, n=4 * nside, norm="forward").real

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
    # print(f"Inverse FFT execution time: {time.time() - start_time0:.6f} seconds")

    end_time = time.time()
    print(f"data_interpolation execution time: {end_time - start_time:.6f} seconds")
    return upsampled_data, fft_coeff


# The inverse transformation
def transform_grid_to_healpix(
    grid_data: jnp.array, fft_coeff: jnp.array = None
) -> jnp.array:
    """
    Transform a tensor product latitude-longitude grid into a HEALPix map,
    correctly handling shifted rings.

    Parameters:
        grid_data (ndarray): HEALPix map data.

    Returns:
        healpix_map (ndarray): Data mapped to the structured grid.
    """

    # get general info
    nside = fft_coeff.shape[1] // 4
    n_rings = 4 * nside - 1
    ring_info = get_ring_indices(nside)  # [start_id, end_id, ring_id]

    # find the ring sizes
    i = np.arange(1, n_rings + 1)
    ring_sizes = np.full(n_rings, 4 * nside)
    ring_sizes[:nside] = 4 * i[:nside]
    ring_sizes[3 * nside :] = 4 * (4 * nside - i[3 * nside :])

    healpix_map = np.empty(12 * nside**2)

    # Define function for vectorized FFT processing
    def calc_fft(ring_data):
        fft_coeffs = jnp.fft.fft(ring_data, n=4 * nside, norm="forward")
        return fft_coeffs

    def process_polar_ring(fft_coeff, num_pts):
        mid = num_pts // 2
        corrected_coeffs_back = np.zeros(num_pts, dtype=complex)
        corrected_coeffs_back[:mid] = fft_coeff[:mid]
        corrected_coeffs_back[-mid:] = fft_coeff[-mid:]

        # numpy (not jax) FFT for the same per-ring-dispatch reason as the forward.
        fft_coeffs = np.fft.ifft(corrected_coeffs_back, n=num_pts, norm="forward").real
        # Mirror of the forward change: no num_pts/(4*nside) rescaling. The
        # forward FFT (norm='forward') already carries the 1/num_pts, so the
        # ifft (also norm='forward') inverts it exactly with no extra factor.
        return fft_coeffs

    def process_equatorial_ring(fft_coeffs):
        return jnp.fft.ifft(fft_coeffs, n=4 * nside, norm="forward").real

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
