import jax.numpy as jnp
import numpy as np
import time
import jax
import jax_healpy as jhp


def get_ring_indices(nside):
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
    ring_sizes[3 * nside:] = 4 * (4 * nside - i[3 * nside:])

    # find the start and end indices
    start_indices = jnp.cumsum(ring_sizes) - ring_sizes
    end_indices = start_indices + ring_sizes - 1
    return jnp.vstack((start_indices, end_indices, i)).T



def transform_healpix_to_grid(healpix_map):
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

    ring_info = get_ring_indices(nside) # [start_id, end_id, ring_id]
    upsampled_data = jnp.empty((n_rings, 4 * nside))
    
    # Define function for vectorized FFT processing
    def process_equatorial_ring(ring_data):
        fft_coeffs = jnp.fft.fft(ring_data, n=4 * nside)
        shift = np.exp(-1j * np.pi * np.arange(4 * nside) / (2 * nside))
        fft_coeffs = jax.lax.select((nside <= i) & (i < (3 * nside)), fft_coeffs * shift, fft_coeffs)
        return fft_coeffs
    
    def process_polar_ring(ring_data):
        num_pts = len(ring_data)
        coeffs = jnp.fft.fft(ring_data, n=num_pts)
        k_vals = np.fft.fftfreq(num_pts) * num_pts
        phase_shift = jnp.exp(-1j * jnp.pi * k_vals / (2 * nside))
        corrected_coeffs = coeffs * phase_shift
        coeffs_padded = jnp.pad(corrected_coeffs, (0, 4 * nside - num_pts), mode='constant')
        return coeffs_padded
    
    def inverse_fft(fft_coeffs):
        return jnp.fft.ifft(fft_coeffs, n=4 * nside).real
    

    
    # Processing of polar rings

    '''
    As an idea I can switch to Julia FFTW library for the FFT computation in polar rings

    In python:
    import multiprocessing

    num_cores = multiprocessing.cpu_count()
    print(f"Number of CPU cores: {num_cores}")

    And then in Julia:
    using FFTW
    FFTW.set_num_threads(num_cores)
    '''
    start_time0 = time.time()
    ring_data = [healpix_map[start:end] for start, end, nring in ring_info]
    fft_coeff = np.zeros((n_rings, 4 * nside), dtype=complex)
    print(f"Ring selection execution time: {time.time() - start_time0:.6f} seconds")
    start_time0 = time.time()
    for i in range(nside):
        fft_coeff[i] = process_polar_ring(ring_data[i])
        fft_coeff[n_rings-1 - i] = process_polar_ring(ring_data[n_rings-1 - i])
    print(f"Polar ring execution time: {time.time() - start_time0:.6f} seconds")

    # Processing of equatorial rings
    start_time0 = time.time()
    fft_coeff[nside: 3 * nside] = jax.vmap(process_equatorial_ring)(jnp.array(ring_data[nside: 3 * nside]))
    print(f"Equatorial ring execution time: {time.time() - start_time0:.6f} seconds")

    # Inverse FFT
    start_time0 = time.time()
    upsampled_data = jax.vmap(inverse_fft)(fft_coeff)
    print(f"Inverse FFT execution time: {time.time() - start_time0:.6f} seconds")

    end_time = time.time()
    print(f"data_interpolation execution time: {end_time - start_time:.6f} seconds")
    return upsampled_data, fft_coeff


# For visualisation 
def create_upsampled_grid(nside):
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
    longitudes = np.linspace(0, 360, max_lon_points, endpoint=False)  # Longitudes in degrees

    latitudes = create_latitude_array(nside)

    return np.meshgrid(longitudes, latitudes)


def create_latitude_array(nside):
    # HEALPix j-values for polar and equatorial regions
    j_north_south = np.arange(1, nside)  # j-values for polar region rings
    j_equatorial = np.arange(nside, 3 * nside + 1)  # j-values for equatorial region rings

    # Compute colatitudes
    colatitudes_north = np.arccos(1 - (j_north_south**2) / (3 * nside**2))
    colatitudes_equatorial = np.arccos(2 * (2 * nside - j_equatorial) / (3 * nside))

    # Fix: Compute southern latitudes correctly using the same formula
    colatitudes_south = np.arccos(-(1 - (j_north_south**2) / (3 * nside**2)))
    colatitudes_south = np.flip(colatitudes_south)

    # Convert colatitudes to latitudes (latitude = 90° - colatitude)
    latitudes_north = np.degrees(np.pi / 2 - colatitudes_north)
    latitudes_equatorial = np.degrees(np.pi / 2 - colatitudes_equatorial)
    latitudes_south = np.degrees(np.pi / 2 - colatitudes_south)  # Fix: Ensure smooth transition

    # Concatenate all latitudes in order: North → Equator → South
    latitudes = np.concatenate([latitudes_north, latitudes_equatorial, latitudes_south])

    return latitudes