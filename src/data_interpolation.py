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
    i = np.arange(1, num_rings+1)
    
    # find the ring sizes
    ring_sizes = np.full(num_rings, 4 * nside)
    ring_sizes[:nside] = 4 * i[:nside]
    ring_sizes[3 * nside:] = 4 * (4 * nside - i[3 * nside:])

    # find the start and end indices
    start_indices = jnp.cumsum(ring_sizes) - ring_sizes
    end_indices = start_indices + ring_sizes - 1
    return jnp.vstack((start_indices, end_indices, i)).T



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

    ring_info = get_ring_indices(nside) # [start_id, end_id, ring_id]
    upsampled_data = jnp.empty((n_rings, 4 * nside))
    fft_coeff = np.zeros((n_rings, 4 * nside), dtype=complex)
    
    # Define function for vectorized FFT processing
    def process_equatorial_ring(ring_data):
        fft_coeffs = jnp.fft.fft(ring_data, n=4 * nside, norm='forward')
        return fft_coeffs
    
    def process_polar_ring(ring_data):
        num_pts = len(ring_data)
        coeffs = jnp.fft.fft(ring_data, n=num_pts, norm='forward') 
        k_vals = np.fft.fftfreq(num_pts) * num_pts
        phase_shift = jnp.exp(-1j * jnp.pi * k_vals / (2 * nside))
        corrected_coeffs = coeffs * phase_shift

        # this padding correctly accounts for fft frequencies position in the array
        mid = num_pts // 2
        coeffs_padded = np.zeros(4 * nside, dtype=complex)
        coeffs_padded[:mid] = corrected_coeffs[:mid]  # Positive frequencies
        coeffs_padded[-mid:] = corrected_coeffs[-mid:]  # Negative frequencies

        return coeffs_padded * num_pts / (4 * nside)
    
    def inverse_fft(fft_coeffs):
        return jnp.fft.ifft(fft_coeffs, n=4 * nside, norm='forward').real
    

    # Diving data into rings
    #start_time0 = time.time()
    ring_data = [healpix_map[start:end+1] for start, end, nring in ring_info]
    #print(f"Ring selection execution time: {time.time() - start_time0:.6f} seconds")

    
    # Processing of equatorial rings
    #start_time0 = time.time()
    fft_coeff[nside-1: 3 * nside] = jax.vmap(process_equatorial_ring)(jnp.array(ring_data[nside-1: 3 * nside]))
    shift = np.exp(-1j * np.pi * np.arange(4 * nside) / (2 * nside))
    fft_coeff[nside-1: 3 * nside:2] *= shift
    #print(f"Equatorial ring execution time: {time.time() - start_time0:.6f} seconds")

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
    #start_time0 = time.time()
    for i in range(nside-1):
        fft_coeff[i] = process_polar_ring(ring_data[i])
        fft_coeff[n_rings-1 - i] = process_polar_ring(ring_data[n_rings-1 - i])
    #print(f"Polar ring execution time: {time.time() - start_time0:.6f} seconds")

    # Inverse FFT
    #start_time0 = time.time()
    upsampled_data = jax.vmap(inverse_fft)(fft_coeff)
    #print(f"Inverse FFT execution time: {time.time() - start_time0:.6f} seconds")

    end_time = time.time()
    print(f"data_interpolation execution time: {end_time - start_time:.6f} seconds")
    return upsampled_data, fft_coeff





# The inverse transformation
def transform_grid_to_healpix(grid_data: jnp.array, fft_coeff: jnp.array = None) -> jnp.array:
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
    ring_info = get_ring_indices(nside) # [start_id, end_id, ring_id]
    
    # find the ring sizes
    i = np.arange(1, n_rings + 1)
    ring_sizes = np.full(n_rings, 4 * nside)
    ring_sizes[:nside] = 4 * i[:nside]
    ring_sizes[3 * nside:] = 4 * (4 * nside - i[3 * nside:])

    healpix_map = np.empty(12 * nside**2)
    

    # Define function for vectorized FFT processing
    def calc_fft(ring_data):
        fft_coeffs = jnp.fft.fft(ring_data, n=4 * nside, norm='forward')
        return fft_coeffs
    
    def process_polar_ring(fft_coeff, num_pts):
        mid = num_pts // 2
        corrected_coeffs_back = np.zeros(num_pts, dtype=complex)
        corrected_coeffs_back[:mid] = fft_coeff[:mid]
        corrected_coeffs_back[-mid:] = fft_coeff[-mid:]

        fft_coeffs = jnp.fft.ifft(corrected_coeffs_back, n=num_pts, norm='forward').real
        return fft_coeffs *(4*nside) / num_pts
    
    def process_equatorial_ring(fft_coeffs):
        return jnp.fft.ifft(fft_coeffs, n=4 * nside, norm='forward').real
    
    if fft_coeff is None:
        fft_coeff = np.zeros((n_rings, 4 * nside), dtype=complex)
        fft_coeff[:] = jax.vmap(calc_fft)(grid_data) # Processing all ring with process_equatorial_ring

    # Applying equatorial shift
    shift = jnp.exp(+1j * jnp.pi * jnp.arange(4 * nside) / (2 * nside))
    fft_coeff[nside-1: 3 * nside: 2] *= shift

    eq_rings = jax.vmap(process_equatorial_ring)(fft_coeff[nside-1: 3 * nside]) 
    start_id, _, _ = ring_info[nside-1]
    _, end_id, _ = ring_info[3 * nside-1]
    healpix_map[start_id:end_id+1] = jnp.concatenate(eq_rings)


    # Applying the polar shift
    fft_coeff[:nside-1] *= shift # we can apply the shift to all rings at once
    fft_coeff[3*nside:] *= shift
    for i in range(nside-1):
        num_pts = ring_sizes[i]

        start_id, end_id, _ = ring_info[i]
        healpix_map[start_id:end_id+1] = process_polar_ring(fft_coeff[i], num_pts) 

        start_id, end_id, _ = ring_info[-1 - i]
        healpix_map[start_id:end_id+1] = process_polar_ring(fft_coeff[-1 - i], num_pts)

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
    longitudes = np.linspace(0, 360, max_lon_points, endpoint=False)  # Longitudes in degrees

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