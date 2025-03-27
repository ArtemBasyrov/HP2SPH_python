import jax
import jax.numpy as jnp
import numpy as np
import finufft

from HP2SPH_python.src.data_interpolation import create_latitude_array

def apply_nuFFT(mp: jnp.array) -> jnp.array: 
    nside = mp.shape[1] // 4

    # Calculate the latitude values for the upsampled map
    latitudes = create_latitude_array(nside)
    DFT_upsampled_lat = np.zeros(len(latitudes)*2+2)
    DFT_upsampled_lat[0] = 90
    DFT_upsampled_lat[1:len(latitudes)+1] = latitudes
    DFT_upsampled_lat[len(latitudes)+1] = -90
    DFT_upsampled_lat[len(latitudes)+2:] = -180 + latitudes

    # transform to [-pi, pi) range
    DFT_upsampled_lat = DFT_upsampled_lat * np.pi / 180 + np.pi/2

    # setting up the nufft calculations
    mp = mp.astype(complex)
    plan = finufft.Plan(1, (4*nside+1,), n_trans=4*nside) # allows to reuse the same plan for all latitudes
    plan.setpts(DFT_upsampled_lat)
    fft_lat = plan.execute(mp.T) # batch accelerated calcualtion

    return fft_lat.T


def inverse_nuFFT(fft_lat: jnp.array) -> jnp.array:
    """
    Perform inverse NUFFT (Type-2) to reconstruct signal at non-uniform latitudes.

    Parameters:
    - fft_lat (jnp.array): Fourier coefficients from uniform frequency space.

    Returns:
    - jnp.array: Reconstructed signal at non-uniform latitude samples.

    Note:
    [NB!] Probably this function is useless because I will have bivariate Fourier coefficients,
    which I will need to transform back into grid space. Not individual latitude and
    longitude Fourier coefficients.
    """
    nside = fft_lat.shape[1] // 4
    
    # Calculate the latitude values for the upsampled map
    latitudes = create_latitude_array(nside)
    DFT_upsampled_lat = np.zeros(len(latitudes)*2+2)
    DFT_upsampled_lat[0] = 90
    DFT_upsampled_lat[1:len(latitudes)+1] = latitudes
    DFT_upsampled_lat[len(latitudes)+1] = -90
    DFT_upsampled_lat[len(latitudes)+2:] = -180 + latitudes

    # transform to [-pi, pi) range
    DFT_upsampled_lat = DFT_upsampled_lat * np.pi / 180 + np.pi/2

    fft_lat = np.array(fft_lat)

    # setting up the nufft calculations
    M = 4 * nside + 1  # Number of Fourier modes
    N = len(latitudes)*2+2 # Number of non-uniform latitude samples
    plan = finufft.Plan(2, (M,), n_trans=4*nside) # allows to reuse the same plan for all latitudes
    plan.setpts(DFT_upsampled_lat)
    mp_reconstructed = plan.execute(fft_lat.T) # batch accelerated calcualtion

    mp_reconstructed = mp_reconstructed.T / N # normalise the output

    return mp_reconstructed.real 


