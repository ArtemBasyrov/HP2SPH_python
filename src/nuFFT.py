import jax
import jax.numpy as jnp
import numpy as np
import finufft

from src.data_interpolation import create_latitude_array

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
    plan = finufft.Plan(1, [4*nside+1], n_trans=4*nside) # allows to reuse the same plan for all latitudes
    plan.setpts(DFT_upsampled_lat)
    fft_lat = plan.execute(mp.T) # batch accelerated calcualtion

    return fft_lat.T