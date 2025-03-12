import numpy as np
import jax.numpy as jnp
import jax
from functools import partial

from src.data_interpolation import create_latitude_array


def DFS(mp: jnp.array, fft_coeff: jnp.array) -> (jnp.array, jnp.array):
    south_part = jnp.flipud(mp)
    south_part = jnp.flip(south_part, axis=1)
    double_map = jnp.concatenate((mp, south_part), axis=0)

    double_map = interpolate_polar_rings(double_map)
    
    south_part = np.flipud(np.array(fft_coeff))
    south_part[:,1::2] *= (-1) # flip every odd wave number in the mirrored part by -1

    # double the fft coefficients
    n_rings = fft_coeff.shape[0]
    double_fft = np.zeros((2*n_rings+2, fft_coeff.shape[1]))
    double_fft[0] = np.fft.fft(double_map[0], n=fft_coeff.shape[1])
    double_fft[1:n_rings+1] = fft_coeff[:]
    double_fft[n_rings+1] = np.fft.fft(double_map[n_rings], n=fft_coeff.shape[1])
    double_fft[n_rings+2:] = south_part
    return double_map, double_fft


def interpolate_polar_rings(mp: jnp.array) -> jnp.array:
    nside = mp.shape[1] / 4
    n_rings = mp.shape[0] // 2

    latitudes = create_latitude_array(nside)


    # Calculate the values at the south pole
    def south_pole(fp: jnp.array, theta: jnp.array) -> jnp.array:
        return jnp.interp(90, -theta, fp)
    
    south_theta = jnp.concatenate((latitudes[-3:], -180 - jnp.flip(latitudes[-3:])))
    south_fp = jnp.concatenate((mp[n_rings-3:n_rings], mp[n_rings:n_rings+3])).T

    spole = partial(south_pole, theta=south_theta)
    south_pole_mp = jax.vmap(spole)(south_fp)


    # Calculate the values at the north pole
    def north_pole(fp: jnp.array, theta: jnp.array) -> jnp.array:
        return jnp.interp(90, theta, fp)
    
    north_theta = jnp.concatenate((jnp.flip(latitudes[:3]), 180 - latitudes[:3]))
    north_fp = jnp.concatenate((jnp.flip(mp[:3]), jnp.flip(mp[-3:]))).T

    npole = partial(north_pole, theta=north_theta)
    north_pole_mp = jax.vmap(npole)(north_fp)
    
    
    # Add the polar rings to the map
    double_map = np.zeros((mp.shape[0]+2, mp.shape[1]))
    double_map[0] = north_pole_mp
    double_map[1:n_rings+1] = mp[:n_rings]
    double_map[n_rings+1] = south_pole_mp
    double_map[n_rings+2:] = mp[n_rings:]


    return double_map

