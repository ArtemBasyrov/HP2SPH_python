import numpy as np
import jax.numpy as jnp
import subprocess
import json

def preparation(bivar_coeffs: jnp.array) -> jnp.array:
    NSIDE = bivar_coeffs.shape[1]//4

    # expanding the coefficients in longitude to 4*NSIDE+1 
    X_coeff = np.zeros((4*NSIDE+1, 4*NSIDE+1), dtype=complex)
    neg_column = bivar_coeffs[:,0] # at k = -2*NSIDE, it's 0 index because of natural ordering of FFT
    X_coeff[:, 1:] = bivar_coeffs
    X_coeff[:,0] = 0.5*neg_column
    X_coeff[:,-1] = 0.5*neg_column

    # transform X into g array
    g = np.zeros((2*NSIDE+1, 4*NSIDE+1), dtype=complex) # size (p, 2p+1)

    # rearange X into [0 ,-1, 1, -2, 2, ...] order along k
    indx = np.fft.fftfreq(4*NSIDE+1, d=1) * (4*NSIDE+1)
    indx = np.fft.fftshift(indx)
    sel = np.argsort(np.abs(indx), kind='stable')
    indx = indx[sel]
    X_sort = X_coeff[:,sel]

    X_pos_ell = X_sort[2*NSIDE:] # including 0 and positive ell
    X_neg_ell = X_sort[:2*NSIDE]

    # create sel for odd and even k
    sel_even = (indx[1:]%2 == 0)
    sel_odd = ~sel_even

    # first row is zero j = 0
    g[0,1:] = (X_pos_ell[0,1:] + X_pos_ell[0,1:])* np.sqrt(1./np.pi)
    g[0,1:][sel_odd] = 0

    # first column is zero k = m = 0
    g[:,0] = X_pos_ell[:,0]*2 * np.sqrt(0.5/np.pi)

    # everyhting inside the matrix except the zero row and column
    g_k_even = (X_pos_ell[1:,1:] + X_neg_ell[:,1:])* np.sqrt(1./np.pi) # k even
    g_k_odd = 1j*(X_pos_ell[1:,1:] - X_neg_ell[:,1:])* np.sqrt(1./np.pi) # k odd

    g[1:,1:][:,sel_even] = g_k_even[:,sel_even]
    g[1:,1:][:,sel_odd] = g_k_odd[:,sel_odd]

    return g


def call_Julia(g: jnp.array) -> jnp.array:
    json_data = json.dumps({'real': g.real.tolist(), 'imag': g.imag.tolist()})

    result = subprocess.run(
        ["julia", "src/julia_sph.jl"],
        input=json_data,  # Pass JSON as input
        text=True,
        capture_output=True
    )
    
    output_array = json.loads(result.stdout)
    return output_array
    

def FSHT(bivar_coeffs: jnp.array) -> jnp.array:
    NSIDE = bivar_coeffs.shape[1]//4

    g = preparation(bivar_coeffs)

    output_array = call_Julia(g)

    # very suspicious move! [probably slow]
    # Extract real and imaginary parts into separate arrays
    C = np.array([el['re'] + 1j*el['im'] for el in output_array])

    return C



