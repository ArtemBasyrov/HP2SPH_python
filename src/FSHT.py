import numpy as np
import jax.numpy as jnp
import subprocess
import json

def preparation(bivar_coeffs: jnp.array) -> jnp.array:
    NSIDE = bivar_coeffs.shape[1]//4

    # expanding the coefficients in longitude to 4*NSIDE+1 
    X_coeff = np.zeros((4*NSIDE+1, 4*NSIDE+1), dtype=complex)
    neg_column = bivar_coeffs[:,0] # at k = -2*NSIDE, it's 0 index because of natural ordering of FFT
    X_coeff[:, :4*NSIDE] = bivar_coeffs
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

    X_pos_ell = X_sort[2*NSIDE:] # including 0 and positive ell = [0, 1, 2, ..., 2*NSIDE]
    X_neg_ell = X_sort[:2*NSIDE] # negative ell = [-2*NSIDE, ..., -2, -1]
    X_neg_ell = np.flip(X_neg_ell, axis=0) # [-1, -2, ..., -2*NSIDE]

    # create sel for odd and even k
    sel_even = (indx[1:]%2 == 0)
    sel_odd = ~sel_even

    # first row is zero j = 0
    g[0,1:] = 2* X_pos_ell[0,1:]* np.sqrt(1./np.pi)
    g[0,1:][sel_odd] = 1j*(X_pos_ell[1,1:] - X_neg_ell[0,1:])[sel_odd]*np.sqrt(1./np.pi)
    
    # first column is zero k = m = 0
    g[0,0] = X_pos_ell[0,0]*2 * np.sqrt(0.5/np.pi)
    g[1:,0] = (X_pos_ell[1:,0]+X_neg_ell[:,0]) * np.sqrt(0.5/np.pi)
    
    # everyhting inside the matrix except the zero row and column
    g_k_even = (X_pos_ell[1:,1:] + X_neg_ell[:,1:])* np.sqrt(1./np.pi) # k even
    g_k_odd = 1j*(X_pos_ell[1+1:,1:] - X_neg_ell[1:,1:])* np.sqrt(1./np.pi) # k odd

    g[1:,1:][:,sel_even] = g_k_even[:,sel_even]
    g[1:2*NSIDE,1:][:,sel_odd] = g_k_odd[:,sel_odd] # all odd m at l = lmax are zero
    
    return g


def call_Julia(g: jnp.array, scriptname: str) -> jnp.array:
    json_data = json.dumps({'real': g.real.tolist(), 'imag': g.imag.tolist()})

    result = subprocess.run(
        ["julia", scriptname],
        input=json_data,  # Pass JSON as input
        text=True,
        capture_output=True
    )

    def complex_decoder(obj):
        if "__complex__" in obj:
            return complex(obj["real"], obj["imag"])
        return obj
    
    output_array = json.loads(result.stdout, object_hook=complex_decoder)
    return output_array
    

def FSHT(bivar_coeffs: jnp.array) -> jnp.array:
    g = preparation(bivar_coeffs)

    output_array = call_Julia(g, scriptname='src/julia_sph.jl')
    C = np.array(output_array)

    return C


def convert_to_bivar_coeffs(g: jnp.array) -> jnp.array:
    # converting 2D array of g coefficients of Fourier-Chebyshev series 
    # into 2D array of bivariate Fourier coefficients

    NSIDE = (g.shape[1]-1)//4
    X_coeff = np.zeros((4*NSIDE+1, 4*NSIDE+1), dtype=complex)

    # m = 0
    X_pos_ell = g[:,0] * np.sqrt(2*np.pi) / 2
    X_coeff[2*NSIDE:,2*NSIDE] = X_pos_ell # including ell = 0
    X_coeff[:2*NSIDE,2*NSIDE] = np.flip(X_pos_ell[1:]) 
    
    # m != 0
    # rearange X into [0 ,-1, 1, -2, 2, ...] order along k
    indx = np.fft.fftfreq(4*NSIDE+1, d=1) * (4*NSIDE+1)
    indx = np.fft.fftshift(indx)

    #sel = np.argsort(np.abs(indx), kind='stable')
    #indx = indx[sel]

    g_m_neg = g[:,1::2] # [-1, -2, -3, ..., -2*NSIDE]
    g_m_pos = g[:,2::2] # [ 1,  2,  3, ...,  2*NSIDE]
    sel_even = (np.arange(1, 2*NSIDE+1)%2 == 0)
    sel_odd = ~sel_even

    # m > 0
    X_pos_ell = g_m_pos * np.sqrt(np.pi) / 2
    X_pos_ell[0, sel_odd] = 0 # odd m, ell = 0
    X_pos_ell[1:,sel_odd] = -1j*g_m_pos[:2*NSIDE,sel_odd] * np.sqrt(np.pi) / 2

    X_coeff[2*NSIDE:,2*NSIDE+1:] = X_pos_ell # including ell = 0
    X_coeff[:2*NSIDE,2*NSIDE+1:][:,sel_even] = np.flip(X_pos_ell[1:], axis=0)[:,sel_even]
    X_coeff[:2*NSIDE,2*NSIDE+1:][:,sel_odd] = -np.flip(X_pos_ell[1:], axis=0)[:,sel_odd]
    
    # m < 0
    g_m_neg = np.flip(g_m_neg, axis=1) # [-2*NSIDE , ..., -3, -2, -1]
    sel_odd = np.flip(sel_odd)
    sel_even = np.flip(sel_even)

    X_pos_ell = g_m_neg * np.sqrt(np.pi) / 2
    X_pos_ell[0, sel_odd] = 0 # odd m, ell = 0
    X_pos_ell[1:,sel_odd] = -1j*g_m_neg[:2*NSIDE,sel_odd] * np.sqrt(np.pi) / 2

    X_coeff[2*NSIDE:,:2*NSIDE] = X_pos_ell # including ell = 0
    X_coeff[:2*NSIDE,:2*NSIDE][:,sel_even] = np.flip(X_pos_ell[1:], axis=0)[:,sel_even]
    X_coeff[:2*NSIDE,:2*NSIDE][:,sel_odd] = -np.flip(X_pos_ell[1:], axis=0)[:,sel_odd]
    
    # de-expanding the coefficients in longitude to 4*NSIDE 
    bivar_coeff = np.zeros((4*NSIDE+1, 4*NSIDE), dtype=complex)
    neg_column = X_coeff[:,0] # m=-2*NSIDE
    bivar_coeff = X_coeff[:,:4*NSIDE]
    bivar_coeff[:,0] = 2*neg_column
    
    return bivar_coeff

def inverse_FSHT(alm: jnp.array) -> jnp.array:
    output_array = call_Julia(alm, scriptname='src/julia_sph_inverse.jl')
    bivar_coeffs = np.array(output_array)

    C = convert_to_bivar_coeffs(bivar_coeffs)

    return bivar_coeffs, C



