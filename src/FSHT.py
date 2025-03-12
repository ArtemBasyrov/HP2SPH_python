import numpy as np
import jax.numpy as jnp
import subprocess
import json

def preparation(bivar_coeffs: jnp.array) -> jnp.array:
    NSIDE = bivar_coeffs.shape[1]//4

    # expanding the coefficients in longitude to 4*NSIDE+1 
    X = np.zeros((4*NSIDE+1, 4*NSIDE+1), dtype=complex)
    X[:, 1:] = bivar_coeffs
    X[:,0] = 0.5*bivar_coeffs[:,-1]
    X[:,-1] = 0.5*bivar_coeffs[:,0]

    # transform X into g array
    g = np.zeros((2*NSIDE+1, 4*NSIDE+1), dtype=complex)
    X_pos = X[2*NSIDE:]
    X_neg = np.conj(np.flip(X[:2*NSIDE+1]))
    g[:,0::2] = X_pos[:,0::2] + X_neg[:,0::2] # k even
    g[:,1::2] = X_pos[:,1::2] - X_neg[:,1::2] # k odd

    # sort the g array
    k = np.arange(-2*NSIDE, 2*NSIDE+1, dtype=float)
    k[k < 0] = k[k < 0] + 0.1
    idx = np.argsort(np.abs(k))
    g = g[:,idx]

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



