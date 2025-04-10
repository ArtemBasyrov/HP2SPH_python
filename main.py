import healpy as hp
import numpy as np
import time
from astropy.io import fits

from src.data_interpolation import transform_healpix_to_grid, transform_grid_to_healpix
from src.double_fourier_sphere import DFS, DFS_inverse
from src.nuFFT import apply_nuFFT, inverse_nuFFT
from src.FSHT import FSHT, inverse_FSHT


def save_to_fits(data, filename):
    """
    Save a numpy array to a FITS file.
    
    Parameters:
    - data (numpy.ndarray): The data to save.
    - filename (str): The name of the FITS file to create.
    """
    # Separate real and imaginary parts
    data_real = np.real(data)
    data_imag = np.imag(data)

    # Save real and imaginary parts to FITS file
    hdu_real = fits.PrimaryHDU(data_real)
    hdu_imag = fits.ImageHDU(data_imag, name='IMAGINARY')
    hdul = fits.HDUList([hdu_real, hdu_imag])
    hdul.writeto(filename, overwrite=True)
    print(filename+' saved as FITS file')

def read_fits(filename):
    """
    Read a FITS file and return the data.
    
    Parameters:
    - filename (str): The name of the FITS file to read.
    """
    # Open the FITS file
    with fits.open(filename) as hdul:
        alm_real = hdul[0].data
        alm_imag = hdul['IMAGINARY'].data
        
        # Combine real and imaginary parts into a complex array
        alm = alm_real + 1j * alm_imag
        
    return alm


def forward(mp: np.array) -> np.array:
    """
    Perform the forward transformation from HEALPix map to spherical harmonics.
    
    Parameters:
    - mp (numpy.ndarray): The HEALPix map data.
    
    Returns:
    - alm (numpy.ndarray): The spherical harmonics coefficients.
    """
    # Perform the forward transformation

    assert mp.shape[0] == 3, "Input map must have I, Q, U components" # temporary limit, Q and U do nothing
    NSIDE = hp.get_nside(mp[0])
    lmax = 2*NSIDE
    upsampled_map, fft_coeff = transform_healpix_to_grid(mp[0])

    start = time.time()
    _, fft_coeff_DFS = DFS(upsampled_map, fft_coeff)
    print('DFS time:', time.time() - start)

    start = time.time()
    fft_lat_nufft = apply_nuFFT(fft_coeff_DFS)
    print('nuFFT time:', time.time() - start)

    start = time.time()
    alm = FSHT(fft_lat_nufft) 
    print('FSHT time:', time.time() - start)

    save_to_fits(alm, 'alm_array_cg_{0}.fits'.format(NSIDE))

    return alm


def inverse(alm: np.array) -> np.array:
    """
    Perform the inverse transformation from spherical harmonics to HEALPix map.
    
    Parameters:
    - alm (numpy.ndarray): The spherical harmonics coefficients.
    
    Returns:
    - mp (numpy.ndarray): The HEALPix map data.
    """
    
    NSIDE = (alm.shape[1]-1)//4
    bivar_coeff, C = inverse_FSHT(alm) 
    fft_lat = inverse_nuFFT(bivar_coeff)
    fft_coeff = DFS_inverse(fft_lat)
    mp = transform_grid_to_healpix(fft_coeff, fft_coeff)

    return mp


if __name__ == "__main__":
    pass