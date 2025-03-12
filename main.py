import healpy as hp
import numpy as np
import time
from astropy.io import fits

def main():
    filename = 'mapname.fits'
    mp = hp.read_map(filename, field=(0,1,2))

    NSIDE = 16
    lmax = 2*NSIDE

    import src.data_interpolation as data_int
    upsampled_map, fft_coeff = data_int.transform_healpix_to_grid(mp[0])


    from src.double_fourier_sphere import DFS
    start = time.time()
    DFS_upsampled_map, fft_coeff_DFS = DFS(upsampled_map, fft_coeff)
    print('DFS time:', time.time() - start)

    from src.nuFFT import apply_nuFFT
    start = time.time()
    fft_lat_nufft = apply_nuFFT(DFS_upsampled_map)
    print('nuFFT time:', time.time() - start)

    import src.FSHT as FSHT
    start = time.time()
    # prepare the bivarivative fourier decomposition coefficients
    bivar_coeffs = fft_coeff_DFS[:4*NSIDE+1] * fft_lat_nufft
    alm = FSHT.FSHT(bivar_coeffs) # healpy like
    print('FSHT time:', time.time() - start)

    # Separate real and imaginary parts
    alm_real = np.real(alm)
    alm_imag = np.imag(alm)

    # Save real and imaginary parts to FITS file
    hdu_real = fits.PrimaryHDU(alm_real)
    hdu_imag = fits.ImageHDU(alm_imag, name='IMAGINARY')
    hdul = fits.HDUList([hdu_real, hdu_imag])
    hdul.writeto('alm_array_{0}.fits'.format(NSIDE), overwrite=True)
    print('alm array saved to FITS file')

if __name__ == "__main__":
    main()