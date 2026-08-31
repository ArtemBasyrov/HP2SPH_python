import logging
import time

import numpy as np
from scipy.special import gammaln

from ._threads import default_workers, run_blocks

logger = logging.getLogger(__name__)

# Rows per block when applying the per-ring phase reference. Chosen so the temporary
# stays a few MB at any nside rather than scaling with the map.
_PHASE_BLOCK = 64

# Byte budget for one equatorial FFT call's transient. Rows per call are derived from
# it, so the temporary is a few MB at any nside instead of scaling with the map -- and
# so small maps, where the transient was never the problem, still go in a single call.
_FFT_BLOCK_BYTES = 8 << 20

# Below this many rings the thread hand-off costs more than splitting the rows saves.
# Measured (threaded against serial, same process, t_min of 3): 0.55x at nside 32,
# 0.88x at 64, 1.01x at 128, 1.35x at 256, 1.87x at 512, 2.44x at 1024, 3.01x at 2048.
# nside 128 is 4*128-1 = 511 rings, which is where the split starts paying.
_MIN_THREADED_RINGS = 511


def get_ring_indices(nside: int) -> np.ndarray:
    """
    Compute the indices of the pixels in each equatorial ring.

    Parameters:
        nside (int): HEALPix resolution parameter.

    Returns:
        ring_indices (list): List of pixel indices for each ring.
    """
    num_rings = 4 * nside - 1
    i = np.arange(1, num_rings + 1)
    ring_sizes = ring_pixel_counts(nside)

    # find the start and end indices
    start_indices = np.cumsum(ring_sizes) - ring_sizes
    end_indices = start_indices + ring_sizes - 1
    return np.vstack((start_indices, end_indices, i)).T


def npix2nside(npix: int) -> int:
    """HEALPix pixel count -> nside, with a validity check.

    Replaces the single ``jax_healpy.npix2nside`` call this module used to make --
    the only reason ``jax_healpy`` was a dependency at all.
    """
    nside = int(round(np.sqrt(npix / 12.0)))
    if 12 * nside**2 != npix:
        raise ValueError(f"{npix} is not a valid HEALPix pixel count")
    return nside


def ring_pixel_counts(nside: int) -> np.ndarray:
    """Number of HEALPix pixels in each RING-ordered ring (north -> south)."""
    n_rings = 4 * nside - 1
    i = np.arange(1, n_rings + 1)
    sizes = np.full(n_rings, 4 * nside)
    sizes[:nside] = 4 * i[:nside]
    sizes[3 * nside :] = 4 * (4 * nside - i[3 * nside :])
    return sizes


def ring_alias_target(nside: int) -> (np.ndarray, np.ndarray, np.ndarray):
    """Where each longitude mode is actually MEASURED on each ring.

    A ring of ``npix`` pixels samples the longitude field at ``npix`` points, so it does
    not measure mode ``m``: it measures the whole alias family of ``m``. Writing
    ``b = m mod npix`` folded into ``[-npix//2, npix//2)``, what
    ``transform_healpix_to_grid`` puts in slot ``b`` of ring ``r`` is

        M[r, b] = sum over {m : m == b (mod npix)} of c_m(theta_r) exp(i (m - b) phi0_r)

    -- the exponential because the phi=0 referencing at the end of that function
    multiplies each slot by ``exp(-i b phi0)``, using the SLOT index rather than the mode
    that contributed, so an aliased contribution also arrives with the wrong longitude
    phase. Verified against the pipeline to ~1e-12 against O(10) coefficients.

    The pipeline then asserts ``M[r, b] = c_b(theta_r)``, which is false whenever another
    member of the family carries amplitude. This is the analysis-side mirror of the
    synthesis-side fold in ``transform_grid_to_healpix.process_polar_ring``.

    Returns ``(target, phase, resolved)``, each ``(n_rings, 4*nside)`` in NATURAL
    (fftshifted) longitude order -- column ``j`` is mode ``m = j - 2*nside``, the order
    ``DFS`` and ``apply_nuFFT`` use:

    * ``target[r, j]`` -- the slot mode ``j`` folds onto,
    * ``phase[r, j]``  -- the ``exp(i (m - b) phi0)`` it arrives with,
    * ``resolved[r, j]`` -- whether the ring produces slot ``j`` at all. The Nyquist slot
      ``-npix//2`` counts as produced: under the fold it is a genuine constraint on the
      ``+-npix//2`` sum, not the mis-assignment the zero-padding makes of it.

    This describes the ONE-SIDED Nyquist layout, so a consumer of this plan must call
    ``transform_healpix_to_grid(..., nyquist_split=False)``. With the split on, each
    Nyquist family scatters half onto ``-npix//2`` and half onto ``+npix//2`` with two
    different phases, which the single-target ``(target, phase)`` form cannot express.
    """
    sizes = ring_pixel_counts(nside)
    n_lon = 4 * nside
    phi0 = ring_first_longitude(nside)
    m = np.arange(n_lon) - n_lon // 2
    mid = (sizes // 2)[:, None]
    b = (m[None, :] + mid) % sizes[:, None] - mid
    target = b + n_lon // 2
    phase = np.exp(1j * (m[None, :] - b) * phi0[:, None])
    j = np.arange(n_lon)[None, :]
    resolved = (j >= n_lon // 2 - mid) & (j < n_lon // 2 + mid)
    return target, phase, resolved


def mode_pole_envelope(
    nside: int, spin: int = 0, lmax: int = None, rings: slice = None
) -> np.ndarray:
    """Largest ``|c_m(theta_r)|`` a band-limited spin-s field can carry, as a fraction of
    that mode's own peak over latitude.

    A spin-s mode-m latitude profile is the Wigner d function ``d^l_{-s,m}``, i.e.
    ``sin^|m+s|(theta/2) cos^|m-s|(theta/2) P^(|m-s|,|m+s|)(cos theta)``. Near a pole the
    polynomial is NOT O(1): the uniform asymptotic is Bessel, ``J_a(l*theta)`` with
    ``a = |m+s|`` at the north pole and ``a = |m-s|`` at the south, so for a small
    argument the amplitude is ``(l*theta/2)^a / a!`` -- THE SCALE IS ``l*theta``, NOT
    ``theta``. Dropping the ``l`` factor underestimates by six orders at the band edge
    (measured at nside 32, l = 63, m = +2: the innermost ring carries 1.6e-2 of the
    mode's peak; ``(theta/2)^4`` predicts 2.6e-8, ``(l*theta/2)^4/4!`` predicts 1.7e-2).

    ``lmax`` defaults to the pipeline band ``2*nside``, i.e. the worst case over the band.
    ``rings`` restricts the result to a slice of rings, so a caller working in blocks
    never has to hold the whole ``(4*nside-1, 4*nside)`` array.
    """
    if lmax is None:
        lmax = 2 * nside
    theta = np.deg2rad(90.0 - create_latitude_array(nside))
    if rings is not None:
        theta = theta[rings]
    theta = theta[:, None]
    n_lon = 4 * nside
    m = np.arange(n_lon) - n_lon // 2
    a = np.abs(m + spin).astype(float)[None, :]
    b = np.abs(m - spin).astype(float)[None, :]
    log_north = np.log(np.maximum(lmax * theta / 2, np.finfo(float).tiny))
    log_south = np.log(np.maximum(lmax * (np.pi - theta) / 2, np.finfo(float).tiny))
    north = np.exp(np.minimum(a * log_north - gammaln(a + 1), 0.0))
    south = np.exp(np.minimum(b * log_south - gammaln(b + 1), 0.0))
    return np.minimum(north, south)


def ring_fold_plan(
    nside: int, spin: int = 0, tol: float = 1e-2, lmax: int = None
) -> (np.ndarray, np.ndarray, np.ndarray):
    """The forward latitude operator's longitude layout: fold + selective zero-assertion.

    The zero-padding the pipeline does today is exactly equivalent to "fold, AND assert
    ``c_m(theta_r) = 0`` for every mode the ring does not resolve" -- when the other
    members of an alias family vanish the fold degenerates to the identity. Those zero
    assertions are not a bug, they are what makes the latitude least squares WELL
    CONDITIONED. Dropping them all -- which is what folding without them does, and what
    the superseded mask fix did -- leaves the polar caps constrained only through the
    alias sums: the system stays consistent -- the true solution satisfies it to ~7e-13 --
    but becomes so ill conditioned that LSMR is still 10% off after 40000 iterations.

    Almost all of the assertions are true to many digits, because an unresolved mode is
    negligible on a polar ring unless it decays slowly into the nearby pole, which happens
    only for ``|m|`` near ``|spin|``. So keep the assertion where ``mode_pole_envelope``
    says the mode is below ``tol``, RELAX it where it is not, and FOLD so the retained
    data equations account exactly for the relaxed content aliasing into them.

    Returns ``(target, phase, keep)``. ``keep[r, j]`` False marks an entry dropped from
    the fit -- an unresolved mode with non-negligible amplitude there; its content is
    still carried onto the ring's own bin by ``target``/``phase``. A trusted unresolved
    mode is its own target, so it asserts ``c_m = 0`` in place (the data is already 0
    there, from the zero-padding).

    With an empty relax set this is the identity, i.e. the current unmasked path
    bit-for-bit.
    """
    target, phase, resolved = ring_alias_target(nside)
    relax = (~resolved) & (mode_pole_envelope(nside, spin, lmax) > tol)
    trusted = (~resolved) & ~relax
    j = np.broadcast_to(np.arange(target.shape[1]), target.shape)
    return np.where(trusted, j, target), np.where(trusted, 1.0, phase), ~relax


def ring_first_longitude(nside: int) -> np.ndarray:
    """Longitude (radians) of the first pixel in each RING-ordered HEALPix ring.

    HEALPix rings are not aligned to phi=0: each ring's first pixel sits half a
    pixel in (phi = pi/npix) for the polar rings, while the equatorial rings
    alternate between phi = pi/(4*nside) (a half equatorial pixel) and phi = 0.
    To reference every ring's longitude Fourier coefficients to a common phi=0
    origin (the convention healpy's a_lm use), each ring's mode-m coefficient
    must be multiplied by exp(-i * m * phi_first). Getting this wrong leaves an
    m-dependent longitude phase in the output alm. Matches ``hp.pix2ang`` exactly.
    """
    n_rings = 4 * nside - 1
    phi0 = np.zeros(n_rings)
    for r in range(n_rings):
        i = r + 1  # ring number 1 .. 4*nside-1 (north -> south)
        if i < nside:  # north polar cap
            phi0[r] = np.pi / (4 * i)
        elif i <= 3 * nside:  # equatorial belt
            phi0[r] = np.pi / (4 * nside) if (i - nside) % 2 == 0 else 0.0
        else:  # south polar cap
            phi0[r] = np.pi / (4 * (4 * nside - i))
    return phi0


def transform_healpix_to_grid(
    healpix_map: np.ndarray, map_rows: int = None, nyquist_split: bool = True
) -> (np.ndarray, np.ndarray):
    start_time = time.time()
    """
    Step 1: Transform a HEALPix map into a tensor product latitude-longitude grid,
    correctly handling shifted equatorial rings.
    
    Parameters:
        healpix_map (ndarray): HEALPix map data.
    
    Returns:
        upsampled_data (ndarray): Data mapped to the structured grid. With ``map_rows``
            set to ``k``, only the first and last ``k`` rings, stacked into a
            ``(2*k, 4*nside)`` array -- what the pole fill needs and nothing else.
        fft_coeff (ndarray): the per-ring longitude Fourier coefficients.

    ``nyquist_split`` (default True) splits each polar ring's Nyquist coefficient
    equally between ``m = -npix/2`` and ``m = +npix/2`` instead of assigning all of
    it to the negative slot. A ring of ``npix`` points cannot tell the two apart, so
    the one-sided assignment turns a cosine into a complex exponential and injects a
    spurious sine of the same amplitude; the even split is the assignment consistent
    with a real map. Set it False when the consumer models the ring aliasing itself
    (:func:`ring_alias_target` and the fold plans built on it describe the one-sided
    layout). Either setting round-trips exactly through
    :func:`transform_grid_to_healpix`, which folds both slots back onto the same bin.
    """
    healpix_map = np.asarray(healpix_map)
    nside = npix2nside(healpix_map.shape[0])
    n_rings = 4 * nside - 1

    ring_info = get_ring_indices(nside)  # [start_id, end_id, ring_id]
    fft_coeff = np.zeros((n_rings, 4 * nside), dtype=complex)

    def process_polar_ring(ring_data):
        num_pts = len(ring_data)
        # Each polar ring has a different length, so the polar rings cannot be
        # batched into one array FFT the way the equatorial belt is; they are
        # transformed one at a time. numpy's FFT on these small arrays is
        # ~microseconds, so ~2*nside of them are cheap.
        coeffs = np.fft.fft(ring_data, n=num_pts, norm="forward")

        # this padding correctly accounts for fft frequencies position in the array
        mid = num_pts // 2
        coeffs_padded = np.zeros(4 * nside, dtype=complex)
        coeffs_padded[:mid] = coeffs[:mid]  # Positive frequencies
        coeffs_padded[-mid:] = coeffs[-mid:]  # Negative frequencies

        # The ring's Nyquist bin measures the SUM of the m = +mid and m = -mid
        # content: at num_pts samples the two exponentials are the same function.
        # numpy parks it at index mid, i.e. frequency -mid, so the slice above hands
        # all of it to m = -mid and leaves m = +mid empty -- which asserts a complex
        # exponential where the truth is a cosine, and injects a spurious
        # sin(mid*phi) of equal amplitude. Half in each slot is the assignment
        # consistent with a real map. This is the whole of the anti-Hermitian part
        # of the post-NUFFT array on a smooth sky: it drops from ~1e-4 relative to
        # ~4e-15 at nside 128-512, and buys 1.22-1.34x in the 0.75-0.875 lmax band.
        if nyquist_split:
            coeffs_padded[-mid] *= 0.5
            coeffs_padded[mid] = coeffs_padded[-mid]

        # NB: do NOT rescale by num_pts/(4*nside). With norm='forward' the FFT
        # coefficients are already the true longitude Fourier coefficients of the
        # ring (DC = ring mean), independent of how many pixels the ring has.
        # Zero-padding to 4*nside only extends the (empty) high-frequency band,
        # so the populated coefficients must be left untouched. The old factor
        # shrank every polar-ring coefficient by i/nside, which cancelled against
        # the inverse (so round trips stayed exact) but fed the wrong Fourier
        # coefficients to DFS/nuFFT/FSHT -- the dominant forward-alm error.
        return coeffs_padded

    # A spin field (Q + iU) is carried as a complex map; an intensity (I) map is
    # real. Keep complex content when present, but drop the (zero) imaginary part
    # for a real input so the scalar path stays bit-identical to before.
    is_complex = np.iscomplexobj(healpix_map)

    # Dividing data into rings
    ring_data = [healpix_map[start : end + 1] for start, end, nring in ring_info]

    # Every stage below writes disjoint row ranges of ``fft_coeff``, so splitting the
    # rows across threads gives exactly the serial answer. numpy's pocketfft releases
    # the GIL, so the split scales; ``workers`` is 1 below the crossover.
    workers = default_workers(n_rings) if n_rings >= _MIN_THREADED_RINGS else 1

    # Processing of equatorial rings. Every equatorial ring has 4*nside pixels, so the
    # belt is a batched FFT over the last axis rather than a loop over rings.
    n_belt = 2 * nside + 1

    def equatorial_block(lo, hi):
        # In sub-blocks, because stacking the whole belt and transforming it in one
        # call holds two arrays the size of ``fft_coeff`` itself -- 1.1 GB of transient
        # at nside 2048 -- and with the rows split across threads every worker would
        # hold its own. pocketfft transforms each row independently, so a shorter batch
        # is bit-identical to the corresponding slice of a longer one.
        step = max(1, _FFT_BLOCK_BYTES // (4 * nside * 16))
        for a in range(lo, hi, step):
            b = min(a + step, hi)
            fft_coeff[nside - 1 + a : nside - 1 + b] = np.fft.fft(
                np.array(ring_data[nside - 1 + a : nside - 1 + b]),
                n=4 * nside,
                axis=-1,
                norm="forward",
            )

    run_blocks(equatorial_block, n_belt, workers)

    # Processing of polar rings. The rings are ragged, so this stays a loop over
    # individual FFTs; what is threaded is a contiguous BLOCK of rings per worker,
    # since one ring is far too little work to hand off.
    def polar_block(lo, hi):
        for i in range(lo, hi):
            fft_coeff[i] = process_polar_ring(ring_data[i])
            fft_coeff[n_rings - 1 - i] = process_polar_ring(ring_data[n_rings - 1 - i])

    run_blocks(polar_block, nside - 1, workers)

    # Reference every ring's coefficients to a common phi=0 origin. Each ring's
    # first pixel is offset by phi_first, so its mode-m FFT coefficient carries a
    # spurious exp(+i*m*phi_first); divide it out with exp(-i*m*phi_first). m is
    # the SIGNED frequency (numpy FFT order), so use fftfreq, not arange.
    m_signed = np.fft.fftfreq(4 * nside) * (4 * nside)
    phi0 = ring_first_longitude(nside)

    # Blocked, because the full phase array is the same size as ``fft_coeff`` itself and
    # ``np.outer`` builds a second one of the same size before ``np.exp`` does. At nside
    # 1024 that pair is 0.4 GB of transient for a multiply that is done row by row
    # anyway; at nside 2048 it is 1.6 GB.
    def phase_block(lo, hi):
        for a in range(lo, hi, _PHASE_BLOCK):
            b = min(a + _PHASE_BLOCK, hi)
            fft_coeff[a:b] *= np.exp(-1j * np.outer(phi0[a:b], m_signed))

    run_blocks(phase_block, n_rings, workers)

    # Back to map space. ``map_rows = k`` returns only the first and last k rings,
    # stacked, and transforms just those: the only consumer of the map is the polar
    # pole fill, which reads exactly that many rows from each end. The full inverse FFT
    # is 72 ms and 0.27 GB at nside 1024 against 0.19 ms for twelve rows, and the subset
    # is bit-identical to the corresponding slice of the whole transform.
    if map_rows is None:
        source = fft_coeff
    else:
        if 2 * map_rows > n_rings:
            raise ValueError(f"map_rows={map_rows} exceeds half of {n_rings} rings")
        source = np.concatenate((fft_coeff[:map_rows], fft_coeff[-map_rows:]))
    upsampled_data = np.fft.ifft(source, n=4 * nside, axis=-1, norm="forward")
    if not is_complex:
        upsampled_data = upsampled_data.real

    logger.debug(
        "transform_healpix_to_grid(nside=%d): %.6f s", nside, time.time() - start_time
    )
    return upsampled_data, fft_coeff


# The inverse transformation
def transform_grid_to_healpix(
    grid_data: np.ndarray, fft_coeff: np.ndarray = None, real_output: bool = True
) -> np.ndarray:
    """
    Transform a tensor product latitude-longitude grid into a HEALPix map,
    correctly handling shifted rings.

    Parameters:
        grid_data (ndarray): HEALPix map data.
        real_output (bool): take the real part of each ring (the intensity / I
            path, default). Set False to keep a complex (Q + iU) spin field --
            the inverse of carrying a complex map through the forward transform.

    Returns:
        healpix_map (ndarray): Data mapped to the structured grid.
    """

    # get general info
    nside = fft_coeff.shape[1] // 4
    ring_info = get_ring_indices(nside)  # [start_id, end_id, ring_id]

    ring_sizes = ring_pixel_counts(nside)

    map_dtype = float if real_output else complex
    healpix_map = np.empty(12 * nside**2, dtype=map_dtype)

    def _maybe_real(arr):
        return arr.real if real_output else arr

    def process_polar_ring(fft_coeff, num_pts):
        # ALIAS (fold), do not truncate. A ring of num_pts pixels samples the
        # longitude field at num_pts points, so mode m lands in bin m % num_pts:
        # the correct pixel values are sum_m c_m exp(i m phi_j), which is the FOLD
        # of the 4*nside-wide spectrum onto num_pts bins. Simply DROPPING
        # |m| >= num_pts//2 (what this did) throws that content away instead.
        #
        # For a spectrum produced by ``transform_healpix_to_grid`` the two are
        # bit-identical -- the forward zero-pads, so every folded-in entry is either
        # exactly 0 or the other half of a split Nyquist bin, which this add.at sums
        # back -- which is why the round trip never noticed. They differ when
        # the spectrum comes from the SYNTHESIS side (``main.backward`` /
        # ``spin_transform.backward_spin``), where the high-|m| entries carry real
        # signal. For a SPIN field that is the whole ball game: |m| = |spin| is O(1)
        # AT the pole and is exactly the Nyquist of the innermost 4-pixel ring (the
        # same asymmetry ``ring_fold_plan`` handles on the analysis side), and
        # truncating it left the innermost polar rings ~100% wrong.
        m_signed = np.rint(np.fft.fftfreq(len(fft_coeff)) * len(fft_coeff)).astype(int)
        corrected_coeffs_back = np.zeros(num_pts, dtype=complex)
        np.add.at(corrected_coeffs_back, m_signed % num_pts, np.asarray(fft_coeff))

        # One FFT per ring, for the same ragged-length reason as the forward.
        fft_coeffs = _maybe_real(
            np.fft.ifft(corrected_coeffs_back, n=num_pts, norm="forward")
        )
        # Mirror of the forward change: no num_pts/(4*nside) rescaling. The
        # forward FFT (norm='forward') already carries the 1/num_pts, so the
        # ifft (also norm='forward') inverts it exactly with no extra factor.
        return fft_coeffs

    if fft_coeff is None:
        # Every row of the grid has 4*nside samples, so this is one batched FFT.
        fft_coeff = np.fft.fft(
            np.asarray(grid_data), n=4 * nside, axis=-1, norm="forward"
        )

    # Undo the phi=0 referencing applied in the forward transform: shift each
    # ring's mode-m coefficient back to its native first-pixel longitude with
    # exp(+i*m*phi_first) (the conjugate of the forward correction).
    m_signed = np.fft.fftfreq(4 * nside) * (4 * nside)
    phi0 = ring_first_longitude(nside)
    fft_coeff = np.asarray(fft_coeff) * np.exp(+1j * np.outer(phi0, m_signed))

    eq_rings = _maybe_real(
        np.fft.ifft(
            fft_coeff[nside - 1 : 3 * nside], n=4 * nside, axis=-1, norm="forward"
        )
    )
    start_id, _, _ = ring_info[nside - 1]
    _, end_id, _ = ring_info[3 * nside - 1]
    healpix_map[start_id : end_id + 1] = eq_rings.ravel()

    # Polar rings: the phi=0 referencing was already undone above for all rows.
    for i in range(nside - 1):
        num_pts = ring_sizes[i]

        start_id, end_id, _ = ring_info[i]
        healpix_map[start_id : end_id + 1] = process_polar_ring(fft_coeff[i], num_pts)

        start_id, end_id, _ = ring_info[-1 - i]
        healpix_map[start_id : end_id + 1] = process_polar_ring(
            fft_coeff[-1 - i], num_pts
        )

    return healpix_map


# For visualisation
def create_upsampled_grid(nside: int) -> (np.ndarray, np.ndarray):
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
    longitudes = np.linspace(
        0, 360, max_lon_points, endpoint=False
    )  # Longitudes in degrees

    latitudes = create_latitude_array(nside)

    return np.meshgrid(longitudes, latitudes)


def create_latitude_array(nside: int) -> np.ndarray:
    """
    Generate latitude values for HEALPix rings, covering polar and equatorial regions.

    Parameters:
    - nside: HEALPix resolution parameter.

    Returns:
    - latitudes: 1D NumPy array of latitude values (in degrees).
    """
    # HEALPix j-values for polar and equatorial regions
    j_north_south = np.arange(1, nside)  # j-values for polar region rings
    j_equatorial = np.arange(
        nside, 3 * nside + 1
    )  # j-values for equatorial region rings

    # Compute colatitudes
    colatitudes_north = np.arccos(1 - (j_north_south**2) / (3 * nside**2))
    colatitudes_equatorial = np.arccos(2 * (2 * nside - j_equatorial) / (3 * nside))

    # Fix: Compute southern latitudes correctly using the same formula
    colatitudes_south = np.arccos(-(1 - (j_north_south**2) / (3 * nside**2)))
    colatitudes_south = np.flip(colatitudes_south)

    # Convert colatitudes to latitudes (latitude = 90° - colatitude)
    latitudes_north = np.degrees(np.pi / 2 - colatitudes_north)
    latitudes_equatorial = np.degrees(np.pi / 2 - colatitudes_equatorial)
    latitudes_south = np.degrees(
        np.pi / 2 - colatitudes_south
    )  # Fix: Ensure smooth transition

    # Concatenate all latitudes in order: North → Equator → South
    latitudes = np.concatenate([latitudes_north, latitudes_equatorial, latitudes_south])

    return latitudes
