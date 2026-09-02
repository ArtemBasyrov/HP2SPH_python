"""Stage 1: HEALPix rings <-> an equiangular longitude grid.

The forward direction FFTs each HEALPix ring in longitude, references every ring to a
common ``phi = 0`` origin, and zero-pads the short polar rings up to the equatorial
``4*nside`` samples, giving the tensor-product ``(4*nside-1, 4*nside)`` grid the Double
Fourier Sphere stage consumes. ``transform_grid_to_healpix`` inverts it.

Conventions established here and assumed downstream:

* Rings are RING-ordered, north to south, numbered ``1 .. 4*nside-1`` in the formulas
  and indexed from 0 in the arrays.
* Every FFT uses ``norm='forward'``, so a coefficient is the true longitude Fourier
  coefficient of its ring -- the ``m = 0`` entry is the ring mean -- whatever the
  ring's pixel count.
* Coefficient arrays are in numpy FFT order, ``m = 0, 1, .. , -2*nside, .. , -1``.
  ``ring_alias_target``, ``mode_pole_envelope`` and ``ring_fold_plan`` instead describe
  the NATURAL (fftshifted) order used from ``DFS`` onward.
* The zero-pad asserts that a short ring measures nothing above its own Nyquist. That
  is false, and ``ring_alias_target`` states what the ring measures instead.
"""

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
    """Return the pixel index range of every RING-ordered HEALPix ring.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.

    Returns
    -------
    ndarray of int, shape (4*nside-1, 3)
        One row per ring, north to south. Column 0 is the index of the ring's first
        pixel in a RING-ordered map, column 1 the index of its last (inclusive), and
        column 2 the 1-based ring number ``1 .. 4*nside-1``. Dimensionless.

    Examples
    --------
    >>> from hp2sph.data_interpolation import get_ring_indices
    >>> get_ring_indices(1).tolist()
    [[0, 3, 1], [4, 7, 2], [8, 11, 3]]
    """
    num_rings = 4 * nside - 1
    i = np.arange(1, num_rings + 1)
    ring_sizes = ring_pixel_counts(nside)

    start_indices = np.cumsum(ring_sizes) - ring_sizes
    end_indices = start_indices + ring_sizes - 1
    return np.vstack((start_indices, end_indices, i)).T


def npix2nside(npix: int) -> int:
    """Convert a HEALPix pixel count to its resolution parameter.

    Parameters
    ----------
    npix : int
        Number of pixels in a full-sky HEALPix map. Must equal ``12 * nside**2``.

    Returns
    -------
    int
        The resolution parameter ``nside``. Not checked for being a power of two,
        so ``108`` returns ``3``.

    Raises
    ------
    ValueError
        If ``npix`` is not twelve times a perfect square.

    Examples
    --------
    >>> from hp2sph.data_interpolation import npix2nside
    >>> npix2nside(3072)
    16
    """
    nside = int(round(np.sqrt(npix / 12.0)))
    if 12 * nside**2 != npix:
        raise ValueError(f"{npix} is not a valid HEALPix pixel count")
    return nside


def ring_pixel_counts(nside: int) -> np.ndarray:
    """Return the pixel count of every RING-ordered HEALPix ring.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.

    Returns
    -------
    ndarray of int, shape (4*nside-1,)
        Pixels per ring, north to south: ``4*i`` on the north polar cap, ``4*nside``
        throughout the equatorial belt, and the mirror image on the south cap.
        Dimensionless, and summing to ``12*nside**2``.

    Examples
    --------
    >>> from hp2sph.data_interpolation import ring_pixel_counts
    >>> ring_pixel_counts(2).tolist()
    [4, 8, 8, 8, 8, 8, 4]
    """
    n_rings = 4 * nside - 1
    i = np.arange(1, n_rings + 1)
    sizes = np.full(n_rings, 4 * nside)
    sizes[:nside] = 4 * i[:nside]
    sizes[3 * nside :] = 4 * (4 * nside - i[3 * nside :])
    return sizes


def ring_alias_target(nside: int) -> (np.ndarray, np.ndarray, np.ndarray):
    """Map each longitude mode to the slot the ring actually measures it in.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.

    Returns
    -------
    target : ndarray of int, shape (4*nside-1, 4*nside)
        The slot mode ``m`` folds onto, as a column index in the same layout.
    phase : ndarray of complex128, shape (4*nside-1, 4*nside)
        The factor ``exp(i (m - b) phi0_r)`` the aliased contribution arrives with.
        Unit modulus, dimensionless.
    resolved : ndarray of bool, shape (4*nside-1, 4*nside)
        Whether the ring produces slot ``j`` at all. The Nyquist slot ``-npix//2``
        counts as produced: under the fold it is a genuine constraint on the
        ``+-npix//2`` sum, not the mis-assignment the zero-pad makes of it.

    All three are in NATURAL (fftshifted) longitude order -- column ``j`` is mode
    ``m = j - 2*nside``, the order ``DFS`` and ``apply_nuFFT`` use, not the numpy FFT
    order ``transform_healpix_to_grid`` returns. Rows run north to south.

    Notes
    -----
    A ring of ``npix`` pixels samples the longitude field at ``npix`` points, so it
    does not measure mode ``m``: it measures the whole alias family of ``m``. Writing
    ``b = m mod npix`` folded into ``[-npix//2, npix//2)``, what
    ``transform_healpix_to_grid`` puts in slot ``b`` of ring ``r`` is

        M[r, b] = sum over {m : m == b (mod npix)} of c_m(theta_r) exp(i (m - b) phi0_r)

    -- the exponential because the ``phi = 0`` referencing at the end of that function
    multiplies each slot by ``exp(-i b phi0)``, using the SLOT index rather than the
    mode that contributed, so an aliased contribution also arrives with the wrong
    longitude phase.

    The pipeline then asserts ``M[r, b] = c_b(theta_r)``, which is false whenever
    another member of the family carries amplitude. This is the analysis-side mirror of
    the synthesis-side fold in ``transform_grid_to_healpix``.

    This describes the ONE-SIDED Nyquist layout, so a consumer of this plan must call
    ``transform_healpix_to_grid(..., nyquist_split=False)``. With the split on, each
    Nyquist family scatters half onto ``-npix//2`` and half onto ``+npix//2`` with two
    different phases, which the single-target ``(target, phase)`` form cannot express.

    The model is measured against the pipeline to a relative 1e-12 on O(10) individual
    coefficients.
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
    """Bound the polar amplitude a band-limited spin-s longitude mode can carry.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.
    spin : int, default 0
        Spin weight of the field. Dimensionless. The bound is asymmetric in it: the
        exponent is ``|m + spin|`` at the north pole and ``|m - spin|`` at the south.
    lmax : int, optional
        Band limit to take the worst case over. Defaults to the pipeline band
        ``2*nside``. Raising it loosens every bound, because the profile of a
        higher-``l`` mode reaches closer to the pole; lowering it below the band
        actually present asserts a decay the field does not have.
    rings : slice, optional
        Restrict the result to a slice of rings, so a caller working in blocks never
        has to hold the whole ``(4*nside-1, 4*nside)`` array. Defaults to every ring.

    Returns
    -------
    ndarray of float64, shape (n_rings, 4*nside)
        For each ring and each mode, the largest ``|c_m(theta_r)|`` the mode can carry
        there as a fraction of its own peak over latitude. In ``[0, 1]``,
        dimensionless. Rows run north to south, in NATURAL (fftshifted) longitude
        order. ``n_rings`` is ``4*nside-1``, or the length of ``rings``.

    Notes
    -----
    A spin-s mode-m latitude profile is the Wigner d function ``d^l_{-s,m}``, i.e.
    ``sin^|m+s|(theta/2) cos^|m-s|(theta/2) P^(|m-s|,|m+s|)(cos theta)``. Near a pole
    the polynomial is NOT O(1): the uniform asymptotic is Bessel, ``J_a(l*theta)`` with
    ``a = |m+s|`` at the north pole and ``a = |m-s|`` at the south, so for a small
    argument the amplitude is ``(l*theta/2)^a / a!``. THE SCALE IS ``l*theta``, NOT
    ``theta``.

    Reproducibility
    ~~~~~~~~~~~~~~~
    Accuracy: derived, and it is a bound rather than a value -- the leading Bessel term
    with the oscillatory factor replaced by its maximum, clipped at 1. Dropping the
    ``l`` factor understates it by six orders at the band edge (measured, nside 32,
    ``l = 63``, ``m = +2``: the innermost ring carries 1.6e-2 of the mode's peak;
    ``(theta/2)^4`` predicts 2.6e-8 and ``(l*theta/2)^4/4!`` predicts 1.7e-2).
    Platform dependence: assumed. Elementary functions only, agreeing across libm
    implementations to a few ulp.
    Determinism: not established.
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
    # In logs, because a**700 / 700! overflows both factors at nside 512 while their
    # ratio is representable. The clip at 0 is the "fraction of its own peak" cap: the
    # small-argument form exceeds 1 once l*theta leaves the Bessel core.
    log_north = np.log(np.maximum(lmax * theta / 2, np.finfo(float).tiny))
    log_south = np.log(np.maximum(lmax * (np.pi - theta) / 2, np.finfo(float).tiny))
    north = np.exp(np.minimum(a * log_north - gammaln(a + 1), 0.0))
    south = np.exp(np.minimum(b * log_south - gammaln(b + 1), 0.0))
    # Both poles bound the same mode, and a ring near neither is bounded by whichever
    # is tighter; a ring near one pole has the other's factor saturated at 1.
    return np.minimum(north, south)


def ring_fold_plan(
    nside: int, spin: int = 0, tol: float = 1e-2, lmax: int = None
) -> (np.ndarray, np.ndarray, np.ndarray):
    """Build the forward latitude operator's fold plus selective zero-assertion.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.
    spin : int, default 0
        Spin weight of the field, passed through to ``mode_pole_envelope``.
    tol : float, default 1e-2
        Amplitude, as a fraction of a mode's own peak, below which an unresolved mode
        is asserted to be zero on that ring. Raising it relaxes fewer entries, so the
        fit is better conditioned and the modelling error larger; lowering it relaxes
        more, and each relaxed entry costs iterations while shrinking a term that stops
        dominating below about 1e-2.
    lmax : int, optional
        Band limit for the envelope. Defaults to the pipeline band ``2*nside``.

    Returns
    -------
    target : ndarray of int, shape (4*nside-1, 4*nside)
        The slot each mode's content is carried onto. A trusted unresolved mode is its
        own target, so it asserts ``c_m = 0`` in place.
    phase : ndarray of complex128, shape (4*nside-1, 4*nside)
        The factor that content arrives with; exactly 1 for a self-target.
    keep : ndarray of bool, shape (4*nside-1, 4*nside)
        False marks an entry dropped from the fit -- an unresolved mode with
        non-negligible amplitude there. Its content is still carried onto the ring's
        own bin by ``target`` and ``phase``.

    All three are in NATURAL (fftshifted) longitude order, rows north to south.

    Notes
    -----
    Zero-padding a short ring is exactly equivalent to "fold, AND assert
    ``c_m(theta_r) = 0`` for every mode the ring does not resolve" -- when the other
    members of an alias family vanish the fold degenerates to the identity. Those zero
    assertions are what make the latitude least squares WELL CONDITIONED, and almost
    all of them are true to many digits, because an unresolved mode is negligible on a
    polar ring unless it decays slowly into the nearby pole, which happens only for
    ``|m|`` near ``|spin|``.

    So the plan keeps the assertion where ``mode_pole_envelope`` says the mode is below
    ``tol``, RELAXES it where it is not, and FOLDS so that the retained data equations
    account exactly for the relaxed content aliasing into them. With an empty relax set
    it is the identity, i.e. the plain zero-padded operator bit for bit.
    """
    target, phase, resolved = ring_alias_target(nside)
    # Dropping every zero assertion instead of only the untrustworthy ones leaves the
    # polar caps constrained through the alias sums alone. The system stays consistent
    # -- the true solution satisfies it to 7e-13 -- but is so ill conditioned that LSMR
    # is still 10% off after 40000 iterations. That is the negative result behind the
    # `tol` threshold: relax the fewest entries that make the model honest.
    relax = (~resolved) & (mode_pole_envelope(nside, spin, lmax) > tol)
    trusted = (~resolved) & ~relax
    j = np.broadcast_to(np.arange(target.shape[1]), target.shape)
    return np.where(trusted, j, target), np.where(trusted, 1.0, phase), ~relax


def ring_first_longitude(nside: int) -> np.ndarray:
    """Return the longitude of the first pixel of each RING-ordered ring.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.

    Returns
    -------
    ndarray of float64, shape (4*nside-1,)
        Longitude of each ring's first pixel, in radians, north to south. In
        ``[0, pi/4]``, and matching ``healpy.pix2ang`` on the corresponding pixel.

    Notes
    -----
    HEALPix rings are not aligned to ``phi = 0``: a polar ring's first pixel sits half
    a pixel in, at ``phi = pi/npix``, while the equatorial rings alternate between
    ``phi = pi/(4*nside)`` and ``phi = 0``. To reference every ring's longitude Fourier
    coefficients to a common ``phi = 0`` origin -- the convention healpy's ``alm`` use
    -- each ring's mode-m coefficient is multiplied by ``exp(-i m phi_first)``. Getting
    this wrong leaves an m-dependent longitude phase in the output ``alm``.

    Examples
    --------
    >>> import numpy as np
    >>> from hp2sph.data_interpolation import ring_first_longitude
    >>> phi0 = ring_first_longitude(2)
    >>> bool(np.allclose(phi0, [np.pi / 4, np.pi / 8, 0, np.pi / 8, 0, np.pi / 8,
    ...                         np.pi / 4]))
    True
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
    """HEALPix map -> equiangular longitude grid and its ring Fourier coefficients.

    Parameters
    ----------
    healpix_map : ndarray of float64 or complex128, shape (12*nside**2,)
        A full-sky RING-ordered HEALPix map. Real for an intensity field, complex for a
        spin field carried as ``Q + iU``. Units are carried through unchanged.
    map_rows : int, optional
        Return and inverse-transform only the first and last ``map_rows`` rings instead
        of the whole grid, which is all the pole fill downstream reads. Defaults to the
        whole grid. Must not exceed half the ring count.
    nyquist_split : bool, default True
        Split each polar ring's Nyquist coefficient equally between ``m = -npix/2`` and
        ``m = +npix/2`` instead of assigning all of it to the negative slot. Set it
        False when the consumer models the ring aliasing itself: ``ring_alias_target``
        and the fold plans built on it describe the one-sided layout, which the split
        invalidates. Either setting round-trips exactly through
        ``transform_grid_to_healpix``, which folds both slots back onto the same bin.

    Returns
    -------
    upsampled_data : ndarray, shape (4*nside-1, 4*nside)
        The map resampled onto the tensor-product grid, rows north to south, columns at
        ``phi = 2*pi*j/(4*nside)``. float64 for a real input, complex128 for a complex
        one. With ``map_rows = k`` it is instead the first and last ``k`` rings stacked
        into a ``(2*k, 4*nside)`` array, north block first.
    fft_coeff : ndarray of complex128, shape (4*nside-1, 4*nside)
        Per-ring longitude Fourier coefficients, referenced to ``phi = 0``, in numpy
        FFT order, so column 0 holds ``m = 0``. Normalised with ``norm='forward'``, so
        for a constant map column 0 equals that constant on every ring and every other
        column is zero. Always the full ring set, whatever ``map_rows`` is.

    Raises
    ------
    ValueError
        If ``healpix_map`` has no valid HEALPix length, or if ``2*map_rows`` exceeds
        the ring count.

    Notes
    -----
    The polar rings are zero-padded from their own pixel count up to ``4*nside``, which
    asserts that they measure nothing above their own Nyquist. That assertion is the
    stage's modelling error, not a round-off one; ``ring_alias_target`` gives what the
    ring measures instead, and ``ring_fold_plan`` relaxes the assertion where it fails.

    The row work is split across threads above ``511`` rings, i.e. from nside 128. The
    thread count comes from ``HP2SPH_NUFFT_WORKERS`` when it is set, and from the
    usable core count otherwise; ``1`` disables the split. The function therefore
    creates threads, and is not meant to be called concurrently with itself.

    Reproducibility
    ~~~~~~~~~~~~~~~
    Accuracy: derived. Every step is an FFT, a zero-pad and a unit-modulus multiply, so
    the coefficients are those of the sampled ring up to floating-point round-off.
    Platform dependence: assumed, at the round-off level only, through numpy's FFT
    build. The row split is not a source of variation: pocketfft transforms each row
    independently and each worker writes a disjoint row range, so a threaded run gives
    the serial answer bit for bit.
    Determinism: not established.
    """
    start_time = time.time()
    healpix_map = np.asarray(healpix_map)
    nside = npix2nside(healpix_map.shape[0])
    n_rings = 4 * nside - 1

    ring_info = get_ring_indices(nside)  # [start_id, end_id, ring_id]
    fft_coeff = np.zeros((n_rings, 4 * nside), dtype=complex)

    def process_polar_ring(ring_data):
        # One FFT per ring: the polar rings have different lengths, so they cannot be
        # batched into one array FFT the way the belt is. numpy's FFT on arrays this
        # small is microseconds, so 2*nside of them are cheap.
        num_pts = len(ring_data)
        coeffs = np.fft.fft(ring_data, n=num_pts, norm="forward")

        # Zero-pad in numpy FFT order, which puts the positive frequencies at the front
        # and the negative ones at the back, so the padding goes in the middle.
        mid = num_pts // 2
        coeffs_padded = np.zeros(4 * nside, dtype=complex)
        coeffs_padded[:mid] = coeffs[:mid]
        coeffs_padded[-mid:] = coeffs[-mid:]

        # The ring's Nyquist bin measures the SUM of the m = +mid and m = -mid content:
        # at num_pts samples the two exponentials are the same function. numpy parks it
        # at index mid, i.e. frequency -mid, so the slice above hands all of it to
        # m = -mid and leaves m = +mid empty -- which asserts a complex exponential
        # where the truth is a cosine, and injects a spurious sin(mid*phi) of equal
        # amplitude. Half in each slot is the assignment consistent with a real map.
        # Measured: this is the whole anti-Hermitian part of the post-NUFFT array on a
        # smooth sky, which drops from 1e-4 relative to 4e-15, and it is worth
        # 1.22-1.34x in the 0.75-0.875 lmax band. nside 128 to 512, intensity,
        # lmax = 2*nside, mmax = lmax-1, median of 4 seeds.
        if nyquist_split:
            coeffs_padded[-mid] *= 0.5
            coeffs_padded[mid] = coeffs_padded[-mid]

        # NB: do NOT rescale by num_pts/(4*nside). With norm='forward' these already
        # ARE the ring's true longitude Fourier coefficients (m = 0 is the ring mean),
        # independent of its pixel count, and the padding only extends the empty
        # high-frequency band. Such a factor cancels against its inverse in
        # transform_grid_to_healpix, so a round trip stays exact and does not catch it;
        # what it corrupts is the coefficients handed to every later stage. Compare a
        # single-harmonic map's polar-ring coefficients against the equatorial ones to
        # see it.
        return coeffs_padded

    # A spin field (Q + iU) is carried as a complex map; an intensity map is real. Keep
    # complex content when it is there, and drop the identically zero imaginary part
    # otherwise, so the scalar path returns a real grid.
    is_complex = np.iscomplexobj(healpix_map)

    ring_data = [healpix_map[start : end + 1] for start, end, nring in ring_info]

    # Every stage below writes disjoint row ranges of `fft_coeff`, so splitting the rows
    # across threads gives exactly the serial answer. numpy's pocketfft releases the
    # GIL, so the split scales.
    workers = default_workers(n_rings) if n_rings >= _MIN_THREADED_RINGS else 1

    # Every equatorial ring has 4*nside pixels, so the belt is one batched FFT over the
    # last axis rather than a loop over rings.
    n_belt = 2 * nside + 1

    def equatorial_block(lo, hi):
        # In sub-blocks, because stacking the whole belt and transforming it in one call
        # holds two arrays the size of `fft_coeff` itself -- 1.1 GB of transient at
        # nside 2048 -- and with the rows split across threads every worker would hold
        # its own. pocketfft transforms each row independently, so a shorter batch is
        # bit-identical to the corresponding slice of a longer one.
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

    # The polar rings are ragged, so this stays a loop over individual FFTs; what is
    # threaded is a contiguous BLOCK of rings per worker, one ring being far too little
    # work to hand off. Each iteration takes the ring and its southern mirror, so the
    # two caps are covered by a loop over nside-1 rather than over every polar ring.
    def polar_block(lo, hi):
        for i in range(lo, hi):
            fft_coeff[i] = process_polar_ring(ring_data[i])
            fft_coeff[n_rings - 1 - i] = process_polar_ring(ring_data[n_rings - 1 - i])

    run_blocks(polar_block, nside - 1, workers)

    # Reference every ring to a common phi = 0 origin: the ring's first pixel is offset
    # by phi_first, so its mode-m coefficient carries a spurious exp(+i*m*phi_first),
    # divided out here. m is the SIGNED frequency in numpy FFT order, so it comes from
    # fftfreq, not arange -- arange would put the wrong phase on every negative mode.
    m_signed = np.fft.fftfreq(4 * nside) * (4 * nside)
    phi0 = ring_first_longitude(nside)

    # Blocked, because the full phase array is the size of `fft_coeff` itself and
    # np.outer builds a second one of that size before np.exp does. At nside 1024 that
    # pair is 0.4 GB of transient for a multiply that is row-local anyway; at nside 2048
    # it is 1.6 GB.
    def phase_block(lo, hi):
        for a in range(lo, hi, _PHASE_BLOCK):
            b = min(a + _PHASE_BLOCK, hi)
            fft_coeff[a:b] *= np.exp(-1j * np.outer(phi0[a:b], m_signed))

    run_blocks(phase_block, n_rings, workers)

    # Back to map space. With `map_rows` set, only the polar rows the pole fill reads
    # are transformed: the inverse FFT of the whole grid is 72 ms and 0.27 GB at nside
    # 1024 against 0.19 ms for twelve rows, and a row's inverse transform does not
    # depend on the other rows, so the subset is bit-identical to the slice.
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


def transform_grid_to_healpix(
    grid_data: np.ndarray, fft_coeff: np.ndarray = None, real_output: bool = True
) -> np.ndarray:
    """Ring Fourier coefficients -> HEALPix map, aliasing the short polar rings.

    Parameters
    ----------
    grid_data : ndarray, shape (4*nside-1, 4*nside)
        The equiangular grid, rows north to south. Read only when ``fft_coeff`` is
        None, and every caller in this package passes the coefficients as both
        arguments.
    fft_coeff : ndarray of complex128, shape (4*nside-1, 4*nside)
        Per-ring longitude Fourier coefficients referenced to ``phi = 0``, in numpy FFT
        order and normalised with ``norm='forward'`` -- the second return value of
        ``transform_healpix_to_grid``. Required in practice: it also carries the ``nside``
        this function works at, so passing None raises ``AttributeError`` rather than
        falling back to ``grid_data``.
    real_output : bool, default True
        Take the real part of each ring, which is the intensity path. Set False to keep
        a complex ``Q + iU`` spin field, the inverse of carrying a complex map through
        the forward transform. Leaving it True on a spin field silently discards ``U``.

    Returns
    -------
    ndarray, shape (12*nside**2,)
        A full-sky RING-ordered HEALPix map, float64 when ``real_output`` is True and
        complex128 otherwise. Units are those of the coefficients.

    Notes
    -----
    Reproducibility
    ~~~~~~~~~~~~~~~
    Accuracy: derived. The inverse FFTs and the phase unreferencing invert their
    forward counterparts exactly, and the polar fold is the exact adjoint of the
    forward zero-pad on any spectrum the forward produced, so the round trip is exact
    up to floating-point round-off. On a spectrum from the SYNTHESIS side the fold is
    an alias, not an inverse, and the map is the band-limited field sampled on the
    ring rather than a recovery of anything.
    Platform dependence: assumed, at the round-off level only, through numpy's FFT
    build.
    Determinism: not established.
    """
    nside = fft_coeff.shape[1] // 4
    ring_info = get_ring_indices(nside)  # [start_id, end_id, ring_id]

    ring_sizes = ring_pixel_counts(nside)

    map_dtype = float if real_output else complex
    healpix_map = np.empty(12 * nside**2, dtype=map_dtype)

    def _maybe_real(arr):
        return arr.real if real_output else arr

    def process_polar_ring(fft_coeff, num_pts):
        # ALIAS (fold), do not truncate. A ring of num_pts pixels samples the longitude
        # field at num_pts points, so mode m lands in bin m % num_pts: the pixel values
        # are sum_m c_m exp(i m phi_j), which is the FOLD of the 4*nside-wide spectrum
        # onto num_pts bins. Dropping |m| >= num_pts//2 throws that content away.
        #
        # On a spectrum from transform_healpix_to_grid the two agree bit for bit -- the
        # forward zero-pads, so every folded-in entry is either exactly 0 or the other
        # half of a split Nyquist bin, which this add.at sums back -- so a round trip
        # cannot tell them apart. They differ on a spectrum from the SYNTHESIS side,
        # where the high-|m| entries carry real signal. For a SPIN field that is the
        # whole ball game: |m| = |spin| is O(1) AT the pole and is exactly the Nyquist
        # of the innermost 4-pixel ring, and truncating it leaves the innermost polar
        # rings 100% wrong. The contract is stated executably by
        # test_polar_rings_alias_not_truncate.
        m_signed = np.rint(np.fft.fftfreq(len(fft_coeff)) * len(fft_coeff)).astype(int)
        corrected_coeffs_back = np.zeros(num_pts, dtype=complex)
        np.add.at(corrected_coeffs_back, m_signed % num_pts, np.asarray(fft_coeff))

        # One FFT per ring, for the same ragged-length reason as the forward. No
        # num_pts/(4*nside) rescaling: the forward FFT (norm='forward') carries the
        # 1/num_pts, and this ifft (also norm='forward') inverts it with no extra
        # factor.
        return _maybe_real(
            np.fft.ifft(corrected_coeffs_back, n=num_pts, norm="forward")
        )

    if fft_coeff is None:
        # Every row of the grid has 4*nside samples, so this is one batched FFT.
        fft_coeff = np.fft.fft(
            np.asarray(grid_data), n=4 * nside, axis=-1, norm="forward"
        )

    # Undo the phi = 0 referencing: shift each ring's mode-m coefficient back to its
    # native first-pixel longitude with exp(+i*m*phi_first), the conjugate of the
    # forward correction. Unblocked, unlike the forward: the synthesis side has no
    # nside 2048 caller and the transient has not been a limit here.
    m_signed = np.fft.fftfreq(4 * nside) * (4 * nside)
    phi0 = ring_first_longitude(nside)
    fft_coeff = np.asarray(fft_coeff) * np.exp(+1j * np.outer(phi0, m_signed))

    # The belt rings all have 4*nside pixels and are contiguous in RING order, so one
    # batched inverse FFT fills that whole span of the map.
    eq_rings = _maybe_real(
        np.fft.ifft(
            fft_coeff[nside - 1 : 3 * nside], n=4 * nside, axis=-1, norm="forward"
        )
    )
    start_id, _, _ = ring_info[nside - 1]
    _, end_id, _ = ring_info[3 * nside - 1]
    healpix_map[start_id : end_id + 1] = eq_rings.ravel()

    # Polar rings, north and its southern mirror per iteration. The phi = 0 referencing
    # was already undone above for every row, including these.
    for i in range(nside - 1):
        num_pts = ring_sizes[i]

        start_id, end_id, _ = ring_info[i]
        healpix_map[start_id : end_id + 1] = process_polar_ring(fft_coeff[i], num_pts)

        start_id, end_id, _ = ring_info[-1 - i]
        healpix_map[start_id : end_id + 1] = process_polar_ring(
            fft_coeff[-1 - i], num_pts
        )

    return healpix_map


def create_upsampled_grid(nside: int) -> (np.ndarray, np.ndarray):
    """Return the longitude-latitude coordinates of the upsampled HEALPix grid.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.

    Returns
    -------
    longitudes : ndarray of float64, shape (4*nside-1, 4*nside)
        Longitude of every grid point, in degrees, uniform on ``[0, 360)`` along each
        row and constant down each column.
    latitudes : ndarray of float64, shape (4*nside-1, 4*nside)
        Latitude of every grid point, in degrees, constant along each row and running
        north to south down the columns.

    The pair is a ``numpy.meshgrid``, i.e. what a plotting routine wants; the grid the
    transform works on is the same one, so this is the coordinate system of
    ``transform_healpix_to_grid``'s first return value.

    Notes
    -----
    Reproducibility
    ~~~~~~~~~~~~~~~
    Accuracy: derived. Closed-form coordinates, exact up to floating-point round-off.
    Platform dependence: assumed, at the round-off level only, through ``arccos``.
    Determinism: not established.

    Examples
    --------
    >>> import numpy as np
    >>> from hp2sph.data_interpolation import create_upsampled_grid
    >>> lon, lat = create_upsampled_grid(2)
    >>> lon.shape
    (7, 8)
    >>> bool(np.allclose(lon[0], [0, 45, 90, 135, 180, 225, 270, 315]))
    True
    """
    max_lon_points = 4 * nside
    longitudes = np.linspace(0, 360, max_lon_points, endpoint=False)  # degrees

    latitudes = create_latitude_array(nside)

    return np.meshgrid(longitudes, latitudes)


def create_latitude_array(nside: int) -> np.ndarray:
    """Return the latitude of every RING-ordered HEALPix ring.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.

    Returns
    -------
    ndarray of float64, shape (4*nside-1,)
        Ring latitudes in degrees, strictly decreasing from just under ``+90`` to just
        over ``-90``. Symmetric about the equator, and entry ``2*nside-1`` is exactly
        ``0``. These are the non-uniform latitudes the nuFFT stage solves at.

    Notes
    -----
    The colatitude of ring ``j`` is ``arccos(1 - j**2/(3*nside**2))`` on the polar caps
    and ``arccos(2*(2*nside - j)/(3*nside))`` through the equatorial belt, the standard
    HEALPix ring formulas.

    Reproducibility
    ~~~~~~~~~~~~~~~
    Accuracy: derived. Closed form, exact up to floating-point round-off. The polar
    formula is evaluated where ``arccos`` has its square-root singularity, so the
    innermost ring loses about half its significant digits relative to the equatorial
    ones -- unavoidable in this parametrisation and small against the stage's other
    errors.
    Platform dependence: assumed, at the round-off level only, through ``arccos``.
    Determinism: not established.

    Examples
    --------
    >>> import numpy as np
    >>> from hp2sph.data_interpolation import create_latitude_array
    >>> lat = create_latitude_array(2)
    >>> bool(np.allclose(lat, -lat[::-1]))
    True
    >>> float(round(lat[3], 12))
    0.0
    """
    j_north_south = np.arange(1, nside)
    j_equatorial = np.arange(nside, 3 * nside + 1)

    colatitudes_north = np.arccos(1 - (j_north_south**2) / (3 * nside**2))
    colatitudes_equatorial = np.arccos(2 * (2 * nside - j_equatorial) / (3 * nside))

    # The south cap is the north cap reflected, so it uses the north formula with the
    # sign of the cosine flipped and the ring order reversed. Deriving it from a
    # separate south-going index instead is where an off-by-one ring gets in.
    colatitudes_south = np.arccos(-(1 - (j_north_south**2) / (3 * nside**2)))
    colatitudes_south = np.flip(colatitudes_south)

    latitudes_north = np.degrees(np.pi / 2 - colatitudes_north)
    latitudes_equatorial = np.degrees(np.pi / 2 - colatitudes_equatorial)
    latitudes_south = np.degrees(np.pi / 2 - colatitudes_south)

    return np.concatenate([latitudes_north, latitudes_equatorial, latitudes_south])
