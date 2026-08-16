from typing import NamedTuple

import numpy as np
import healpy as hp

from ._threads import default_workers, run_blocks
from .data_interpolation import (
    create_latitude_array,
    mode_pole_envelope,
    ring_first_longitude,
    ring_fold_plan,
    ring_pixel_counts,
)

# Number of HEALPix rings on EACH side of a pole used to extrapolate the (unsampled)
# pole-ring value. Capped at the polar-cap size (nside-1) so the stencil never
# reaches into the equatorial belt, where the long extrapolation to the pole would
# be unstable. The Lagrange weights are well behaved (L1 ~ 3.4, sum = 1) and stable
# across nside, so 6 is safe everywhere. See ``interpolate_polar_rings``.
POLE_INTERP_NPTS = 6


def _mirror_odd_mask(n_lon: int, spin: int) -> np.ndarray:
    """Columns (numpy-FFT longitude order) whose mode m has (m+spin) odd.

    The DFS mirrors the map across the pole via the glide reflection
    theta -> 2*pi - theta, phi -> phi + pi. The phi+pi shift multiplies mode m by
    e^{i*m*pi} = (-1)^m, and a spin-s field picks up an extra (-1)^s, so the
    mirrored half is multiplied by (-1)^(m+s): columns with (m+s) odd flip sign.
    For the scalar field (spin=0) this is exactly "flip every odd wavenumber".
    Since 4*nside is even, the numpy-order column index j has the same parity as
    its signed mode m, so the mask is a pure parity test on the column index.
    """
    j = np.arange(n_lon)
    return ((j + spin) % 2) == 1


def _mirror_map(mp: np.ndarray, spin: int) -> np.ndarray:
    """The southern (mirrored) half of the DFS doubling, in MAP space.

    The DFS glide reflection is theta -> 2*pi - theta, phi -> phi + pi, and a spin-s
    field picks up (-1)^s across the pole, so

        mirror(theta, phi) = (-1)^s * mp(2*pi - theta, phi + pi).

    In grid terms phi + pi is a ROLL by half the longitude samples, not a reversal.
    The previous code used ``flip(mp)`` with no axis, which reverses BOTH axes and
    so applied phi -> -phi. That is wrong for every |m| > 0, but it only ever showed up
    through the pole fill (the only consumer of this array), where the sole mode with a
    nonzero pole value is |m| = |s|:

      * scalar (s = 0): that is m = 0, which is invariant under both -phi and phi+pi,
        so the bug was invisible -- every other mode vanishes at the pole either way;
      * spin +-2: that is m = -+2, whose pole value is NOT zero, so the reversed
        longitude injected a wrong pole ring and spread a single harmonic over all l of
        the same parity. This was the "m != 0 is broken" symptom (SPIN2_PLAN.md Phase 3).
    """
    n_lon = mp.shape[1]
    mirrored = np.roll(np.flip(mp, axis=0), n_lon // 2, axis=1)
    return mirrored * ((-1.0) ** spin)


def pole_stencil_rows(nside: int) -> int:
    """How many rings from each end the pole fill actually reads.

    ``transform_healpix_to_grid(map_rows=...)`` takes this so it can return just those
    rows instead of the whole grid.
    """
    return max(2, min(POLE_INTERP_NPTS, nside - 1))


def _pole_stencils(orig: np.ndarray, spin: int):
    """The two pole rows, computed from the edge rings alone.

    ``interpolate_polar_rings`` reads its stencils out of the fully doubled map, but
    they only ever involve the ``npts`` rings nearest each pole and the mirror images of
    those same rings. The mirror is a glide reflection -- a row flip plus a half-turn in
    longitude -- so those few rows can be formed directly and the southern half of the
    map never has to exist.

    ``orig`` is the ``(4*nside-1, 4*nside)`` grid of original rings, or just its first
    and last ``pole_stencil_rows(nside)`` rows stacked -- only those are read, so
    ``transform_healpix_to_grid(map_rows=...)`` can supply the short form and skip the
    inverse FFT over the whole grid. Returns the north and south pole rows, identical to
    what ``interpolate_polar_rings`` produces.
    """
    n_rings, n_lon = orig.shape
    nside = n_lon // 4
    npts = pole_stencil_rows(nside)
    if n_rings < 2 * npts:
        raise ValueError(
            f"the pole fill needs {npts} rings from each end; got {n_rings} rows"
        )
    latitudes = create_latitude_array(nside)
    sgn = (-1.0) ** spin
    half_turn = n_lon // 2

    north_theta = np.concatenate((np.flip(latitudes[:npts]), 180 - latitudes[:npts]))
    north_fp = np.concatenate(
        (np.flip(orig[:npts], axis=0), np.roll(orig[:npts], half_turn, axis=1) * sgn)
    )
    north = _pole_lagrange_weights(north_theta, 90.0) @ north_fp

    south_theta = np.concatenate((latitudes[-npts:], -180 - np.flip(latitudes[-npts:])))
    south_fp = np.concatenate(
        (
            orig[n_rings - npts :],
            np.roll(np.flip(orig[n_rings - npts :], axis=0), half_turn, axis=1) * sgn,
        )
    )
    south = _pole_lagrange_weights(-south_theta, 90.0) @ south_fp
    return north, south


# Below this many rings the thread hand-off costs more than splitting the rows saves;
# the same crossover the interpolation stage measures.
_MIN_THREADED_RINGS = 511


def _shifted_into(dst: np.ndarray, src: np.ndarray) -> None:
    """Write ``np.fft.fftshift(src, axes=1)`` into ``dst`` with a single copy."""
    n_lon = src.shape[1]
    shift = n_lon // 2
    dst[:, :shift] = src[:, n_lon - shift :]
    dst[:, shift:] = src[:, : n_lon - shift]


def DFS(
    mp: np.ndarray, fft_coeff: np.ndarray, spin: int = 0, half: bool = False
) -> (np.ndarray, np.ndarray):
    """Double the map across the poles and return ``(map, fourier)`` rows.

    The full layout is ``[north pole, rings, south pole, mirrored rings]``.

    ``half=True`` returns only ``[north pole, rings, south pole]``, and returns ``None``
    in place of the map, which no caller uses. The mirrored rings
    are an exact reflection -- ``d[mu(r), c] = (-1)^(m+spin) d[r, c]`` -- and the
    latitude solve exploits that symmetry to work on this half anyway, so materialising
    them costs memory and buys nothing. Pass the result to ``apply_nuFFT`` together with
    ``half_domain=True`` and with fold arrays from ``dfs_fold_plan(..., half=True)``.
    """
    if half:
        n_rings, n_lon = fft_coeff.shape
        north, south = _pole_stencils(mp, spin)
        # The map half of the return value is not built. Every caller of ``DFS`` uses
        # only the Fourier array, and at nside 1024 the map is another 0.27 GB whose
        # only content beyond the two pole rows is a verbatim copy of ``mp``.
        half_map = None
        half_fft = np.empty((n_rings + 2, n_lon), dtype=complex)
        # The fftshift to natural ordering is folded into the copy. Doing it afterwards
        # reads and writes the whole array a second time and holds a second full-size
        # array while it does -- 0.27 GB extra at nside 1024, and the array is the
        # largest thing this stage produces. A shift along one axis is a permutation of
        # columns, so writing the two column ranges straight into their destination is
        # exactly equal, not merely close.
        _shifted_into(half_fft[0:1], np.fft.fft(north, n=n_lon, norm="forward")[None])
        _shifted_into(
            half_fft[n_rings + 1 : n_rings + 2],
            np.fft.fft(south, n=n_lon, norm="forward")[None],
        )
        # Rows are independent, so the body is split across threads; each block writes
        # only its own rows.
        body = half_fft[1 : n_rings + 1]
        run_blocks(
            lambda lo, hi: _shifted_into(body[lo:hi], fft_coeff[lo:hi]),
            n_rings,
            default_workers(n_rings) if n_rings >= _MIN_THREADED_RINGS else 1,
        )
        return half_map, half_fft

    south_part = _mirror_map(mp, spin)
    double_map = np.concatenate((mp, south_part), axis=0)

    double_map = interpolate_polar_rings(double_map)

    south_part = np.flipud(np.array(fft_coeff))
    # flip the mirrored half by (-1)^(m+spin) (scalar: every odd wavenumber)
    odd = _mirror_odd_mask(fft_coeff.shape[1], spin)
    south_part[:, odd] *= -1

    # double the fft coefficients
    n_rings = fft_coeff.shape[0]
    double_fft = np.zeros((2 * n_rings + 2, fft_coeff.shape[1]), dtype=complex)
    double_fft[0] = np.fft.fft(double_map[0], n=fft_coeff.shape[1], norm="forward")
    double_fft[1 : n_rings + 1] = fft_coeff[:]
    # ``interpolate_polar_rings`` lays the doubled map out as
    #   [north pole, original rings (n_rings), south pole, mirrored rings],
    # so the SOUTH pole is row n_rings+1; row n_rings is the last original ring.
    # This read used ``double_map[n_rings]``, which duplicated that last ring into the
    # pole slot -- the south pole never received the polynomial pole fill at all (the
    # north pole did). The fill is worth 5-15x at the band edge, so half of that gain
    # was being thrown away.
    double_fft[n_rings + 1] = np.fft.fft(
        double_map[n_rings + 1], n=fft_coeff.shape[1], norm="forward"
    )
    double_fft[n_rings + 2 :] = south_part

    """# apply weights correction
    weights = compute_ring_area_weights(fft_coeff.shape[1] // 4) # both poles + original map
    double_fft[:n_rings+2] *= weights[:, np.newaxis]
    double_fft[n_rings+2:] *= np.flip(weights[1:-1])[:, np.newaxis] # flip weights for the mirrored part 
    """
    # apply FFT shift from numpy ordering to natural ordering
    double_fft = np.fft.fftshift(double_fft, axes=1)

    return double_map, double_fft


def DFS_inverse(double_fft: np.ndarray, spin: int = 0) -> np.ndarray:
    nside = double_fft.shape[1] // 4
    n_rings = 4 * nside - 1

    # selecting the upper part of the double map without added poles
    fft_coeff = double_fft[1 : n_rings + 1]

    # apply weights correction
    # weights = compute_ring_area_weights(nside) # both poles + original map
    # fft_coeff /= weights[1:-1][:, np.newaxis]

    # apply FFT shift from natural ordering to numpy ordering
    fft_coeff = np.fft.ifftshift(fft_coeff, axes=1)

    return fft_coeff


def dfs_fold_plan(
    nside: int, spin: int = 0, tol: float = 1e-2, lmax: int = None, half: bool = False
) -> (np.ndarray, np.ndarray, np.ndarray):
    """``data_interpolation.ring_fold_plan`` in the row layout ``DFS`` returns.

    Same rows as ``interpolate_polar_rings``/``_upsampled_latitudes``
    (``[north pole, rings, south pole, mirrored rings]``) and the same natural
    (fftshifted) longitude order, so ``(target, phase)`` can be handed straight to
    ``nuFFT.apply_nuFFT(fold=...)`` and ``keep`` to its ``sample_mask``.

    A mirrored row reuses its original's plan: the DFS sign flip is ``(-1)^(m+spin)`` and
    every ring size is even, so ``m == b (mod npix)`` implies ``(-1)^m == (-1)^b`` and the
    sign commutes with the fold.

    A pole row is data only in the weak sense that it is a Lagrange extrapolation of the
    ring rows, so it inherits their aliasing: a relaxed mode is both MISSING from its own
    pole slot (the measurement zero-padded it) and PRESENT in the slot it folds onto. Both
    of those pole slots are dropped. That is far less restrictive than the superseded
    mask fix, which kept only ``|m| <= 1`` there.
    """
    target, phase, keep = ring_fold_plan(nside, spin, tol, lmax)
    n_rings, n_lon = target.shape
    ident = np.arange(n_lon)[None, :]

    pole_keep = np.ones((2, n_lon), dtype=bool)
    npts = max(2, min(POLE_INTERP_NPTS, nside - 1))
    for i, rings in enumerate((np.arange(npts), np.arange(n_rings - npts, n_rings))):
        relaxed = ~keep[rings]
        pole_keep[i, np.unique(np.nonzero(relaxed)[1])] = False
        pole_keep[i, np.unique(target[rings][relaxed])] = False

    head = (
        np.concatenate((ident, target, ident)),
        np.concatenate((np.ones((1, n_lon)), phase, np.ones((1, n_lon)))),
        np.concatenate((pole_keep[:1], keep, pole_keep[1:])),
    )
    if half:
        return head
    return (
        np.concatenate((head[0], np.flip(target, axis=0))),
        np.concatenate((head[1], np.flip(phase, axis=0))),
        np.concatenate((head[2], np.flip(keep, axis=0))),
    )


class FoldPlan(NamedTuple):
    """The polar-ring alias fold as flat index arrays, in the half-DFS layout.

    ``src``, ``dst`` and ``drop`` index a C-contiguous ``(n_trans, n_rows)`` solver
    buffer, i.e. entry ``(column c, row r)`` sits at ``c * n_rows + r``.

    * ``src``  -- entries whose content moves onto another slot,
    * ``dst``  -- where each of them lands,
    * ``phase``-- the ``exp(i (m - b) phi0)`` it arrives with,
    * ``drop`` -- entries carrying no data, which get zero quadrature weight.

    ``drop`` is a superset of ``src``: a relaxed entry is exactly one the ring never
    measured, so its equation is dropped while its content is still folded on. The only
    additions are the pole slots a relaxed mode can have corrupted.
    """

    src: np.ndarray
    dst: np.ndarray
    phase: np.ndarray
    drop: np.ndarray
    n_trans: int
    n_rows: int


def _relax_block(nside, spin, tol, lmax, lo, hi):
    """``(target, phase, relax)`` for rings ``lo:hi`` only, without any full-grid array."""
    n_lon = 4 * nside
    sizes = ring_pixel_counts(nside)[lo:hi]
    phi0 = ring_first_longitude(nside)[lo:hi]
    m = np.arange(n_lon) - n_lon // 2
    mid = (sizes // 2)[:, None]
    b = (m[None, :] + mid) % sizes[:, None] - mid
    target = b + n_lon // 2
    j = np.arange(n_lon)[None, :]
    resolved = (j >= n_lon // 2 - mid) & (j < n_lon // 2 + mid)
    env = mode_pole_envelope(nside, spin, lmax, rings=slice(lo, hi))
    relax = (~resolved) & (env > tol)
    return target, b, phi0, m, relax


def dfs_fold_sparse(
    nside: int, spin: int = 0, tol: float = 1e-2, lmax: int = None, block: int = 64
) -> FoldPlan:
    """``dfs_fold_plan(half=True)`` without ever forming a full-grid array.

    The dense plan is three ``(4*nside+1, 4*nside)`` arrays -- 0.4 GB at nside 1024 and
    1.6 GB at 2048 -- of which only a few percent of entries say anything: the rest is
    "this slot stays where it is". This walks the rings in blocks and keeps only the
    entries that move, which is the same information in a few MB.

    Equivalent to deriving the same indices from ``dfs_fold_plan(..., half=True)``, and
    tested to be bit-identical to it.
    """
    n_rings = 4 * nside - 1
    n_lon = 4 * nside
    n_rows = n_rings + 2  # [north pole, rings, south pole]
    src, dst, phases = [], [], []
    for lo in range(0, n_rings, block):
        hi = min(lo + block, n_rings)
        target, b, phi0, m, relax = _relax_block(nside, spin, tol, lmax, lo, hi)
        rr, cc = np.nonzero(relax)
        if rr.size == 0:
            continue
        rows = rr + lo + 1  # ring i sits at DFS row i+1
        src.append(cc * n_rows + rows)
        dst.append(target[rr, cc] * n_rows + rows)
        phases.append(np.exp(1j * (m[cc] - b[rr, cc]) * phi0[rr]))

    empty_i = np.empty(0, dtype=np.intp)
    src = np.concatenate(src).astype(np.intp) if src else empty_i
    dst = np.concatenate(dst).astype(np.intp) if dst else empty_i
    phase = np.concatenate(phases) if phases else np.empty(0, dtype=complex)

    # A pole row inherits the aliasing of the rings its Lagrange fill reads, so a slot
    # is dropped there if a relaxed mode lives in it or folds onto it.
    npts = pole_stencil_rows(nside)
    drops = [src]
    for pole_row, (lo, hi) in enumerate(((0, npts), (n_rings - npts, n_rings))):
        target, _, _, _, relax = _relax_block(nside, spin, tol, lmax, lo, hi)
        rr, cc = np.nonzero(relax)
        bad = np.unique(np.concatenate((cc, target[rr, cc])))
        row = 0 if pole_row == 0 else n_rings + 1
        drops.append(bad * n_rows + row)
    drop = np.unique(np.concatenate(drops)).astype(np.intp)
    return FoldPlan(src, dst, phase, drop, n_lon, n_rows)


def _pole_lagrange_weights(nodes: np.ndarray, x0: float) -> np.ndarray:
    """Lagrange weights to evaluate, at ``x0``, the degree-(n-1) polynomial that
    interpolates values sampled at ``nodes``. The nodes are the same for every
    longitude column, so the whole pole ring is one matvec ``weights @ stencil``.
    """
    n = len(nodes)
    w = np.ones(n)
    for i in range(n):
        for j in range(n):
            if j != i:
                w[i] *= (x0 - nodes[j]) / (nodes[i] - nodes[j])
    return w


def interpolate_polar_rings(mp: np.ndarray) -> np.ndarray:
    """Fill the two (HEALPix-unsampled) pole rings of the DFS-doubled map.

    Each pole value is a polynomial extrapolation in latitude: fit a degree-(2*npts-1)
    polynomial through a stencil symmetric about the pole -- the ``npts`` rings nearest
    the pole and their mirror images across it -- and evaluate it AT the pole. This
    replaces the old piecewise-LINEAR ``interp`` fill, which was the dominant
    high-l forward error: the m=0 latitude profile P_l(cos theta) peaks at the poles,
    so a crude pole value injects a large zonal error that grows with l. The
    higher-order fit cuts the m=0 error ~5-15x at the band edge (now on par with /
    better than healpy ring weights). See tests/test_double_fourier_sphere.py.
    """
    nside = mp.shape[1] // 4
    n_rings = mp.shape[0] // 2
    npts = max(2, min(POLE_INTERP_NPTS, nside - 1))

    latitudes = create_latitude_array(nside)
    mp = np.asarray(mp)

    # North pole (latitude 90): stencil = the npts northmost rings (latitudes
    # `latitudes[:npts]`, just below the pole) and their mirror images across the
    # pole (the southern-hemisphere rings of the DFS doubling, at 180 - latitude).
    north_theta = np.concatenate((np.flip(latitudes[:npts]), 180 - latitudes[:npts]))
    north_fp = np.concatenate((np.flip(mp[:npts], axis=0), np.flip(mp[-npts:], axis=0)))
    north_pole_mp = _pole_lagrange_weights(north_theta, 90.0) @ north_fp

    # South pole (latitude -90): same construction, mirrored. The original code
    # interpolated on the negated-latitude axis to reuse x0 = 90; keep that.
    south_theta = np.concatenate((latitudes[-npts:], -180 - np.flip(latitudes[-npts:])))
    south_fp = np.concatenate(
        (mp[n_rings - npts : n_rings], mp[n_rings : n_rings + npts])
    )
    south_pole_mp = _pole_lagrange_weights(-south_theta, 90.0) @ south_fp

    # Add the polar rings to the map. Keep the dtype of the input so a complex
    # spin field (Q + iU) is not silently truncated to real. The Lagrange pole
    # fill is linear in the samples, so it applies unchanged to the complex field;
    # a spin-|s|>=1 field vanishes at the pole (sin^|m+s|(theta/2) cos^|m-s|(theta/2)),
    # which the extrapolation of the genuine ring values reproduces (-> ~0).
    double_map = np.zeros((mp.shape[0] + 2, mp.shape[1]), dtype=mp.dtype)
    double_map[0] = north_pole_mp
    double_map[1 : n_rings + 1] = mp[:n_rings]
    double_map[n_rings + 1] = south_pole_mp
    double_map[n_rings + 2 :] = mp[n_rings:]

    return double_map


def compute_ring_area_weights(nside):
    theta = create_latitude_array(nside)
    theta = np.concatenate(([90.0], theta, [-90.0]))  # [90, ..., -90]

    ring_borders = np.zeros(len(theta) + 1)
    ring_borders[1:-1] = theta[:-1] + np.diff(theta) / 2
    ring_borders[0] = 90
    ring_borders[-1] = -90
    ring_borders = np.deg2rad(ring_borders + 90.0)  # [90, -90] -> [pi, 0]

    ring_areas = np.zeros(len(ring_borders) - 1)
    ring_areas = -2 * np.pi * (np.cos(ring_borders[:-1]) - np.cos(ring_borders[1:]))

    assert np.isclose(np.sum(ring_areas), 4 * np.pi), (
        "Sum of ring areas should be equal to 4*pi"
    )

    hp_pix_area = hp.nside2pixarea(nside)
    pixel_area = ring_areas / (4 * nside)
    correction = pixel_area / hp_pix_area

    return correction
