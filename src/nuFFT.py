"""Latitude nuFFT -- the pipeline's only ill-conditioned stage.

After the DFS doubling there are ~8*nside latitude samples at the (clustered)
HEALPix colatitudes, and the analysis fits latitude Fourier modes to them. How many
modes you fit decides everything, because the Vandermonde's conditioning depends on
the band:

    nside                     16     32      64       128
    cond(A), |k|<=4*nside    ~2e2   ~1e5   ~8e10    ~6e15 (~1/eps)   <- SQUARE
    cond(A), |k|<=2*nside    ~1.15  ~1.15  ~1.16    ~1.16            <- band-limited

``apply_nuFFT`` exposes two regimes (``solve_modes`` + ``solver``):

* DEFAULT -- ``solve_modes = 4*nside+1`` (|k| <= 2*nside), ``solver="cg"``,
  ``N_modes = solve_modes`` (no zero-pad). This is the band a band-limited
  (lmax = 2*nside) signal lives in; the Vandermonde is WELL conditioned at every
  nside, so finufft + CG converges in a few iterations at machine precision and stays
  O(N log N). The result feeds the FSHT at the NATURAL band L = lmax = 2*nside (the
  compact (L+1, 2L+1) g-array) -- half the rows/cols and, since ``fourier2sph`` is
  ~O(L^3), ~8x faster than the old L = 4*nside array, at bit-identical accuracy.
  Accurate (forward & round trip both converge, ~1e-3 by nside 256, ~1e-4 by 1024)
  and SCALABLE to nside 1024-2048. The small round-trip residual is above-band polar
  aliasing the band can't (and arguably shouldn't) represent -- the paper's truncation.
  (Set ``N_modes > solve_modes`` to zero-pad into a wider FSHT band; rarely needed.)

* EXACT round trip -- ``solve_modes = 8*nside+1`` (the SQUARE interpolation, one mode
  per sample), ``solver="svd"``. Reproduces the map bit-for-bit, but the square
  Vandermonde is severely ill-conditioned: CG on the normal equations sees the
  SQUARED condition number and finufft's error is amplified by cond (round trip
  floors ~8e-3 by nside 64), so it needs the dense truncated-SVD ``svd_nufft_forward``
  (exact matrix, cached & shared across longitude columns; O(nside^3), fine to
  ~nside 256). Good for bit-exact round trips up to nside ~64; the square band is a
  genuine 1/eps wall by nside 128 that NO solver escapes.

Synthesis (``cg_nufft_backward`` / ``inverse_nuFFT``) is a plain NUFFT evaluation
(well conditioned), so it stays O(N log N) in both regimes.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import finufft
from scipy.sparse.linalg import cg, lsmr, LinearOperator

from . import _openmp
from .cg import cg_normal_equations, weighted_norm2
from .data_interpolation import create_latitude_array
from .double_fourier_sphere import FoldPlan

logger = logging.getLogger(__name__)


def compute_voronoi_weights_1d(x, period=2 * np.pi):
    """Voronoi cell widths for monotonic samples on a PERIODIC axis.

    The DFS doubling makes the latitude variable periodic with period ``2*pi``, so the
    samples live on a circle and the first and last cells meet at the seam. Each point
    gets half the gap to its neighbour on either side, with the neighbour of the first
    point being the last one, wrapped:

        g[i] = x[i] - x[i+1]  (descending input),  g[-1] = x[-1] - x[0] + period
        w[i] = (g[i-1] + g[i]) / 2                 (indices mod M)

    The widths sum to ``period`` and are all positive.

    This used to clamp the two end cells to fixed boundaries ``(pi, -pi)`` instead of
    wrapping, which is wrong on a circle and was not symmetric. The north pole sits
    exactly ON the seam at ``x = pi``, so it was given only the half of its cell that
    falls below ``pi`` and the southernmost mirrored ring absorbed the other half:
    measured at nside 8, ``0.0510`` against a true ``0.1021``, and ``0.1533`` against a
    true ``0.1022``. Every other weight was already symmetric to 4.4e-16 and the total
    was already exactly ``2*pi`` -- the old code conserved the measure and misallocated
    it. The starved row is the Lagrange pole fill, which carries the m=0 high-l
    accuracy, and the asymmetry also blocked exploiting the DFS mirror symmetry to
    halve the solve (see fix_pass_2.md item 1).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError(f"need at least 2 samples along one axis, got shape {x.shape}")
    step = np.diff(x)
    if np.all(step < 0):
        gaps = np.concatenate((-step, [x[-1] - x[0] + period]))
    elif np.all(step > 0):
        gaps = np.concatenate((step, [x[0] - x[-1] + period]))
    else:
        raise ValueError("samples must be strictly monotonic to have Voronoi cells")
    return 0.5 * (np.roll(gaps, 1) + gaps)


def _default_N_modes(M_samples):
    # Number of latitude Fourier modes to solve for. The latitude bandwidth is
    # set by the number of latitude SAMPLES (~8*nside after the DFS doubling),
    # NOT by the longitude count. Using only 4*nside+1 modes (the old
    # `n_trans + 1`) under-resolves latitude by ~2x and aliases high-frequency
    # latitude content down into low ell. Default to the smallest ODD count
    # >= M_samples: odd so the modes are symmetric (-L..L) for the downstream
    # spherical-harmonic step, and >= M_samples so the Vandermonde system
    # interpolates the samples (the extra mode sits in the null space). NOTE the
    # FSHT `preparation` is calibrated for L = (N_modes-1)//2 = 4*nside, i.e. this
    # exact count -- it does NOT work at other latitude band limits, so changing
    # N_modes also requires reworking preparation. See module docstring / CLAUDE.md.
    return M_samples + (M_samples % 2 == 0)


# Cache of the latitude Vandermonde SVD, keyed by (sample locations, N_modes).
# The Vandermonde A is identical for every longitude column, so one factorization
# serves all n_trans transforms and is reused across forward calls at a given nside.
_SVD_CACHE = {}


def _vandermonde_svd(x, N_modes, weights):
    """SVD of the weighted latitude Vandermonde sqrt(W) A (cached).

    A[j, k] = exp(+i * k * x_j) with modes k = -L..L (L = (N_modes-1)//2), the same
    -L..L ordering finufft produces with modeord=0 (row L = DC), so the resulting
    f_hat aligns with what `FSHT.preparation` expects. ``weights`` are the
    density-compensation (Voronoi) weights; the solve is the weighted least squares
    min || sqrt(W) (A f_hat - samples) ||, the same problem the CG path solves via
    the weighted normal equations (so the two solvers agree where both are stable).
    """
    key = (x.shape[0], N_modes, hash(x.tobytes()))
    cached = _SVD_CACHE.get(key)
    if cached is None:
        L = (N_modes - 1) // 2
        k = np.arange(-L, L + 1)
        A = np.exp(1j * np.outer(x, k))  # (M_samples, N_modes), modes -L..L
        sw = np.sqrt(np.abs(weights))
        U, s, Vh = np.linalg.svd(sw[:, None] * A, full_matrices=False)
        cached = (sw, U, s, Vh)
        _SVD_CACHE[key] = cached
    return cached


def svd_nufft_forward(x, f_samples, N_modes=None, rcond=1e-13):
    """Latitude analysis via a dense truncated-SVD pseudo-inverse.

    Solves the same weighted least-squares latitude fit as ``cg_nufft_forward`` but
    with an explicit, cached SVD of the (small, shared) Vandermonde instead of CG on
    the normal equations. This matters at high nside: the latitude Vandermonde at the
    clustered HEALPix colatitudes is severely ill-conditioned (cond ~ 8e10 at
    nside=64), and CG on the *normal* equations works at the SQUARED condition number
    while finufft's transform error gets amplified by the condition number -- together
    they floor the round trip at ~1e-2 by nside=64. A direct SVD solve on the exact
    matrix avoids both, restoring near machine-precision invertibility (round trip
    ~1.8e-6 at nside=64) at the same forward accuracy. ``rcond`` truncates singular
    directions below ``rcond * sigma_max`` -- a small value keeps the (well-determined)
    modes and exposes the conditioning/accuracy/invertibility trade-off.

    Cost is O(M_samples^2 * N_modes) for the one-off factorization (cached per nside),
    i.e. O(nside^3); fine for research resolutions up to ~nside 256. For nside >= 128
    the Vandermonde is numerically rank-deficient (cond ~ 1/eps) and NO solver
    recovers invertibility -- that regime needs the FSHT reworked at L = lmax = 2*nside
    so the latitude solve can drop to the well-conditioned 4*nside+1 modes.

    Returns (f_hat of shape (N_modes, n_trans), info=0) to match ``cg_nufft_forward``.
    """
    n_trans = f_samples.shape[0]
    M_samples = f_samples.shape[1]
    if N_modes is None:
        N_modes = _default_N_modes(M_samples)

    weights = compute_voronoi_weights_1d(x)
    sw, U, s, Vh = _vandermonde_svd(x, N_modes, weights)

    data = f_samples.T  # (M_samples, n_trans)
    weighted = sw[:, None] * data
    s_inv = np.where(s > rcond * s[0], 1.0 / s, 0.0)
    f_hat = (Vh.conj().T * s_inv) @ (U.conj().T @ weighted)  # (N_modes, n_trans)
    return f_hat, 0


_EXECUTORS = {}


def _executor(workers):
    """A process-lifetime thread pool of the given size, created on first use."""
    ex = _EXECUTORS.get(workers)
    if ex is None:
        ex = ThreadPoolExecutor(max_workers=workers)
        _EXECUTORS[workers] = ex
    return ex


# Below this many transforms the thread hand-off costs more than the split saves.
# Measured crossover, spin forward, alternating A/B: 0.31x at 32 transforms, 0.72x at
# 64, 1.24x at 128, 1.63x at 256.
WORKER_MIN_TRANSFORMS = 128  # nside 32


def default_workers(n_trans: int = None) -> int:
    """How many threads the latitude solve splits its transform batch over.

    Set ``HP2SPH_NUFFT_WORKERS`` to override; 1 disables the split. Otherwise it is the
    core count capped at 7, above which the measured gain flattens.

    ``n_trans`` is the batch size. Small batches return 1: the per-call thread hand-off
    is fixed while the work per thread shrinks, so below ``WORKER_MIN_TRANSFORMS`` the
    split is a slowdown rather than a speed-up.
    """
    env = os.environ.get("HP2SPH_NUFFT_WORKERS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    if n_trans is not None and n_trans < WORKER_MIN_TRANSFORMS:
        return 1
    return max(1, min(7, os.cpu_count() or 1))


class _BatchPlan:
    """A ``finufft.Plan`` whose transform batch is split across threads.

    The latitude solve runs ``4*nside`` independent 1-D transforms, each only a few
    thousand points. finufft parallelises WITHIN one transform, not across a batch, so
    on this shape its own threading is at best marginal and at small thread counts it is
    much worse than serial. Splitting the batch across threads and giving each chunk a
    single-threaded plan scales nearly linearly instead, because the finufft call
    releases the GIL.

    Interchangeable with ``finufft.Plan`` for the ``setpts`` and ``execute`` calls used
    here. ``workers=1`` builds one plan and adds no threading.

    The chunks are contiguous row blocks of a C-ordered ``(n_trans, ...)`` array, so
    every chunk is a view and no data is copied. Each thread owns its own plan, since
    a plan may not be executed concurrently with itself.
    """

    def __init__(self, kind, n_modes, n_trans, isign, eps, workers):
        workers = max(1, min(int(workers), n_trans))
        self.bounds = np.linspace(0, n_trans, workers + 1).astype(int)
        self.plans = [
            finufft.Plan(
                kind,
                (n_modes,),
                n_trans=int(hi - lo),
                isign=isign,
                dtype=np.complex128,
                eps=eps,
                nthreads=1,
            )
            for lo, hi in zip(self.bounds[:-1], self.bounds[1:])
        ]
        self.workers = workers

    def setpts(self, x):
        for p in self.plans:
            p.setpts(x)

    def execute(self, data, out):
        if self.workers == 1:
            return self.plans[0].execute(data, out)
        lo, hi = self.bounds[:-1], self.bounds[1:]
        ex = _executor(self.workers)
        list(
            ex.map(
                lambda i: self.plans[i].execute(
                    data[lo[i] : hi[i]], out[lo[i] : hi[i]]
                ),
                range(self.workers),
            )
        )
        return out


def nyquist_discrepancy(mp: np.ndarray) -> float:
    """How badly the longitude grid cannot represent its own Nyquist mode.

    The ``4*nside``-point longitude grid gives modes ``m = +2*nside`` and
    ``m = -2*nside`` the same slot, and the HEALPix rings do not agree on their relative
    phase: a ring whose first pixel sits at ``phi0`` sees them combined as
    ``c_-(theta) + c_+(theta) * exp(i * 4*nside * phi0)``, and ``phi0`` alternates
    between ``0`` and ``pi/(4*nside)`` along the equatorial belt, so the sign of the
    second term flips from one ring to the next. A single latitude column can carry
    ``c_-`` or ``c_+`` but not both, so ``c_+`` cannot be fitted at all.

    That makes ``c_+`` an irreducible error in the forward model, and it is what stops
    the latitude least squares from reaching zero residual. This returns an estimate of
    its size, in the same weighted norm the solver's residual uses, so it can be passed
    to ``apply_nuFFT(level=...)`` as the data-error level of the discrepancy principle.

    Because the two terms differ by a sign from ring to ring, ``c_+`` is the
    ring-to-ring ALTERNATING part of the measured Nyquist slot, and that is available
    from the data before any solve. Cost is O(nside).

    Parameters
    ----------
    mp : ndarray, shape (8*nside, 4*nside)
        The DFS array, in the natural (fftshifted) longitude order that ``DFS``
        produces, so that column 0 is the Nyquist slot ``m = -2*nside``.

    Returns
    -------
    float
        The estimated level, comparable to ``||b - A f_hat||_W``.

    Notes
    -----
    The estimate covers only this one defect. It is close to the whole least-squares
    residual when the field carries power at ``|m| = 2*nside``, and an underestimate
    when it does not -- a field band-limited below the grid Nyquist has no such content,
    and its residual floor is set by something else. Combine it with a fallback rule
    rather than relying on it alone.
    """
    mp = np.asarray(mp)
    n_trans = mp.shape[1]
    nside = n_trans // 4
    w = compute_voronoi_weights_1d(_upsampled_latitudes(nside))
    v = mp[:, 0]
    belt = np.arange(nside, 3 * nside)
    # half the difference of neighbouring belt rings: the smooth c_- cancels to first
    # order in the ring spacing, the sign-flipped c_+ adds.
    d = 0.5 * (v[belt] - v[belt + 1])
    # factor 2: the DFS doubling carries every ring twice.
    return float(np.sqrt(2.0 * np.sum(w[belt] * (d.real**2 + d.imag**2))))


def _fold_ops(fold, n_trans, M_samples):
    """(apply, adjoint) for the polar-ring longitude alias, or ``(None, None)``.

    ``fold`` is the ``(target, phase)`` pair from
    ``double_fourier_sphere.dfs_fold_plan``, both ``(M_samples, n_trans)``. ``apply``
    turns the model's wide-band longitude spectrum at each latitude into what the
    HEALPix ring at that latitude actually measures -- a scatter-add over alias
    families -- and ``adjoint`` is the matching gather, so the pair is an exact adjoint
    and LSMR/CG stay valid.

    Both work on the ``(n_trans, M_samples)`` layout the solvers use, and both return
    that layout C-contiguous.

    Each takes an optional ``out``. Passing the input array itself is allowed and makes
    the operation in place, which is what keeps the solver's peak memory down; see the
    disjointness note below for why that is safe.

    Representation
    --------------
    The fold is stored as IDENTITY PLUS A SPARSE CORRECTION rather than as a dense
    per-entry index and phase. Three facts make that exact:

    * an entry that stays in its own slot carries phase exactly ``1.0``;
    * only the RELAXED entries move, 2.8% of the array at nside 256 and falling with
      resolution;
    * an entry always moves onto a slot that itself stays put, so the set of sources
      and the set of destinations are disjoint.

    So the correction is three arrays of length R, the number of relaxed entries, in
    place of six of shape ``(n_trans, M_samples)``. Disjointness is also what licenses
    the in-place form: zeroing the sources cannot destroy a destination, and writing
    the gathered sources cannot disturb a value still to be read.

    Duplicate destinations (several relaxed modes aliasing onto one slot) are summed by
    ``np.bincount`` over the COMPACT destination set, so the accumulator is O(R) rather
    than the size of the array.
    """
    if fold is None:
        return None, None
    if isinstance(fold, FoldPlan):
        if (fold.n_trans, fold.n_rows) != (n_trans, M_samples):
            raise ValueError(
                f"fold plan is for ({fold.n_trans}, {fold.n_rows}), "
                f"solver wants ({n_trans}, {M_samples})"
            )
        src, dst, ph = fold.src, fold.dst, fold.phase
    else:
        target = np.asarray(fold[0])
        phase = np.asarray(fold[1])
        if target.shape != (M_samples, n_trans) or phase.shape != target.shape:
            raise ValueError(
                f"fold arrays must be ({M_samples}, {n_trans}), got {target.shape}"
            )
        # (column c, row r) sits at c * M + r in a C-contiguous (n_trans, M) buffer.
        moved = target != np.arange(n_trans)[None, :]
        rel_r, rel_c = np.nonzero(moved)
        src = (rel_c * M_samples + rel_r).astype(np.intp)
        dst = (target[rel_r, rel_c] * M_samples + rel_r).astype(np.intp)
        ph = phase[rel_r, rel_c]
    ph_conj = np.conj(ph)

    # compact the destinations so the scatter accumulator is O(R), not O(n_trans * M)
    uniq, inv = np.unique(dst, return_inverse=True)
    inv = inv.astype(np.intp).ravel()
    pair = np.empty(2 * inv.size, dtype=np.intp)
    pair[0::2] = 2 * inv
    pair[1::2] = 2 * inv + 1
    acc_len = 2 * uniq.size

    # apply and adjoint keep SEPARATE lazy buffers, so a caller may hold both results
    # at once; sharing one silently makes the second call clobber the first.
    scratch = [None, None]

    def _out_for(model, out, slot):
        if out is None:
            b = scratch[slot]
            if b is None or b.shape != model.shape:
                b = np.empty(model.shape, dtype=np.complex128)
                scratch[slot] = b
            out = b
        if out is not model:
            np.copyto(out, model)
        return out

    def apply(model, out=None):
        flat_in = model.reshape(-1)
        vals = flat_in[src] * ph  # read the sources before anything is overwritten
        out = _out_for(model, out, 0)
        flat = out.reshape(-1)
        flat[src] = 0.0
        acc = np.bincount(pair, weights=vals.view(np.float64), minlength=acc_len)
        flat[uniq] += acc.view(np.complex128)
        return out

    def adjoint(residual, out=None):
        gathered = residual.reshape(-1)[dst] * ph_conj
        out = _out_for(residual, out, 1)
        out.reshape(-1)[src] = gathered
        return out

    return apply, adjoint


def _mirror_plan(x, spin, n_trans, N_modes, tol=1e-9):
    """Half-domain plan from the DFS mirror symmetry, or ``None``.

    The DFS doubling makes the latitude samples closed under ``x -> -x`` with the two
    poles as fixed points, and makes the array obey ``d[mu(r), c] = parity[c] d[r, c]``
    with ``parity = (-1)^(m + spin)``. The latitude coefficients then obey
    ``c_{-k} = parity[c] c_k``, so the whole least squares restricts to ``4*nside + 1``
    rows instead of ``8*nside`` and ``2*nside + 1`` coefficients per column instead of
    ``4*nside + 1``. This is an EXACT restructuring: the minimiser of the full problem
    already satisfies the symmetry.

    Returns ``(mu, rows, mult, parity, scale, even)``.

    Two details are load-bearing.  ``c_{-k} = -c_k`` forces ``c_0 = 0`` on odd-parity
    columns, so ``even`` pins that slot; writing it for every column makes the embedded
    vector asymmetric and the half operator disagrees with the full one by 52%.  And the
    embedding must be an ISOMETRY -- a coefficient with ``k >= 1`` feeds both ``+k`` and
    ``-k``, so it carries ``1/sqrt(2)`` -- otherwise ``P^H N P`` has a different spectrum
    from ``N`` on the symmetric subspace and CG needs 257 iterations instead of 115 at
    nside 64, i.e. a cheaper product and a slower solve.
    """
    if spin is None or N_modes % 2 == 0:
        return None
    M = len(x)
    xx = np.mod(x, 2 * np.pi)
    xm = np.mod(-x, 2 * np.pi)
    order = np.argsort(xx)
    xs = xx[order]
    pos = np.searchsorted(xs, xm)
    c0, c1 = (pos - 1) % M, pos % M

    def circ(a, b):
        d = np.abs(a - b)
        return np.minimum(d, 2 * np.pi - d)

    d0, d1 = circ(xs[c0], xm), circ(xs[c1], xm)
    mu = order[np.where(d0 <= d1, c0, c1)]
    if np.minimum(d0, d1).max() > tol:
        return None  # the sample set is not closed under the involution
    idx = np.arange(M)
    if not np.array_equal(mu[mu], idx):
        return None
    rows = np.nonzero(idx <= mu)[0]
    mult = np.where(mu[rows] == rows, 1.0, 2.0)
    m = np.arange(n_trans) - n_trans // 2
    parity = ((-1.0) ** (m + spin)).astype(float)
    K = (N_modes - 1) // 2
    scale = np.full(K + 1, 1.0 / np.sqrt(2.0))
    scale[0] = 1.0
    even = (parity > 0).astype(float)
    return mu, rows, mult, parity, scale, even


def _is_mirror_symmetric(f_samples, mu, parity, rtol=1e-8):
    """Does the data obey ``d[mu(r), c] = parity[c] d[r, c]``?

    ``cg_nufft_forward`` is handed a bare array, so this is checked rather than assumed;
    an asymmetric input falls back to the full domain instead of being silently
    symmetrised. One pass over the array, about 2% of a single matrix-vector product.
    """
    scale = np.abs(f_samples).max()
    if scale == 0.0:
        return True
    diff = np.abs(f_samples[:, mu] - parity[:, None] * f_samples).max()
    return diff <= rtol * scale


def _cg_nufft_forward_half(
    x,
    f_samples,
    N_modes,
    plan_half,
    rtol,
    maxiter,
    eps,
    sample_mask,
    fold,
    level=None,
    eta=None,
    delay=5,
    monitor=None,
    workers=1,
):
    """``cg_nufft_forward`` restricted to the mirror-symmetric subspace."""
    mu, rows, mult, parity, scale, even = plan_half
    n_trans, M_samples = f_samples.shape
    Mh = len(rows)
    K = (N_modes - 1) // 2

    plan_forward = _BatchPlan(2, N_modes, n_trans, 1, eps, workers)
    plan_adjoint = _BatchPlan(1, N_modes, n_trans, -1, eps, workers)
    xh = np.ascontiguousarray(x[rows])
    plan_forward.setpts(xh)
    plan_adjoint.setpts(xh)

    # A mirrored row contributes exactly what its partner does, so dropping it and
    # doubling the partner's weight leaves the least squares (and ``norm``) unchanged.
    w = compute_voronoi_weights_1d(x)[rows] * mult
    # The weight of entry (c, r) is w[r] where the mask keeps it and 0 where it does
    # not, so the dense (n_trans, Mh) product is a 1-D vector plus a list of dropped
    # positions. Storing it that way costs a few MB instead of a few hundred, and the
    # weighting becomes a broadcast multiply plus a sparse zeroing.
    drop = None
    if isinstance(fold, FoldPlan):
        if sample_mask is not None:
            raise ValueError(
                "a fold plan carries its own dropped entries; pass sample_mask=None"
            )
        drop = fold.drop
        lost = np.bincount(drop // Mh, weights=w[drop % Mh], minlength=n_trans)
        norm = float((w.sum() - lost).mean())
    elif sample_mask is None:
        norm = np.sum(w)
    else:
        mask = np.asarray(sample_mask)
        if mask.shape == (M_samples, n_trans):
            mask = mask.T
        if mask.shape != (n_trans, M_samples):
            raise ValueError(
                f"sample_mask must be ({n_trans}, {M_samples}) or its transpose, "
                f"got {np.shape(sample_mask)}"
            )
        drop = np.flatnonzero(~np.asarray(mask[:, rows], dtype=bool).reshape(-1))
        lost = np.bincount(drop // Mh, weights=w[drop % Mh], minlength=n_trans)
        norm = (w.sum() - lost)[:, None]
        if fold is not None:
            norm = norm.mean()
        elif np.any(norm == 0):
            raise ValueError("sample_mask leaves a longitude mode with no samples")

    def weight(buf):
        """Multiply by the (masked) quadrature weights, in place."""
        np.multiply(buf, w, out=buf)
        if drop is not None:
            buf.reshape(-1)[drop] = 0.0
        return buf

    if fold is None or isinstance(fold, FoldPlan):
        fold_h = fold
    else:
        fold_h = (fold[0][rows], fold[1][rows])
    fold_apply, fold_adjoint = _fold_ops(fold_h, n_trans, Mh)
    # ``_fold_ops`` has copied everything it needs into its own index arrays, and these
    # row selections are two more full-size arrays. At nside 1024 they are 0.4 GB.
    del fold_h

    # THREE buffers carry the whole matrix-vector product, and every step below writes
    # into one of them rather than allocating. ``coef`` doubles as the expanded
    # coefficient vector on the way out and the adjoint NUFFT's output on the way back
    # -- the first is dead by the time the second is written -- and the entire data-side
    # chain (fold, weights, adjoint fold) runs in place on ``gbuf``, which the fold's
    # source/destination disjointness makes exact. This is what the solver's peak
    # memory is, and it gates nside 2048.
    coef = np.zeros((n_trans, N_modes), dtype=np.complex128)
    gbuf = np.zeros((n_trans, Mh), dtype=np.complex128)
    rbuf = np.empty((n_trans, K + 1), dtype=np.complex128)
    rflat = rbuf.reshape(-1)
    par = parity[:, None]

    def expand(ch):
        # Write the symmetric half straight into ``coef`` and read the mirrored half
        # back out of it. Columns K..2K and 0..K-1 are disjoint, so the reversed read
        # and the write do not overlap and no scratch copy is needed.
        np.multiply(ch, scale, out=coef[:, K:])
        np.multiply(par, coef[:, 2 * K : K : -1], out=coef[:, :K])
        coef[:, K] *= even
        return coef

    def restrict(fl):
        # ``out[:, 0] = fl[:, K]`` and ``out[:, 1:] = fl[:, K+1:] + par * reversed``.
        # Forming the reversed product directly in ``rbuf[:, 1:]`` and adding the
        # forward half to it drops the scratch the other order would need.
        np.multiply(par, fl[:, K - 1 :: -1], out=rbuf[:, 1:])
        rbuf[:, 1:] += fl[:, K + 1 :]
        rbuf[:, 0] = fl[:, K]
        np.multiply(rbuf, scale, out=rbuf)
        rbuf[:, 0] *= even
        return rbuf

    def adjoint_of(samples):
        if samples is not gbuf:
            np.copyto(gbuf, samples)
        weight(gbuf)
        if fold_adjoint is not None:
            fold_adjoint(gbuf, out=gbuf)
        plan_adjoint.execute(gbuf, coef)
        restrict(coef)
        np.divide(rbuf, norm, out=rbuf)
        return rflat

    def apply_AHA(vec):
        plan_forward.execute(expand(vec.reshape(n_trans, K + 1)), gbuf)
        if fold_apply is not None:
            fold_apply(gbuf, out=gbuf)
        return adjoint_of(gbuf)

    bh = np.ascontiguousarray(f_samples[:, rows])
    # ||b||_W^2 = <b, W b>, evaluated through the same in-place weighting rather than
    # by forming a dense weight array just to square against it.
    np.copyto(gbuf, bh)
    bw2 = float(np.vdot(bh, weight(gbuf)).real)
    # ``adjoint_of`` returns a view of ``rbuf``, which the iteration overwrites, so the
    # right-hand side -- which must survive the whole solve -- takes a copy.
    rhs = adjoint_of(bh).copy()
    n_half = n_trans * (K + 1)
    if level is None and eta is None and monitor is None:
        A = LinearOperator(
            shape=(n_half, n_half), matvec=apply_AHA, dtype=np.complex128
        )
        sol, info = cg(
            A, rhs, x0=np.zeros(n_half, dtype=np.complex128), rtol=rtol, maxiter=maxiter
        )
    else:
        mon = None
        if monitor is not None:

            def mon(k, rho, rrel, vec):
                monitor(
                    k,
                    rho,
                    rrel,
                    lambda: expand(vec.reshape(n_trans, K + 1)).copy().T,
                )

        if np.size(norm) != 1:
            raise NotImplementedError(
                "the data residual is defined against one least-squares problem; a "
                "per-column norm rescales each block separately. Use fold=..."
            )
        sol, info = cg_normal_equations(
            apply_AHA,
            rhs,
            bw2,
            norm=float(norm),
            rtol=rtol,
            maxiter=maxiter,
            level=level,
            eta=eta,
            delay=delay,
            monitor=mon,
        )
    return expand(sol.reshape(n_trans, K + 1)).copy().T, info


def lsmr_nufft_forward(
    x,
    f_samples,
    N_modes=None,
    sample_mask=None,
    rtol=1e-9,
    maxiter=None,
    eps=1e-12,
    fold=None,
):
    """Latitude analysis via LSMR on the least-squares problem itself.

    Solves ``min || sqrt(W) (A f_hat - samples) ||`` directly, rather than forming the
    normal equations. Two reasons this is the solver to use with a ``sample_mask``:

    * With entries masked out, the high-``|m|`` columns keep samples only in the
      equatorial belt and the problem becomes RANK-DEFICIENT. Started from zero, LSMR
      converges to the MINIMUM-NORM least-squares solution, which is exactly the right
      answer for "this mode was never sampled here" -- do not invent content in the
      null space. CG on the normal equations has no such property and drifts into the
      null space instead.
    * LSMR works with ``cond(A)``, not ``cond(A)^2``.

    Cost per iteration is one forward + one adjoint NUFFT, the same as a CG iteration.
    Returns ``(f_hat of shape (N_modes, n_trans), info)`` like ``cg_nufft_forward``.
    """
    n_trans = f_samples.shape[0]
    M_samples = f_samples.shape[1]
    if N_modes is None:
        N_modes = _default_N_modes(M_samples)

    plan_forward = finufft.Plan(
        2, (N_modes,), n_trans=n_trans, isign=1, dtype=np.complex128, eps=eps
    )
    plan_adjoint = finufft.Plan(
        1, (N_modes,), n_trans=n_trans, isign=-1, dtype=np.complex128, eps=eps
    )
    plan_forward.setpts(x)
    plan_adjoint.setpts(x)

    weights = np.abs(compute_voronoi_weights_1d(x))
    sw = np.sqrt(weights)[None, :]  # (1, M_samples), broadcasts over columns
    if sample_mask is not None:
        mask = np.asarray(sample_mask, dtype=float)
        if mask.shape == (M_samples, n_trans):
            mask = mask.T
        if mask.shape != (n_trans, M_samples):
            raise ValueError(
                f"sample_mask must be ({n_trans}, {M_samples}) or its transpose, "
                f"got {np.shape(sample_mask)}"
            )
        sw = sw * np.sqrt(mask)  # zero rows drop out of the least squares entirely

    fold_apply, fold_adjoint = _fold_ops(fold, n_trans, M_samples)

    def matvec(f_hat_vec):
        out = np.zeros((n_trans, M_samples), dtype=np.complex128)
        plan_forward.execute(
            np.ascontiguousarray(f_hat_vec.reshape(n_trans, N_modes)), out
        )
        if fold_apply is not None:
            out = fold_apply(out)
        return (sw * out).ravel()

    def rmatvec(res_vec):
        out = np.zeros((n_trans, N_modes), dtype=np.complex128)
        weighted = sw * res_vec.reshape(n_trans, M_samples)
        if fold_adjoint is not None:
            weighted = fold_adjoint(weighted)
        plan_adjoint.execute(np.ascontiguousarray(weighted), out)
        return out.ravel()

    B = LinearOperator(
        shape=(n_trans * M_samples, n_trans * N_modes),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=np.complex128,
    )
    b = (sw * np.asarray(f_samples)).ravel()
    result = lsmr(B, b, atol=rtol, btol=rtol, maxiter=maxiter)
    f_hat, istop = result[0], result[1]
    # istop 1/2 = converged to a solution / least-squares solution; 7 = hit maxiter
    return f_hat.reshape(n_trans, N_modes).T, (0 if istop in (1, 2) else int(istop))


def cg_nufft_forward(
    x,
    f_samples,
    N_modes=None,
    rtol=1e-9,
    maxiter=None,
    eps=1e-12,
    sample_mask=None,
    fold=None,
    spin=None,
    level=None,
    eta=None,
    delay=5,
    monitor=None,
    workers=1,
    half_domain=False,
):
    # Get dimensions
    n_trans = f_samples.shape[0]  # = 4*nside (number of longitude transforms)
    M_samples = f_samples.shape[1]  # = 8*nside (number of latitude samples)
    if N_modes is None:
        N_modes = _default_N_modes(M_samples)

    # With ``spin`` known, the DFS mirror symmetry halves both the latitude samples and
    # the unknowns at no cost in accuracy -- same iteration count, same answer to ~7e-15,
    # 1.5-1.65x wall from nside 32 to 256 (see ``_mirror_plan``). The symmetry of the
    # DATA is checked rather than assumed, so an array that does not have it falls back
    # to the full domain below.
    plan_half = _mirror_plan(x, spin, n_trans, N_modes)
    # ``half_domain`` says the caller already dropped the mirrored rows, so the symmetry
    # holds by construction and there is nothing left to check. The rows the plan keeps
    # are exactly ``0 .. 4*nside``, so every ``[..., rows]`` below is then the identity
    # and the half arrays flow through untouched.
    if plan_half is not None and (
        half_domain or _is_mirror_symmetric(f_samples, plan_half[0], plan_half[3])
    ):
        return _cg_nufft_forward_half(
            x,
            f_samples,
            N_modes,
            plan_half,
            rtol,
            maxiter,
            eps,
            sample_mask,
            fold,
            level=level,
            eta=eta,
            delay=delay,
            monitor=monitor,
            workers=workers,
        )

    # Precompute NUFFT plans with batch processing (n_trans transforms)
    plan_forward = _BatchPlan(2, N_modes, n_trans, 1, eps, workers)
    plan_adjoint = _BatchPlan(1, N_modes, n_trans, -1, eps, workers)

    # Set nonuniform points (same for all transforms)
    plan_forward.setpts(x)
    plan_adjoint.setpts(x)

    # Calculate Voronoi weights. With a ``sample_mask`` the weights become per-column:
    # a (ring, mode) entry the ring never resolved gets weight 0, so it drops out of the
    # least squares as MISSING rather than being asserted to be zero (see
    # ``data_interpolation.ring_fold_plan``). The blocks stay independent and each
    # A^H W A is still Hermitian positive semi-definite, so batched CG is unaffected;
    # only the elementwise weighting below changes shape from (M,) to (n_trans, M).
    weights = compute_voronoi_weights_1d(x)
    if sample_mask is None:
        norm = np.sum(weights)  # M_samples if weights = 1
    else:
        mask = np.asarray(sample_mask, dtype=float)
        if mask.shape == (M_samples, n_trans):
            mask = mask.T
        if mask.shape != (n_trans, M_samples):
            raise ValueError(
                f"sample_mask must be ({n_trans}, {M_samples}) or its transpose, "
                f"got {np.shape(sample_mask)}"
            )
        weights = weights[None, :] * mask
        norm = weights.sum(axis=1, keepdims=True)
        if fold is not None:
            # A per-column norm is a per-block scaling. That keeps A^H W A Hermitian only
            # while the blocks are independent; the fold couples them, so scaling row
            # block j alone would break the symmetry CG requires. Use a scalar.
            norm = norm.mean()
        elif np.any(norm == 0):
            raise ValueError("sample_mask leaves a longitude mode with no samples")

    # Reshape helpers
    def vec_to_mat_hat(vec):
        return vec.reshape(n_trans, N_modes)

    def vec_to_mat_samples(vec):
        return vec.reshape(n_trans, M_samples)

    def mat_to_vec(mat):
        return mat.ravel()

    if isinstance(fold, FoldPlan):
        raise ValueError(
            "a fold plan is in the half-DFS layout; the full-domain solve needs the "
            "dense (target, phase) pair. Pass spin= so the half-domain path is used."
        )
    fold_apply, fold_adjoint = _fold_ops(fold, n_trans, M_samples)

    # Define NUFFT operators with batch processing
    def forward_op(f_hat_vec):
        """A @ f_hat for all transforms (batched)"""
        f_hat_mat = vec_to_mat_hat(f_hat_vec)
        out = np.zeros((n_trans, M_samples), dtype=np.complex128)
        plan_forward.execute(f_hat_mat, out)
        if fold_apply is not None:
            out = fold_apply(out)
        return mat_to_vec(out)

    def calc_rhs(f_sample_init):
        """A^H @ f_samples for all transforms (batched)"""
        weighted_samples = f_sample_init * weights
        if fold_adjoint is not None:
            weighted_samples = fold_adjoint(weighted_samples)
        out = np.zeros((n_trans, N_modes), dtype=np.complex128)
        plan_adjoint.execute(np.ascontiguousarray(weighted_samples), out)
        return mat_to_vec(out / norm)

    def adjoint_op(f_samples_vec):
        """A^H @ f_samples for all transforms (batched)"""
        return calc_rhs(vec_to_mat_samples(f_samples_vec))

    # Solve (A^H A) f_hat = A^H f_samples using CG (batched)
    def apply_AHA(f_hat_vec):
        return adjoint_op(forward_op(f_hat_vec))

    # Reshape RHS and initial guess
    rhs = calc_rhs(f_samples)  # Flatten input data
    f_hat_guess = np.zeros(n_trans * N_modes, dtype=np.complex128)

    # Linear operator for CG
    A = LinearOperator(
        shape=(n_trans * N_modes, n_trans * N_modes),
        matvec=apply_AHA,
        dtype=np.complex128,
    )

    # Run CG. NOTE: CG runs on the NORMAL equations (A^H A), so it sees the SQUARED
    # condition number; the default rtol is tightened to 1e-9 (from 1e-6) because the
    # loose tol left the nside=32 round trip at ~1e-2 instead of ~1e-9. At nside>=64
    # the conditioning defeats CG regardless of rtol -- use solver="svd" there.
    if level is None and eta is None and monitor is None:
        f_hat_recon_flat, info = cg(A, rhs, x0=f_hat_guess, rtol=rtol, maxiter=maxiter)
    else:
        if np.size(norm) != 1:
            raise NotImplementedError(
                "the data residual is defined against one least-squares problem; a "
                "per-column norm rescales each block separately. Use fold=..."
            )
        mon = None
        if monitor is not None:

            def mon(k, rho, rrel, vec):
                monitor(k, rho, rrel, lambda: vec.reshape(n_trans, N_modes).T.copy())

        f_hat_recon_flat, info = cg_normal_equations(
            apply_AHA,
            rhs,
            weighted_norm2(f_samples, weights),
            norm=float(norm),
            rtol=rtol,
            maxiter=maxiter,
            level=level,
            eta=eta,
            delay=delay,
            monitor=mon,
        )

    # Reshape back to (4*nside, N_modes)
    return f_hat_recon_flat.reshape(n_trans, N_modes).T, info


def cg_nufft_backward(x, f_hat, eps=1e-12):
    """
    Synthesis: evaluate the latitude Fourier modes f_hat at the nonuniform
    sample locations x. This is the adjoint/forward direction of the analysis
    problem and is a plain type-2 NUFFT, s = A @ f_hat -- no linear solve is
    needed.

    The previous implementation solved (A A^H) s = A f_hat with CG. That system
    is SINGULAR whenever there are more samples than modes (A A^H has rank
    <= N_modes < M_samples), so CG returned a min-norm solution that does not
    equal A f_hat and injected a large reconstruction error. Direct evaluation
    is both correct and faster.

    Synthesis is well conditioned (a plain evaluation), so finufft's ``eps`` is
    NOT amplified here -- unlike the analysis solve. Keeping it as a NUFFT (rather
    than a dense matvec) preserves the O(N log N) scaling of the inverse transform.
    """
    # Get dimensions
    n_trans = f_hat.shape[0]  # number of longitude transforms (= 4*nside)
    N_modes = f_hat.shape[1]  # number of latitude Fourier modes
    M_samples = len(x)

    plan_forward = finufft.Plan(
        2, (N_modes,), n_trans=n_trans, isign=1, dtype=np.complex128, eps=eps
    )
    plan_forward.setpts(x)

    out = np.zeros((n_trans, M_samples), dtype=np.complex128)
    plan_forward.execute(np.ascontiguousarray(f_hat), out)

    # info kept for API compatibility; direct evaluation always "converges".
    return out.T, 0


def _upsampled_latitudes(nside):
    """The DFS latitude sample locations (radians) for the given nside.

    The original HEALPix ring colatitudes, mirrored across the poles by the DFS
    step and bracketed by the two pole rings, mapped into the [-pi, pi) period the
    NUFFT/Vandermonde use.
    """
    latitudes = create_latitude_array(nside)
    lat = np.zeros(len(latitudes) * 2 + 2)
    lat[0] = 90
    lat[1 : len(latitudes) + 1] = latitudes
    lat[len(latitudes) + 1] = -90
    lat[len(latitudes) + 2 :] = -180 + latitudes
    return lat * np.pi / 180 + np.pi / 2


def _embed_centered(f_hat, N_full):
    """Zero-pad a (-Lw..Lw)-ordered latitude spectrum into a (-Lf..Lf) array.

    The solved spectrum (rows in finufft modeord=0, i.e. -Lw..Lw with DC at the
    centre) is placed centred inside the wider N_full array, zeroing |k|>Lw. This
    lets a solve at the well-conditioned band feed the FSHT, which is calibrated for
    the wider L = 4*nside array (``FSHT.preparation``).
    """
    Nw, ncols = f_hat.shape
    Lw = (Nw - 1) // 2
    Lf = (N_full - 1) // 2
    out = np.zeros((N_full, ncols), dtype=f_hat.dtype)
    out[Lf - Lw : Lf + Lw + 1] = f_hat
    return out


def apply_nuFFT(
    mp: np.ndarray,
    solver: str = "cg",
    N_modes=None,
    solve_modes=None,
    rtol: float = None,
    maxiter: int = None,
    eps: float = 1e-12,
    rcond: float = 1e-13,
    sample_mask=None,
    fold=None,
    spin=None,
    level=None,
    eta=None,
    delay=5,
    monitor=None,
    workers=None,
    half_domain=False,
) -> np.ndarray:
    """Latitude analysis (the DFS grid's only ill-conditioned stage).

    Two knobs decide accuracy vs invertibility vs scalability:

    ``solve_modes`` -- how many latitude Fourier modes to actually fit:

    * ``4*nside+1`` (default, |k| <= 2*nside): the band a band-limited (lmax =
      2*nside) signal lives in. The Vandermonde here is WELL conditioned (cond ~
      1.15 at every nside), so the fit is stable and CG converges in a few
      iterations -- this is what scales to nside 256/512. The result is zero-padded
      up to ``N_modes`` for the FSHT. Forward accuracy and round trip both converge
      with nside (~1e-3 by nside 256); the small round-trip residual is above-band
      polar aliasing the band cannot (and arguably should not) represent.
    * ``8*nside+1`` (= ``N_modes``, the SQUARE interpolation): fits one mode per DFS
      sample, so the synthesis reproduces the map bit-for-bit (exact round trip) at
      low nside. But this Vandermonde is severely ill-conditioned (cond ~ 8e10 at
      nside 64, ~1/eps at nside 128), so it needs ``solver="svd"`` and walls out by
      nside ~128. Use it only for bit-exact round trips at nside <= 64.

    ``solver`` -- how to solve that fit:

    * ``"cg"`` (default): finufft + conjugate gradient on the normal equations.
      O(N log N) and scalable; ideal for the well-conditioned ``solve_modes`` above.
    * ``"svd"``: dense truncated-SVD pseudo-inverse of the (cached, shared)
      Vandermonde. O(nside^3) one-off factorisation; needed only for the
      ill-conditioned square band, where it reaches ~1e-6 round trip up to nside 64.

    ``sample_mask`` (not available with ``svd``) marks which (latitude sample, longitude
    mode) entries the HEALPix grid actually resolved -- see
    ``double_fourier_sphere.dfs_fold_plan``. Masked entries get zero
    weight, so they are treated as MISSING instead of zero. This matters only where a mode
    carries real amplitude on a ring too coarse to sample it, i.e. ``|m|`` near ``|spin|``
    on the innermost polar rings of a SPIN field; the scalar path is unaffected and
    defaults to ``None``.

    ``fold`` is the ``(target, phase)`` pair from ``dfs_fold_plan``, which makes the
    forward operator model what a coarse HEALPix ring actually measures -- the ALIAS SUM
    over each longitude mode's residue family -- instead of asserting that the ring
    measured each mode separately. Use it together with that function's ``keep`` as the
    ``sample_mask``; the two are one plan and are not meaningful apart.

    ``N_modes`` is the latitude band handed to the FSHT (-> L = (N_modes-1)//2).
    It defaults to ``solve_modes``, i.e. the FSHT runs at the natural band L = lmax =
    2*nside (the compact `(L+1, 2L+1)` g-array). The FastTransforms ``fourier2sph`` is
    ~O(L^3), so this compact band is ~8x faster and uses half the memory of the old
    L = 4*nside array, with bit-identical accuracy now that the `preparation` float-
    parity bug is fixed. Set ``N_modes`` larger than ``solve_modes`` to zero-pad the
    solved spectrum into a wider FSHT band (rarely needed). ``rcond`` regularises the
    SVD; ``rtol``/``maxiter``/``eps`` tune CG and the NUFFT. ``maxiter=None`` (the
    default) leaves the iteration cap to the solver; every solve here is stopped by
    ``rtol`` long before it, so the cap is a safety net rather than a knob.

    ``level`` (``solver="cg"`` only) stops the iteration by the discrepancy principle
    instead: halt at the first iterate whose weighted data residual
    ``||b - A f_hat||_W`` falls to ``level``. Give it the size of the error in the
    forward model, which for a folded spin solve is set by ``alias_tol``. Unlike
    ``rtol`` it is stated in the units of the data, so it does not need recalibrating
    per resolution. ``rtol`` remains active as the fallback for the case where the
    least-squares residual never reaches ``level``.

    ``monitor``, if given, is called as ``monitor(k, rho, rrel, get_spectrum)`` after
    each CG iteration. ``rho`` is the data residual relative to ``||b||_W``, ``rrel``
    is the ratio ``rtol`` tests, and ``get_spectrum()`` returns the current iterate in
    the same layout as the return value.
    """
    _openmp.pin()  # finufft's OpenMP runtime is one of several; see src/_openmp.py
    nside = mp.shape[1] // 4
    if half_domain and mp.shape[0] != 4 * nside + 1:
        raise ValueError(
            f"half_domain expects {4 * nside + 1} latitude rows "
            f"(pole, rings, pole), got {mp.shape[0]}"
        )
    if rtol is None:
        rtol = 1e-9
    if solve_modes is None:
        solve_modes = 4 * nside + 1  # well-conditioned latitude band (|k| <= 2*nside)
    if N_modes is None:
        N_modes = solve_modes  # compact: FSHT at L = (solve_modes-1)//2 = lmax

    DFT_upsampled_lat = _upsampled_latitudes(nside)

    if solver == "svd":
        if fold is not None:
            raise NotImplementedError(
                "the alias fold couples the longitude columns; the SVD path factorizes "
                "one shared per-column Vandermonde. Use solver='cg' or 'lsmr'."
            )
        if sample_mask is not None:
            raise NotImplementedError(
                "sample_mask needs a per-column solve; the SVD path shares one cached "
                "Vandermonde factorization across all longitude columns. Use solver='cg'."
            )
        fft_lat, info = svd_nufft_forward(
            DFT_upsampled_lat, np.asarray(mp.T), N_modes=solve_modes, rcond=rcond
        )
    elif solver == "lsmr":
        fft_lat, info = lsmr_nufft_forward(
            DFT_upsampled_lat,
            np.asarray(mp.T).copy(),
            N_modes=solve_modes,
            sample_mask=sample_mask,
            rtol=rtol,
            maxiter=maxiter,
            eps=eps,
            fold=fold,
        )
        if info != 0:
            logger.warning("LSMR did not converge (istop=%s)", info)
    elif solver == "cg":
        fft_lat, info = cg_nufft_forward(
            DFT_upsampled_lat,
            np.asarray(mp.T).copy(),
            N_modes=solve_modes,
            rtol=rtol,
            maxiter=maxiter,
            eps=eps,
            sample_mask=sample_mask,
            fold=fold,
            spin=spin,
            level=level,
            eta=eta,
            delay=delay,
            monitor=monitor,
            workers=default_workers(4 * nside) if workers is None else workers,
            half_domain=half_domain,
        )
        if info != 0:
            logger.warning("CG did not converge (info=%s)", info)
    else:
        raise ValueError(f"unknown solver {solver!r}; use 'svd', 'cg' or 'lsmr'")

    if solve_modes < N_modes:
        fft_lat = _embed_centered(fft_lat, N_modes)

    return fft_lat


def inverse_nuFFT(fft_lat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Perform inverse NUFFT (Type-2) to reconstruct signal at non-uniform latitudes.

    Synthesis is a plain evaluation s = A @ f_hat (well conditioned), so it stays a
    NUFFT regardless of which solver the forward analysis used.

    Parameters:
    - fft_lat (np.ndarray): Fourier coefficients from uniform frequency space.
    - eps (float): NUFFT precision.

    Returns:
    - np.ndarray: Reconstructed signal at non-uniform latitude samples.
    """
    _openmp.pin()  # finufft's OpenMP runtime is one of several; see src/_openmp.py
    nside = fft_lat.shape[1] // 4
    DFT_upsampled_lat = _upsampled_latitudes(nside)

    fft_lat = np.array(fft_lat)
    mp_reconstructed, info = cg_nufft_backward(
        DFT_upsampled_lat, fft_lat.T.copy(), eps=eps
    )

    if info != 0:
        logger.warning("CG did not converge (info=%s)", info)

    return mp_reconstructed
