"""D39: solve the latitude system on HALF the DFS domain.

The DFS doubling makes the latitude samples closed under x -> -x, with the two poles as
fixed points, and makes the data obey  data[mirror(r), c] = parity[c] * data[r, c]  with
parity = (-1)^(m + spin).  The coefficients then obey c_{-k} = parity * c_k.

So the whole system can be restricted to
  * M_half = 4 * nside + 1 latitude rows instead of 8 * nside, and
  * K + 1 = 2 * nside + 1 latitude coefficients per column instead of 4 * nside + 1.

This is an EXACT restructuring, not an approximation: the minimiser of the full least
squares already satisfies the symmetry, so the constrained half system returns the same
coefficients.  It halves the fold and glue exactly, shrinks the CG vectors by two, and
speeds the NUFFT by the ratio measured in fix_pass_2 section 5.1.  It also REDUCES
memory, which matters most above nside 256 where the coarse solve has been given up.

Everything here is verified against the full operator before it is timed.
"""

import sys, time, contextlib, io
import numpy as np
import healpy as hp
import finufft

import precond_common  # noqa: F401
from precond_common import make_spin_dfs
from src.nuFFT import compute_voronoi_weights_1d, _upsampled_latitudes
from src.double_fourier_sphere import dfs_fold_plan
from d38_fold_fast import fold_ops_fast


def involution(x):
    """Permutation mu with x[mu[r]] == -x[r] (mod 2 pi), and the fixed points."""
    xx = np.mod(x, 2 * np.pi)
    xm = np.mod(-x, 2 * np.pi)
    mu = np.array(
        [
            int(np.argmin(np.abs(np.mod(xm[r] - xx + np.pi, 2 * np.pi) - np.pi)))
            for r in range(len(x))
        ]
    )
    err = np.max(np.abs(np.mod(xx[mu] - xm + np.pi, 2 * np.pi) - np.pi))
    return mu, err


def half_plan(nside, spin, tol):
    """Rows to keep, their multiplicity, and the parity sign per longitude column."""
    x = _upsampled_latitudes(nside)
    M = len(x)
    mu, err = involution(x)
    fixed = mu == np.arange(M)
    keep_rows = np.array([r for r in range(M) if r <= mu[r]])
    mult = np.where(fixed[keep_rows], 1.0, 2.0)
    C = 4 * nside
    m = np.arange(C) - C // 2
    parity = ((-1.0) ** (m + spin)).astype(float)
    return x, mu, keep_rows, mult, parity, err


def check_symmetry(nside, spin=2, tol=1e-2):
    dfs, _ = make_spin_dfs(nside, spin=spin)
    x, mu, keep_rows, mult, parity, err = half_plan(nside, spin, tol)
    d = np.asarray(dfs)  # (M, C)
    lhs = d[mu]
    rhs = parity[None, :] * d
    scale = np.abs(d).max()
    print(
        f"nside {nside}: involution error {err:.2e}   "
        f"data symmetry |d[mu] - parity*d| / max|d| = "
        f"{np.abs(lhs - rhs).max() / scale:.2e}   "
        f"rows {len(x)} -> {len(keep_rows)}"
    )
    return np.abs(lhs - rhs).max() / scale


class HalfOperator:
    """N restricted to the mirror-symmetric subspace."""

    def __init__(self, nside, spin=2, tol=1e-2, eps=1e-12, fast_fold=True):
        self.nside = nside
        C = self.n_trans = 4 * nside
        self.N_modes = 4 * nside + 1
        self.K = 2 * nside  # coefficients k = 0 .. K
        x, mu, keep_rows, mult, parity, _ = half_plan(nside, spin, tol)
        self.x_half = x[keep_rows]
        self.keep_rows = keep_rows
        self.Mh = len(keep_rows)
        target, phase, keep = dfs_fold_plan(nside, spin, tol)
        th, ph, kh = target[keep_rows], phase[keep_rows], keep[keep_rows]
        self.fold_apply, self.fold_adjoint = (
            fold_ops_fast if fast_fold else _orig_fold
        )((th, ph), C, self.Mh)
        w = compute_voronoi_weights_1d(x)[keep_rows] * mult
        self.weights = w[None, :] * kh.T.astype(float)
        self.norm = self.weights.sum(axis=1, keepdims=True).mean()
        self.parity = parity
        # c_{-k} = parity * c_k forces c_0 = 0 on odd-parity columns, so those carry K
        # free coefficients and the even ones K + 1.  The k = 0 slot of an odd column is
        # pinned to zero in both expand and restrict so CG never moves it.
        self.even = (parity > 0).astype(float)
        # P must be an ISOMETRY, or N_half = P^H N P has a different spectrum from N on
        # the symmetric subspace and CG converges more slowly (measured: 115 -> 257
        # iterations at nside 64).  c_k feeds BOTH +k and -k, so it carries 1/sqrt(2).
        self.scale = np.full(self.K + 1, 1.0 / np.sqrt(2.0))
        self.scale[0] = 1.0

        self.plan_f = finufft.Plan(
            2, (self.N_modes,), n_trans=C, isign=1, dtype=np.complex128, eps=eps
        )
        self.plan_a = finufft.Plan(
            1, (self.N_modes,), n_trans=C, isign=-1, dtype=np.complex128, eps=eps
        )
        self.plan_f.setpts(self.x_half)
        self.plan_a.setpts(self.x_half)
        self._full = np.zeros((C, self.N_modes), dtype=np.complex128)
        self._g = np.zeros((C, self.Mh), dtype=np.complex128)
        self._out = np.zeros((C, self.N_modes), dtype=np.complex128)

    def expand(self, ch):
        """half coefficients (n_trans, K+1) -> full (n_trans, N_modes), modes -K..K."""
        K = self.K
        full = self._full
        cs = ch * self.scale  # isometric embedding
        full[:, K:] = cs  # k = 0 .. K
        full[:, K] *= self.even  # c_0 = 0 where parity = -1
        full[:, :K] = self.parity[:, None] * cs[:, :0:-1]  # k = -K .. -1
        return full

    def restrict(self, full):
        """adjoint of expand."""
        K = self.K
        out = full[:, K:].copy()
        out[:, 1:] += self.parity[:, None] * full[:, K - 1 :: -1]
        out *= self.scale
        out[:, 0] *= self.even
        return out

    def matvec(self, vec):
        ch = vec.reshape(self.n_trans, self.K + 1)
        self.plan_f.execute(self.expand(ch), self._g)
        y = self.fold_apply(self._g)
        y *= self.weights
        y = self.fold_adjoint(y)
        self.plan_a.execute(y, self._out)
        return (self.restrict(self._out) / self.norm).ravel()

    def rhs(self, dfs):
        """A0^H F^H W d restricted to the symmetric subspace."""
        d = np.ascontiguousarray(np.asarray(dfs)[self.keep_rows].T)  # (n_trans, Mh)
        y = self.fold_adjoint(d * self.weights)
        self.plan_a.execute(y, self._out)
        return (self.restrict(self._out) / self.norm).ravel()


def _orig_fold(fold, n_trans, M):
    from src.nuFFT import _fold_ops

    return _fold_ops(fold, n_trans, M)


def full_matvec_factory(nside, spin=2, tol=1e-2, eps=1e-12, fast_fold=False):
    C, M = 4 * nside, 8 * nside
    N_modes = 4 * nside + 1
    x = _upsampled_latitudes(nside)
    target, phase, keep = dfs_fold_plan(nside, spin, tol)
    fa, fj = (fold_ops_fast if fast_fold else _orig_fold)((target, phase), C, M)
    w = compute_voronoi_weights_1d(x)
    weights = w[None, :] * keep.T.astype(float)
    norm = weights.sum(axis=1, keepdims=True).mean()
    pf = finufft.Plan(2, (N_modes,), n_trans=C, isign=1, dtype=np.complex128, eps=eps)
    pa = finufft.Plan(1, (N_modes,), n_trans=C, isign=-1, dtype=np.complex128, eps=eps)
    pf.setpts(x)
    pa.setpts(x)
    g = np.zeros((C, M), dtype=np.complex128)
    out = np.zeros((C, N_modes), dtype=np.complex128)

    def mv(vec):
        pf.execute(vec.reshape(C, N_modes), g)
        y = fa(g)
        y = y * weights
        y = fj(y)
        pa.execute(np.ascontiguousarray(y), out)
        return (out / norm).ravel()

    return mv, C, N_modes


def full_rhs(nside, dfs, spin=2, tol=1e-2, eps=1e-12):
    C, M = 4 * nside, 8 * nside
    N_modes = 4 * nside + 1
    x = _upsampled_latitudes(nside)
    target, phase, keep = dfs_fold_plan(nside, spin, tol)
    _, fj = fold_ops_fast((target, phase), C, M)
    w = compute_voronoi_weights_1d(x)
    weights = w[None, :] * keep.T.astype(float)
    norm = weights.sum(axis=1, keepdims=True).mean()
    pa = finufft.Plan(1, (N_modes,), n_trans=C, isign=-1, dtype=np.complex128, eps=eps)
    pa.setpts(x)
    out = np.zeros((C, N_modes), dtype=np.complex128)
    d = np.ascontiguousarray(np.asarray(dfs).T)
    pa.execute(fj(d * weights), out)
    return (out / norm).ravel()


def solve_compare(nside, spin=2, tol=1e-2, rtol=1e-7):
    """Solve the full and the half system and compare answers, iterations and time."""
    from scipy.sparse.linalg import LinearOperator, cg

    dfs, _ = make_spin_dfs(nside, spin=spin)
    mv, C, N_modes = full_matvec_factory(nside, spin, tol, fast_fold=True)
    H = HalfOperator(nside, spin, tol)

    def count(op, b, n):
        k = [0]
        A = LinearOperator((n, n), matvec=op, dtype=np.complex128)
        t = time.perf_counter()
        x, _ = cg(
            A,
            b,
            rtol=rtol,
            maxiter=20000,
            callback=lambda z: k.__setitem__(0, k[0] + 1),
        )
        return x, k[0], time.perf_counter() - t

    bf = full_rhs(nside, dfs, spin, tol)
    xf, itf, tf = count(mv, bf, C * N_modes)
    bh = H.rhs(dfs)
    xh, ith, th = count(H.matvec, bh, C * (H.K + 1))

    xh_full = H.expand(xh.reshape(C, H.K + 1)).copy()
    rel = np.linalg.norm(xh_full - xf.reshape(C, N_modes)) / np.linalg.norm(xf)
    print(
        f"    solve: full {itf:4d} its {tf:7.3f} s | half {ith:4d} its {th:7.3f} s "
        f"({tf / th:4.2f}x)   |x_half - x_full| / |x_full| = {rel:.2e}"
    )


def timeit(fn, n=5):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts)


def main(nside, spin=2, tol=1e-2):
    check_symmetry(nside, spin, tol)
    H = HalfOperator(nside, spin, tol)
    mv_slow, C, N_modes = full_matvec_factory(nside, spin, tol, fast_fold=False)
    mv_fast, _, _ = full_matvec_factory(nside, spin, tol, fast_fold=True)

    rng = np.random.default_rng(0)
    ch = rng.standard_normal((C, H.K + 1)) + 1j * rng.standard_normal((C, H.K + 1))
    full = H.expand(ch).copy()
    ref = mv_slow(full.ravel()).reshape(C, N_modes)
    got = H.matvec(ch.ravel()).reshape(C, H.K + 1)
    exp = H.restrict(ref)
    rel = np.abs(got - exp).max() / np.abs(exp).max()
    print(f"    half operator vs full, on a symmetric vector: {rel:.2e}")

    t_slow = timeit(lambda: mv_slow(full.ravel()))
    t_fast = timeit(lambda: mv_fast(full.ravel()))
    t_half = timeit(lambda: H.matvec(ch.ravel()))
    print(
        f"    matvec: current {1e3 * t_slow:8.2f} ms | fast fold {1e3 * t_fast:8.2f} ms "
        f"({t_slow / t_fast:4.2f}x) | half domain {1e3 * t_half:8.2f} ms "
        f"({t_slow / t_half:4.2f}x)"
    )
    print(f"    unknowns {C * N_modes} -> {C * (H.K + 1)}   rows {8 * nside} -> {H.Mh}")


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [128]:
        main(ns)
