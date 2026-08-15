"""Two-level coarse-space preconditioner for the folded latitude solve.

Diagnosis
---------
The folded normal operator is  N = A0^H D A0 / norm  with

  * A0  the per-longitude-column latitude NUDFT (block diagonal by COLUMN),
  * D = F^H Wt F  the position-space weight operator (block diagonal by latitude ROW).

On a polar row an alias family {slot b, relaxed c_1..c_q} contributes the block
w_r * u u^H with u = (1, p_1, ..., p_q):  RANK ONE where the fold-free operator has
rank q+1.  D therefore has an explicit null space of dimension = the number of relaxed
entries, and the near-null space of N is its band-limited pull-back  Z = A0^H W Psi.
That is why Jacobi, block-diagonal-by-column and "use the plain operator" all failed:
the bad directions couple two longitude columns at one latitude, which is invisible to
all three.

Preconditioner
--------------
M^-1 = I + Z E^-1 Z^H,  E = Z^H N Z.  Additive (not multiplicative/BNN) because the
fine level is already the identity to 1%: the fold-free operator's spectrum is
[0.992, 1.003].  An exact eigenvector of eigenvalue lam in span(Z) is mapped to
lam + 1 ~ 1, so the whole plunge region is lifted at no extra matrix-vector product.

E is data independent -- it depends only on (nside, spin, alias_tol) -- and has a
CLOSED FORM needing no NUFFT at all:

    E = Psi^H W G D G W Psi / norm,     G = A0 A0^H  the Dirichlet kernel
    G[r, r'] = sin((L + 1/2)(x_r - x_r')) / sin((x_r - x_r')/2).

E is also EXACTLY SPARSE.  Two generators couple only if the longitude columns they
touch land on the same slot of some ring, which is a combinatorial condition, not a
decay condition: 23.3% / 18.4% / 14.3% / 12.9% of entries are structurally nonzero at
nside 8 / 16 / 32 / 64.  A sparse LU fills in by only 1.08x to 1.41x, so the coarse
solve is a sparse triangular pair rather than a dense O(R^2) one, and E is never
materialised densely.  Note that this is EXACT.  Thresholding E instead destroys
positive definiteness and fails outright at nside 64 (fix_pass_3.md section 8).
"""

import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
from scipy.sparse.linalg import LinearOperator, splu

from src.double_fourier_sphere import dfs_fold_plan
from src.nuFFT import _upsampled_latitudes, compute_voronoi_weights_1d


def coarse_generators(nside, spin, tol):
    """One generator of null(D) per relaxed entry.

    Returns (rows, cols, coeffs) with cols/coeffs of shape (R, 2); a generator that
    touches only one column carries coeff 0 in its second slot.
    """
    target, phase, keep = dfs_fold_plan(nside, spin, tol)
    rr, cc = np.nonzero(~keep)
    b = target[rr, cc]
    p = phase[rr, cc]
    single = b == cc
    cols = np.stack([cc, np.where(single, cc, b)], axis=1)
    coeffs = np.stack(
        [np.where(single, 1.0, np.conj(p)), np.where(single, 0.0, -1.0)], axis=1
    ).astype(complex)
    return rr, cols, coeffs, (target, phase, keep)


def _dirichlet(x, L):
    u = x[:, None] - x[None, :]
    s = np.sin(0.5 * u)
    tiny = np.abs(s) < 1e-13
    return np.where(tiny, 2.0 * L + 1.0, np.sin((L + 0.5) * u) / np.where(tiny, 1.0, s))


def _groups(rows, cols, coeffs):
    """Group generators by the (c, b) column pair they touch."""
    R = len(rows)
    key = [
        (int(cols[j, 0]), int(cols[j, 1]), bool(coeffs[j, 1] != 0)) for j in range(R)
    ]
    seen = {}
    for j, k in enumerate(key):
        seen.setdefault(k, []).append(j)
    gk = list(seen)
    gidx = [np.asarray(seen[k]) for k in gk]
    gcols = [(c0, c1) if two else (c0,) for (c0, c1, two) in gk]
    return gidx, gcols


def _interacting(target, gcols):
    """Boolean P x P: which group pairs can have a nonzero block.

    Groups gi and gj interact iff one of gi's columns and one of gj's land on the same
    slot of some ring.  Only the DISTINCT rows of ``target`` matter, and there are only
    about nside of those (one per ring size, plus one shared by the whole equatorial
    belt), so this costs O(nside * C) rather than O(M * C).
    """
    P = len(gcols)
    slot_group = np.concatenate([np.full(len(cs), gi) for gi, cs in enumerate(gcols)])
    slot_col = np.concatenate([np.asarray(cs) for cs in gcols])
    uniq = np.unique(target, axis=0)
    inter = np.zeros((P, P), dtype=bool)
    for trow in uniq:
        t = trow[slot_col]
        order = np.argsort(t, kind="stable")
        bnd = np.flatnonzero(np.diff(t[order])) + 1
        for seg in np.split(order, bnd):
            if seg.size > 1:
                gg = np.unique(slot_group[seg])
                if gg.size > 1:
                    inter[np.ix_(gg, gg)] = True
    np.fill_diagonal(inter, True)
    return inter


def closed_form_E(
    nside, spin, tol, rows, cols, coeffs, plan, norm, sparse=True, row_weight=None
):
    """E = Z^H N Z without a single NUFFT, assembled block by block.

    Only the structurally nonzero group pairs are visited, and with ``sparse`` the dense
    R x R intermediate is never allocated.

    ``row_weight`` scales the latitude rows of the position-space operator D inside the
    COARSE Galerkin only, i.e. it builds  Psi^H W G D_S G W Psi  with D_S = S D S for a
    diagonal S >= 0.  That is positive semi-definite for ANY such S, so the preconditioner
    stays valid -- unlike thresholding the assembled E, which destroys definiteness.
    Zeroing the innermost rings is the cheap way to cut nnz(E), since those rings alias
    every longitude mode into a handful of slots and generate most of the coupling.
    """
    target, phase, keep = plan
    x = _upsampled_latitudes(nside)
    w = np.abs(compute_voronoi_weights_1d(x))
    # w weights Z itself (Z = A0^H W Psi); wD weights D inside the Galerkin.  Only the
    # latter is masked -- masking the former would change the coarse space, not the
    # operator it is a Galerkin of.
    wD = w if row_weight is None else w * np.asarray(row_weight, dtype=float)
    G = _dirichlet(x, 2 * nside)
    R = len(rows)
    M = len(x)
    ar = np.arange(M)

    gidx, gcols = _groups(rows, cols, coeffs)
    inter = _interacting(target, gcols)

    # per group column: the row-indexed target / phase / keep it carries
    tgt = {c: target[:, c] for cs in gcols for c in cs}
    pha = {c: phase[:, c] for cs in gcols for c in cs}
    kep = {c: keep[ar, target[:, c]] for cs in gcols for c in cs}

    E = None if sparse else np.zeros((R, R), dtype=complex)
    data, ii, jj = [], [], []

    for gi in range(len(gidx)):
        ig, ci = gidx[gi], gcols[gi]
        Gi = G[:, rows[ig]]
        for gj in np.nonzero(inter[gi, gi:])[0] + gi:
            jg, cj = gidx[gj], gcols[gj]
            Gj = G[:, rows[jg]]
            blk = np.zeros((len(ig), len(jg)), dtype=complex)
            hit = False
            for a, ca in enumerate(ci):
                Ta, Pa, Ka = tgt[ca], pha[ca], kep[ca]
                for bq, cb in enumerate(cj):
                    K = wD * np.conj(Pa) * pha[cb] * (Ta == tgt[cb]) * Ka
                    if not np.any(K):
                        continue
                    hit = True
                    blk += (
                        np.conj(coeffs[ig, a])[:, None]
                        * (Gi.conj().T @ (K[:, None] * Gj))
                        * coeffs[jg, bq][None, :]
                    )
            if not hit:
                continue
            blk *= np.outer(w[rows[ig]], w[rows[jg]]) / norm
            if sparse:
                I, J = np.meshgrid(ig, jg, indexing="ij")
                data.append(blk.ravel())
                ii.append(I.ravel())
                jj.append(J.ravel())
                if gj != gi:
                    data.append(blk.conj().T.ravel())
                    ii.append(J.T.ravel())
                    jj.append(I.T.ravel())
            else:
                E[np.ix_(ig, jg)] = blk
                if gj != gi:
                    E[np.ix_(jg, ig)] = blk.conj().T
    if not sparse:
        return 0.5 * (E + E.conj().T)
    Es = sp.coo_matrix(
        (np.concatenate(data), (np.concatenate(ii), np.concatenate(jj))), shape=(R, R)
    ).tocsc()
    return (Es + Es.conj().T.tocsc()) * 0.5


class TwoLevel:
    """M^-1 = I + Z E^-1 Z^H as a scipy LinearOperator."""

    def __init__(
        self,
        nside,
        spin=2,
        tol=1e-2,
        N_modes=None,
        n_trans=None,
        rcond=1e-10,
        E=None,
        sparse=True,
        row_weight=None,
        ridge=1e-8,
    ):
        self.nside = nside
        self.n_trans = n_trans or 4 * nside
        self.N_modes = N_modes or 4 * nside + 1
        self.n = self.n_trans * self.N_modes
        self.sparse = sparse
        rows, cols, coeffs, plan = coarse_generators(nside, spin, tol)
        self.rows, self.cols, self.coeffs, self.plan = rows, cols, coeffs, plan
        self.R = len(rows)

        x = _upsampled_latitudes(nside)
        w = np.abs(compute_voronoi_weights_1d(x))
        L = (self.N_modes - 1) // 2
        k = np.arange(-L, L + 1)
        Erow = w[:, None] * np.exp(-1j * np.outer(x, k))

        # Z as one sparse matrix: n x R, 2 * N_modes nonzeros per column
        data, ri, ci = [], [], []
        for a in range(2):
            live = coeffs[:, a] != 0
            blk = coeffs[live, a][:, None] * Erow[rows[live]]
            base = cols[live, a][:, None] * self.N_modes + np.arange(self.N_modes)
            data.append(blk.ravel())
            ri.append(base.ravel())
            ci.append(np.repeat(np.nonzero(live)[0], self.N_modes))
        self.Zs = sp.csr_matrix(
            (np.concatenate(data), (np.concatenate(ri), np.concatenate(ci))),
            shape=(self.n, self.R),
        )
        self.ZsH = sp.csr_matrix(self.Zs.conj().T)

        if E is None:
            keepm = plan[2]
            norm = (w[None, :] * keepm.T).sum(axis=1).mean()
            E = closed_form_E(
                nside,
                spin,
                tol,
                rows,
                cols,
                coeffs,
                plan,
                norm,
                sparse=sparse,
                row_weight=row_weight,
            )
        self.E = E
        if sp.issparse(E):
            # E = Z^H N Z is Hermitian positive definite and EXACTLY sparse, so this
            # factorisation is exact -- no dropping, no loss of definiteness.  The
            # ridge is needed because cond(Z^H Z) grows with nside (1.1e2 at nside 8,
            # 1.7e6 at nside 64) and by nside 128 E is numerically singular; without it
            # the factorisation returns a solve with residual 2.75.
            shift = ridge * E.diagonal().real.mean()
            self.lu = splu((E + shift * sp.eye(E.shape[0], format="csc")).tocsc())
            self._solve = self.lu.solve
        else:
            ridge = rcond * np.trace(E).real / max(E.shape[0], 1)
            chol = sla.cho_factor(E + ridge * np.eye(E.shape[0]), lower=True)
            self.chol = chol
            self._solve = lambda v: sla.cho_solve(chol, v)

    def operator(self):
        solve, Zs, ZsH = self._solve, self.Zs, self.ZsH

        def apply(v):
            return v + Zs @ solve(ZsH @ v)

        return LinearOperator((self.n, self.n), matvec=apply, dtype=complex)
