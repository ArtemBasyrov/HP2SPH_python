"""D46: what is the least-squares residual floor MADE OF?

The stopping level the regularisation literature asks for in the perturbed-operator case
is ``Theta * (h ||x_k|| + eps)`` with ``h`` bounding ``||A - A_h||`` and ``eps`` the data
error (Nemirovsky 1986 eq. 2 and its introduction; Neubauer 2022 §3). Every quantity in
it except ``h`` is available at run time, so bounding ``h`` replaces the calibrated
stagnation threshold with an application of the theorem.

``h`` cannot be bounded before it is known WHAT is in it.

This splits the converged weighted residual by equation type. Every latitude row and
longitude slot of the DFS least squares falls into exactly one of:

  ring / resolved      a real HEALPix ring measurement of that mode
  ring / asserted      an unresolved mode whose envelope is below alias_tol, so the plan
                       asserts c_m(theta_r) = 0 against a zero-padded datum. The residual
                       here IS the alias modelling error, directly.
  pole  / kept         the two Lagrange-extrapolated pole rows. These are not
                       measurements, so their residual is a DATA error (eps), not an
                       operator error (h).

Dropped entries (``keep`` False) carry zero weight and are excluded.

The split says which term any bound on ``h`` has to cover, and how much of the floor is
``eps`` rather than ``h`` at all.

Pass ``signal_lmax`` below ``2*nside`` to remove the longitude-Nyquist column, which
otherwise dominates the floor and hides everything else.

WARNING: a tighter ``alias_tol`` relaxes more entries and needs far more iterations. At
nside 32 and ``alias_tol=1e-4`` the reported floor is 6.6e-6, 1.1e-6 and 1.5e-7 at
``maxiter`` 400, 800 and 3000. An under-converged run reports a floor that is really its
own iteration cap, and reading one as a new error term is a mistake this script has
already caused once. Raise ``maxiter`` until the number stops moving before quoting it.

Usage: python d46_residual_budget.py [nside ...]
"""

import sys
import contextlib
import io

import numpy as np
import healpy as hp
import finufft

import precond_common  # noqa: F401
from src.data_interpolation import (
    ring_alias_target,
    transform_healpix_to_grid,
    mode_pole_envelope,
)
from src.double_fourier_sphere import DFS, dfs_fold_plan
from src.nuFFT import apply_nuFFT, compute_voronoi_weights_1d, _upsampled_latitudes


def _sky(nside, seed=0, slope=1.5, signal_lmax=None):
    lmax = 2 * nside
    slmax = signal_lmax if signal_lmax is not None else lmax
    ell = np.arange(slmax + 1)
    np.random.seed(seed)
    aE = hp.synalm((1.0 + ell) ** (-2.0 * slope), lmax=slmax, new=True)
    aB = hp.synalm((1.0 + ell) ** (-2.0 * slope), lmax=slmax, new=True)
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, slmax)
    return Q, U, lmax


def budget(
    nside, seed=0, alias_tol=1e-2, spin=2, rtol=1e-14, maxiter=6000, signal_lmax=None
):
    """Converged residual, split by equation type. Returns a dict of squared norms."""
    Q, U, lmax = _sky(nside, seed, signal_lmax=signal_lmax)
    z = Q + 1j * U
    with contextlib.redirect_stdout(io.StringIO()):
        up, fc = transform_healpix_to_grid(z)
        _, dfs = DFS(up, fc, spin=spin)
        target, phase, keep = dfs_fold_plan(nside, spin, alias_tol)
        fft_lat = apply_nuFFT(
            dfs,
            solver="cg",
            sample_mask=keep,
            fold=(target, phase),
            rtol=rtol,
            maxiter=maxiter,
            eta=None,
            spin=spin,
        )

    # rebuild the forward model at the converged iterate, on the full DFS domain
    x = _upsampled_latitudes(nside)
    b = np.ascontiguousarray(np.asarray(dfs).T)  # (n_trans, M)
    n_trans, M = b.shape
    N_modes = fft_lat.shape[0]
    plan = finufft.Plan(2, (N_modes,), n_trans=n_trans, isign=1, dtype=np.complex128)
    plan.setpts(x)
    model = np.zeros((n_trans, M), dtype=np.complex128)
    plan.execute(np.ascontiguousarray(fft_lat.T), model)

    # apply the fold: slot (r, c) receives from every c' with target[r, c'] == c
    folded = np.zeros_like(model)
    tt = np.asarray(target).T  # (n_trans, M) -> indices into the column axis
    ph = np.asarray(phase).T
    t = model * ph
    for c in range(n_trans):
        np.add.at(folded, (tt[c], np.arange(M)), t[c])

    w = compute_voronoi_weights_1d(x)
    resid2 = w[None, :] * np.abs(b - folded) ** 2
    kept = np.asarray(keep).T.astype(bool)
    resid2 = np.where(kept, resid2, 0.0)

    # classify the rows and columns
    n_rings = 4 * nside - 1
    is_pole = np.zeros(M, dtype=bool)
    is_pole[0] = True
    is_pole[n_rings + 1] = True
    _, _, resolved = ring_alias_target(nside)
    res_rows = np.concatenate(
        (
            np.zeros((1, n_trans), bool),
            resolved,
            np.zeros((1, n_trans), bool),
            np.flip(resolved, axis=0),
        )
    ).T  # (n_trans, M)

    ring = ~is_pole[None, :]
    return {
        "total": float(resid2.sum()),
        "ring/resolved": float(resid2[ring & res_rows].sum()),
        "ring/asserted": float(resid2[ring & ~res_rows].sum()),
        "pole/kept": float(resid2[:, is_pole].sum()),
        "bw2": float((w[None, :] * np.abs(b) ** 2 * kept).sum()),
        "n_asserted": int((kept & ring & ~res_rows).sum()),
        "n_resolved": int((kept & ring & res_rows).sum()),
        "n_pole": int(kept[:, is_pole].sum()),
    }


def main(nside, tols=(1e-1, 1e-2, 1e-3), seeds=(0, 1)):
    print(f"\n=== nside {nside}, spin 2, cosmology, converged residual budget ===")
    print(
        "  alias_tol   rho_inf     ring/resolved   ring/asserted    pole/kept"
        "     n_assert"
    )
    for tol in tols:
        rows = [budget(nside, s, tol) for s in seeds]
        tot = np.median([r["total"] for r in rows])
        bw2 = np.median([r["bw2"] for r in rows])
        frac = {
            k: np.median([r[k] for r in rows]) / tot
            for k in ("ring/resolved", "ring/asserted", "pole/kept")
        }
        print(
            f"   {tol:.0e}    {np.sqrt(tot / bw2):.4e}   "
            f"{frac['ring/resolved']:12.1%}   {frac['ring/asserted']:12.1%}   "
            f"{frac['pole/kept']:10.1%}   {int(np.median([r['n_asserted'] for r in rows])):8d}"
        )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [16, 32, 64]:
        main(ns)
