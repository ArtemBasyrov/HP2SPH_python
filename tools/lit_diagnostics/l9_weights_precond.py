"""L9: are the Voronoi weights doing the preconditioning, as the literature claims?

Feichtinger, Gröchenig & Strohmer (1995), abstract, verbatim: "The choice of
'adaptive weights' can be seen as a simple but very efficient method of
preconditioning." Their weights are w_i = (t_{i+1} - t_{i-1}) / 2, which is exactly
``nuFFT.compute_voronoi_weights_1d``.

Strohmer (1997), Experiment 2, compares T. Chan's circulant preconditioner on the
UNWEIGHTED system against preconditioning by the adaptive weights and finds them
"about equally attractive" -- substitutes, not complements.

Earlier sessions used that pairing to EXPLAIN why an exact Toeplitz preconditioner
changed our iteration count by nothing (137 -> 137 at nside 32): the preconditioning
was already being done by the weights. That explanation was asserted, never tested.
This tests it, by removing the weights and seeing whether the conditioning and the
iteration count degrade.

Two things it can show.
  * If unweighted is much worse, the claim holds and there is no room left for a
    preconditioner -- the shipped solver is already preconditioned.
  * If unweighted is no worse, the claim is wrong for our geometry, the weights are
    doing nothing for convergence, and the Toeplitz null result needs another
    explanation.

IMPORTANT. Unlike MRI density compensation, our W is a quadrature weight defining
the weighted least-squares functional, so an unweighted solve fits a DIFFERENT
problem. The accuracy column is reported for that reason. Uniform weights are a
diagnostic here and never a shipping option.

Usage: PYTHONPATH=. python tools/lit_diagnostics/l9_weights_precond.py
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from src.nuFFT import compute_voronoi_weights_1d, _mirror_plan, _upsampled_latitudes
from tools.lit_diagnostics._common import make_sky, build_operator


def unfolded_cond(nside, spin, weights):
    """cond of the half-restricted normal operator WITHOUT the fold, built densely."""
    N = 4 * nside + 1
    n_trans = 4 * nside
    x = _upsampled_latitudes(nside)
    _, rows, mult, _, scale, _ = _mirror_plan(x, spin, n_trans, N)
    xh = x[rows]
    w = compute_voronoi_weights_1d(x)[rows] * mult
    if weights == "uniform":
        w = np.full_like(w, w.sum() / mult.sum()) * mult
    K = (N - 1) // 2
    k = np.arange(-K, K + 1)
    A = np.exp(1j * np.outer(xh, k))
    T = A.conj().T @ (w[:, None] * A)
    out = {}
    for p, label in ((1.0, "even"), (-1.0, "odd")):
        P = np.zeros((N, K + 1), complex)
        for i in range(K + 1):
            P[K + i, i] = scale[i]
            if i > 0:
                P[K - i, i] = p * scale[i]
        if p < 0:
            P[:, 0] = 0.0
        B = (P.conj().T @ T @ P) / w.sum()
        ev = np.linalg.eigvalsh((B + B.conj().T) / 2)
        ev = ev[np.abs(ev) > 1e-12 * np.abs(ev).max()]
        out[label] = ev.max() / ev.min()
    return out


def main(nsides=(32, 64, 128), spin=2, rtol=1e-7):
    print("Unfolded half-restricted normal operator: does the weight choice matter?")
    print(f"{'nside':>6} {'weights':>9} {'cond even':>12} {'cond odd':>12}")
    for nside in nsides:
        for wm in ("voronoi", "uniform"):
            c = unfolded_cond(nside, spin, wm)
            print(f"{nside:6d} {wm:>9} {c['even']:12.5f} {c['odd']:12.5f}")

    print("\nFolded solve: iterations and answer, Voronoi against uniform weights")
    print(
        f"{'nside':>6} {'voronoi its':>12} {'uniform its':>12} {'ratio':>8} "
        f"{'||dx||/||x||':>13}"
    )
    for nside in nsides:
        dfs, plan = make_sky(nside, spin=spin)
        res = {}
        for wm in ("voronoi", "uniform"):
            AHA, rhs, n, _ = build_operator(
                nside, dfs, plan, use_fold=True, spin=spin, weights=wm
            )
            A = LinearOperator((n, n), matvec=AHA, dtype=complex)
            it = [0]
            sol, _ = cg(
                A,
                rhs,
                x0=np.zeros(n, complex),
                rtol=rtol,
                maxiter=6000,
                callback=lambda v: it.__setitem__(0, it[0] + 1),
            )
            res[wm] = (it[0], sol)
        d = np.linalg.norm(res["uniform"][1] - res["voronoi"][1]) / np.linalg.norm(
            res["voronoi"][1]
        )
        print(
            f"{nside:6d} {res['voronoi'][0]:12d} {res['uniform'][0]:12d} "
            f"{res['uniform'][0] / res['voronoi'][0]:7.2f}x {d:13.2e}"
        )


if __name__ == "__main__":
    main()
