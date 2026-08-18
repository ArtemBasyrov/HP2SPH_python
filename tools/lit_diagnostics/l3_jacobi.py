"""Ong, Uecker & Lustig (2020): diagonal preconditioners as the cheap alternative
to circulant ones. Unlike a spectral preconditioner, a diagonal one does not
amplify the small-eigenvalue subspace, so it is the one class worth testing on an
ill-posed system.

Result: no effect, and the reason rules out the WHOLE diagonal class -- the
diagonal of the normal operator is flat to 2%, so all the ill-conditioning is
off-diagonal, in the fold coupling.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from tools.lit_diagnostics._common import make_sky, build_operator

for nside in (32, 64):
    dfs, plan = make_sky(nside)
    AHA, rhs, n, _ = build_operator(nside, dfs, plan, use_fold=True)
    A = LinearOperator((n, n), matvec=AHA, dtype=complex)

    # exact diagonal by probing; a shipped version would derive it in O(n)
    d = np.empty(n)
    e = np.zeros(n, complex)
    for i in range(n):
        e[i] = 1.0
        d[i] = AHA(e)[i].real
        e[i] = 0.0
    pos = d > 0
    print(
        f"\nnside {nside}: n={n}  diag range [{d[pos].min():.3e}, {d.max():.3e}]  "
        f"ratio {d.max() / d[pos].min():.2e}  ({np.sum(~pos)} non-positive, the "
        f"pinned k=0 slots on odd-parity columns)"
    )

    dinv = np.where(pos, 1.0 / np.maximum(d, 1e-300), 1.0)
    M = LinearOperator((n, n), matvec=lambda v: dinv * v, dtype=complex)
    res = {}
    for label, P in (("plain", None), ("Jacobi", M)):
        it = [0]
        sol, _ = cg(
            A,
            rhs,
            x0=np.zeros(n, complex),
            rtol=1e-7,
            maxiter=6000,
            M=P,
            callback=lambda v: it.__setitem__(0, it[0] + 1),
        )
        res[label] = (it[0], sol)
    err = np.linalg.norm(res["Jacobi"][1] - res["plain"][1]) / np.linalg.norm(
        res["plain"][1]
    )
    print(
        f"  plain {res['plain'][0]:4d} its   Jacobi {res['Jacobi'][0]:4d} its   "
        f"({res['plain'][0] / max(res['Jacobi'][0], 1):.2f}x)   agree {err:.1e}"
    )
