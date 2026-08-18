"""Papež, Grigori & Stompor (2018), Sect. 5: PCG convergence depends on the
initial vector; they start map-making from the cheap binned map.

Our analogue: start the FOLDED solve from the UNFOLDED one, which converges in
2-3 iterations and is therefore nearly free.

Result: the gain decays to 1.00x by nside 256, which is their own caveat that the
effect is negligible for low signal-to-noise solutions.
"""

import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from tools.lit_diagnostics._common import make_sky, build_operator

for nside in (32, 64, 128, 256):
    dfs, plan = make_sky(nside)
    AHA_f, rhs_f, n, _ = build_operator(nside, dfs, plan, use_fold=True)
    AHA_u, rhs_u, _, _ = build_operator(nside, dfs, plan, use_fold=False)
    A_f = LinearOperator((n, n), matvec=AHA_f, dtype=complex)
    A_u = LinearOperator((n, n), matvec=AHA_u, dtype=complex)

    its_u = [0]
    x0, _ = cg(
        A_u,
        rhs_u,
        x0=np.zeros(n, complex),
        rtol=1e-7,
        maxiter=200,
        callback=lambda v: its_u.__setitem__(0, its_u[0] + 1),
    )

    out = {}
    for label, start in (("cold", np.zeros(n, complex)), ("warm", x0)):
        it = [0]
        t0 = time.perf_counter()
        sol, _ = cg(
            A_f,
            rhs_f,
            x0=start.copy(),
            rtol=1e-7,
            maxiter=4000,
            callback=lambda v: it.__setitem__(0, it[0] + 1),
        )
        out[label] = (it[0], time.perf_counter() - t0, sol)
    d = np.linalg.norm(out["cold"][2] - out["warm"][2]) / np.linalg.norm(out["cold"][2])
    print(
        f"nside {nside:4d}: unfolded presolve {its_u[0]:2d} its | "
        f"cold {out['cold'][0]:4d} its | warm {out['warm'][0]:4d} its "
        f"({out['cold'][0] / max(out['warm'][0], 1):.2f}x) | agree {d:.2e}",
        flush=True,
    )
