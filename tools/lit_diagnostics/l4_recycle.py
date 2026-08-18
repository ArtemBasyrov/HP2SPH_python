"""Papež, Grigori & Stompor (2021): subspace recycling across a SEQUENCE of
similar systems, building progressively better two-level preconditioners.

Our case looks stronger than theirs: at fixed (nside, spin, alias_tol) the
operator is IDENTICAL across transforms and only the right-hand side changes,
which is the Monte-Carlo workload of any CMB pipeline.

Build a spectral subspace from solve #1 (smallest Ritz values = the slow
directions), then precondition solve #2 on an independent sky with
    M^-1 = I + U ((U^H A U)^-1 - I) U^H.

Result: worse at every subspace size, never reaching parity. Deflating the
small-eigenvalue subspace accelerates convergence into the semiconvergent tail
that the stopping rule exists to avoid.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from tools.lit_diagnostics._common import make_sky, build_operator

for nside in (32, 64):
    dfs1, plan = make_sky(nside, seed=0)
    dfs2, _ = make_sky(nside, seed=1)
    AHA, rhs1, n, _ = build_operator(nside, dfs1, plan, use_fold=True)
    _, rhs2, _, _ = build_operator(nside, dfs2, plan, use_fold=True)
    A = LinearOperator((n, n), matvec=AHA, dtype=complex)

    # solve #1, keeping the residual Krylov basis
    V = []
    r = rhs1.copy()
    x = np.zeros(n, complex)
    p = r.copy()
    rs = np.vdot(r, r).real
    for _ in range(400):
        V.append(r / np.linalg.norm(r))
        Ap = AHA(p)
        al = rs / np.vdot(p, Ap).real
        x += al * p
        r -= al * Ap
        rs2 = np.vdot(r, r).real
        if np.sqrt(rs2) / np.linalg.norm(rhs1) < 1e-7:
            break
        p = r + (rs2 / rs) * p
        rs = rs2
    print(f"\nnside {nside}: solve #1 took {len(V)} iterations, n={n}")

    V, _ = np.linalg.qr(np.array(V).T)
    AV = np.column_stack([AHA(V[:, j].copy()) for j in range(V.shape[1])])
    H = V.conj().T @ AV
    H = (H + H.conj().T) / 2
    _, evec = np.linalg.eigh(H)

    base = [0]
    ref, _ = cg(
        A,
        rhs2,
        x0=np.zeros(n, complex),
        rtol=1e-7,
        maxiter=4000,
        callback=lambda v: base.__setitem__(0, base[0] + 1),
    )
    print(f"  solve #2 plain CG: {base[0]} its")

    for k in (10, 20, 40, 80):
        if k > V.shape[1]:
            continue
        U, _ = np.linalg.qr(V @ evec[:, :k])
        AU = np.column_stack([AHA(U[:, j].copy()) for j in range(k)])
        Hk = U.conj().T @ AU
        Hinv = np.linalg.inv((Hk + Hk.conj().T) / 2)

        def Minv(v, U=U, Hinv=Hinv):
            c = U.conj().T @ v
            return v + U @ (Hinv @ c - c)

        it = [0]
        sol, _ = cg(
            A,
            rhs2,
            x0=np.zeros(n, complex),
            rtol=1e-7,
            maxiter=4000,
            M=LinearOperator((n, n), matvec=Minv, dtype=complex),
            callback=lambda v: it.__setitem__(0, it[0] + 1),
        )
        err = np.linalg.norm(sol - ref) / np.linalg.norm(ref)
        print(
            f"  recycle k={k:3d}: {it[0]:4d} its "
            f"({base[0] / max(it[0], 1):.2f}x)   agree {err:.1e}"
        )
