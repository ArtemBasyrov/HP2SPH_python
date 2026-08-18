"""Is the operator built on EQUISPACED latitudes a good preconditioner for the
folded solve? The HEALPix colatitudes are perturbed-equispaced, so the uniform
operator is close in norm and its inverse would be an FFT.

Result: no. It is close in norm and MISALIGNED in its eigenvectors, so its
inverse amplifies directions the true operator does not have. The damage grows
by 172x per resolution doubling.

This is the cleanest demonstration of the pattern in literature_review.md: a
preconditioner for an ill-posed system must be spectrally aligned, not merely
norm-close.
"""

import sys
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from tools.lit_diagnostics._common import make_sky, build_operator

# nside 32 forms two dense (8320, 8320) complex matrices; nside 16 is cheap.
for nside in (16, 32) if len(sys.argv) < 2 else (int(sys.argv[1]),):
    dfs, plan = make_sky(nside)
    AHA, rhs, n, parity = build_operator(nside, dfs, plan, use_fold=True, eps=1e-12)
    AHA_u, _, _, _ = build_operator(
        nside, dfs, plan, use_fold=True, eps=1e-12, uniform=True
    )
    print(f"\n=== nside {nside}, n = {n} ===", flush=True)

    M = np.empty((n, n), complex)
    Mt = np.empty((n, n), complex)
    e = np.zeros(n, complex)
    for i in range(n):
        e[i] = 1.0
        M[:, i] = AHA(e)
        Mt[:, i] = AHA_u(e)
        e[i] = 0.0

    # odd-parity columns pin their k=0 coefficient, an exact nullspace of BOTH
    pin = np.zeros(n, bool)
    pin.reshape(parity.size, -1)[parity < 0, 0] = True
    keep = ~pin
    Mk = np.ascontiguousarray(M[np.ix_(keep, keep)])
    Mtk = np.ascontiguousarray(Mt[np.ix_(keep, keep)])
    sm = np.linalg.svd(Mk, compute_uv=False)
    st = np.linalg.svd(Mtk, compute_uv=False)
    sp = np.linalg.svd(np.linalg.solve(Mtk, Mk), compute_uv=False)
    print(f"  cond(M)        = {sm[0] / sm[-1]:.4e}   (the folded operator)")
    print(f"  cond(Mt)       = {st[0] / st[-1]:.4e}   (uniform nodes)")
    print(f"  cond(Mt^-1 M)  = {sp[0] / sp[-1]:.4e}   (what PCG would see)", flush=True)

    Mt[pin, :] = 0
    Mt[:, pin] = 0
    Mt[pin, pin] = 1.0
    inv = np.linalg.inv(Mt)
    A = LinearOperator((n, n), matvec=AHA, dtype=complex)
    P = LinearOperator((n, n), matvec=lambda v: inv @ v, dtype=complex)
    for label, prec in (("plain CG", None), ("uniform-node PC", P)):
        it = [0]
        cg(
            A,
            rhs,
            x0=np.zeros(n, complex),
            rtol=1e-7,
            maxiter=4000,
            M=prec,
            callback=lambda v: it.__setitem__(0, it[0] + 1),
        )
        print(f"  {label:18s} {it[0]:5d} its")
