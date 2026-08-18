"""L8: the alias fold is block-diagonal in the longitude mode residue class mod 4.

DERIVED. Every HEALPix ring has a pixel count divisible by 4 -- the polar rings
carry 4k pixels for ring index k, the equatorial rings 4*nside. A ring of n_r
pixels aliases longitude mode m into slot b with m = b (mod n_r). Since 4 | n_r
for every ring, m = b (mod 4). So the fold never moves a mode out of its residue
class mod 4, and the folded normal operator is block diagonal with respect to
that partition.

This bounds the coupling, it does not fix it. The derivation says there are AT
LEAST 4 blocks and that no block straddles a class. Whether each class is fully
connected is a separate, configuration-dependent question, and it is not always
yes -- see ``residue_sweep`` below, where spin 0 at alias_tol=1e-1 fragments into
17 components. An implementation should compute the components rather than assume
four.

MEASURED at the shipped configuration (spin 2, alias_tol=1e-2): exactly 4
components of size nside at nside 16 to 256, with the label a function of m mod 4
at every resolution tested.

What it could buy. CG on a block-diagonal system builds ONE Krylov space and
applies ONE polynomial to every block. Four separate solves each get their own
polynomial, which is at least as good per block and strictly better when the
blocks have different spectra. So splitting could cut total work even though it
does not cut the total problem size. It would also quarter the solver's working
set, which is what gates nside 2048.

This measures whether the blocks differ enough for that to pay.

Because the operator is block diagonal, applying it to a vector supported on one
class returns a vector supported on that class. So the block solve is measured
here by running CG on full-length vectors restricted to a class's support, which
is mathematically identical to the block solve and needs no re-plumbing. In a
real implementation each matvec would cost a quarter as much, so the comparison
is  sum_r k_r / 4  against the global k.

Usage: PYTHONPATH=. python tools/lit_diagnostics/l8_residue_blocks.py [nside ...]
"""

import sys
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from tools.lit_diagnostics._common import make_sky, build_operator


def run(nside, spin=2, rtol=1e-7):
    dfs, plan = make_sky(nside, spin=spin)
    AHA, rhs, n, parity = build_operator(nside, dfs, plan, use_fold=True)
    n_trans = parity.size
    K1 = n // n_trans  # coefficients per column
    A = LinearOperator((n, n), matvec=AHA, dtype=complex)

    m = np.arange(n_trans) - n_trans // 2
    cls = m % 4

    its = [0]
    cg(
        A,
        rhs,
        x0=np.zeros(n, complex),
        rtol=rtol,
        maxiter=6000,
        callback=lambda v: its.__setitem__(0, its[0] + 1),
    )
    k_global = its[0]

    per = []
    for r in range(4):
        mask = np.zeros((n_trans, K1), bool)
        mask[cls == r] = True
        mask = mask.reshape(-1)
        sub = np.flatnonzero(mask)

        def block_matvec(v, sub=sub):
            full = np.zeros(n, complex)
            full[sub] = v
            return AHA(full)[sub]

        nb = sub.size
        B = LinearOperator((nb, nb), matvec=block_matvec, dtype=complex)
        it = [0]
        cg(
            B,
            rhs[sub],
            x0=np.zeros(nb, complex),
            rtol=rtol,
            maxiter=6000,
            callback=lambda v: it.__setitem__(0, it[0] + 1),
        )
        per.append(it[0])

    # leakage check: the operator really is block diagonal
    probe = np.zeros(n, complex)
    p_mask = np.zeros((n_trans, K1), bool)
    p_mask[cls == 0] = True
    p_mask = p_mask.reshape(-1)
    probe[p_mask] = np.random.randn(p_mask.sum()) + 0j
    out = AHA(probe)
    leak = np.abs(out[~p_mask]).max() / np.abs(out[p_mask]).max()

    work = sum(per) / 4.0
    print(
        f"nside {nside:4d}: global {k_global:4d} its | per class "
        f"{per} | effective work {work:6.1f} vs {k_global:4d} "
        f"({k_global / work:.2f}x) | off-block leakage {leak:.1e}"
    )
    return k_global, per


def residue_sweep(nsides=(32, 64), spins=(0, 2), tols=(1e-1, 1e-2, 1e-3, 1e-4)):
    """Is the mod-4 partition the ACTUAL component structure, or only a bound?"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from src.double_fourier_sphere import dfs_fold_sparse

    print(
        f"\n{'nside':>6} {'spin':>5} {'alias_tol':>10} {'components':>11} {'= m mod 4?':>11}"
    )
    for nside in nsides:
        for spin in spins:
            for tol in tols:
                p = dfs_fold_sparse(nside, spin, tol)
                Mh, nt = p.n_rows, p.n_trans
                sc, dc = p.src // Mh, p.dst // Mh
                if sc.size == 0:
                    print(f"{nside:6d} {spin:5d} {tol:10.0e} {'(no fold)':>11}")
                    continue
                ncomp, lab = connected_components(
                    coo_matrix((np.ones(sc.size), (sc, dc)), shape=(nt, nt)),
                    directed=False,
                )
                m = np.arange(nt) - nt // 2
                ok = all(len(set(lab[m % 4 == r])) == 1 for r in range(4))
                print(f"{nside:6d} {spin:5d} {tol:10.0e} {ncomp:11d} {str(ok):>11}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    if args and args[0] == "sweep":
        residue_sweep()
    else:
        for ns in [int(a) for a in args] or [32, 64, 128]:
            run(ns)
