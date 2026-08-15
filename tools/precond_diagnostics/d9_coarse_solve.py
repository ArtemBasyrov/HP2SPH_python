"""D9: how to make the coarse solve cheap.

The full coarse space (R = O(n)) is needed -- screening and the naive generalised
eigenproblem both fail -- so the question is how to apply E^-1 in less than O(R^2).

  A  proper compression: orthonormalise Z first (via eigh of S = Z^H Z with the
     numerically null directions TRUNCATED, not ridged), then eigh of Q^H N Q.
  B  block-diagonal E by latitude ROW (the fold couples columns within a row).
  C  threshold-sparsified E with a sparse factorisation.
"""

import sys, time
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, splu

from precond_common import make_spin_dfs, capture, cg_count
from src.double_fourier_sphere import dfs_fold_plan
from precond_twolevel import TwoLevel


def orthonormal_compress(T, rs):
    S = np.asarray((T.ZsH @ T.Zs).todense())
    S = 0.5 * (S + S.conj().T)
    s, Vs = sla.eigh(S)
    live = s > 1e-9 * s[-1]
    Tmat = Vs[:, live] / np.sqrt(s[live])  # Z Tmat has orthonormal columns
    Et = Tmat.conj().T @ T.E @ Tmat
    lam, W = sla.eigh(0.5 * (Et + Et.conj().T))
    print(
        f"    S rank {live.sum()}/{len(s)}   orthonormal Rayleigh range "
        f"[{lam[0]:.2e}, {lam[-1]:.3f}]"
    )
    out = []
    for r in rs:
        r = min(r, len(lam))
        B = Tmat @ W[:, :r]
        out.append((r, B, lam[:r]))
    return out


def op_from_basis(T, B, lam):
    Zs, ZsH = T.Zs, T.ZsH

    def apply(v):
        return v + Zs @ (B @ ((B.conj().T @ (ZsH @ v)) / lam))

    return LinearOperator((T.n, T.n), matvec=apply, dtype=complex)


def row_blocks(T):
    order = np.argsort(T.rows, kind="stable")
    _, starts = np.unique(T.rows[order], return_index=True)
    return np.split(order, starts[1:])


def block_op(T, blocks):
    fac = []
    for g in blocks:
        Eg = T.E[np.ix_(g, g)]
        fac.append(
            (
                g,
                sla.cho_factor(
                    Eg + 1e-10 * np.trace(Eg).real / len(g) * np.eye(len(g)), lower=True
                ),
            )
        )
    Zs, ZsH = T.Zs, T.ZsH

    def apply(v):
        c = ZsH @ v
        out = np.zeros_like(c)
        for g, ch in fac:
            out[g] = sla.cho_solve(ch, c[g])
        return v + Zs @ out

    return LinearOperator((T.n, T.n), matvec=apply, dtype=complex)


def sparse_op(T, thresh):
    d = np.sqrt(np.abs(np.diag(T.E)))
    Escaled = np.abs(T.E) / np.outer(d, d)
    mask = Escaled > thresh
    Esp = sp.csc_matrix(np.where(mask, T.E, 0.0))
    lu = splu(Esp.tocsc())
    Zs, ZsH = T.Zs, T.ZsH
    dens = Esp.nnz / T.R**2

    def apply(v):
        return v + Zs @ lu.solve(ZsH @ v)

    return LinearOperator((T.n, T.n), matvec=apply, dtype=complex), dens


def main(nside, spin=2, tol=1e-2, rtol=1e-7):
    dfs, _ = make_spin_dfs(nside, spin=spin)
    A, b = capture(dfs, nside, dfs_fold_plan(nside, spin, tol), rtol=rtol)
    t = time.perf_counter()
    x0, it0, _ = cg_count(A, b, rtol=rtol, maxiter=20000)
    t0 = time.perf_counter() - t
    T = TwoLevel(nside, spin, tol)
    print(
        f"nside {nside}  n {T.n}  R {T.R}  plain {it0} its {t0:.3f} s "
        f"({1e3 * t0 / it0:.2f} ms/it)"
    )

    print("  A  orthonormalised compression")
    for r, B, lam in orthonormal_compress(T, [50, 100, 200, 400, 800]):
        Mop = op_from_basis(T, B, lam)
        t = time.perf_counter()
        _, it, _ = cg_count(A, b, M=Mop, rtol=rtol, maxiter=20000)
        dt = time.perf_counter() - t
        print(
            f"     r {r:5d}  {it:4d} its  {1e3 * dt / it:6.2f} ms/it  "
            f"iters {it0 / it:5.1f}x  wall {t0 / dt:5.2f}x"
        )

    print("  B  block-diagonal E by latitude row")
    blocks = row_blocks(T)
    Mop = block_op(T, blocks)
    t = time.perf_counter()
    _, it, _ = cg_count(A, b, M=Mop, rtol=rtol, maxiter=20000)
    dt = time.perf_counter() - t
    print(
        f"     {len(blocks)} blocks (max {max(len(g) for g in blocks)})  {it:4d} its  "
        f"iters {it0 / it:5.1f}x  wall {t0 / dt:5.2f}x"
    )

    print("  C  threshold-sparsified E")
    for th in (1e-3, 1e-2, 5e-2):
        Mop, dens = sparse_op(T, th)
        t = time.perf_counter()
        _, it, _ = cg_count(A, b, M=Mop, rtol=rtol, maxiter=20000)
        dt = time.perf_counter() - t
        print(
            f"     thresh {th:.0e}  density {100 * dens:5.2f}%  {it:4d} its  "
            f"{1e3 * dt / it:6.2f} ms/it  iters {it0 / it:5.1f}x  wall {t0 / dt:5.2f}x"
        )


if __name__ == "__main__":
    for ns in [int(a) for a in sys.argv[1:]] or [32]:
        main(ns)
