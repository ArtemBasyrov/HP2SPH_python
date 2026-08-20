"""Time the REAL pieces of the shipped half-domain matvec, by pulling the closure
cells out of ``apply_AHA`` rather than reimplementing it."""

import sys, time, contextlib, io

sys.path.insert(0, "tools/precond_diagnostics")
import numpy as np, healpy as hp
import src.nuFFT as _nu
from src.cg import _axpy_for
from src.spin_transform import forward_spin


def sky(ns, seed=0, slope=1.5):
    lmax = 2 * ns
    ell = np.arange(lmax + 1)
    np.random.seed(seed)
    aE = hp.synalm((1.0 + ell) ** (-2.0 * slope), lmax=lmax, new=True)
    Q, U = hp.alm2map_spin([aE, np.zeros_like(aE)], ns, 2, lmax)
    return Q, U, lmax


def cells(fn):
    return dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in fn.__closure__)))


def timeit(fn, n=7):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts)


def run(ns):
    Q, U, lmax = sky(ns)
    box = {}
    orig = _nu.cg_normal_equations

    def spy(matvec, rhs, bw2, **kw):
        box["mv"] = matvec
        box["rhs"] = rhs
        return np.zeros_like(rhs), 0

    _nu.cg_normal_equations = spy
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            forward_spin(Q, U, lmax)
    finally:
        _nu.cg_normal_equations = orig

    mv, rhs = box["mv"], box["rhs"]
    A = cells(mv)  # apply_AHA
    B = cells(A["adjoint_of"])  # adjoint_of
    n_trans, K = A["n_trans"], A["K"]
    expand, fold_apply, gbuf, plan_f = (
        A["expand"],
        A["fold_apply"],
        A["gbuf"],
        A["plan_forward"],
    )
    restrict, weight, fold_adj = B["restrict"], B["weight"], B["fold_adjoint"]
    plan_a, coef, rbuf, norm = B["plan_adjoint"], B["coef"], B["rbuf"], B["norm"]
    # ``/norm`` is folded into ``restrict``'s scale whenever ``norm`` is scalar, so the
    # separate pass only exists on the per-column path.
    folded = cells(restrict).get("scale_out") is not cells(expand).get("scale")

    rng = np.random.default_rng(0)
    vec = (rng.standard_normal(len(rhs)) + 1j * rng.standard_normal(len(rhs))).astype(
        np.complex128
    )
    ch = vec.reshape(n_trans, K + 1)

    total_mv = timeit(lambda: mv(vec))
    t_exp = timeit(lambda: expand(ch))
    ex = expand(ch)
    t_fwd = timeit(lambda: plan_f.execute(ex, gbuf))
    plan_f.execute(ex, gbuf)
    t_fold = timeit(lambda: fold_apply(gbuf, out=gbuf)) if fold_apply else 0.0
    t_w = timeit(lambda: weight(gbuf))
    t_fadj = timeit(lambda: fold_adj(gbuf, out=gbuf)) if fold_adj else 0.0
    t_adj = timeit(lambda: plan_a.execute(gbuf, coef))
    plan_a.execute(gbuf, coef)
    t_res = timeit(lambda: restrict(coef))
    t_div = 0.0 if folded else timeit(lambda: np.divide(rbuf, norm, out=rbuf))

    # CG's own per-iteration vector work, as ``src.cg.cg_normal_equations`` now does
    # it: two BLAS axpy, one scaled update, and three inner products.
    x = np.zeros_like(rhs)
    r = rhs.copy()
    p = rhs.copy()
    Ap = rhs.copy()
    axpy = _axpy_for(x, r, p)

    def cg_vec():
        pAp = np.vdot(p, Ap).real
        a = 1.0 / pAp
        if axpy is not None:
            axpy(p, x, a=a)
            axpy(Ap, r, a=-a)
        else:
            np.add(x, a * p, out=x)
            np.subtract(r, a * Ap, out=r)
        rs = np.vdot(r, r).real
        np.vdot(x, rhs)
        np.vdot(x, r)
        np.multiply(p, rs / (rs + 1.0), out=p)
        np.add(p, r, out=p)

    t_cg = timeit(cg_vec)

    parts = [
        ("expand (embed half->full)", t_exp),
        ("forward NUFFT (type 2)", t_fwd),
        ("fold_apply", t_fold),
        ("weights", t_w),
        ("fold_adjoint", t_fadj),
        ("adjoint NUFFT (type 1)", t_adj),
        ("restrict (full->half)", t_res),
        ("/norm (folded into restrict)" if folded else "/norm", t_div),
        ("CG vector ops + dots", t_cg),
    ]
    s = sum(t for _, t in parts)
    print(f"\n=== nside {ns}: n_trans {n_trans}, K+1 {K + 1}, unknowns {len(rhs)} ===")
    print(
        f"  measured whole matvec: {1e3 * total_mv:7.2f} ms   "
        f"(sum of pieces below, excl. CG ops: {1e3 * (s - t_cg):7.2f} ms)"
    )
    for name, t in parts:
        print(f"    {name:28s} {1e3 * t:8.2f} ms  {100 * t / s:5.1f}%")
    nufft = t_fwd + t_adj
    print(
        f"    -> NUFFT {100 * nufft / s:4.1f}% | half-domain glue (expand+restrict) "
        f"{100 * (t_exp + t_res) / s:4.1f}% | fold+weights {100 * (t_fold + t_w + t_fadj) / s:4.1f}%"
        f" | CG ops {100 * t_cg / s:4.1f}%"
    )


for ns in (256, 512, 1024):
    run(ns)
