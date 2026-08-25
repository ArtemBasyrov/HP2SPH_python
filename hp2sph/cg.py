"""Conjugate gradients on normal equations, with the data residual available.

A least-squares fit ``min ||b - A x||_W`` is usually handed to CG as the normal
equations ``A^H W A x = A^H W b``. CG then measures its own progress by the
NORMAL-equation residual ``||A^H W (b - Ax)||``, which is what
``scipy.sparse.linalg.cg(rtol=...)`` tests. That is the wrong quantity for deciding
when to stop an ill-posed fit: the criteria from regularisation theory are stated for
the DATA residual ``||b - A x||_W``, and the two behave very differently -- the data
residual reaches the least-squares floor long before the normal-equation residual
settles.

``cg_normal_equations`` runs the same iteration and reports both, at a cost of two
inner products per step and no extra operator application. It offers three stopping
rules, which can be combined:

* ``rtol`` -- the usual normal-equation test, kept as a fallback and as the default.
* ``level`` -- stop once the data residual reaches a known error level.
* ``eta`` -- stop once the data residual STOPS FALLING, for when that level is not
  known in advance.

See ``solve`` for the identity that makes the data residual free, and the caveats
section for what these rules do and do not guarantee.

Choosing between ``level`` and ``eta``
-------------------------------------
``level`` is Morozov's discrepancy principle, for which CG on the normal equations has
order-optimality results (Nemirovskii 1986; Neubauer 2022). Use it when the error is
known well enough to bound. Where the error is in the OPERATOR rather than in the data,
the level those results ask for is

    ||b - A_h x_k||  <=  Theta * ( h * ||x_k|| + eps )

with ``h`` bounding the operator perturbation, ``eps`` the data noise, and
``Theta > 1``. Note the ``||x_k||`` factor and that ``h`` covers the WHOLE perturbation,
including any discretisation or band truncation, not just the part one happens to have
a tolerance for.

``eta`` is for when no such bound is available. It stops once the data residual reaches
its own floor, which is an estimate of the same level from the iteration itself. This
does not inherit the rate guarantee: those results require ``h`` known in advance and
``Theta`` strictly above 1, whereas stagnation puts the stop at ``Theta`` of about 1.
Calibrate it by measurement on the problem at hand.
"""

import numpy as np
from scipy.linalg.blas import get_blas_funcs

__all__ = ["cg_normal_equations", "weighted_norm2"]


def _axpy_for(*arrays):
    """``y += a * x`` in one pass, or ``None`` if BLAS cannot be used here.

    The vector updates are single passes over an array the size of the unknowns, and at
    nside 1024 that is 0.25 GB running at DRAM bandwidth, so the temporary that
    ``y += a * x`` allocates costs a whole extra pass. BLAS ``axpy`` writes in place with
    no temporary. It needs contiguous same-dtype arrays, so callers that do not have
    them fall back to the numpy expression.
    """
    if not all(a.flags.c_contiguous for a in arrays):
        return None
    if len({a.dtype for a in arrays}) != 1:
        return None
    try:
        return get_blas_funcs("axpy", arrays)
    except (ValueError, TypeError):
        return None


def weighted_norm2(b, weights):
    """``||b||_W^2 = sum_r w_r |b_r|^2``.

    ``weights`` broadcasts against ``b``, so a per-sample vector and a per-block
    array both work.
    """
    b = np.asarray(b)
    return float(np.sum(weights * (b.real**2 + b.imag**2)))


def cg_normal_equations(
    matvec,
    rhs,
    bw2,
    norm=1.0,
    rtol=1e-9,
    maxiter=None,
    level=None,
    eta=None,
    delay=5,
    monitor=None,
):
    """Solve ``N x = rhs`` by conjugate gradients, tracking the data residual.

    ``N`` must be Hermitian positive definite. It is the normal operator of a
    weighted least-squares fit, scaled by ``norm``:

        N = A^H W A / norm,      rhs = A^H W b / norm.

    The iteration starts from ``x = 0``.

    Parameters
    ----------
    matvec : callable
        Applies ``N`` to a flat vector and returns a flat vector.
    rhs : ndarray
        Right-hand side, flat. Real or complex.
    bw2 : float
        ``||b||_W^2``, the weighted squared norm of the data being fitted. Use
        ``weighted_norm2``. This is what makes the data residual computable.
    norm : float
        The scalar by which ``A^H W A`` and ``A^H W b`` were divided to form ``N``
        and ``rhs``. Pass 1.0 if they were not scaled.
    rtol : float
        Stop when ``||rhs - N x_k|| <= rtol * ||rhs||``, the convention
        ``scipy.sparse.linalg.cg`` uses.
    maxiter : int or None
        Iteration cap. ``None`` means ``10 * len(rhs)``.
    level : float or None
        Stop at the first iterate with ``||b - A x_k||_W <= level``. Use the size of
        the error in the data or in the forward model; see the module docstring for the
        form the regularisation literature asks for when the error is in the operator.
        ``None`` disables it.
    eta : float or None
        Stop at the first ``k >= delay`` whose squared data residual has fallen by
        less than a fraction ``eta`` over the preceding ``delay`` iterations, i.e.
        once it has stopped improving. Use when the error level is not known in
        advance. ``None`` disables it.
    delay : int
        Window, in iterations, over which ``eta`` is measured.
    monitor : callable or None
        Called as ``monitor(k, rho, rrel, x)`` after each iteration. ``k`` counts from
        1, ``rho`` is ``||b - A x_k||_W / ||b||_W``, and ``rrel`` is the ratio ``rtol``
        tests. ``x`` is the LIVE iterate buffer and is overwritten in place on the next
        step, so copy it if you intend to keep it.

    Returns
    -------
    (x, info) : (ndarray, int)
        ``info`` is 0 if any stopping rule fired, and ``maxiter`` if none did. This
        matches ``scipy.sparse.linalg.cg``.

    Notes
    -----
    Whichever rule fires first stops the iteration; they are checked in the order
    ``level``, ``eta``, ``rtol``.

    The data residual is obtained from quantities the iteration already holds. With
    ``J(x) = ||b - A x||_W^2``, expanding the square gives

        J(x) = ||b||_W^2 - 2 Re<x, A^H W b> + <x, A^H W A x>,

    and substituting ``A^H W A x_k = norm * (rhs - r_k)`` and ``A^H W b = norm * rhs``,
    where ``r_k`` is the CG residual, leaves

        J(x_k) = ||b||_W^2 - norm * ( Re<x_k, rhs> + Re<x_k, r_k> ).

    No second application of the operator is needed, so the cost is two inner products
    per iteration.
    """
    x = np.zeros_like(rhs)
    r = np.array(rhs, copy=True)
    p = r.copy()
    rs = np.vdot(r, r).real
    b0 = np.sqrt(np.vdot(rhs, rhs).real)
    bnorm = np.sqrt(bw2) if bw2 > 0 else 1.0
    if maxiter is None:
        maxiter = 10 * len(rhs)
    if b0 == 0.0:
        return x, 0

    info = maxiter
    hist = [float(bw2)]  # squared data residual at x = 0
    axpy = _axpy_for(x, r, p)
    for k in range(1, maxiter + 1):
        Ap = matvec(p)
        pAp = np.vdot(p, Ap).real
        if pAp <= 0:
            # N is not positive definite, or rounding has destroyed the property.
            # Stop on the last good iterate rather than take a divergent step.
            break
        alpha = rs / pAp
        if axpy is not None and Ap.flags.c_contiguous and Ap.dtype == x.dtype:
            axpy(p, x, a=alpha)
            axpy(Ap, r, a=-alpha)
        else:
            x += alpha * p
            r -= alpha * Ap
        rs_new = np.vdot(r, r).real
        rrel = np.sqrt(rs_new) / b0
        # J is a difference against ||b||_W^2 and can go a few ulp negative once the
        # residual has stagnated at the least-squares floor.
        J = bw2 - norm * (np.vdot(x, rhs).real + np.vdot(x, r).real)
        rho = np.sqrt(max(J, 0.0)) / bnorm

        if monitor is not None:
            monitor(k, rho, rrel, x)

        if level is not None and rho * bnorm <= level:
            info = 0
            break
        if eta is not None:
            hist.append(max(J, 0.0))
            if len(hist) > delay + 1:
                hist.pop(0)
            if len(hist) == delay + 1 and hist[-1] > 0:
                if (hist[0] - hist[-1]) / hist[-1] <= eta:
                    info = 0
                    break
        if rrel <= rtol:
            info = 0
            break

        # p = r + beta * p, written as two in-place passes rather than three plus a
        # fresh allocation every iteration.
        np.multiply(p, rs_new / rs, out=p)
        p += r
        rs = rs_new
    return x, info
