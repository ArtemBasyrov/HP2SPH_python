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

Caveats
-------
``level`` implements Morozov's discrepancy principle, for which CG on the normal
equations has classical order-optimality results (Nemirovskii 1986; Hanke 1995;
Hansen 1998 ch. 6). Those results assume the error level is KNOWN.

``eta`` does not have that backing. It estimates the level from the iteration's own
stagnation, which places it with the heuristic rules for an unknown noise level
(Hanke-Raus, generalised cross-validation, the L-curve) rather than with the
discrepancy principle proper. It is motivated by the theory and calibrated by
measurement, not certified by a theorem.
"""

import numpy as np

__all__ = ["cg_normal_equations", "weighted_norm2"]


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
        the error in the data or in the forward model. ``None`` disables it.
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
    for k in range(1, maxiter + 1):
        Ap = matvec(p)
        pAp = np.vdot(p, Ap).real
        if pAp <= 0:
            # N is not positive definite, or rounding has destroyed the property.
            # Stop on the last good iterate rather than take a divergent step.
            break
        alpha = rs / pAp
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

        p = r + (rs_new / rs) * p
        rs = rs_new
    return x, info
