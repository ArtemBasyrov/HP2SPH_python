"""Paper-style accuracy test: error vs *known* coefficients, resolved per ell.

This reproduces the validation methodology of Drake & Wright (arXiv:1904.10514,
Sec. 4): rather than a forward->backward round trip (which only tests
self-consistency), build a signal whose spherical-harmonic coefficients are
KNOWN, analyse it, and measure the actual error against the truth -- per
multipole ell and as a function of resolution nside. The paper's headline claim
is that HEALPix's own ``map2alm`` quadrature loses accuracy at high ell while
HP2SPH stays accurate; the metric that exposes this is the per-ell error curve,
not an integrated norm.

Ground truth is a random alm with a chosen spectrum and known coefficients. The
map is an *exact* sample of the function (azimuthal band capped at the grid's
longitude Nyquist), so a correct analysis returns the in-band coefficients up to
the pipeline's global ``scale``.

What this measurement found at the resolutions reachable here (nside<=32):

* Band-limited signal: HP2SPH converges (sub-band rms ~9.5e-3 -> 1.1e-3 over
  nside 8->32) and beats healpy's *ring weights*, but healpy's default
  *iterative* map2alm is a near-exact inverse (~1e-6) -- a rigged round trip,
  not a real accuracy test.
* Smooth, not-band-limited signal (the paper's regime): the sub-band error is
  dominated by grid aliasing common to all methods, so HP2SPH and both healpy
  configurations agree to within ~the aliasing floor. HP2SPH reproduces
  healpy-quality analysis but does not beat it here.
* The paper's headline accuracy *advantage* over libsharp is a high-ell /
  high-nside (Nside~2048, ell~2000) effect where libsharp's quadrature error
  accumulates and iteration stops converging; that regime is out of reach of the
  slow Julia-subprocess FSHT, so the superiority claim is NOT reproduced here --
  only the methodology, the convergence, and parity with healpy at low nside.

Run (needs Julia / FastTransforms.jl, or the in-process libfasttransforms
backend; use the interpreter from an env with the pipeline deps)::

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
        python -m pytest tests/test_paper_accuracy.py -s

or as a standalone report (prints per-ell tables + saves a plot)::

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
        python tests/test_paper_accuracy.py
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import jax

jax.config.update("jax_enable_x64", True)

import sys

# Allow running as a standalone script (python tests/test_paper_accuracy.py),
# where the repo root is not automatically on sys.path the way pytest puts it.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import healpy as hp
import pytest

from tests.pipeline_helpers import forward_alm, calibrate_scale


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def per_ell_alm_error(alm_rec, alm_true, lmax):
    """Relative L2 error of the recovered alm at each ell (over all m).

    e(ell) = ||a_rec(ell,:) - a_true(ell,:)|| / ||a_true(ell,:)||
    """
    ells, _ = hp.Alm.getlm(lmax, np.arange(len(alm_true)))
    out = np.full(lmax + 1, np.nan)
    for ell in range(lmax + 1):
        sel = ells == ell
        num = np.linalg.norm(alm_rec[sel] - alm_true[sel])
        den = np.linalg.norm(alm_true[sel])
        out[ell] = num / den if den > 0 else np.nan
    return out


def per_ell_cl_error(alm_rec, alm_true, lmax):
    """Relative error of the angular power spectrum C_ell (the paper's metric)."""
    cl_rec = hp.alm2cl(alm_rec, lmax=lmax)
    cl_true = hp.alm2cl(alm_true, lmax=lmax)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(cl_rec - cl_true) / cl_true


def _known_alm(lmax, mmax_cap, seed=20260620, slope=0.0):
    """Random alm in standard (mmax=lmax) packing, with m>mmax_cap zeroed.

    ``slope`` sets the amplitude spectrum sqrt(C_l) ~ (1+l)^(-slope):
    ``slope=0`` is flat (white noise on the sphere -- a maximally rough function
    whose aliasing tail swamps quadrature differences); a positive slope makes a
    smooth field, the regime the paper's test functions live in, where the small
    above-band tail lets each method's quadrature accuracy show through.

    Capping m at ``mmax_cap`` keeps ``alm2map`` an exact sampler (nonzero
    azimuthal content stays below the grid's longitude Nyquist); the array stays
    standard-packed so ``getidx`` works normally.
    """
    rng = np.random.default_rng(seed)
    ncoeff = hp.Alm.getsize(lmax)
    alm = rng.standard_normal(ncoeff) + 1j * rng.standard_normal(ncoeff)
    ells, ems = hp.Alm.getlm(lmax, np.arange(ncoeff))
    alm *= (1.0 + ells.astype(float)) ** (-slope)
    alm[ems == 0] = alm[ems == 0].real
    alm[ems > mmax_cap] = 0.0
    return alm.astype(np.complex128)


def _restrict_to_band(alm_full, lmax_full, lmax_band):
    """Pull the in-band coefficients of a higher-lmax alm into ``lmax_band`` order.

    These are the *true* coefficients an analysis at ``lmax_band`` should recover;
    everything above ``lmax_band`` is the aliasing source, not a target.
    """
    out = np.zeros(hp.Alm.getsize(lmax_band), dtype=np.complex128)
    for ell in range(lmax_band + 1):
        for m in range(min(ell, lmax_band) + 1):
            out[hp.Alm.getidx(lmax_band, ell, m)] = alm_full[
                hp.Alm.getidx(lmax_full, ell, m)
            ]
    return out


def measure(nside, signal_lmax=None, seed=20260620, slope=0.0):
    """Analyse a known map with HP2SPH and healpy; return per-ell errors.

    The analysis band is the grid's natural ``lmax = 2*nside``. ``signal_lmax``
    controls the *content* of the test function:

    * ``signal_lmax == 2*nside`` (or None): a band-limited map. This is the easy,
      self-consistency regime where healpy's iterative ``map2alm`` is a near-exact
      inverse of ``alm2map`` -- not a real accuracy test.
    * ``signal_lmax > 2*nside``: the paper's discriminating regime. Extra content
      sits above the analysis band and ALIASES during analysis, exposing each
      method's latitude/pixel quadrature accuracy. To keep the map an *exact*
      sample of the function (no synthesis aliasing in longitude), the azimuthal
      band is capped at ``mmax = 2*nside - 1`` (< the grid's longitude Nyquist).
    """
    lmax = 2 * nside  # analysis band, fixed by the grid / pipeline
    mmax = 2 * nside - 1  # keep alm2map an exact sampler of the function
    if signal_lmax is None:
        signal_lmax = lmax

    alm_full = _known_alm(signal_lmax, mmax, seed, slope=slope)
    mp = hp.alm2map(alm_full, nside=nside, lmax=signal_lmax)
    alm_true = _restrict_to_band(alm_full, signal_lmax, lmax)

    scale = calibrate_scale(nside, lmax)
    alm_hp2sph = forward_alm(mp, lmax=lmax, scale=scale)

    alm_hpw = hp.map2alm(mp, lmax=lmax, use_weights=True, iter=0)  # ring weights
    alm_hpi = hp.map2alm(mp, lmax=lmax, use_weights=True, iter=3)  # iterative (default)

    return {
        "nside": nside,
        "lmax": lmax,
        "signal_lmax": signal_lmax,
        "ell": np.arange(lmax + 1),
        "hp2sph": per_ell_alm_error(alm_hp2sph, alm_true, lmax),
        "healpy_ring": per_ell_alm_error(alm_hpw, alm_true, lmax),
        "healpy_iter3": per_ell_alm_error(alm_hpi, alm_true, lmax),
        "hp2sph_cl": per_ell_cl_error(alm_hp2sph, alm_true, lmax),
        "healpy_ring_cl": per_ell_cl_error(alm_hpw, alm_true, lmax),
        "healpy_iter3_cl": per_ell_cl_error(alm_hpi, alm_true, lmax),
    }


def _subband_rms(err, lmax, cut=1):
    """RMS of a per-ell error curve below the Nyquist band (ell <= lmax-cut)."""
    e = err[: lmax - cut + 1]
    return float(np.sqrt(np.nanmean(e**2)))


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.julia
def test_competitive_in_aliasing_regime(nside):
    """In the paper's regime (smooth, not band-limited) HP2SPH matches healpy.

    With above-band content the sub-band error is dominated by grid aliasing,
    which is common to every analysis method; HP2SPH, healpy ring weights and
    healpy's iterative map2alm therefore agree to within the aliasing floor. This
    asserts HP2SPH reproduces healpy-quality analysis here (ratio within 20%) --
    NOT superiority. The paper's accuracy *advantage* over libsharp is a high-ell
    / high-nside (Nside~2048, ell~2000) phenomenon, out of reach of this slow
    Julia-subprocess pipeline; at nside<=16 the methods are equivalent.
    """
    m = measure(nside, signal_lmax=4 * nside, slope=1.5)
    lmax = m["lmax"]
    e_hp2sph = _subband_rms(m["hp2sph"], lmax)
    e_healpy = _subband_rms(m["healpy_iter3"], lmax)
    assert 0.8 < e_hp2sph / e_healpy < 1.2, (
        f"nside={nside}: HP2SPH sub-band rms {e_hp2sph:.3e} not on par with "
        f"healpy iter3 {e_healpy:.3e} (ratio {e_hp2sph / e_healpy:.3f})"
    )


@pytest.mark.julia
def test_forward_converges_with_nside():
    """The known-alm error must shrink as nside grows (spectral convergence).

    The paper's core exactness criterion, measured on a band-limited signal where
    HP2SPH's *own* quadrature error is visible (healpy's iterative map2alm is a
    near-exact inverse here and is not the comparison point): a correct transform
    converges under refinement; a systematic convention/normalization bug would
    leave a constant floor. (Band-limited sub-band rms: ~9.5e-3 @ ns8, ~4.5e-3 @
    ns16, ~1.1e-3 @ ns32.)
    """
    errs = {}
    for nside in (8, 16):
        m = measure(nside)  # band-limited (signal_lmax = 2*nside)
        errs[nside] = _subband_rms(m["hp2sph"], m["lmax"])
    assert errs[16] < errs[8], f"no convergence: {errs}"


# --------------------------------------------------------------------------- #
# Standalone report                                                            #
# --------------------------------------------------------------------------- #
def _print_report(results):
    for m in results:
        lmax = m["lmax"]
        print(
            f"\n=== nside={m['nside']}  lmax={lmax} "
            f"(top band ell={lmax} is the longitude Nyquist edge) ==="
        )
        print(
            f"  {'ell':>4} | {'HP2SPH':>10} {'healpy_ring':>12} "
            f"{'healpy_iter3':>13}   (relative alm error)"
        )
        for ell in m["ell"]:
            print(
                f"  {ell:>4} | {m['hp2sph'][ell]:>10.2e} "
                f"{m['healpy_ring'][ell]:>12.2e} {m['healpy_iter3'][ell]:>13.2e}"
            )
        print(
            f"  sub-band rms (ell<=lmax-1): "
            f"HP2SPH={_subband_rms(m['hp2sph'], lmax):.3e}  "
            f"healpy_ring={_subband_rms(m['healpy_ring'], lmax):.3e}  "
            f"healpy_iter3={_subband_rms(m['healpy_iter3'], lmax):.3e}"
        )

    print("\n=== convergence of sub-band rms alm error vs nside ===")
    print(f"  {'nside':>6} {'HP2SPH':>12} {'healpy_iter3':>14}")
    for m in results:
        print(
            f"  {m['nside']:>6} {_subband_rms(m['hp2sph'], m['lmax']):>12.3e} "
            f"{_subband_rms(m['healpy_iter3'], m['lmax']):>14.3e}"
        )


def _save_plot(results, path="paper_accuracy.png"):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"(skipping plot: {exc})")
        return
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, m in zip(axes[0], results):
        lmax = m["lmax"]
        ax.semilogy(m["ell"], m["hp2sph"], "o-", label="HP2SPH")
        ax.semilogy(m["ell"], m["healpy_ring"], "s--", label="healpy ring wts")
        ax.semilogy(m["ell"], m["healpy_iter3"], "^:", label="healpy iter=3")
        ax.axvline(lmax, color="k", ls=":", alpha=0.4)
        ax.set_title(f"nside={m['nside']}  (lmax={lmax})")
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"relative $a_{\ell m}$ error")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nsaved per-ell error plot -> {os.path.abspath(path)}")


if __name__ == "__main__":
    nsides = [int(x) for x in sys.argv[1:]] or [8, 16, 32]

    print("\n##### BAND-LIMITED regime (signal_lmax = 2*nside) #####")
    print("# healpy's iterative map2alm is a near-exact inverse here -- easy case.")
    bl = []
    for ns in nsides:
        print(f"measuring nside={ns} (band-limited) ...", flush=True)
        bl.append(measure(ns))
    _print_report(bl)

    print("\n\n##### ALIASED SMOOTH regime (signal_lmax=4*nside, sqrt(Cl)~(1+l)^-1.5)")
    print("# the paper's test: a SMOOTH function with a small above-band tail that")
    print("# aliases in, so each method's quadrature accuracy shows through.")
    al = []
    for ns in nsides:
        print(f"measuring nside={ns} (aliased, smooth) ...", flush=True)
        al.append(measure(ns, signal_lmax=4 * ns, slope=1.5))
    _print_report(al)
    _save_plot(al, path="paper_accuracy_aliased.png")
