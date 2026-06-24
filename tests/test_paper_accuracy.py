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
  accumulates and iteration stops converging; that regime needs nside >= 128 and
  is not exercised by the small-nside cases here -- only the methodology, the
  convergence, and parity with healpy at low nside.

Run (needs the libfasttransforms C library; use the interpreter from an env with
the pipeline deps)::

    python -m pytest tests/test_paper_accuracy.py -s

or as a standalone report (prints per-ell tables + saves a plot)::

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

# The pipeline helpers load the C library on import; skip cleanly if it is missing.
pytest.importorskip("src.ft_sphere")

from tests.pipeline_helpers import forward_alm, calibrate_scale  # noqa: E402


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


def _band_abs_error(alm_rec, alm_true, lmax, lo, hi):
    """Total L2 error over ell in [lo, hi], normalised by the in-band amplitude.

    Robust (no per-coefficient division), so it is not inflated by the small
    random m=0 coefficients the per-ell relative metric divides by.
    """
    ells, _ = hp.Alm.getlm(lmax, np.arange(len(alm_true)))
    sel = (ells >= lo) & (ells <= hi)
    return np.linalg.norm(alm_rec[sel] - alm_true[sel]) / np.linalg.norm(alm_true[sel])


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.ft
def test_competitive_in_aliasing_regime(nside):
    """In the paper's regime (smooth, not band-limited) HP2SPH matches healpy.

    With above-band content the sub-band error is dominated by grid aliasing,
    which is common to every analysis method; HP2SPH, healpy ring weights and
    healpy's iterative map2alm therefore agree to within the aliasing floor. This
    asserts HP2SPH reproduces healpy-quality analysis here (ratio within ~35%) --
    NOT superiority. The default well-conditioned (4*nside+1) band drops some
    above-band latitude content, so in this aliasing regime it runs a touch behind
    healpy at coarse nside (ratio ~1.24 @ ns8) -- still on par. The paper's accuracy
    *advantage* over libsharp is a high-ell / high-nside (Nside~2048, ell~2000)
    phenomenon, not exercised by the small-nside cases here.
    """
    m = measure(nside, signal_lmax=4 * nside, slope=1.5)
    lmax = m["lmax"]
    e_hp2sph = _subband_rms(m["hp2sph"], lmax)
    e_healpy = _subband_rms(m["healpy_iter3"], lmax)
    assert 0.8 < e_hp2sph / e_healpy < 1.35, (
        f"nside={nside}: HP2SPH sub-band rms {e_hp2sph:.3e} not on par with "
        f"healpy iter3 {e_healpy:.3e} (ratio {e_hp2sph / e_healpy:.3f})"
    )


@pytest.mark.ft
def test_square_band_beats_healpy_at_band_edge():
    """Square band de-aliases the harsh band edge (NOT the paper's method).

    This is a niche regime, kept as a guard, NOT the paper reproduction (that is
    test_compact_band_reproduces_paper_high_ell, which uses the scalable default
    band -- the paper's actual method). Here the signal is the WORST case for the
    compact band: flat-spectrum, fully band-limited, so its top band carries FULL
    power and the polar-undersampling aliasing the compact band folds in is
    maximally exposed. The exact square latitude band (solve_modes = 8*nside+1)
    resolves that above-|k|=2*nside content, so it beats healpy ring AND pixel
    weights at the top quarter band even at low nside. For realistic SMOOTH signals
    the compact band already matches this, so the square band's O(nside^3)
    ill-conditioned solve is rarely worth it. Robust total-L2 band error.
    """
    nside = 32
    lmax = 2 * nside
    mmax = 2 * nside - 1
    alm_true = _known_alm(lmax, mmax)
    mp = hp.alm2map(alm_true, nside=nside, lmax=lmax)

    rec = forward_alm(mp, lmax=lmax, solver="svd", solve_modes=8 * nside + 1)
    hpr = hp.map2alm(mp, lmax=lmax, use_weights=True, iter=0)  # ring weights
    hpp = hp.map2alm(mp, lmax=lmax, use_pixel_weights=True, iter=0)  # pixel weights

    lo, hi = 3 * lmax // 4 + 1, lmax  # top quarter band, around the Nyquist edge
    e_hp2sph = _band_abs_error(rec, alm_true, lmax, lo, hi)
    e_ring = _band_abs_error(hpr, alm_true, lmax, lo, hi)
    e_pixel = _band_abs_error(hpp, alm_true, lmax, lo, hi)
    assert e_hp2sph < e_ring, f"hp2sph {e_hp2sph:.3e} !< healpy ring {e_ring:.3e}"
    assert e_hp2sph < e_pixel, f"hp2sph {e_hp2sph:.3e} !< healpy pixel {e_pixel:.3e}"


@pytest.mark.ft
def test_compact_band_reproduces_paper_high_ell():
    """THE PAPER REPRODUCTION (Drake & Wright Fig 8b), using the SCALABLE default.

    The paper's own method is the well-conditioned truncated latitude solve (their
    "m = 4*nside+1 modes", = our compact default band) -- NOT a square/exact solve.
    Its headline is a high-ell phenomenon: healpy's single-pass quadrature error
    GROWS toward high ell (ring weights diverge for ell > ~200) while HP2SPH stays
    accurate, so HP2SPH wins by a growing margin in the upper band. This needs
    ell >~ 200, i.e. nside >~ 128 -- at nside <= 64 the two look similar (NOT the
    paper's regime). Metric = the paper's per-ell C_ell relative error.

    nside=128 (lmax=256), smooth signal (sqrt(Cl) ~ (1+l)^-1.5). At the upper band
    HP2SPH beats ring weights by ~5-9x (measured ~6-9x); we assert >=3x with margin.
    Compact-band forward is ~0.5 s here, so the test stays cheap.
    """
    nside = 128
    lmax = 2 * nside
    mmax = 2 * nside - 1
    alm_true = _known_alm(lmax, mmax, slope=1.5)  # smooth (the paper's regime)
    mp = hp.alm2map(alm_true, nside=nside, lmax=lmax)

    rec = forward_alm(mp, lmax=lmax)  # compact default band, scale = 1/(2pi)
    hpr = hp.map2alm(mp, lmax=lmax, use_weights=True, iter=0)  # ring weights

    cl_true = hp.alm2cl(alm_true, lmax=lmax)
    lo, hi = 3 * lmax // 4, 7 * lmax // 8  # upper band, below the very Nyquist edge
    with np.errstate(divide="ignore", invalid="ignore"):
        e_hp2sph = np.abs(hp.alm2cl(rec, lmax=lmax) - cl_true) / cl_true
        e_ring = np.abs(hp.alm2cl(hpr, lmax=lmax) - cl_true) / cl_true
    r_hp2sph = float(np.sqrt(np.nanmean(e_hp2sph[lo : hi + 1] ** 2)))
    r_ring = float(np.sqrt(np.nanmean(e_ring[lo : hi + 1] ** 2)))
    assert r_hp2sph < r_ring / 3.0, (
        f"paper claim not reproduced: HP2SPH C_ell err {r_hp2sph:.3e} not <<"
        f" ring weights {r_ring:.3e} over ell {lo}-{hi}"
    )


@pytest.mark.ft
def test_forward_converges_with_nside():
    """The known-alm error must shrink as nside grows (spectral convergence).

    The paper's core exactness criterion, measured on a band-limited signal where
    HP2SPH's *own* quadrature error is visible (healpy's iterative map2alm is a
    near-exact inverse here and is not the comparison point): a correct transform
    converges under refinement; a systematic convention/normalization bug would
    leave a constant floor. (Sub-band rms with the polynomial pole fill: ~8.0e-3 @
    ns8, ~1.0e-2 @ ns16, ~2.0e-3 @ ns32.)

    The check is coarsest-vs-finest (ns8 vs ns32), NOT strict step monotonicity:
    in this band-limited regime the metric is dominated by the longitude-Nyquist
    top band (ell = lmax), whose error does not fall smoothly with nside, so the
    ns16 midpoint can sit above ns8 even though the transform is converging (and
    the higher-order pole fill, which most helps the coarsest grid, accentuates
    this). The robust global-L2 error is monotone (ns8 1.2e-2 -> ns32 2.9e-3).
    """
    errs = {}
    for nside in (8, 16, 32):
        m = measure(nside)  # band-limited (signal_lmax = 2*nside)
        errs[nside] = _subband_rms(m["hp2sph"], m["lmax"])
    assert errs[32] < 0.5 * errs[8], f"no convergence: {errs}"


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
