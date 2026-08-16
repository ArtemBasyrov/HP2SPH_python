"""Benchmark 2 -- how close the forward (map -> alm) transform is to the truth.

The measurement follows Drake & Wright (arXiv:1904.10514, Sec. 4): do NOT test a
forward against its own inverse, which only proves self-consistency. Instead
start from coefficients that are *known*, synthesise the map, analyse it, and
compare against those known coefficients.

Three signal regimes are run. They differ in whether the sky is BAND LIMITED and
in how smooth it is, and those two axes are independent.

``cosmology`` -- THE PRIMARY REGIME
    ``signal_lmax = lmax = 2*nside`` with ``sqrt(C_l) ~ (1+l)^-1.5``: smooth and
    band limited. This is the regime cosmology actually works in, so it is what
    the suite leads with and what conclusions should be drawn from. It is also
    the exact configuration of the repository's own paper reproduction,
    ``tests/test_paper_accuracy.py::test_compact_band_reproduces_paper_high_ell``.
    Measured here over ``l = 3*lmax/4 .. 7*lmax/8``, median of 4 seeds, HP2SPH
    beats healpy ring weights by 1.97x at nside 32 rising to 4.42x at nside 1024
    -- a real advantage that grows with resolution, which is the paper's claim.

``flat``
    ``signal_lmax = 2*nside``, flat spectrum. Band limited like ``cosmology``, but
    with full power at the band edge, so it is the worst case WITHIN the primary
    regime rather than a different regime. Useful as a stress bound; not
    representative of a sky.

``aliased``
    ``signal_lmax = 4*nside``, same slope. A smooth function with an above-band
    tail that aliases during analysis. Diagnostic only: it exposes latitude
    quadrature, but the resulting error is dominated by grid aliasing common to
    every method, so it does not separate them well and it is not the regime real
    analyses run in. It is the regime in which the spin-2 forward's top-of-band
    defect shows up (see ``benchmarks/README.md``).

In every regime ``m`` is capped at ``2*nside-1`` so ``alm2map`` stays an exact
sampler of the function and the truth is well defined.

Note that on band-limited input healpy's iterative ``map2alm`` and ducc0's
``pseudo_analysis`` are near-exact inverses of ``alm2map`` by construction, and
they score 3 to 4 orders of magnitude better than any single-pass method. That is
a real property, not an artefact, and it is the honest headline of this
benchmark: if you want accuracy on a band-limited sky and can afford the
iterations, they win.

For polarization there is a third measurement, ``leakage``, which neither the
paper nor healpy reports directly: feed a pure-E sky and measure how much power
comes back as B (and vice versa). For CMB work that ratio, not the diagonal
accuracy, is what decides whether a transform is usable.

Metrics per multipole: relative ``C_l`` error (the paper's, phase-blind) and
relative ``a_lm`` L2 error (phase sensitive, so it catches convention bugs a
``C_l`` comparison cannot). Both are summarised over four ``l`` bands, because a
single number across the whole spectrum averages the interesting top of the band
away against the bulk, where everything is near-exact.

Run::

    python -m benchmarks.bench_accuracy --channel I
    python -m benchmarks.bench_accuracy --channel P --nside 8 16 32
"""

import argparse
import os

import numpy as np
import healpy as hp

from benchmarks import backends as bk
from benchmarks import common
from benchmarks.common import (
    RESULTS_DIR,
    band_l2,
    band_rms,
    ell_bands,
    env_metadata,
    markdown_table,
    per_ell_alm_error,
    per_ell_cl_error,
    per_ell_leakage,
    random_EB,
    random_alm,
    restrict_to_band,
)

# Both ladders reach nside 2048, but the cost is 4 scenarios x len(seeds) transforms per
# backend per nside, so the top two rungs are hours rather than minutes and are worth
# running separately with fewer seeds. A cell measured with fewer seeds is still a valid
# cell -- ``n_seeds`` is stored per record, so the seed count is never implied.
DEFAULT_NSIDE_I = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
DEFAULT_NSIDE_P = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
DEFAULT_SEEDS = [0, 1, 2, 3]

SCENARIOS = {
    # name: (signal_lmax multiplier of nside, amplitude slope)
    # Ordered by priority: `cosmology` is the primary regime (see the docstring).
    "cosmology": (2, 1.5),
    "flat": (2, 0.0),
    "aliased": (4, 1.5),
}

PRIMARY_SCENARIO = "cosmology"

KEY_FIELDS = ("channel", "backend", "nside", "scenario", "field")


def _analysis_inputs(channel, nside, lmax, signal_lmax, slope, seed):
    """Known coefficients, the map they generate, and the in-band truth."""
    mmax = 2 * nside - 1
    if channel == "I":
        alm_full = random_alm(signal_lmax, seed, slope=slope, mmax_cap=mmax)
        mp = hp.alm2map(alm_full, nside=nside, lmax=signal_lmax)
        truth = {"T": restrict_to_band(alm_full, signal_lmax, lmax)}
        return {"map": mp}, truth
    aE, aB = random_EB(signal_lmax, seed, slope=slope, mmax_cap=mmax)
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, signal_lmax)
    truth = {
        "E": restrict_to_band(aE, signal_lmax, lmax),
        "B": restrict_to_band(aB, signal_lmax, lmax),
    }
    return {"Q": Q, "U": U}, truth


def _forward(backend, channel, inputs, nside, lmax):
    if channel == "I":
        return {"T": backend.forward_I(inputs["map"], nside, lmax)}
    aE, aB = backend.forward_P(inputs["Q"], inputs["U"], nside, lmax)
    return {"E": aE, "B": aB}


def _leakage_inputs(nside, lmax, seed, pure):
    """A sky with power in exactly one parity channel."""
    mmax = 2 * nside - 1
    aE, aB = random_EB(lmax, seed, slope=1.5, mmax_cap=mmax)
    if pure == "E":
        aB = np.zeros_like(aB)
    else:
        aE = np.zeros_like(aE)
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, lmax)
    return {"Q": Q, "U": U}, aE, aB


def run(channel, nsides, keys, scenarios, seeds, out_path, resume, threads=1):
    chosen = bk.select(keys)
    applied = bk.set_threads(threads)
    lmin = 2 if channel == "P" else 0
    meta = env_metadata(
        {
            "benchmark": "forward_accuracy",
            "channel": channel,
            "seeds": seeds,
            "threads": applied,
            "lmax_rule": "2*nside",
            "mmax_cap_rule": "2*nside-1 (keeps alm2map an exact sampler)",
            "note": "per-l curves are the median across seeds",
        }
    )
    store = common.ResultStore(out_path, meta, KEY_FIELDS)
    fields = ["T"] if channel == "I" else ["E", "B"]

    for nside in nsides:
        lmax = 2 * nside
        bands = ell_bands(lmax, lmin=lmin)

        for scenario in scenarios:
            if scenario not in SCENARIOS:  # "leakage" is handled separately below
                continue
            mult, slope = SCENARIOS[scenario]
            signal_lmax = mult * nside
            todo = [
                b
                for b in chosen
                if b.available_at(nside, channel)[0]
                and not (
                    resume
                    and all(
                        store.has(
                            channel=channel,
                            backend=b.key,
                            nside=nside,
                            scenario=scenario,
                            field=f,
                        )
                        for f in fields
                    )
                )
            ]
            print(
                f"\n=== {channel} nside={nside} lmax={lmax} scenario={scenario} "
                f"(signal_lmax={signal_lmax}, slope={slope}) ===",
                flush=True,
            )
            if not todo:
                print("  (all cached)")
                continue

            # accumulate per-seed curves, then reduce
            acc = {
                (b.key, f): {"cl": [], "alm": [], "l2": []}
                for b in todo
                for f in fields
            }
            for seed in seeds:
                inputs, truth = _analysis_inputs(
                    channel, nside, lmax, signal_lmax, slope, seed
                )
                for backend in todo:
                    try:
                        rec = _forward(backend, channel, inputs, nside, lmax)
                    except Exception as exc:
                        print(f"  {backend.key:<16} seed {seed} FAILED: {exc}")
                        continue
                    for f in fields:
                        a = acc[(backend.key, f)]
                        a["cl"].append(per_ell_cl_error(rec[f], truth[f], lmax))
                        a["alm"].append(per_ell_alm_error(rec[f], truth[f], lmax))
                        a["l2"].append(
                            {
                                lbl: band_l2(rec[f], truth[f], lmax, lo, hi)
                                for lbl, lo, hi in bands
                            }
                        )

            top_label = bands[-1][0]
            for backend in todo:
                shown = []
                for f in fields:
                    a = acc[(backend.key, f)]
                    if not a["cl"]:
                        continue
                    cl = np.nanmedian(np.vstack(a["cl"]), axis=0)
                    alm = np.nanmedian(np.vstack(a["alm"]), axis=0)
                    band_summary = {
                        lbl: {
                            "lo": lo,
                            "hi": hi,
                            "cl_rms": band_rms(cl, lo, hi),
                            "alm_rms": band_rms(alm, lo, hi),
                            "l2": float(np.nanmedian([d[lbl] for d in a["l2"]])),
                        }
                        for lbl, lo, hi in bands
                    }
                    store.add(
                        {
                            "channel": channel,
                            "backend": backend.key,
                            "nside": nside,
                            "lmax": lmax,
                            "scenario": scenario,
                            "field": f,
                            "signal_lmax": signal_lmax,
                            "slope": slope,
                            "lmin": lmin,
                            "cl_err_per_ell": cl,
                            "alm_err_per_ell": alm,
                            "bands": band_summary,
                        }
                    )
                    shown.append(f"{f} Cl_rms={band_summary[top_label]['cl_rms']:.2e}")
                if shown:
                    print(f"  {backend.key:<16} l {top_label}:  " + "   ".join(shown))

        if channel == "P" and "leakage" in scenarios:
            _run_leakage(store, chosen, nside, lmax, seeds, bands, resume)

    print(f"\nwrote {out_path}")
    _summary(store, channel, scenarios)
    return store


def _run_leakage(store, chosen, nside, lmax, seeds, bands, resume):
    """Pure-E and pure-B skies: how much power crosses the parity split."""
    print(f"\n=== P nside={nside} scenario=leakage ===", flush=True)
    for pure, field in (("E", "B_from_E"), ("B", "E_from_B")):
        for backend in chosen:
            if not backend.available_at(nside, "P")[0]:
                continue
            if resume and store.has(
                channel="P",
                backend=backend.key,
                nside=nside,
                scenario="leakage",
                field=field,
            ):
                continue
            curves = []
            for seed in seeds:
                inputs, _, _ = _leakage_inputs(nside, lmax, seed, pure)
                try:
                    aE_rec, aB_rec = backend.forward_P(
                        inputs["Q"], inputs["U"], nside, lmax
                    )
                except Exception as exc:
                    print(f"  {backend.key:<16} {field} FAILED: {exc}")
                    break
                signal, contaminant = (
                    (aE_rec, aB_rec) if pure == "E" else (aB_rec, aE_rec)
                )
                curves.append(per_ell_leakage(contaminant, signal, lmax))
            if not curves:
                continue
            curve = np.nanmedian(np.vstack(curves), axis=0)
            store.add(
                {
                    "channel": "P",
                    "backend": backend.key,
                    "nside": nside,
                    "lmax": lmax,
                    "scenario": "leakage",
                    "field": field,
                    "lmin": 2,
                    "leak_per_ell": curve,
                    "bands": {
                        lbl: {
                            "lo": lo,
                            "hi": hi,
                            "leak_median": float(np.nanmedian(curve[lo : hi + 1])),
                        }
                        for lbl, lo, hi in bands
                    },
                }
            )
            print(
                f"  {backend.key:<16} {field}: median power ratio "
                f"{np.nanmedian(curve):.2e}"
            )


def _summary(store, channel, scenarios):
    records = store.frame()
    for scenario in scenarios:
        rows = []
        sel = [r for r in records if r["scenario"] == scenario and "bands" in r]
        if not sel:
            continue
        nsides = sorted({r["nside"] for r in sel})
        field = "T" if channel == "I" else "E"
        sel = [r for r in sel if r.get("field") in (field, "B_from_E")]
        metric = "leak_median" if scenario == "leakage" else "cl_rms"
        for key in bk.DEFAULT_KEYS:
            row = [key]
            any_val = False
            for n in nsides:
                match = [r for r in sel if r["backend"] == key and r["nside"] == n]
                if not match:
                    row.append(None)
                    continue
                top = list(match[0]["bands"])[-1]
                row.append(match[0]["bands"][top].get(metric))
                any_val = True
            if any_val:
                rows.append(row)
        if rows:
            label = (
                "median E->B power ratio"
                if scenario == "leakage"
                else f"{field} relative C_l error"
            )
            print(f"\n### {channel} / {scenario} -- TOP l band, {label}\n")
            print(markdown_table(rows, ["backend"] + [f"ns{n}" for n in nsides]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--channel", choices=["I", "P"], default="I")
    p.add_argument("--nside", type=int, nargs="+", default=None)
    p.add_argument("--backends", nargs="+", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument(
        "--threads",
        type=int,
        default=1,
        help="threads every backend is asked to use; see benchmarks.backends.set_threads",
    )
    p.add_argument("--scenarios", nargs="+", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args(argv)

    nsides = args.nside or (DEFAULT_NSIDE_I if args.channel == "I" else DEFAULT_NSIDE_P)
    scenarios = args.scenarios or (
        list(SCENARIOS) + (["leakage"] if args.channel == "P" else [])
    )
    out = args.out or os.path.join(RESULTS_DIR, f"accuracy_{args.channel}.json")
    run(
        args.channel,
        nsides,
        args.backends,
        scenarios,
        args.seeds,
        out,
        not args.no_resume,
        args.threads,
    )


if __name__ == "__main__":
    main()
