"""Benchmark 1 -- wall time of forward and backward transforms vs nside.

Every backend is given the *same* work: the same map, the same band
``lmax = 2*nside`` (the HP2SPH grid's natural band), single threaded, float64.
Anything else would compare configurations rather than algorithms.

Single threaded is the headline on purpose. ducc0 and healpy both scale across
cores while the HP2SPH pipeline's CG/LSMR latitude solve is largely serial, so a
multi-core comparison measures OpenMP maturity rather than the algorithm. It is
also what the repo's existing profiling assumes (``OMP_NUM_THREADS=1``, forced by
``src/_bootstrap.py`` because several libomp copies in one process segfault).

Read the result against the scaling claim, not for a winner: HP2SPH's advertised
``O(N log^2 N)`` comes entirely from the FastTransforms butterfly algorithm, and
``nm -gU`` on this build's ``libfasttransforms`` shows no butterfly symbols -- it
runs the plain ``O(n^3)` Givens rotations. So HP2SPH is expected to sit in the
same ``O(N^1.5)`` class as healpy here, and the plot's guide slopes let you check
that rather than take it on faith.

Run::

    python -m benchmarks.bench_speed --channel I
    python -m benchmarks.bench_speed --channel P --nside 8 16 32 64
"""

import argparse
import os

from benchmarks import common
from benchmarks.common import (
    RESULTS_DIR,
    env_metadata,
    markdown_table,
    random_EB,
    random_alm,
    time_call,
)
from benchmarks import backends as bk

import healpy as hp

# Chosen so a full default run lands near half an hour on one core. The
# polarization ladder stops earlier because the spin forward's masked LSMR solve
# costs ~4x per nside doubling (measured 0.12 / 0.29 / 1.12 / 4.45 s at nside
# 8 / 16 / 32 / 64), so nside 256 alone would be ~75 s per call.
DEFAULT_NSIDE_I = [8, 16, 32, 64, 128, 256, 512]
DEFAULT_NSIDE_P = [8, 16, 32, 64, 128]

KEY_FIELDS = ("channel", "backend", "nside", "direction")


def _inputs(channel, nside, lmax, seed=20260801):
    """A band-limited test signal plus its map, shared by every backend."""
    mmax = 2 * nside - 1
    if channel == "I":
        alm = random_alm(lmax, seed, slope=1.5, mmax_cap=mmax)
        mp = hp.alm2map(alm, nside=nside, lmax=lmax)
        return {"alm": alm, "map": mp}
    aE, aB = random_EB(lmax, seed, slope=1.5, mmax_cap=mmax)
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, lmax)
    return {"aE": aE, "aB": aB, "Q": Q, "U": U}


def run(channel, nsides, keys, repeats, warmup, out_path, resume, stages):
    chosen = bk.select(keys)
    meta = env_metadata(
        {
            "benchmark": "speed",
            "channel": channel,
            "repeats": repeats,
            "warmup": warmup,
            "lmax_rule": "2*nside",
            "note": "single-threaded; times are seconds, min of `repeats` timed calls",
        }
    )
    store = common.ResultStore(out_path, meta, KEY_FIELDS)

    for nside in nsides:
        lmax = 2 * nside
        data = _inputs(channel, nside, lmax)
        print(f"\n=== {channel} channel, nside={nside} (lmax={lmax}) ===", flush=True)

        for backend in chosen:
            ok, why = backend.available_at(nside, channel)
            if not ok:
                print(f"  {backend.key:<16} skipped: {why}")
                continue

            for direction in ("forward", "backward"):
                if resume and store.has(
                    channel=channel,
                    backend=backend.key,
                    nside=nside,
                    direction=direction,
                ):
                    print(f"  {backend.key:<16} {direction:<8} (cached)")
                    continue

                fn = _make_call(backend, channel, direction, data, nside, lmax)
                try:
                    _, stats = time_call(fn, repeats=repeats, warmup=warmup)
                except Exception as exc:  # a backend failing must not lose the run
                    print(f"  {backend.key:<16} {direction:<8} FAILED: {exc}")
                    continue

                record = {
                    "channel": channel,
                    "backend": backend.key,
                    "nside": nside,
                    "lmax": lmax,
                    "npix": hp.nside2npix(nside),
                    "direction": direction,
                    **stats,
                }
                if getattr(backend, "last_iterations", None) is not None:
                    record["solver_iterations"] = backend.last_iterations
                store.add(record)
                print(
                    f"  {backend.key:<16} {direction:<8} "
                    f"{stats['t_min']:9.4f} s   (median {stats['t_median']:.4f}, "
                    f"peak RSS +{stats['peak_rss_mb']:.0f} MB)"
                )

        if stages and channel == "I":
            _stage_profile(store, data, nside, lmax)

    print(f"\nwrote {out_path}")
    _summary(store, channel)
    return store


def _make_call(backend, channel, direction, data, nside, lmax):
    if channel == "I":
        if direction == "forward":
            return lambda: backend.forward_I(data["map"], nside, lmax)
        return lambda: backend.backward_I(data["alm"], nside, lmax)
    if direction == "forward":
        return lambda: backend.forward_P(data["Q"], data["U"], nside, lmax)
    return lambda: backend.backward_P(data["aE"], data["aB"], nside, lmax)


def _stage_profile(store, data, nside, lmax):
    """Where HP2SPH's forward time actually goes, stage by stage."""
    backend = bk.BACKENDS["hp2sph"]
    times = backend.stage_times_I(data["map"], nside, lmax)
    for stage, t in times.items():
        store.add(
            {
                "channel": "I",
                "backend": f"hp2sph::{stage}",
                "nside": nside,
                "lmax": lmax,
                "npix": hp.nside2npix(nside),
                "direction": "stage",
                "t_min": t,
                "t_median": t,
                "stage": stage,
            }
        )
    total = sum(times.values())
    parts = "  ".join(
        f"{k} {v:.3f}s ({100 * v / total:.0f}%)" for k, v in times.items()
    )
    print(f"  hp2sph stages:   {parts}")


def _summary(store, channel):
    records = [
        r
        for r in store.frame()
        if r["channel"] == channel and r["direction"] in ("forward", "backward")
    ]
    if not records:
        return
    nsides = sorted({r["nside"] for r in records})
    for direction in ("forward", "backward"):
        rows = []
        for key in bk.DEFAULT_KEYS:
            by_nside = {
                r["nside"]: r["t_min"]
                for r in records
                if r["backend"] == key and r["direction"] == direction
            }
            if not by_nside:
                continue
            rows.append([key] + [by_nside.get(n) for n in nsides])
        if rows:
            print(f"\n### {channel} {direction} -- min wall time (s), 1 thread\n")
            print(
                markdown_table(
                    rows, ["backend"] + [f"ns{n}" for n in nsides], floatfmt="{:.3e}"
                )
            )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--channel", choices=["I", "P"], default="I")
    p.add_argument("--nside", type=int, nargs="+", default=None)
    p.add_argument("--backends", nargs="+", default=None)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--out", default=None)
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="recompute cells already present in the output file",
    )
    p.add_argument(
        "--no-stages",
        action="store_true",
        help="skip the HP2SPH per-stage forward profile",
    )
    args = p.parse_args(argv)

    nsides = args.nside or (DEFAULT_NSIDE_I if args.channel == "I" else DEFAULT_NSIDE_P)
    out = args.out or os.path.join(RESULTS_DIR, f"speed_{args.channel}.json")
    run(
        args.channel,
        nsides,
        args.backends,
        args.repeats,
        args.warmup,
        out,
        not args.no_resume,
        not args.no_stages,
    )


if __name__ == "__main__":
    main()
