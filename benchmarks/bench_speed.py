"""Benchmark 1 -- wall time of forward and backward transforms vs nside.

Every backend is given the *same* work: the same map and the same band
``lmax = 2*nside`` (the HP2SPH grid's natural band), float64.

``--threads N`` asks every backend for N threads, and the run metadata records what
each could actually be told, because they do not all honour it the same way:

* ducc0 takes an explicit ``nthreads`` argument, so it is exact.
* HP2SPH splits its NUFFT batch over N Python threads and threads its FastTransforms
  stage through OpenMP.
* healpy 1.20 exposes NO thread argument; its bundled libsharp reads
  ``OMP_NUM_THREADS``, which ``hp2sph/_bootstrap`` pins to 1. To let healpy thread, set
  ``HP2SPH_OMP_THREADS`` in the ENVIRONMENT before the run -- measured at nside 256,
  that takes ``map2alm`` with ring weights from 19.6 ms to 4.6 ms, and it does not
  slow HP2SPH down (0.92 s to 0.83 s on the spin forward, since its NUFFT plans ask
  for one thread each regardless and only the FSHT stage gains).

**A run without ``HP2SPH_OMP_THREADS`` set therefore under-reports healpy.** Say
which configuration a number came from whenever one is quoted.

Read the result against the scaling claim, not for a winner: HP2SPH's advertised
``O(N log^2 N)`` comes entirely from the FastTransforms butterfly algorithm, and
``nm -gU`` on this build's ``libfasttransforms`` shows no butterfly symbols -- it
runs the plain ``O(n^3)` Givens rotations. So HP2SPH is expected to sit in the
same ``O(N^1.5)`` class as healpy here, and the plot's guide slopes let you check
that rather than take it on faith.

Run::

    python -m benchmarks.bench_speed --channel I
    python -m benchmarks.bench_speed --channel P --nside 8 16 32 64

    # threaded, the configuration the shipped figures use
    HP2SPH_OMP_THREADS=8 python -m benchmarks.bench_speed --channel P --threads 8
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

# Both ladders now reach nside 2048, the project's gating resolution. The polarization
# ladder used to stop at 128 because the spin forward cost ~20 s per call at nside 256;
# it is now ~1 s there and ~100 s at 2048, so the ladders match. At 1024 and 2048 the
# cost is dominated by a handful of very long calls, so those two are worth running with
# ``--repeats 1`` in a separate invocation -- the store merges on
# (channel, backend, nside, direction) and each record keeps its own ``repeats``.
DEFAULT_NSIDE_I = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
DEFAULT_NSIDE_P = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]

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


def run(channel, nsides, keys, repeats, warmup, out_path, resume, stages, threads=1):
    chosen = bk.select(keys)
    applied = bk.set_threads(threads)
    meta = env_metadata(
        {
            "benchmark": "speed",
            "channel": channel,
            "repeats": repeats,
            "warmup": warmup,
            "lmax_rule": "2*nside",
            "threads": applied,
            "note": (
                "times are seconds, min of `repeats` timed calls. `threads` records "
                "what each backend could actually be told to use -- they are not all "
                "the same number, see backends.set_threads"
            ),
        }
    )
    store = common.ResultStore(out_path, meta, KEY_FIELDS)

    for nside in nsides:
        lmax = 2 * nside
        data = _inputs(channel, nside, lmax)
        print(f"\n=== {channel} channel, nside={nside} (lmax={lmax}) ===", flush=True)

        for backend in chosen:
            ok, why = backend.available_at(nside, channel, lmax)
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

        if stages:
            _stage_profile(store, data, nside, lmax, channel)

    print(f"\nwrote {out_path}")
    _summary(store, channel, threads)
    return store


def _make_call(backend, channel, direction, data, nside, lmax):
    if channel == "I":
        if direction == "forward":
            return lambda: backend.forward_I(data["map"], nside, lmax)
        return lambda: backend.backward_I(data["alm"], nside, lmax)
    if direction == "forward":
        return lambda: backend.forward_P(data["Q"], data["U"], nside, lmax)
    return lambda: backend.backward_P(data["aE"], data["aB"], nside, lmax)


def _stage_profile(store, data, nside, lmax, channel):
    """Where HP2SPH's forward time actually goes, stage by stage.

    Time only. The stage MEMORY profile lives in ``bench_memory`` because peak RSS is a
    process-lifetime high-water mark: measured here, every nside after the first would
    report near zero, since the process has already peaked higher. That needs a fresh
    subprocess per cell, which does not belong inside a timing loop.
    """
    backend = bk.BACKENDS["hp2sph"]
    if channel == "I":
        times = backend.stage_times_I(data["map"], nside, lmax)
    else:
        times = backend.stage_times_P(data["Q"], data["U"], nside, lmax)
    for stage, t in times.items():
        store.add(
            {
                "channel": channel,
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


def _summary(store, channel, threads=1):
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
            label = "1 thread" if threads == 1 else f"{threads} threads requested"
            print(f"\n### {channel} {direction} -- min wall time (s), {label}\n")
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
    p.add_argument(
        "--threads",
        type=int,
        default=1,
        help=(
            "threads every backend is asked to use. ducc0 honours it exactly and "
            "HP2SPH splits its NUFFT batch over it; healpy 1.20 has no thread "
            "argument and needs HP2SPH_OMP_THREADS in the environment instead"
        ),
    )
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
        args.threads,
    )


if __name__ == "__main__":
    main()
