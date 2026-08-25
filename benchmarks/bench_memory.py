"""Benchmark 4 -- how the HP2SPH forward's peak memory builds up, stage by stage.

Memory is the constraint that decides whether a resolution runs at all, so it is
measured the same way time is rather than being estimated from array shapes: the
shapes miss the transforms' internal workspace, and at nside 1024 that turned out to
be a larger share than the arrays themselves.

**Each measurement runs in a fresh subprocess, and that is the whole point.** Peak RSS
is a process-lifetime high-water mark. Profiling several resolutions in one process
reports the running maximum of the largest one so far, so every later stage reads as
free -- which is not a small distortion, it silently reports 1 MB where the truth is
gigabytes. The same applies to profiling one resolution twice.

What is reported is CUMULATIVE: peak RSS after each stage, above the baseline taken
once the input map exists and before the pipeline starts. A per-stage cost is not a
well-defined quantity here, because a stage's arrays stay live into the next one; the
running high-water mark is what determines whether the run fits.

Run::

    python -m benchmarks.bench_memory --channel P
    python -m benchmarks.bench_memory --channel I --nside 512 1024 2048
"""

import argparse
import json
import os
import subprocess
import sys


from benchmarks import common
from benchmarks.common import RESULTS_DIR, env_metadata, markdown_table

DEFAULT_NSIDE = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
KEY_FIELDS = ("channel", "nside", "stage")


def _worker(channel, nside):
    """Measure one (channel, nside) in this process, which must be a fresh one."""
    import healpy as hp

    from benchmarks import backends as bk
    from benchmarks.common import random_EB, random_alm

    lmax = 2 * nside
    mmax = 2 * nside - 1
    backend = bk.BACKENDS["hp2sph"]
    if channel == "I":
        alm = random_alm(lmax, 20260801, slope=1.5, mmax_cap=mmax)
        mp = hp.alm2map(alm, nside=nside, lmax=lmax)
        del alm
        out = backend.stage_memory_I(mp, nside, lmax)
    else:
        aE, aB = random_EB(lmax, 20260801, slope=1.5, mmax_cap=mmax)
        Q, U = hp.alm2map_spin([aE, aB], nside, 2, lmax)
        del aE, aB
        out = backend.stage_memory_P(Q, U, nside, lmax)
    print("__MEMJSON__" + json.dumps(out))


def _measure(channel, nside, env):
    """Spawn a fresh interpreter for one cell and read its stage totals back."""
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.bench_memory",
        "--worker",
        "--channel",
        channel,
        "--nside",
        str(nside),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=os.getcwd())
    for line in proc.stdout.splitlines():
        if line.startswith("__MEMJSON__"):
            return json.loads(line[len("__MEMJSON__") :])
    raise RuntimeError(
        f"memory worker failed for {channel} nside={nside}:\n{proc.stderr[-2000:]}"
    )


def run(channel, nsides, out_path, resume):
    meta = env_metadata(
        {
            "benchmark": "memory",
            "channel": channel,
            "note": (
                "peak RSS in MB above the post-input baseline, cumulative through the "
                "forward's stages; one fresh subprocess per cell, because peak RSS is "
                "a process-lifetime high-water mark"
            ),
        }
    )
    store = common.ResultStore(out_path, meta, KEY_FIELDS)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")

    for nside in nsides:
        if resume and store.has(channel=channel, nside=nside, stage="nuFFT"):
            print(f"  nside {nside:<5} (cached)")
            continue
        stages = _measure(channel, nside, env)
        for stage, mb in stages.items():
            store.add(
                {
                    "channel": channel,
                    "nside": nside,
                    "stage": stage,
                    "peak_rss_mb": float(mb),
                }
            )
        parts = "  ".join(f"{k} {v:.0f} MB" for k, v in stages.items())
        print(f"  nside {nside:<5} {parts}", flush=True)

    print(f"\nwrote {out_path}")
    _summary(store, channel)
    return store


def _summary(store, channel):
    records = [r for r in store.frame() if r["channel"] == channel]
    if not records:
        return
    nsides = sorted({r["nside"] for r in records})
    stages = []
    for r in records:
        if r["stage"] not in stages:
            stages.append(r["stage"])
    rows = []
    for stage in stages:
        by = {r["nside"]: r["peak_rss_mb"] for r in records if r["stage"] == stage}
        rows.append([stage] + [by.get(n) for n in nsides])
    print(f"\n### {channel} forward -- peak RSS above baseline (MB), cumulative\n")
    print(
        markdown_table(
            rows, ["after stage"] + [f"ns{n}" for n in nsides], floatfmt="{:.0f}"
        )
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--channel", choices=["I", "P"], default="P")
    p.add_argument("--nside", type=int, nargs="+", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument(
        "--worker",
        action="store_true",
        help="internal: measure ONE cell in this process and print it as JSON",
    )
    args = p.parse_args(argv)

    if args.worker:
        _worker(args.channel, args.nside[0])
        return

    nsides = args.nside or DEFAULT_NSIDE
    out = args.out or os.path.join(RESULTS_DIR, f"memory_{args.channel}.json")
    run(args.channel, nsides, out, not args.no_resume)


if __name__ == "__main__":
    main()
