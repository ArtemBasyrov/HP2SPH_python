"""Benchmark 3 -- how faithfully a transform composed with its inverse is the identity.

Two round trips, because they are not the same test:

``harmonic`` (primary): ``alm -> map -> alm``
    Unambiguous. The starting coefficients are inside the analysis band, so the
    identity really is the right answer and any deviation is the composed
    operator's error. Reported per multipole.

``pixel`` (secondary): ``map -> alm -> map``
    Only meaningful on a BAND-LIMITED map, which is what this runs. On a general
    map the composition is a projection, not the identity, so the "error" would
    be dominated by the above-band content every method discards -- a measure of
    aliasing, not of the round trip.

Both run at ``lmax = 2*nside - 1`` rather than the full ``2*nside``. At exactly
``2*nside`` the single ``l = m = lmax`` coefficient cannot be represented:
``m = +2*nside`` and ``m = -2*nside`` are the same mode on a ``4*nside``-point
longitude grid, and the per-ring ``phi0`` offsets give them different phases, so
no column can carry both. That corner is a property of the HEALPix grid, not of
any implementation, and letting one coefficient dominate every curve would hide
what the rest of the band is doing. ``--include-nyquist-corner`` measures it
explicitly instead.

HP2SPH's round trip is expected to be limited by its FORWARD alone: the native
backward reproduces ``hp.alm2map`` / ``hp.alm2map_spin`` to ~3e-13 for every band
``lmax <= 2*nside - 1``. The square-band HP2SPH variant is the one configuration
that is a bit-exact interpolation, so it is included at low nside as a check that
the exactness claim still holds.

Run::

    python -m benchmarks.bench_roundtrip --channel I
    python -m benchmarks.bench_roundtrip --channel P --nside 8 16 32
"""

import argparse
import os

import numpy as np
import healpy as hp

from benchmarks import backends as bk
from benchmarks import common
from benchmarks.common import (
    RESULTS_DIR,
    band_rms,
    ell_bands,
    env_metadata,
    markdown_table,
    per_ell_alm_error,
    per_ell_cl_error,
    random_EB,
    random_alm,
    rel_l2,
    rel_max,
)

DEFAULT_NSIDE_I = [8, 16, 32, 64, 128, 256]
DEFAULT_NSIDE_P = [8, 16, 32, 64]
DEFAULT_SEEDS = [0, 1, 2]

KEY_FIELDS = ("channel", "backend", "nside", "mode", "field")


def _signal(channel, nside, lmax, seed):
    """A band-limited signal: coefficients, and the map they synthesise to."""
    mmax = min(lmax, 2 * nside - 1)
    if channel == "I":
        alm = random_alm(lmax, seed, slope=1.5, mmax_cap=mmax)
        return {"T": alm}, hp.alm2map(alm, nside=nside, lmax=lmax)
    aE, aB = random_EB(lmax, seed, slope=1.5, mmax_cap=mmax)
    Q, U = hp.alm2map_spin([aE, aB], nside, 2, lmax)
    return {"E": aE, "B": aB}, (Q, U)


def _harmonic_roundtrip(backend, channel, alms, nside, lmax):
    """alm -> map -> alm, entirely through one backend."""
    if channel == "I":
        mp = backend.backward_I(alms["T"], nside, lmax)
        return {"T": backend.forward_I(mp, nside, lmax)}
    Q, U = backend.backward_P(alms["E"], alms["B"], nside, lmax)
    aE, aB = backend.forward_P(Q, U, nside, lmax)
    return {"E": aE, "B": aB}


def _pixel_roundtrip(backend, channel, maps, nside, lmax):
    """map -> alm -> map, entirely through one backend."""
    if channel == "I":
        alm = backend.forward_I(maps, nside, lmax)
        return backend.backward_I(alm, nside, lmax)
    Q, U = maps
    aE, aB = backend.forward_P(Q, U, nside, lmax)
    return backend.backward_P(aE, aB, nside, lmax)


def run(channel, nsides, keys, seeds, modes, out_path, resume, nyquist_corner):
    chosen = bk.select(keys)
    lmin = 2 if channel == "P" else 0
    fields = ["T"] if channel == "I" else ["E", "B"]
    meta = env_metadata(
        {
            "benchmark": "roundtrip",
            "channel": channel,
            "seeds": seeds,
            "lmax_rule": "2*nside - 1 (avoids the l=m=2*nside grid-Nyquist corner)",
            "note": "per-l curves are the median across seeds",
        }
    )
    store = common.ResultStore(out_path, meta, KEY_FIELDS)

    for nside in nsides:
        lmax = 2 * nside - 1
        bands = ell_bands(lmax, lmin=lmin)
        print(f"\n=== {channel} nside={nside} lmax={lmax} ===", flush=True)

        for backend in chosen:
            ok, why = backend.available_at(nside, channel)
            if not ok:
                print(f"  {backend.key:<16} skipped: {why}")
                continue

            if "harmonic" in modes and not (
                resume
                and all(
                    store.has(
                        channel=channel,
                        backend=backend.key,
                        nside=nside,
                        mode="harmonic",
                        field=f,
                    )
                    for f in fields
                )
            ):
                _do_harmonic(
                    store, backend, channel, nside, lmax, seeds, bands, fields, lmin
                )

            if "pixel" in modes and not (
                resume
                and store.has(
                    channel=channel,
                    backend=backend.key,
                    nside=nside,
                    mode="pixel",
                    field="map",
                )
            ):
                _do_pixel(store, backend, channel, nside, lmax, seeds)

            if (
                "native" in modes
                and channel == "I"
                and hasattr(backend, "native_roundtrip_I")
                and not (
                    resume
                    and store.has(
                        channel=channel,
                        backend=backend.key,
                        nside=nside,
                        mode="native",
                        field="map",
                    )
                )
            ):
                _do_native(store, backend, nside, lmax, seeds)

        if nyquist_corner:
            _do_nyquist_corner(store, chosen, channel, nside)

    print(f"\nwrote {out_path}")
    _summary(store, channel, modes)
    return store


def _do_harmonic(store, backend, channel, nside, lmax, seeds, bands, fields, lmin):
    acc = {f: {"cl": [], "alm": [], "l2": []} for f in fields}
    for seed in seeds:
        alms, _ = _signal(channel, nside, lmax, seed)
        try:
            rec = _harmonic_roundtrip(backend, channel, alms, nside, lmax)
        except Exception as exc:
            print(f"  {backend.key:<16} harmonic FAILED: {exc}")
            return
        for f in fields:
            acc[f]["cl"].append(per_ell_cl_error(rec[f], alms[f], lmax))
            acc[f]["alm"].append(per_ell_alm_error(rec[f], alms[f], lmax))
            acc[f]["l2"].append(rel_l2(rec[f], alms[f]))

    shown = []
    for f in fields:
        cl = np.nanmedian(np.vstack(acc[f]["cl"]), axis=0)
        alm = np.nanmedian(np.vstack(acc[f]["alm"]), axis=0)
        store.add(
            {
                "channel": channel,
                "backend": backend.key,
                "nside": nside,
                "lmax": lmax,
                "mode": "harmonic",
                "field": f,
                "lmin": lmin,
                "cl_err_per_ell": cl,
                "alm_err_per_ell": alm,
                "global_l2": float(np.median(acc[f]["l2"])),
                "bands": {
                    lbl: {
                        "lo": lo,
                        "hi": hi,
                        "cl_rms": band_rms(cl, lo, hi),
                        "alm_rms": band_rms(alm, lo, hi),
                    }
                    for lbl, lo, hi in bands
                },
            }
        )
        shown.append(f"{f} L2={np.median(acc[f]['l2']):.2e}")
    print(f"  {backend.key:<16} harmonic:  " + "   ".join(shown))


def _do_pixel(store, backend, channel, nside, lmax, seeds):
    l2, mx = [], []
    for seed in seeds:
        _, maps = _signal(channel, nside, lmax, seed)
        try:
            out = _pixel_roundtrip(backend, channel, maps, nside, lmax)
        except Exception as exc:
            print(f"  {backend.key:<16} pixel    FAILED: {exc}")
            return
        if channel == "I":
            l2.append(rel_l2(out, maps))
            mx.append(rel_max(out, maps))
        else:
            # One figure for the polarization pair: Q and U are two components of
            # the same field, so their errors are combined rather than averaged.
            ref = np.concatenate(maps)
            got = np.concatenate(out)
            l2.append(rel_l2(got, ref))
            mx.append(rel_max(got, ref))
    store.add(
        {
            "channel": channel,
            "backend": backend.key,
            "nside": nside,
            "lmax": lmax,
            "mode": "pixel",
            "field": "map",
            "map_rel_l2": float(np.median(l2)),
            "map_rel_max": float(np.median(mx)),
        }
    )
    print(
        f"  {backend.key:<16} pixel:     L2={np.median(l2):.2e}  "
        f"max={np.median(mx):.2e}"
    )


def _do_native(store, backend, nside, lmax, seeds):
    """HP2SPH-only: the round trip in the pipeline's own coefficient representation."""
    l2, mx = [], []
    for seed in seeds:
        _, mp = _signal("I", nside, lmax, seed)
        try:
            out = backend.native_roundtrip_I(mp, nside, lmax)
        except Exception as exc:
            print(f"  {backend.key:<16} native   FAILED: {exc}")
            return
        l2.append(rel_l2(out, mp))
        mx.append(rel_max(out, mp))
    store.add(
        {
            "channel": "I",
            "backend": backend.key,
            "nside": nside,
            "lmax": lmax,
            "mode": "native",
            "field": "map",
            "map_rel_l2": float(np.median(l2)),
            "map_rel_max": float(np.median(mx)),
        }
    )
    print(
        f"  {backend.key:<16} native:    L2={np.median(l2):.2e}  "
        f"max={np.median(mx):.2e}   (map -> C -> map, no alm conversion)"
    )


def _do_nyquist_corner(store, chosen, channel, nside):
    """Measure the one coefficient the grid cannot carry, in isolation.

    A unit ``a_{lmax,lmax}`` at ``lmax = 2*nside`` round-tripped on its own. It
    is excluded from every other curve here, so this records what excluding it
    costs instead of leaving the exclusion unquantified.
    """
    lmax = 2 * nside
    n = hp.Alm.getsize(lmax)
    i = hp.Alm.getidx(lmax, lmax, lmax)
    print(f"  -- l=m={lmax} grid-Nyquist corner --")
    for backend in chosen:
        if not backend.available_at(nside, channel)[0]:
            continue
        try:
            if channel == "I":
                alm = np.zeros(n, dtype=np.complex128)
                alm[i] = 1.0
                mp = backend.backward_I(alm, nside, lmax)
                rec = backend.forward_I(mp, nside, lmax)[i]
            else:
                aE = np.zeros(n, dtype=np.complex128)
                aB = np.zeros(n, dtype=np.complex128)
                aE[i] = 1.0
                Q, U = backend.backward_P(aE, aB, nside, lmax)
                rec = backend.forward_P(Q, U, nside, lmax)[0][i]
        except Exception as exc:
            print(f"     {backend.key:<16} n/a ({type(exc).__name__}: {exc})")
            continue
        store.add(
            {
                "channel": channel,
                "backend": backend.key,
                "nside": nside,
                "lmax": lmax,
                "mode": "nyquist_corner",
                "field": "T" if channel == "I" else "E",
                "gain_abs": float(abs(rec)),
                "rel_err": float(abs(rec - 1.0)),
            }
        )
        print(f"     {backend.key:<16} gain |a|={abs(rec):.4f}  (1.0 is exact)")


def _summary(store, channel, modes):
    records = store.frame()
    if "harmonic" in modes:
        field = "T" if channel == "I" else "E"
        sel = [
            r for r in records if r["mode"] == "harmonic" and r.get("field") == field
        ]
        if sel:
            nsides = sorted({r["nside"] for r in sel})
            rows = []
            for key in bk.DEFAULT_KEYS:
                by = {r["nside"]: r["global_l2"] for r in sel if r["backend"] == key}
                if by:
                    rows.append([key] + [by.get(n) for n in nsides])
            print(
                f"\n### {channel} harmonic round trip -- global relative "
                f"L2 coefficient error ({field})\n"
            )
            print(markdown_table(rows, ["backend"] + [f"ns{n}" for n in nsides]))

    for mode, title in (
        ("pixel", "pixel round trip -- relative map L2 error"),
        (
            "native",
            "HP2SPH native round trip (map -> C -> map, no alm conversion)"
            " -- relative map L2 error",
        ),
    ):
        if mode not in modes:
            continue
        sel = [r for r in records if r["mode"] == mode]
        if sel:
            nsides = sorted({r["nside"] for r in sel})
            rows = []
            for key in bk.DEFAULT_KEYS:
                by = {r["nside"]: r["map_rel_l2"] for r in sel if r["backend"] == key}
                if by:
                    rows.append([key] + [by.get(n) for n in nsides])
            print(f"\n### {channel} {title}\n")
            print(markdown_table(rows, ["backend"] + [f"ns{n}" for n in nsides]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--channel", choices=["I", "P"], default="I")
    p.add_argument("--nside", type=int, nargs="+", default=None)
    p.add_argument("--backends", nargs="+", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument(
        "--modes",
        nargs="+",
        choices=["harmonic", "pixel", "native"],
        default=["harmonic", "pixel", "native"],
    )
    p.add_argument("--out", default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument(
        "--include-nyquist-corner",
        action="store_true",
        help="also probe the l=m=2*nside coefficient the grid cannot represent",
    )
    args = p.parse_args(argv)

    nsides = args.nside or (DEFAULT_NSIDE_I if args.channel == "I" else DEFAULT_NSIDE_P)
    out = args.out or os.path.join(RESULTS_DIR, f"roundtrip_{args.channel}.json")
    run(
        args.channel,
        nsides,
        args.backends,
        args.seeds,
        args.modes,
        out,
        not args.no_resume,
        args.include_nyquist_corner,
    )


if __name__ == "__main__":
    main()
