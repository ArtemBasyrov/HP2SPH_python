"""Recompute every benchmark number the docs quote, and diff two result sets.

Run from the repo root::

    python -m benchmarks.compare_results [OLD_DIR] [NEW_DIR]

Defaults to ``benchmarks/results_s2fft`` (archived) against ``benchmarks/results``
(current). Every metric here corresponds to a specific claim made in the project
READMEs, so a re-baseline can be checked against the docs in one command rather
than by hand.

Why this exists: the recorded benchmark numbers went stale when the project
environment changed, and nobody noticed because there was no cheap way to
re-derive them. Two traps this guards against:

* ``--no-resume`` MERGES rather than replaces -- it recomputes only the cells in
  the current ``--nside`` ladder and leaves every other row untouched. A high-nside
  run without it silently skips cells that already exist, so stale rows survive a
  "full" re-run. The staleness check below catches that by testing whether the
  per-ell arrays are bit-identical to the archive.
* Speed rows are wall clock and are NOT comparable across days. Accuracy rows are
  deterministic and are. Treat a speed ratio here as a hint, then confirm it with
  alternating A/B runs in one sitting.
"""

import json
import os
import sys

import numpy as np

OLD = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results_s2fft"
NEW = sys.argv[2] if len(sys.argv) > 2 else "benchmarks/results"


def load(root, name):
    p = os.path.join(root, f"{name}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def recs(root, name, **filt):
    d = load(root, name)
    if d is None:
        return []
    out = []
    for r in d["records"]:
        if all(r.get(k) == v for k, v in filt.items()):
            out.append(r)
    return out


def top_band(r):
    """cl_rms of the top ell quarter."""
    key = sorted(r["bands"], key=lambda k: int(k.split("-")[0]))[-1]
    return r["bands"][key]["cl_rms"]


def band_rms(r, lo_frac, hi_frac):
    """RMS relative C_l error over ell in [lo_frac*lmax, hi_frac*lmax]."""
    lmax = r["lmax"]
    lmin = r["lmin"]
    e = np.asarray(r["cl_err_per_ell"], dtype=float)
    ell = np.arange(lmin, lmin + len(e))
    sel = (ell >= lo_frac * lmax) & (ell <= hi_frac * lmax)
    return float(np.sqrt(np.mean(e[sel] ** 2))) if sel.any() else float("nan")


def fmt(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3e}"


def cmp_line(label, old, new):
    if old is None and new is None:
        return
    ratio = ""
    if old and new and old > 0:
        ratio = f"  ({new / old:.2f}x old)"
    print(f"  {label:34s} old {fmt(old):>10s}   new {fmt(new):>10s}{ratio}")


def get(root, name, backend, **filt):
    r = recs(root, name, backend=backend, **filt)
    return r[0] if r else None


# --------------------------------------------------------------------------- #
print("=" * 78)
print("A. INTENSITY, cosmology, RMS rel C_l over the TOP ELL QUARTER")
print("=" * 78)
BK = [
    "healpy-iter3",
    "ducc0-pseudo",
    "healpy-pixel",
    "hp2sph",
    "healpy-ring",
    "ducc0-adjoint",
    "healpy-plain",
]
for ns in (256, 512, 1024):
    print(f"\n-- nside {ns} --")
    for b in BK:
        o = get(OLD, "accuracy_I", b, nside=ns, scenario="cosmology", field="T")
        n = get(NEW, "accuracy_I", b, nside=ns, scenario="cosmology", field="T")
        cmp_line(b, top_band(o) if o else None, top_band(n) if n else None)

print()
print("=" * 78)
print("B. RING-WEIGHTS / HP2SPH ratio, cosmology, ell in [3*lmax/4, 7*lmax/8]")
print("=" * 78)
print(
    f"  {'nside':>6s}  {'old ratio':>10s}  {'new ratio':>10s}   "
    f"{'new hp2sph':>11s}  {'new ring':>11s}"
)
for ns in (32, 64, 128, 256, 512, 1024):
    row = [f"  {ns:6d}"]
    vals = {}
    for root, tag in ((OLD, "old"), (NEW, "new")):
        h = get(root, "accuracy_I", "hp2sph", nside=ns, scenario="cosmology", field="T")
        g = get(
            root, "accuracy_I", "healpy-ring", nside=ns, scenario="cosmology", field="T"
        )
        if h and g:
            hv, gv = band_rms(h, 0.75, 0.875), band_rms(g, 0.75, 0.875)
            vals[tag] = (gv / hv, hv, gv)
        else:
            vals[tag] = None
    for tag in ("old", "new"):
        row.append(f"  {vals[tag][0]:10.2f}" if vals[tag] else f"  {'--':>10s}")
    row.append(f"   {fmt(vals['new'][1]) if vals['new'] else '--':>11s}")
    row.append(f"  {fmt(vals['new'][2]) if vals['new'] else '--':>11s}")
    print("".join(row))

print()
print("=" * 78)
print("C. POLARIZATION, cosmology, field E, top-quarter band + forward t_min")
print("=" * 78)
for ns in (8, 16, 32, 64):
    print(f"\n-- nside {ns} --")
    for b in [
        "hp2sph",
        "healpy-pixel",
        "healpy-ring",
        "ducc0-adjoint",
        "healpy-iter3",
        "ducc0-pseudo",
    ]:
        o = get(OLD, "accuracy_P", b, nside=ns, scenario="cosmology", field="E")
        n = get(NEW, "accuracy_P", b, nside=ns, scenario="cosmology", field="E")
        cmp_line(b, top_band(o) if o else None, top_band(n) if n else None)
    o = get(OLD, "speed_P", "hp2sph", nside=ns, direction="forward")
    n = get(NEW, "speed_P", "hp2sph", nside=ns, direction="forward")
    cmp_line("t forward (s)", o["t_min"] if o else None, n["t_min"] if n else None)

print()
print("=" * 78)
print("D. E->B LEAKAGE, scenario=leakage, top-quarter band, nside 64")
print("=" * 78)


def leak_top(r):
    key = sorted(r["bands"], key=lambda k: int(k.split("-")[0]))[-1]
    bd = r["bands"][key]
    return bd.get("leak_median", bd.get("leak_rms", bd.get("cl_rms")))


for b in [
    "hp2sph",
    "healpy-ring",
    "healpy-pixel",
    "ducc0-adjoint",
    "healpy-iter3",
    "ducc0-pseudo",
]:
    for fld in ("B_from_E", "E_from_B"):
        o = get(OLD, "accuracy_P", b, nside=64, scenario="leakage", field=fld)
        n = get(NEW, "accuracy_P", b, nside=64, scenario="leakage", field=fld)
        if o or n:
            cmp_line(
                f"{b} [{fld}]", leak_top(o) if o else None, leak_top(n) if n else None
            )

print()
print("=" * 78)
print("E. SPEED: forward t_min at nside 512, and fitted t ~ N^p over ns 64-512")
print("=" * 78)
for b in [
    "hp2sph",
    "ducc0-adjoint",
    "healpy-plain",
    "healpy-ring",
    "healpy-iter3",
    "ducc0-pseudo",
]:
    o = get(OLD, "speed_I", b, nside=512, direction="forward")
    n = get(NEW, "speed_I", b, nside=512, direction="forward")
    cmp_line(f"{b} ns512", o["t_min"] if o else None, n["t_min"] if n else None)
print()
for b in [
    "hp2sph",
    "healpy-ring",
    "ducc0-pseudo",
    "ducc0-adjoint",
    "healpy-iter3",
    "healpy-plain",
]:
    out = []
    for root in (OLD, NEW):
        pts = [
            (r["npix"], r["t_min"])
            for r in recs(root, "speed_I", backend=b, direction="forward")
            if 64 <= r["nside"] <= 512
        ]
        if len(pts) >= 3:
            x = np.log([p[0] for p in pts])
            y = np.log([p[1] for p in pts])
            out.append(np.polyfit(x, y, 1)[0])
        else:
            out.append(float("nan"))
    print(f"  {b:20s} p_old {out[0]:5.2f}   p_new {out[1]:5.2f}")

print()
print("  stage split at nside 512 (forward, new env):")
tot = get(NEW, "speed_I", "hp2sph", nside=512, direction="forward")
if tot:
    for st in [
        "hp2sph::data_interpolation",
        "hp2sph::DFS",
        "hp2sph::nuFFT",
        "hp2sph::FSHT",
    ]:
        s = get(NEW, "speed_I", st, nside=512, direction="stage")
        if s:
            print(
                f"    {st.split('::')[1]:22s} {s['t_min']:.4f} s "
                f"({100 * s['t_min'] / tot['t_min']:4.1f}%)"
            )

print()
print("=" * 78)
print("F. ROUND TRIP")
print("=" * 78)
for ch, fld in (("I", "T"), ("P", "E")):
    for ns in (8, 16, 32, 64):
        o = get(OLD, f"roundtrip_{ch}", "hp2sph", nside=ns, mode="harmonic", field=fld)
        n = get(NEW, f"roundtrip_{ch}", "hp2sph", nside=ns, mode="harmonic", field=fld)
        if o or n:
            cmp_line(
                f"{ch} harmonic {fld} ns{ns}",
                o["global_l2"] if o else None,
                n["global_l2"] if n else None,
            )

print()
print("  nyquist_corner gain (should be 0.5 for hp2sph, 1.0 for the rest):")
for root, tag in ((OLD, "old"), (NEW, "new")):
    for b in ("hp2sph", "healpy-plain"):
        vals = {
            r["nside"]: r.get("gain_abs")
            for r in recs(root, "roundtrip_I", backend=b, mode="nyquist_corner")
            if r.get("gain_abs") is not None
        }
        if vals:
            print(
                f"    {tag} {b:14s} "
                + " ".join(f"ns{k}={v:.4f}" for k, v in sorted(vals.items()))
            )

print()
print("=" * 78)
print("G. HARNESS SELF-CHECK: healpy-plain vs ducc0-adjoint (cosmology, ns256)")
print("=" * 78)
for root, tag in ((OLD, "old"), (NEW, "new")):
    a = get(
        root, "accuracy_I", "healpy-plain", nside=256, scenario="cosmology", field="T"
    )
    b = get(
        root, "accuracy_I", "ducc0-adjoint", nside=256, scenario="cosmology", field="T"
    )
    if a and b:
        d = np.abs(np.asarray(a["cl_err_per_ell"]) - np.asarray(b["cl_err_per_ell"]))
        print(f"  {tag}: max |per-ell difference| = {d.max():.2e}")

print()
print("=" * 78)
print("H. STALENESS CHECK: when was each cell actually computed?")
print("=" * 78)
# Bit-identical numbers do NOT prove a cell was skipped: a deterministic backend
# reproduces exactly, and that is correct. The only reliable signal is the
# per-record ``_computed_utc`` stamp that ResultStore.add writes. Records written
# before that stamp existed report as "unstamped" rather than being guessed at.
for name in (
    "accuracy_I",
    "accuracy_P",
    "speed_I",
    "speed_P",
    "roundtrip_I",
    "roundtrip_P",
):
    rows = recs(NEW, name)
    if not rows:
        continue
    days = {}
    unstamped = 0
    for r in rows:
        ts = r.get("_computed_utc")
        if ts is None:
            unstamped += 1
        else:
            days[ts[:10]] = days.get(ts[:10], 0) + 1
    parts = [f"{d}: {n}" for d, n in sorted(days.items())]
    if unstamped:
        parts.append(f"unstamped: {unstamped}")
    flag = ""
    if len(days) > 1:
        flag = "   <-- MIXED DATES, some cells are older than this run"
    print(f"  {name:14s} {'  '.join(parts)}{flag}")
print()
print("  A cell keeps its old stamp when a re-run SKIPS it. --no-resume merges")
print("  rather than replaces, so it only refreshes the cells in the current")
print("  --nside ladder; anything outside that ladder survives untouched.")

print()
d = load(NEW, "accuracy_I")
if d:
    m = d["meta"]
    print(
        "NEW env:",
        m["created_utc"],
        "| omp",
        m["omp_num_threads"],
        "|",
        {
            k: v
            for k, v in m["versions"].items()
            if k in ("healpy", "finufft", "numpy", "ducc0")
        },
    )
