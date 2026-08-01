"""Figures for the benchmark suite.

Reads only the JSON written by the benchmark scripts, so plots can be restyled
or regenerated without re-running any transforms.

Encoding: **colour is the library, linestyle and marker are the configuration**.
Eight distinct hues would not survive a colour-vision check; three do with room
to spare (worst all-pairs CVD dE 9.2, normal-vision 24.0 against the light
surface). Every chart carries a legend, and each benchmark also prints and stores
the same numbers as a table -- which is the required relief for the one palette
slot that sits below 3:1 contrast.

Run::

    python -m benchmarks.plots            # everything it can find
    python -m benchmarks.plots --only speed
"""

import argparse
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from benchmarks import backends as bk  # noqa: E402
from benchmarks.bench_accuracy import SCENARIOS  # noqa: E402
from benchmarks.common import FIGURES_DIR, RESULTS_DIR, load_results  # noqa: E402

# Chart chrome (light surface). Recessive grid and axes, ink for all text.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Ordinal ramp for the pipeline stages: one hue, light->dark, ordered because the
# stages have a natural order. Starts at step 250, the lightest that still clears
# 2:1 on this surface.
STAGE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]

LINE_KW = dict(linewidth=1.8, markersize=5.0, markeredgewidth=0)


def _fig(nrows=1, ncols=1, width=5.0, height=3.9, **kw):
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(width * ncols, height * nrows), squeeze=False, **kw
    )
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.ravel():
        _style(ax)
    return fig, axes


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.7, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)


def _plot_series(ax, key, x, y, **kw):
    color, ls, marker = bk.style_for(key)
    opts = {
        "color": color,
        "linestyle": ls,
        "marker": marker,
        "label": bk.label_for(key),
        **LINE_KW,
        **kw,  # callers may override, e.g. marker="" on dense per-l curves
    }
    ax.plot(
        x,
        y,
        **opts,
    )


def _legend(fig, axes, ncol=4):
    handles, labels = [], []
    for ax in np.ravel(axes):
        for h, lb in zip(*ax.get_legend_handles_labels()):
            if lb not in labels:
                handles.append(h)
                labels.append(lb)
    if not handles:
        return
    leg = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 0.0),
    )
    for text in leg.get_texts():
        text.set_color(INK_2)


def _title(ax, text, subtitle=None):
    # Drawn as free text rather than set_title: a subtitle needs its own, smaller
    # type, and set_title plus a second text object collide.
    ax.text(
        0.0,
        1.13 if subtitle else 1.03,
        text,
        transform=ax.transAxes,
        color=INK,
        fontsize=10.5,
        ha="left",
        va="bottom",
    )
    if subtitle:
        ax.text(
            0.0,
            1.03,
            subtitle,
            transform=ax.transAxes,
            color=MUTED,
            fontsize=8,
            ha="left",
            va="bottom",
        )


def _save(fig, name, legend_rows=2):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.tight_layout(rect=(0, 0.04 * legend_rows, 1, 1))
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")


def _by(records, **filters):
    return [r for r in records if all(r.get(k) == v for k, v in filters.items())]


def _ordered_keys(records):
    present = {r["backend"] for r in records}
    return [k for k in bk.DEFAULT_KEYS if k in present]


# --------------------------------------------------------------------------- #
# Benchmark 1 -- speed                                                         #
# --------------------------------------------------------------------------- #
def plot_speed(channel):
    path = os.path.join(RESULTS_DIR, f"speed_{channel}.json")
    if not os.path.exists(path):
        return
    _, records = load_results(path)
    timed = [r for r in records if r["direction"] in ("forward", "backward")]
    if not timed:
        return

    fig, axes = _fig(1, 2, width=5.4, height=4.1)
    for ax, direction in zip(axes[0], ("forward", "backward")):
        sel = _by(timed, direction=direction)
        for key in _ordered_keys(sel):
            pts = sorted(((r["npix"], r["t_min"]) for r in sel if r["backend"] == key))
            if pts:
                _plot_series(ax, key, [p[0] for p in pts], [p[1] for p in pts])
        _guide_slopes(ax, sel)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$N$ (pixels)", color=INK_2, fontsize=9)
        ax.set_ylabel("wall time (s)", color=INK_2, fontsize=9)
        _title(
            ax,
            f"{'Intensity' if channel == 'I' else 'Polarization'} {direction}",
            "1 thread, float64, lmax = 2·nside; min of the timed repeats",
        )
    _legend(fig, axes)
    _save(fig, f"speed_{channel}.png")

    _plot_stage_profile(records, channel)


def _guide_slopes(ax, sel):
    """Reference slopes so the measured scaling can be checked, not assumed."""
    npix = sorted({r["npix"] for r in sel})
    if len(npix) < 2:
        return
    ref = [r for r in sel if r["backend"] == "hp2sph"]
    if not ref:
        ref = sel
    # Anchor at the LARGEST size, not the smallest: these lines exist to be
    # compared by slope, and anchoring at the small end throws them orders of
    # magnitude above every measured point by the right-hand edge.
    anchor = max(ref, key=lambda r: r["npix"])
    n0, t0 = anchor["npix"], anchor["t_min"]
    # Only the upper half of the x range: extended to the small end these lines
    # drag the y-limits down by decades and squash every measured curve.
    x = np.array(npix[len(npix) // 2 :], dtype=float)
    for expo, label in ((1.5, r"$O(N^{1.5})$"), (1.0, r"$O(N\log^2 N)$")):
        if expo == 1.0:
            y = t0 * (x / n0) * (np.log2(x) / np.log2(n0)) ** 2
        else:
            y = t0 * (x / n0) ** expo
        ax.plot(x, y, color=MUTED, linewidth=0.9, linestyle="--", zorder=0)
        ax.annotate(
            label,
            xy=(x[0], y[0]),
            xytext=(3, 3),
            textcoords="offset points",
            color=MUTED,
            fontsize=7.5,
            ha="left",
            va="bottom",
        )


def _plot_stage_profile(records, channel):
    stages = [r for r in records if r["direction"] == "stage"]
    if not stages:
        return
    nsides = sorted({r["nside"] for r in stages})
    names = []
    for r in stages:
        if r["stage"] not in names:
            names.append(r["stage"])

    # Share of total, on a LINEAR axis. Stacking on a log axis is a lie: segment
    # heights there are not proportional to their values and do not sum to the
    # total, so the eye reads the split wrong. The absolute seconds are in the
    # JSON and in the printed table; the question this chart answers is where the
    # time goes, which is a share.
    totals = np.array(
        [sum(r["t_min"] for r in stages if r["nside"] == n) for n in nsides]
    )
    fig, axes = _fig(1, 1, width=6.6, height=4.0)
    ax = axes[0, 0]
    bottom = np.zeros(len(nsides))
    x = np.arange(len(nsides), dtype=float)
    for i, stage in enumerate(names):
        vals = np.array(
            [
                next(
                    (
                        r["t_min"]
                        for r in stages
                        if r["nside"] == n and r["stage"] == stage
                    ),
                    0.0,
                )
                for n in nsides
            ]
        )
        share = 100.0 * vals / np.where(totals > 0, totals, 1.0)
        ax.bar(
            x,
            share,
            bottom=bottom,
            width=0.62,
            label=stage,
            color=STAGE_RAMP[i % len(STAGE_RAMP)],
            edgecolor=SURFACE,
            linewidth=1.4,  # 2px surface gap between segments
        )
        bottom += share
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"nside {n}\n{t:.3g} s" for n, t in zip(nsides, totals)], fontsize=8
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("share of forward wall time (%)", color=INK_2, fontsize=9)
    ax.grid(axis="x", visible=False)
    _title(
        ax,
        "Where the HP2SPH forward spends its time",
        "compact-band scalar forward, 1 thread; total time under each bar",
    )
    # Below the axes, not inside: the bars fill the plot area to 100% at every
    # nside, so an in-axes legend sits on top of the data wherever it is placed.
    _legend(fig, axes, ncol=4)
    _save(fig, f"speed_stages_{channel}.png", legend_rows=1)


# --------------------------------------------------------------------------- #
# Benchmark 2 -- forward accuracy                                              #
# --------------------------------------------------------------------------- #
def plot_accuracy(channel):
    path = os.path.join(RESULTS_DIR, f"accuracy_{channel}.json")
    if not os.path.exists(path):
        return
    _, records = load_results(path)
    fields = ["T"] if channel == "I" else ["E", "B"]

    # Canonical order, not alphabetical: the primary regime comes first.
    present = {r["scenario"] for r in records} - {"leakage"}
    ordered = [s for s in SCENARIOS if s in present]
    ordered += sorted(present - set(ordered))
    for scenario in ordered:
        for field in fields:
            sel = _by(records, scenario=scenario, field=field)
            if not sel:
                continue
            _plot_per_ell(sel, channel, scenario, field)
            _plot_bands(sel, channel, scenario, field)

    if channel == "P":
        _plot_leakage(_by(records, scenario="leakage"))


def _plot_per_ell(sel, channel, scenario, field):
    nsides = sorted({r["nside"] for r in sel})
    ncol = min(3, len(nsides))
    nrow = int(np.ceil(len(nsides) / ncol))
    fig, axes = _fig(nrow, ncol, width=4.6, height=3.6)
    for ax, nside in zip(axes.ravel(), nsides):
        panel = _by(sel, nside=nside)
        lmin = panel[0].get("lmin", 0)
        for key in _ordered_keys(panel):
            rec = next(r for r in panel if r["backend"] == key)
            y = np.array(rec["cl_err_per_ell"], dtype=float)
            ell = np.arange(len(y))
            m = ell >= max(lmin, 1)
            _plot_series(ax, key, ell[m], y[m], marker="", linewidth=1.6)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\ell$", color=INK_2, fontsize=9)
        ax.set_ylabel(r"relative $C_\ell$ error", color=INK_2, fontsize=9)
        _title(ax, f"nside {nside}", f"lmax {panel[0]['lmax']}")
    for ax in axes.ravel()[len(nsides) :]:
        ax.set_visible(False)
    fig.suptitle(
        f"Forward accuracy vs known coefficients -- {channel}/{field}, "
        f"{scenario} signal",
        color=INK,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    _legend(fig, axes)
    _save(fig, f"accuracy_{channel}_{scenario}_{field}.png")


#  Bands are addressed by POSITION, never by their label. The labels are absolute
#  l ranges ("13-16" at nside 8, "25-32" at nside 16), so keying on them puts every
#  nside in a different panel and each curve collapses to a single point.
BAND_TITLES = [
    r"lowest quarter: $\ell \leq \ell_{max}/4$",
    r"second quarter of the band",
    r"third quarter of the band",
    r"top quarter: $\ell \to \ell_{max}$",
]


def _plot_bands(sel, channel, scenario, field):
    nbands = len(sel[0]["bands"])
    fig, axes = _fig(1, nbands, width=3.9, height=3.5)
    for i, ax in enumerate(axes[0]):
        for key in _ordered_keys(sel):
            pts = []
            for r in sel:
                if r["backend"] != key or i >= len(r["bands"]):
                    continue
                band = list(r["bands"].values())[i]
                pts.append((r["nside"], band.get("cl_rms")))
            pts = sorted((n, v) for n, v in pts if v is not None and np.isfinite(v))
            if pts:
                _plot_series(ax, key, [p[0] for p in pts], [p[1] for p in pts])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("nside", color=INK_2, fontsize=9)
        ax.set_ylabel(r"RMS relative $C_\ell$ error", color=INK_2, fontsize=9)
        _title(
            ax,
            BAND_TITLES[i] if i < len(BAND_TITLES) else f"band {i + 1}",
            r"$\ell_{max} = 2\cdot$nside",
        )
    fig.suptitle(
        f"Convergence by band -- {channel}/{field}, {scenario} signal",
        color=INK,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    _legend(fig, axes)
    _save(fig, f"accuracy_bands_{channel}_{scenario}_{field}.png")


def _plot_leakage(sel):
    if not sel:
        return
    fields = sorted({r["field"] for r in sel})
    nsides = sorted({r["nside"] for r in sel})
    fig, axes = _fig(len(fields), len(nsides), width=4.2, height=3.4)
    for i, field in enumerate(fields):
        for j, nside in enumerate(nsides):
            ax = axes[i, j]
            panel = _by(sel, field=field, nside=nside)
            for key in _ordered_keys(panel):
                rec = next(r for r in panel if r["backend"] == key)
                y = np.array(rec["leak_per_ell"], dtype=float)
                ell = np.arange(len(y))
                m = ell >= 2
                _plot_series(ax, key, ell[m], y[m], marker="", linewidth=1.6)
            ax.set_yscale("log")
            ax.set_xlabel(r"$\ell$", color=INK_2, fontsize=9)
            ax.set_ylabel("leaked power fraction", color=INK_2, fontsize=9)
            _title(ax, f"{field.replace('_', ' ')}, nside {nside}")
    fig.suptitle(
        "Parity leakage from a pure-E (and pure-B) sky",
        color=INK,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    _legend(fig, axes)
    _save(fig, "leakage_P.png")


# --------------------------------------------------------------------------- #
# Benchmark 3 -- round trip                                                    #
# --------------------------------------------------------------------------- #
def plot_roundtrip(channel):
    path = os.path.join(RESULTS_DIR, f"roundtrip_{channel}.json")
    if not os.path.exists(path):
        return
    _, records = load_results(path)
    field = "T" if channel == "I" else "E"
    harmonic = _by(records, mode="harmonic", field=field)
    pixel = _by(records, mode="pixel")
    if not harmonic and not pixel:
        return

    fig, axes = _fig(1, 2, width=5.4, height=4.1)
    ax = axes[0, 0]
    for key in _ordered_keys(harmonic):
        pts = sorted(
            (r["nside"], r["global_l2"]) for r in harmonic if r["backend"] == key
        )
        if pts:
            _plot_series(ax, key, [p[0] for p in pts], [p[1] for p in pts])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("nside", color=INK_2, fontsize=9)
    ax.set_ylabel("relative L2 coefficient error", color=INK_2, fontsize=9)
    _title(
        ax,
        r"Harmonic round trip: $a_{\ell m}\to$ map $\to a_{\ell m}$",
        "lmax = 2·nside − 1",
    )

    ax = axes[0, 1]
    for key in _ordered_keys(pixel):
        pts = sorted(
            (r["nside"], r["map_rel_l2"]) for r in pixel if r["backend"] == key
        )
        if pts:
            _plot_series(ax, key, [p[0] for p in pts], [p[1] for p in pts])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("nside", color=INK_2, fontsize=9)
    ax.set_ylabel("relative L2 map error", color=INK_2, fontsize=9)
    _title(
        ax,
        r"Pixel round trip: map $\to a_{\ell m}\to$ map",
        "band-limited map, so identity is the right answer",
    )
    _legend(fig, axes)
    _save(fig, f"roundtrip_{channel}.png")

    _plot_roundtrip_per_ell(harmonic, channel, field)


def _plot_roundtrip_per_ell(harmonic, channel, field):
    if not harmonic:
        return
    nsides = sorted({r["nside"] for r in harmonic})[-3:]
    fig, axes = _fig(1, len(nsides), width=4.6, height=3.6)
    for ax, nside in zip(axes[0], nsides):
        panel = _by(harmonic, nside=nside)
        lmin = panel[0].get("lmin", 0)
        for key in _ordered_keys(panel):
            rec = next(r for r in panel if r["backend"] == key)
            y = np.array(rec["alm_err_per_ell"], dtype=float)
            ell = np.arange(len(y))
            m = ell >= max(lmin, 1)
            _plot_series(ax, key, ell[m], y[m], marker="", linewidth=1.6)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\ell$", color=INK_2, fontsize=9)
        ax.set_ylabel(r"relative $a_{\ell m}$ error", color=INK_2, fontsize=9)
        _title(ax, f"nside {nside}", f"lmax {panel[0]['lmax']}")
    fig.suptitle(
        f"Harmonic round-trip error per multipole -- {channel}/{field}",
        color=INK,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    _legend(fig, axes)
    _save(fig, f"roundtrip_per_ell_{channel}_{field}.png")


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--only",
        nargs="+",
        choices=["speed", "accuracy", "roundtrip"],
        default=["speed", "accuracy", "roundtrip"],
    )
    p.add_argument("--channels", nargs="+", choices=["I", "P"], default=["I", "P"])
    args = p.parse_args(argv)

    print("rendering figures...")
    for channel in args.channels:
        if "speed" in args.only:
            plot_speed(channel)
        if "accuracy" in args.only:
            plot_accuracy(channel)
        if "roundtrip" in args.only:
            plot_roundtrip(channel)


if __name__ == "__main__":
    main()
