#!/usr/bin/env python3
"""Reproduce static historical performance figures from verified archive inputs.

Requires Matplotlib. CPU campaigns remain independent, CUDA event timing is
shown separately from process wall time, and presentation totals are explicitly
reported evidence with unavailable raw timing logs/settings.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import math
from pathlib import Path
import re
import subprocess
import sys


COLORS = {"openmp": "#2166A7", "mpi": "#C76319", "cuda": "#596D33"}
TEXT = "#24313D"
MUTED = "#586572"
GRID = "#E2E6EA"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def cuda_event_seconds(directory: Path) -> Decimal:
    text = (directory / "cuda_steps.txt").read_text(encoding="utf-8")
    matches = re.findall(r"^Total GPU time:\s*([0-9]+(?:\.[0-9]+)?)\s*ms\s*$", text, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError("cuda_steps.txt must contain exactly one Total GPU time observation")
    seconds = Decimal(matches[0]) / 1000
    if not seconds.is_finite() or seconds <= 0:
        raise ValueError("CUDA event timing must be finite and positive")
    return seconds


def save_figure(figure, output: Path, name: str) -> None:
    # Fixed SVG hash salt is set in main; remove time-varying SVG metadata.
    figure.savefig(output / f"{name}.png", dpi=160, facecolor="white",
                   metadata={"Software": "hipc-airfoil performance plotting"})
    figure.savefig(output / f"{name}.svg", facecolor="white",
                   metadata={"Date": None, "Creator": "hipc-airfoil performance plotting"})


def style_axis(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#84919C")
    axis.tick_params(axis="both", colors=MUTED, labelsize=12, length=4, pad=7)
    axis.grid(axis="y", color=GRID, linewidth=.8)
    axis.set_axisbelow(True)


def raw_performance(plt, directory: Path, output: Path) -> None:
    series = []
    for filename, campaign in (("strong_scaling.csv", "Primary"),
                               ("additional_scaling.csv", "Additional")):
        rows = read_rows(directory / filename)
        for backend in ("openmp", "mpi"):
            selected = sorted((row for row in rows if row["backend"] == backend), key=lambda row: int(row["workers"]))
            if len(selected) != 5 or [int(row["workers"]) for row in selected] != [1, 2, 4, 8, 16]:
                raise ValueError(f"{filename}: expected five worker counts for {backend}")
            if any((int(row["nx"]), int(row["ny"]), int(row["samples"])) != (1024, 256, 1) for row in selected):
                raise ValueError("Raw chart requires the archived 1024 x 256 single-observation runs")
            wall = [float(row["wall_seconds"]) for row in selected]
            # Recompute each independent denominator, never a pooled baseline.
            speedup = [wall[0] / value for value in wall]
            series.append((backend, campaign, wall, speedup))
    event = cuda_event_seconds(directory)
    figure = plt.figure(figsize=(18, 8.8), facecolor="white")
    layout = figure.add_gridspec(1, 3, left=.065, right=.975, bottom=.31, top=.70,
                                wspace=.34, width_ratios=[1.05, 1.05, .94])
    wall_axis, speed_axis, cuda_axis = [figure.add_subplot(layout[0, i]) for i in range(3)]
    figure.text(.065, .955, "Historical performance | 1024 x 256 grid", fontsize=23, weight="bold", color=TEXT)
    figure.text(.065, .900, "2098 steps | Single observations | Archived implementations", fontsize=15, color=MUTED)
    handles = []
    labels = []
    workers = [1, 2, 4, 8, 16]
    for backend, campaign, wall, speedup in series:
        additional = campaign == "Additional"
        marker = ({"openmp": "D", "mpi": "^"} if additional else {"openmp": "o", "mpi": "s"})[backend]
        settings = {"color": COLORS[backend], "linewidth": 2.2,
                    "linestyle": (0, (5, 3)) if additional else "-",
                    "marker": marker, "markersize": 6.8, "markeredgewidth": 1.5,
                    "markerfacecolor": "white" if additional else COLORS[backend]}
        line, = wall_axis.plot(workers, wall, **settings)
        speed_axis.plot(workers, speedup, **settings)
        handles.append(line)
        labels.append(f"{'OpenMP' if backend == 'openmp' else 'MPI'} / {campaign}")
    figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(.060, .852), ncol=2,
                  frameon=False, fontsize=12.5, handlelength=2.9, columnspacing=2.0, labelspacing=.8)
    for axis, title in ((wall_axis, "A | CPU process wall time"),
                        (speed_axis, "B | CPU strong scaling")):
        style_axis(axis)
        axis.set_title(title, loc="left", fontsize=16, weight="bold", color=TEXT, pad=18)
        axis.set_xscale("log", base=2)
        axis.set_xticks(workers, [str(value) for value in workers])
        axis.set_xlim(.85, 18.8)
        axis.set_xlabel("Workers\nOpenMP threads / MPI processes", fontsize=12, color=MUTED, labelpad=10)
    wall_axis.set_ylabel("Process wall time (s)", fontsize=13, color=TEXT, labelpad=10)
    wall_axis.set_ylim(0, 460)
    wall_axis.set_yticks([0, 100, 200, 300, 400])
    speed_axis.set_ylabel("Own-baseline speedup (x)", fontsize=13, color=TEXT, labelpad=10)
    speed_axis.set_ylim(0, 18)
    speed_axis.set_yticks([0, 4, 8, 12, 16])
    speed_axis.plot(workers, workers, color="#65707A", linestyle=(0, (2, 3)), linewidth=1.4, zorder=1)
    speed_axis.text(8.5, 16.2, "Ideal", fontsize=11.5, color=MUTED)

    cuda_axis.set_axis_off()
    cuda_axis.set_title("C | CUDA event interval", loc="left", fontsize=16, weight="bold", color=TEXT, pad=18)
    cuda_axis.text(.015, .93, "Simulation-loop event", transform=cuda_axis.transAxes, fontsize=13, color=MUTED)
    cuda_axis.text(.015, .745, f"{event:.3f} s", transform=cuda_axis.transAxes, fontsize=34, weight="bold", color=TEXT)
    cuda_axis.plot([.015, .97], [.67, .67], transform=cuda_axis.transAxes, color=GRID, linewidth=1.2)
    cuda_axis.text(.015, .585, "Different timing scope", transform=cuda_axis.transAxes, fontsize=14, weight="bold", color=TEXT)
    cuda_axis.text(.015, .50,
                   "No CPU/GPU speedup computed.\n"
                   "Not process wall time.\n\n"
                   "Excludes initial upload,\n"
                   "final download and output.\n\n"
                   "Old implementation;\n"
                   "single observation.",
                   transform=cuda_axis.transAxes, va="top", fontsize=12.5, linespacing=1.30, color=MUTED)

    figure.text(.065, .165, "CPU: shell real process wall time; independent campaigns, one observation per point (no error bars).", fontsize=12.2, color=MUTED)
    figure.text(.065, .125, "CPU speedup = the same campaign's T(1) / T(workers). CUDA event: 46,290.664 ms = 46.290664 s.", fontsize=12.2, color=MUTED)
    figure.text(.065, .064, "Sources: benchmarks/historical/strong_scaling.csv, additional_scaling.csv and cuda_steps.txt", fontsize=11.6, color=MUTED)
    save_figure(figure, output, "performance")
    plt.close(figure)


def problem_sizes(plt, directory: Path, output: Path) -> None:
    rows = read_rows(directory / "presentation" / "problem_sizes.csv")
    if len(rows) != 12:
        raise ValueError("Presentation size chart requires twelve source rows")
    figure, axis = plt.subplots(figsize=(16, 9.3), facecolor="white")
    figure.subplots_adjust(left=.095, right=.955, top=.77, bottom=.335)
    figure.text(.095, .950, "Presentation-reported runtime by problem size", fontsize=23, weight="bold", color=TEXT)
    figure.text(.095, .893, "Independent historical campaign | CPU: 16 workers | GPU: Tesla T4 (reported)", fontsize=14.5, color=MUTED)
    style_axis(axis)
    categories = ["Small", "Medium", "Large", "ExtraLarge"]
    x_values = []
    tick_labels = []
    for backend, label, marker, line_style in (
            ("openmp", "OpenMP / 16 threads", "o", "-"),
            ("mpi", "MPI / 16 processes", "s", (0, (5, 3))),
            ("cuda", "CUDA / Tesla T4 (reported)", "D", (0, (2, 2)))):
        selected = sorted((row for row in rows if row["backend"] == backend), key=lambda row: int(row["total_cells"]))
        if len(selected) != 4 or [row["size_category"] for row in selected] != categories:
            raise ValueError(f"Presentation chart has incomplete size categories for {backend}")
        x = [int(row["total_cells"]) for row in selected]
        y = [float(row["total_seconds"]) for row in selected]
        if any(not math.isfinite(value) or value <= 0 for value in y):
            raise ValueError("Presentation totals must be finite and positive")
        if any(int(row["nx"]) * int(row["ny"]) != cells for row, cells in zip(selected, x)):
            raise ValueError("Presentation total_cells must match the declared grid")
        if backend != "cuda" and any(row["workers"] != "16" for row in selected):
            raise ValueError("Presentation CPU series requires sixteen workers")
        if backend == "cuda" and any(row["workers"] for row in selected):
            raise ValueError("CUDA workers are unspecified; do not equate a GPU with a CPU worker")
        if not x_values:
            x_values = x
            tick_labels = [f"{row['size_category'].replace('ExtraLarge', 'Extra large')}\n{row['nx']} x {row['ny']}\n{cells:,} cells"
                           for row, cells in zip(selected, x)]
        elif x != x_values:
            raise ValueError("Presentation backends must cover the same four grid sizes")
        axis.plot(x, y, label=label, color=COLORS[backend], marker=marker, linestyle=line_style,
                  linewidth=2.3, markersize=8, markeredgewidth=1.4,
                  markerfacecolor="white" if backend == "cuda" else COLORS[backend])
    axis.set_xscale("log", base=2)
    axis.set_yscale("log", base=10)
    axis.set_xlim(x_values[0] / 1.45, x_values[-1] * 1.45)
    axis.set_ylim(.9, 15000)
    axis.set_xticks(x_values, tick_labels)
    axis.set_yticks([1, 10, 100, 1000, 10000], ["1", "10", "100", "1,000", "10,000"])
    axis.minorticks_off()
    axis.set_xlabel("Total cells (log scale; grid dimensions shown)", fontsize=13, color=TEXT, labelpad=12)
    axis.set_ylabel("Reported total time (s, log scale)", fontsize=13, color=TEXT, labelpad=12)
    axis.legend(loc="lower left", bbox_to_anchor=(-.006, 1.025), ncol=3, frameon=False,
                fontsize=12.6, handlelength=3.0, columnspacing=2.0)
    figure.text(.095, .182, "Presentation-reported times. Raw timing logs and timing scope/settings are unavailable.", fontsize=13.2, weight="bold", color=TEXT)
    figure.text(.095, .137, "Separate campaign from the 46.33 s summary and the 46.291 s CUDA event; no cross-scope speedup is inferred.", fontsize=12.2, color=MUTED)
    figure.text(.095, .095, "One reported total per grid/backend. Step counts and per-step units are unspecified; no uncertainty intervals are available.", fontsize=11.8, color=MUTED)
    figure.text(.095, .048, "Source: benchmarks/historical/presentation/problem_sizes.csv (slide 20; problem-size-table.png).", fontsize=11.5, color=MUTED)
    save_figure(figure, output, "performance_problem_sizes")
    plt.close(figure)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=root / "benchmarks" / "historical",
                        help="Verified historical archive directory")
    parser.add_argument("--output-dir", type=Path, default=root / "docs" / "assets")
    args = parser.parse_args()
    checked = subprocess.run([sys.executable, str(root / "scripts" / "check_historical.py"),
                              "--directory", str(args.directory)], check=False)
    if checked.returncode:
        print("Figures were not regenerated: archive verification failed.", file=sys.stderr)
        return 1
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is required to reproduce these static figures.", file=sys.stderr)
        return 1
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT,
                         "svg.hashsalt": "hipc-airfoil-performance-v1", "svg.fonttype": "none",
                         "axes.unicode_minus": False, "savefig.facecolor": "white"})
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        raw_performance(plt, args.directory, args.output_dir)
        problem_sizes(plt, args.directory, args.output_dir)
    except (OSError, ValueError) as error:
        print(f"Performance plotting failed: {error}", file=sys.stderr)
        return 1
    print("Saved performance.png/.svg and performance_problem_sizes.png/.svg.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
