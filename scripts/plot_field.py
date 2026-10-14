#!/usr/bin/env python3
"""Plot an actual 2-D VTR field; optional dependencies: numpy and matplotlib."""
from __future__ import annotations

import argparse
from pathlib import Path

from compare_vtk import read_grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--field", choices=["u", "p"], default="u")
    parser.add_argument("--airfoil", help="Optional four-digit NACA outline; default solver placement")
    args = parser.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = read_grid(args.input, strict=True)
    nx, ny = len(grid.coordinates[0]) - 2, len(grid.coordinates[1]) - 2
    if len(grid.coordinates[2]) != 1 or nx < 4 or ny < 4:
        parser.error("Expected a 2-D ghost-inclusive solver grid")
    dx = grid.coordinates[0][1] - grid.coordinates[0][0]
    dy = grid.coordinates[1][1] - grid.coordinates[1][0]
    x = np.arange(1, nx + 1) * dx
    y = (np.arange(1, ny + 1) - .5) * dy
    if args.field == "u":
        values = np.asarray(grid.fields["uv"].values).reshape(ny + 2, nx + 2, 3)[1:-1, 1:-1, 0]
        label = "Horizontal velocity u (solver units)"
    else:
        values = np.asarray(grid.fields["p"].values).reshape(ny + 2, nx + 2)[1:-1, 1:-1]
        x = x - dx / 2
        label = "Pressure p (solver units)"
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "#f8fafc"}):
        figure, axis = plt.subplots(figsize=(11, 3.6), layout="constrained")
        mesh = axis.pcolormesh(x, y, values, shading="nearest", cmap="RdBu_r", rasterized=True)
        if args.airfoil:
            if len(args.airfoil) != 4 or not args.airfoil.isdigit():
                parser.error("--airfoil must be four digits")
            naca = int(args.airfoil)
            m, position, thickness = (naca // 1000) / 100, ((naca // 100) % 10) / 10, (naca % 100) / 100
            if m and not position:
                parser.error("Cambered NACA profile requires a nonzero camber position")
            chord = np.linspace(0, 1, 500)
            yt = 5 * thickness * (.2969*np.sqrt(chord) - .1260*chord - .3516*chord**2 + .2843*chord**3 - .1015*chord**4)
            if m:
                yc = np.where(chord <= position, m/position**2*(2*position*chord-chord**2),
                              m/(1-position)**2*(1-2*position+2*position*chord-chord**2))
                slope = np.where(chord <= position, 2*m/position**2*(position-chord),
                                 2*m/(1-position)**2*(position-chord))
            else:
                yc, slope = np.zeros_like(chord), np.zeros_like(chord)
            half_height = yt * np.cos(np.arctan(slope))
            midline = ny * dy / 2
            axis.fill_between(chord + .5, midline+yc-half_height, midline+yc+half_height,
                              color="#ffffff", edgecolor="#172033", linewidth=.7, zorder=3)
        axis.set(xlim=(0, nx*dx), ylim=(0, ny*dy), xlabel="x", ylabel="y", aspect="equal")
        axis.set_title(f"NACA {args.airfoil or 'airfoil'} CFD  |  {nx} × {ny} cells  |  t = {grid.time:.3f}", loc="left", pad=12, fontweight="bold")
        figure.colorbar(mesh, ax=axis, label=label, shrink=.84, pad=.025)
        figure.savefig(args.output, dpi=170, bbox_inches="tight")
        plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
