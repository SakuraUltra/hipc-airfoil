"""Render a verified solver result as an offline page using the standard library.

Plots retain the solver's staggered sample positions and reconstruct the setup
geometry mask without inferring geometry from field values. Saved states are
not interpolated, and no physical validation or performance timing is claimed.
"""

from __future__ import annotations

import base64
from html import escape
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import shlex
import struct
from urllib.parse import quote
import zlib

from compare_vtk import Grid
from demo_geometry import fluid_mask


def _format(value: float | int | None) -> str:
    if value is None:
        return "not defined"
    return f"{value:.3e}"


def _basename(value: str) -> str:
    # Also remove Windows drive/user directories when viewing cross-platform logs.
    return PureWindowsPath(value).name if "\\" in value else Path(value).name


def _portable_command(run: dict, output_dir: Path) -> str:
    command = run.get("portable_command") or run.get("command") or []
    if isinstance(command, str):
        command = shlex.split(command)
    parts: list[str] = []
    for index, value in enumerate(command):
        token = str(value)
        if index == 0:
            parts.append(_basename(token))
            continue
        prefix, separator, argument = token.partition("=")
        path_value = argument if separator else token
        if Path(path_value).is_absolute() or PureWindowsPath(path_value).is_absolute():
            try:
                path_value = Path(path_value).resolve().relative_to(output_dir.resolve()).as_posix()
            except ValueError:
                path_value = _basename(path_value)
            token = prefix + "=" + path_value if separator else path_value
        parts.append(token)
    return shlex.join(parts)


def _colour(value: float, limit: float) -> str:
    """A zero-centred navy/ivory/coral scale shared by a field and its legend."""
    fraction = min(1.0, max(-1.0, value / limit))
    centre = (244, 247, 246)
    end = (26, 77, 115) if fraction < 0 else (213, 103, 73)
    amount = abs(fraction)
    rgb = [round(a + amount * (b - a)) for a, b in zip(centre, end)]
    return "#" + "".join(f"{component:02x}" for component in rgb)


def _samples(grid: Grid, field: str, nx: int, ny: int) -> list[float]:
    data = grid.fields["uv" if field == "u" else "p"].values
    pitch = nx + 2
    return [data[3 * (j * pitch + i)] if field == "u" else data[j * pitch + i]
            for j in range(1, ny + 1) for i in range(1, nx + 1)]


def _visible_samples(nx: int, ny: int, dx: float, dy: float,
                     airfoil: str) -> dict[str, list[bool]]:
    mask = fluid_mask(nx, ny, dx, dy, airfoil)
    pitch = nx + 2
    return {
        "p": [mask[j * pitch + i] for j in range(1, ny + 1) for i in range(1, nx + 1)],
        "u": [mask[j * pitch + i] and mask[j * pitch + i + 1]
              for j in range(1, ny + 1) for i in range(1, nx + 1)],
    }


def _png_heatmap(samples: list[float], visible: list[bool], nx: int, ny: int,
                 limit: float) -> str:
    """One RGB pixel per original sample; reverse j for PNG's top-down rows."""
    raw = bytearray()
    for j in range(ny - 1, -1, -1):
        raw.append(0)  # PNG filter type 0; no resampling or interpolation.
        for i in range(nx):
            index = j * nx + i
            colour = _colour(samples[index], limit) if visible[index] else "#e2e7e9"
            raw.extend(bytes.fromhex(colour[1:]))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff))

    image = (b"\x89PNG\r\n\x1a\n"
             + chunk(b"IHDR", struct.pack(">IIBBBBB", nx, ny, 8, 2, 0, 0, 0))
             + chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b""))
    return base64.b64encode(image).decode("ascii")


def _reference_outline(airfoil: str, midline: float, left: float, top: float,
                       scale: float, xmin: float, ymax: float) -> str:
    """Project the setup formula as an unfilled analytic reference, not a mask."""
    naca = int(airfoil)
    camber = (naca // 1000) / 100.0
    position = ((naca // 100) % 10) / 10.0
    thickness = (naca % 100) / 100.0
    if camber and not position:
        raise ValueError("A cambered NACA reference requires a nonzero camber position")
    upper, lower = [], []
    for k in range(241):
        chord = k / 240.0
        yt = 5 * thickness * (.2969 * math.sqrt(chord) - .1260 * chord
                              - .3516 * chord ** 2 + .2843 * chord ** 3 - .1015 * chord ** 4)
        if not camber:
            yc, slope = 0.0, 0.0
        elif chord <= position:
            yc = camber / position ** 2 * (2 * position * chord - chord ** 2)
            slope = 2 * camber / position ** 2 * (position - chord)
        else:
            yc = camber / (1 - position) ** 2 * (1 - 2 * position + 2 * position * chord - chord ** 2)
            slope = 2 * camber / (1 - position) ** 2 * (position - chord)
        half_height = yt * math.cos(math.atan(slope))
        svg_x = left + (chord + .5 - xmin) * scale
        upper.append((svg_x, top + (ymax - midline - yc - half_height) * scale))
        lower.append((svg_x, top + (ymax - midline - yc + half_height) * scale))
    points = upper + list(reversed(lower))
    path = "M " + " L ".join(f"{x:.6f},{y:.6f}" for x, y in points) + " Z"
    return (f'<path class="analytic-reference" d="{path}" fill="none" stroke="#101820" '
            'stroke-width="1.15" stroke-dasharray="4 3" vector-effect="non-scaling-stroke">'
            '<title>Analytic NACA reference outline; not the discrete solver obstacle mask.</title></path>')


def _field_svg(grid: Grid, field: str, nx: int, ny: int, airfoil: str,
               view: str = "full", visible: list[bool] | None = None,
               colour_limit: float | None = None, png: str | None = None) -> str:
    dx = grid.coordinates[0][1] - grid.coordinates[0][0]
    dy = grid.coordinates[1][1] - grid.coordinates[1][0]
    if dx <= 0 or dy <= 0:
        raise ValueError("The demonstration requires positive grid spacing")
    domain_x, domain_y = nx * dx, ny * dy
    samples = _samples(grid, field, nx, ny)
    if not samples or not all(math.isfinite(value) for value in samples):
        raise ValueError("The demonstration requires complete finite interior data")
    if visible is None:
        visible = _visible_samples(nx, ny, dx, dy, airfoil)[field]
    fluid_values = [value for value, shown in zip(samples, visible) if shown]
    if len(visible) != nx * ny or not fluid_values:
        raise ValueError("The demonstration requires visible fluid samples")
    minimum, maximum = min(fluid_values), max(fluid_values)
    limit = colour_limit if colour_limit is not None else max(abs(minimum), abs(maximum)) or 1.0
    if view == "detail":
        xmin, xmax = .25, min(domain_x, 1.9)
        if xmax <= xmin:
            xmin, xmax = 0.0, domain_x
        ymin, ymax = max(0.0, domain_y / 2 - .28), min(domain_y, domain_y / 2 + .28)
    elif view == "full":
        xmin, xmax, ymin, ymax = 0.0, domain_x, 0.0, domain_y
    else:
        raise ValueError("Unknown demonstration view")
    view_label = "Airfoil detail" if view == "detail" else "Full domain"
    left, top, width = 84.0, 84.0, 952.0
    height = width * (ymax - ymin) / (xmax - xmin)
    canvas_height = height + 254.0
    # A single scale preserves equal physical lengths in x and y.
    scale_x = scale_y = width / (xmax - xmin)
    label = "Horizontal velocity u" if field == "u" else "Pressure p"
    position = "x = i·dx, y = (j − ½)·dy" if field == "u" else "x = (i − ½)·dx, y = (j − ½)·dy"
    title = escape(f"{label} · {view_label} · NACA {airfoil} · {nx} × {ny} · TIME {grid.time!r}, CYCLE {grid.cycle}")
    description = escape(
        f"Actual maintained serial-opt solver samples; {nx * ny} interior values. "
        f"Ghost layers are excluded. Sample coordinates: {position}. "
        f"View: {view_label}; bounds x={xmin!r}..{xmax!r}, y={ymin!r}..{ymax!r}. "
        "Solid and boundary samples are neutral grey using the exact setup fluid mask. "
        "The dashed analytic outline is a reference, not that discrete mask. "
        "Both views and all recorded frames use the same visible-fluid colour limits. "
        "The image is a numerical diagnostic, not physical validation."
    )
    metadata = escape(json.dumps({"field": field, "nx": nx, "ny": ny,
                                  "time": grid.time, "cycle": grid.cycle,
                                  "dx": dx, "dy": dy,
                                  "view": view,
                                  "bounds": {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax},
                                  "plot_rect": {"left": left, "top": top, "width": width, "height": height},
                                  "colour_limits": [-limit, limit],
                                  "colour_range_source": "visible fluid across all recorded frames",
                                  "mask_rule": "p: fluid[index]; u: fluid[index] and fluid[index + 1]",
                                  "visible_samples": len(fluid_values),
                                  "masked_samples": nx * ny - len(fluid_values),
                                  "source_vtr": grid.path.name,
                                  "sample_values_sha256": hashlib.sha256(
                                      struct.pack(f">{len(samples)}d", *samples)).hexdigest(),
                                  "image_encoding": "one RGB8 PNG pixel per interior sample" if png else "vector sample rectangles",
                                  "pixel_mapping": "column i-1; row ny-j; i=1..nx and j=1..ny",
                                  "interior_only": True, "sample_positions": position},
                                 allow_nan=False))
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{canvas_height:.3f}" '
        f'viewBox="0 0 1120 {canvas_height:.3f}" role="img" aria-labelledby="field-title field-desc">',
        f'<title id="field-title">{title}</title><desc id="field-desc">{description}</desc>',
        f'<metadata>{metadata}</metadata>',
        '<defs><linearGradient id="colour-scale">'
        '<stop offset="0" stop-color="#1a4d73"/><stop offset=".5" stop-color="#f4f7f6"/>'
        '<stop offset="1" stop-color="#d56749"/></linearGradient>',
        f'<clipPath id="domain"><rect x="{left}" y="{top}" width="{width}" height="{height:.6f}"/></clipPath></defs>',
        f'<rect width="1120" height="{canvas_height:.3f}" rx="16" fill="#fbfcfc"/>',
        '<g font-family="system-ui, sans-serif" fill="#102b40">',
        f'<text x="{left}" y="33" font-size="20" font-weight="700">{escape(label)} · {view_label}</text>',
        f'<text x="{left}" y="58" font-size="12" fill="#576f7c">NACA {escape(airfoil)} · '
        f'{nx} × {ny} interior grid · TIME {grid.time:.6g} · CYCLE {grid.cycle}</text>',
        f'<rect x="{left}" y="{top}" width="{width}" height="{height:.6f}" fill="#e2e7e9"/>',
        '<g clip-path="url(#domain)" shape-rendering="crispEdges">',
    ]
    if png is not None:
        image_left = dx / 2 if field == "u" else 0.0
        pieces.append(
            f'<image class="sample-image" x="{left + (image_left - xmin) * scale_x:.6f}" '
            f'y="{top + (ymax - domain_y) * scale_y:.6f}" width="{domain_x * scale_x:.6f}" '
            f'height="{domain_y * scale_y:.6f}" preserveAspectRatio="none" '
            f'style="image-rendering:pixelated" href="data:image/png;base64,{png}"/>'
        )
    for j in range(1, ny + 1) if png is None else ():
        # SVG y grows downwards; physical y grows upwards.
        cell_bottom, cell_top = (j - 1) * dy, j * dy
        if cell_top <= ymin or cell_bottom >= ymax:
            continue
        svg_y = top + (ymax - cell_top) * scale_y
        for i in range(1, nx + 1):
            index = (j - 1) * nx + i - 1
            if not visible[index]:
                continue
            value = samples[index]
            sample_x = i * dx if field == "u" else (i - .5) * dx
            cell_left, cell_right = sample_x - dx / 2, sample_x + dx / 2
            if cell_right <= xmin or cell_left >= xmax:
                continue
            svg_x = left + (cell_left - xmin) * scale_x
            hover = escape(f"i={i}, j={j}; x={sample_x:.8g}, y={(j - .5) * dy:.8g}; {field}={value:.12g}")
            pieces.append(
                f'<rect class="sample" data-i="{i}" data-j="{j}" data-value="{value:.17g}" data-visible="true" '
                f'x="{svg_x:.6f}" y="{svg_y:.6f}" width="{dx * scale_x:.6f}" '
                f'height="{dy * scale_y:.6f}" fill="{_colour(value, limit)}"><title>{hover}</title></rect>'
            )
    pieces.extend([
        '</g>',
        '<g clip-path="url(#domain)">',
        _reference_outline(airfoil, domain_y / 2, left, top, scale_x, xmin, ymax),
        '</g>',
        f'<rect x="{left}" y="{top}" width="{width}" height="{height:.6f}" fill="none" stroke="#acbfc9"/>',
    ])
    baseline = top + height
    for tick in range(5):
        x = left + width * tick / 4
        pieces.append(f'<path d="M{x:.3f} {baseline:.3f}v6" stroke="#7b929f"/>')
        pieces.append(f'<text x="{x:.3f}" y="{baseline + 25:.3f}" text-anchor="middle" font-size="12">{xmin + (xmax - xmin) * tick / 4:.4g}</text>')
    for tick in range(3):
        y = top + height * (1 - tick / 2)
        pieces.append(f'<path d="M{left - 6:.3f} {y:.3f}h6" stroke="#7b929f"/>')
        pieces.append(f'<text x="{left - 13:.3f}" y="{y + 4:.3f}" text-anchor="end" font-size="12">{ymin + (ymax - ymin) * tick / 2:.4g}</text>')
    legend_y = baseline + 69
    pieces.extend([
        f'<text x="{left + width / 2}" y="{baseline + 46:.3f}" text-anchor="middle" font-size="12">x · solver length units</text>',
        f'<text x="25" y="{top + height / 2:.3f}" font-size="12" text-anchor="middle" '
        f'transform="rotate(-90 25 {top + height / 2:.3f})">y · solver length units</text>',
        f'<rect x="{left}" y="{legend_y:.3f}" width="330" height="12" rx="3" fill="url(#colour-scale)" stroke="#d3dce1"/>',
        f'<text x="{left}" y="{legend_y + 30:.3f}" font-size="12">{escape(_format(-limit))}</text>',
        f'<text x="{left + 165}" y="{legend_y + 30:.3f}" text-anchor="middle" font-size="12">0</text>',
        f'<text x="{left + 330}" y="{legend_y + 30:.3f}" text-anchor="end" font-size="12">{escape(_format(limit))}</text>',
        f'<text x="{left + 375}" y="{legend_y + 11:.3f}" font-size="12" fill="#576f7c">'
        f'Current fluid range: {escape(_format(minimum))} to {escape(_format(maximum))} · solver units</text>',
        f'<text x="{left}" y="{legend_y + 56:.3f}" font-size="12" fill="#576f7c">'
        f'Fluid samples only · {escape(position)} · solid/boundary samples are neutral grey</text>',
        f'<text x="{left}" y="{legend_y + 75:.3f}" font-size="12" fill="#576f7c">'
        'Dashed line: analytic NACA reference; not the solver’s discrete obstacle mask. No smoothing.</text>',
        '</g></svg>',
    ])
    return "\n".join(pieces) + "\n"


def write_demo_page(output_dir: Path, grid: Grid, comparison: dict,
                    runs: list[dict], settings: dict,
                    frames: list[dict] | None = None) -> Path:
    """Write local SVG fields and a page only for an already verified result."""
    if comparison.get("passed") is not True:
        raise ValueError("A failed numerical comparison cannot produce a success page")
    if not runs or any(run.get("status") != "completed" or run.get("returncode") != 0 for run in runs):
        raise ValueError("Every solver run must complete successfully before rendering")
    nx, ny = int(settings["nx"]), int(settings["ny"])
    if tuple(map(len, grid.coordinates)) != (nx + 2, ny + 2, 1):
        raise ValueError("Demo settings and the strict grid dimensions differ")
    if any(comparison["fields"][name].get("passed") is not True for name in ("uv", "p")):
        raise ValueError("Both complete fields must pass before rendering")
    airfoil = str(settings["airfoil"])
    records = sorted(frames if frames is not None else [{"grid": grid, "comparison": comparison}],
                     key=lambda record: record["grid"].time)
    if not records:
        raise ValueError("At least one verified state is required")
    for index, record in enumerate(records):
        state, checked = record["grid"], record["comparison"]
        if (checked.get("passed") is not True
                or any(checked["fields"][name].get("passed") is not True for name in ("uv", "p"))):
            raise ValueError("Every displayed frame must pass the complete-field comparison")
        if state.extent != grid.extent or state.coordinates != grid.coordinates:
            raise ValueError("Every displayed frame must use the same grid geometry")
        if checked["time"] != state.time or checked["cycle"] != state.cycle:
            raise ValueError("Frame comparison metadata differs from its displayed field")
        if index and state.time <= records[index - 1]["grid"].time:
            raise ValueError("Displayed frame times must increase strictly")
    if (records[-1]["grid"].time, records[-1]["grid"].cycle) != (grid.time, grid.cycle):
        raise ValueError("The final displayed state must be the final verified result")
    dx = grid.coordinates[0][1] - grid.coordinates[0][0]
    dy = grid.coordinates[1][1] - grid.coordinates[1][0]
    visible = _visible_samples(nx, ny, dx, dy, airfoil)
    limits: dict[str, float] = {}
    for field in ("u", "p"):
        limits[field] = max((abs(value) for record in records
                             for value, shown in zip(_samples(record["grid"], field, nx, ny), visible[field])
                             if shown), default=0.0) or 1.0
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # The final standalone plots retain auditable vector samples. Timeline
    # states embed one PNG pixel per sample to keep a multi-frame demo small.
    for field, stem in (("u", "velocity"), ("p", "pressure")):
        for view in ("full", "detail"):
            filename = stem + ("-detail" if view == "detail" else "") + ".svg"
            (output_dir / filename).write_text(
                _field_svg(grid, field, nx, ny, airfoil, view, visible[field], limits[field]), encoding="utf-8")
    frame_directory = output_dir / "frames"
    frame_directory.mkdir(exist_ok=True)
    timeline = []
    for index, record in enumerate(records):
        state, checked = record["grid"], record["comparison"]
        files: dict[str, dict[str, str]] = {"u": {}, "p": {}}
        for field, stem in (("u", "velocity"), ("p", "pressure")):
            png = _png_heatmap(_samples(state, field, nx, ny), visible[field], nx, ny, limits[field])
            for view in ("full", "detail"):
                filename = f"frame-{index:04d}-{stem}" + ("-detail" if view == "detail" else "") + ".svg"
                (frame_directory / filename).write_text(
                    _field_svg(state, field, nx, ny, airfoil, view, visible[field], limits[field], png), encoding="utf-8")
                files[field][view] = "frames/" + filename
        timeline.append({"time": state.time, "time_display": f"{state.time:.6g}", "time_exact": repr(state.time),
                         "cycle": state.cycle, "files": files,
                         "fields": {name: {key: checked["fields"][name][key] for key in
                                           ("max_abs_error", "relative_l2_error", "rms_abs_error", "values", "passed")}
                                    for name in ("uv", "p")}, "tolerances": checked["tolerances"]})

    rows = []
    for name, label in (("uv", "Velocity vector (u, v, 0)"), ("p", "Pressure p")):
        result = comparison["fields"][name]
        rows.append(
            f'<tr><th scope="row">{label}<small><span id="{name}-values">{result["values"]:,}</span> values, including ghost layers</small></th>'
            f'<td id="{name}-max">{escape(_format(result["max_abs_error"]))}</td>'
            f'<td id="{name}-l2">{escape(_format(result["relative_l2_error"]))}</td>'
            f'<td id="{name}-rms">{escape(_format(result["rms_abs_error"]))}</td>'
            '<td><span class="status">PASS</span></td></tr>'
        )
    commands = []
    for run in runs:
        backend = escape(str(run.get("backend") or _basename(str(run.get("command", ["solver"])[0]))))
        links = []
        for key in ("stdout", "stderr"):
            if run.get(key):
                basename = _basename(str(run[key]))
                links.append(f'<a href="{escape(quote(basename, safe=""), quote=True)}">{key} log</a>')
        commands.append(
            f'<div class="command"><div class="command-heading"><strong>{backend}</strong>'
            f'<span>{" · ".join(links)}</span></div>'
            f'<code>{escape(_portable_command(run, output_dir))}</code></div>'
        )
    tolerances = comparison["tolerances"]
    actual_time = escape(repr(grid.time))
    requested_time = escape(str(settings["end_time"]))
    timestep = escape(str(settings["dt"]))
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Airfoil CFD · verified offline demonstration</title>
  <style>
    :root { color-scheme:light; --navy:#081927; --ink:#163347; --muted:#526f80;
      --teal:#147e84; --line:#dce7ec; --paper:#f3f7f8; }
    * { box-sizing:border-box; } body { margin:0; background:var(--paper); color:var(--ink);
      font:15px/1.6 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    a { color:#116d78; text-underline-offset:3px; } a:hover { color:#083e49; }
    header { background:var(--navy); color:#eaf4f8; padding:38px max(22px,calc((100vw - 1120px)/2)) 80px; }
    .eyebrow { color:#78d4db; font-size:12px; letter-spacing:2px; font-weight:750; text-transform:uppercase; }
    .hero { display:flex; justify-content:space-between; gap:25px; align-items:center; margin-top:18px; }
    h1 { font-size:clamp(30px,4.3vw,49px); line-height:1.13; letter-spacing:-1.5px; margin:0 0 16px; max-width:740px; }
    .intro { color:#b9cedb; max-width:680px; margin:0; font-size:16px; }
    .success { border:1px solid #2d777b; background:#113b43; color:#a5efe5; padding:13px 17px;
      border-radius:12px; flex-shrink:0; font-weight:750; text-align:center; }
    .success small { display:block; color:#9bccc9; font-weight:400; font-size:11px; }
    main { max-width:1164px; padding:0 22px 42px; margin:-42px auto 0; }
    .stats { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:22px; }
    .stat { background:white; border:1px solid var(--line); border-radius:12px; padding:17px 19px; box-shadow:0 8px 22px #17384a08; min-width:0; }
    .stat span { display:block; color:var(--muted); text-transform:uppercase; font-size:10px; letter-spacing:1.4px; font-weight:750; }
    .stat strong { display:block; margin-top:5px; font-size:clamp(16px,2.2vw,23px); line-height:1.35; overflow-wrap:anywhere; }
    .stat small { display:block; color:var(--muted); font-size:11px; margin-top:5px; }
    section { background:white; border:1px solid var(--line); border-radius:16px; margin:20px 0; padding:24px; }
    h2 { font-size:20px; line-height:1.3; margin:0; letter-spacing:-.35px; }
    .section-heading { display:flex; align-items:center; justify-content:space-between; gap:18px; margin-bottom:16px; }
    .subtle { color:var(--muted); margin:8px 0 0; font-size:13px; }
    .switch { display:inline-flex; padding:4px; background:#edf3f5; border-radius:9px; gap:3px; }
    .view-controls { display:flex; flex-wrap:wrap; gap:10px; justify-content:flex-end; }
    .timeline { display:flex; align-items:center; flex-wrap:wrap; gap:12px; margin:16px 0;
      padding:13px 15px; background:#eef5f6; border:1px solid #dce9ed; border-radius:10px; }
    .timeline button { background:#0d6570; color:white; border:0; border-radius:7px; padding:9px 15px;
      font:650 13px system-ui,sans-serif; cursor:pointer; min-width:78px; }
    .timeline input { flex:1; min-width:130px; accent-color:#147e84; cursor:pointer; }
    .timeline label,.timeline span { font-size:12px; color:var(--muted); }
    .timeline select { margin-left:6px; border:1px solid #bfd3db; border-radius:6px; padding:6px; background:white; color:var(--ink); }
    .switch button { border:0; border-radius:6px; background:transparent; color:var(--muted); padding:8px 13px;
      font:650 13px system-ui,sans-serif; cursor:pointer; white-space:nowrap; }
    .switch button[aria-pressed="true"] { background:var(--navy); color:#e8faff; box-shadow:0 2px 5px #08192720; }
    button:focus-visible,a:focus-visible { outline:3px solid #49b7c4; outline-offset:3px; }
    figure { margin:0; } .field-image { display:block; width:100%; height:auto; border-radius:12px; }
    figcaption { font-size:12px; color:var(--muted); padding:12px 3px 0; }
    .figure-links { display:flex; flex-wrap:wrap; gap:18px; margin-top:12px; font-size:13px; }
    .table-scroll { overflow-x:auto; } table { width:100%; border-collapse:collapse; text-align:left; font-size:13px; }
    thead th { color:var(--muted); font-weight:600; border-bottom:1px solid var(--line); font-size:11px; padding:12px 10px; white-space:nowrap; }
    tbody th,tbody td { border-bottom:1px solid #ecf1f3; padding:16px 10px; font-variant-numeric:tabular-nums; }
    tbody th { font-weight:650; min-width:190px; } tbody small { display:block; font-size:11px; font-weight:400; color:var(--muted); margin-top:4px; }
    .status { color:#11695d; background:#e4f5ef; border-radius:5px; padding:3px 8px; font-weight:750; font-size:11px; }
    .tolerances { font-size:12px; color:var(--muted); margin:14px 0 0; }
    .command { border:1px solid var(--line); border-radius:10px; margin-top:12px; overflow:hidden; }
    .command-heading { display:flex; justify-content:space-between; gap:12px; padding:10px 14px; background:#f3f7f8; font-size:12px; }
    .command-heading span { color:var(--muted); } code { display:block; padding:14px; overflow-wrap:anywhere; white-space:pre-wrap; font:12px/1.8 ui-monospace,SFMono-Regular,Consolas,monospace; }
    .scope { background:#eaf3f4; border-left:3px solid #2a8d95; padding:14px 17px; margin-top:18px; font-size:13px; border-radius:0 8px 8px 0; }
    .scope strong { display:block; margin-bottom:4px; } footer { color:var(--muted); text-align:center; font-size:12px; margin:26px 0 0; }
    @media(max-width:720px) { header { padding-top:27px; padding-bottom:69px; } .hero { align-items:flex-start; flex-direction:column; gap:18px; }
      .success { text-align:left; padding:9px 13px; font-size:13px; } .success small { display:inline; margin-left:10px; }
      .stats { grid-template-columns:repeat(2,1fr); } section { padding:17px 13px; } .section-heading { align-items:flex-start; flex-direction:column; gap:12px; }
      .view-controls { width:100%; } .switch { width:100%; } .switch button { flex:1; } .command-heading { flex-direction:column; gap:4px; }
      main { padding-left:14px; padding-right:14px; } }
  </style>
</head>
<body>
  <header>
    <div class="eyebrow">Airfoil CFD · local numerical demonstration</div>
    <div class="hero"><div><h1>A real flow field.<br>A checked numerical result.</h1>
      <p class="intro">Play the recorded solver states and inspect each state's complete-field agreement with the serial solver. Every frame is real output; everything works offline.</p></div>
      <div class="success">✓ Numerical checks passed<small>within recorded tolerances</small></div></div>
  </header>
  <main>
    <div class="stats">
      <div class="stat"><span>Airfoil / interior grid</span><strong>NACA __AIRFOIL__</strong><small>__NX__ × __NY__ cells</small></div>
      <div class="stat"><span>Actual simulation TIME</span><strong id="actual-time" title="__TIME__">__TIME_DISPLAY__</strong><small>From the displayed frame's VTR metadata</small></div>
      <div class="stat"><span>Actual CYCLE</span><strong id="actual-cycle">__CYCLE__</strong><small>__DT__ fixed timestep</small></div>
      <div class="stat"><span>Verification scope</span><strong>2 complete fields</strong><small>Velocity and pressure, including ghosts</small></div>
    </div>
    <section>
      <div class="section-heading"><div><h2>Explore the computed field</h2><p class="subtle">Real grid samples · fixed colour limits across views · no smoothing</p></div>
        <div class="view-controls"><div class="switch" role="group" aria-label="Displayed field"><button type="button" data-field="u" aria-pressed="true">Velocity u</button><button type="button" data-field="p" aria-pressed="false">Pressure p</button></div>
        <div class="switch" role="group" aria-label="Displayed region"><button type="button" data-view="detail" aria-pressed="true">Airfoil detail</button><button type="button" data-view="full" aria-pressed="false">Full domain</button></div></div></div>
      <div class="timeline"><button id="play-pause" type="button" disabled>Play</button>
        <input id="frame-slider" type="range" min="0" max="__FRAME_MAX__" value="__FRAME_MAX__" step="1" aria-label="Recorded simulation state" disabled>
        <span id="frame-position">Frame __FRAME_COUNT__ / __FRAME_COUNT__</span>
        <label>Playback<select id="playback-speed" aria-label="Playback speed" disabled><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></label></div>
      <figure><img id="field-image" class="field-image" src="velocity-detail.svg" alt="Airfoil detail of actual interior horizontal velocity u from the verified final serial-opt solver result">
        <figcaption>Playback switches between verified saved states without interpolation or invented motion. Each embedded heatmap pixel is one real grid sample, displayed without smoothing. All frames and both views share one visible-fluid colour scale per field. Solid and boundary samples use the exact discrete setup mask and appear neutral grey. The thin dashed line is only an analytic NACA reference. Ghost layers remain part of numerical comparisons.</figcaption></figure>
      <div class="figure-links"><a id="field-download" href="velocity-detail.svg" download>Download displayed SVG</a><a href="pressure-detail.svg">Pressure detail SVG</a><a href="velocity.svg">Full velocity SVG</a><a href="pressure.svg">Full pressure SVG</a></div>
      <noscript><p class="subtle">JavaScript is disabled: the final verified state is displayed. Use the SVG links to inspect and download its velocity and pressure fields.</p></noscript>
    </section>
    <section>
      <div class="section-heading"><div><h2>Current frame: full-field numerical agreement</h2><p class="subtle">Serial compared with serial-opt at the displayed TIME and CYCLE, using strict VTR validation. Every stored value, including ghost layers, is compared.</p></div><a href="report.json">Open local report</a></div>
      <div class="table-scroll"><table><thead><tr><th>Compared field</th><th>Maximum absolute error</th><th>Relative L2 error</th><th>RMS absolute error</th><th>Result</th></tr></thead><tbody>__COMPARISON_ROWS__</tbody></table></div>
      <p class="tolerances">Recorded tolerances: absolute <span id="tol-absolute">__ATOL__</span>; componentwise relative <span id="tol-relative">__RTOL__</span>; relative L2 <span id="tol-l2">__L2TOL__</span>. Grid coordinates, TIME and CYCLE must also match.</p>
      <div class="scope"><strong>Numerical / functional demonstration</strong>This demonstrates that two implementations agree within tolerance at every saved state. It is not a performance benchmark or physical validation of the CFD model. No speedup is measured. Requested end time: __END_TIME__; displayed saved TIME: <span id="actual-time-copy" title="__TIME__">__TIME_DISPLAY__</span>.</div>
    </section>
    <section><h2>Inspect the run evidence</h2><p class="subtle">Commands omit machine-specific paths. Solver logs and the local report are retained beside this page.</p>__COMMANDS__</section>
    <footer>Airfoil CFD · generated from actual solver output · no network connection or plotting packages required</footer>
  </main>
  <script type="application/json" id="timeline-data">__TIMELINE_JSON__</script>
  <script>
    const image = document.getElementById('field-image');
    const download = document.getElementById('field-download');
    const fieldButtons = document.querySelectorAll('[data-field]');
    const viewButtons = document.querySelectorAll('[data-view]');
    const frames = JSON.parse(document.getElementById('timeline-data').textContent);
    const slider = document.getElementById('frame-slider');
    const playButton = document.getElementById('play-pause');
    const speed = document.getElementById('playback-speed');
    let activeField = 'u';
    let activeView = 'detail';
    let frameIndex = frames.length - 1;
    let playing = false;
    let timer = null;
    function scientific(value) { return value === null ? 'not defined' : Number(value).toExponential(3); }
    function updateField() {
      const frame = frames[frameIndex];
      const pressure = activeField === 'p';
      const file = frame.files[activeField][activeView];
      image.src = file;
      image.alt = (activeView === 'detail' ? 'Airfoil detail: ' : 'Full domain: ') + (pressure ? 'actual interior pressure p' : 'actual interior horizontal velocity u') + ' at TIME ' + frame.time_display + ', CYCLE ' + frame.cycle;
      download.href = file;
      fieldButtons.forEach(item => item.setAttribute('aria-pressed', item.dataset.field === activeField ? 'true' : 'false'));
      viewButtons.forEach(item => item.setAttribute('aria-pressed', item.dataset.view === activeView ? 'true' : 'false'));
      ['actual-time', 'actual-time-copy'].forEach(id => {
        const element = document.getElementById(id);
        element.textContent = frame.time_display;
        element.title = frame.time_exact;
      });
      document.getElementById('actual-cycle').textContent = frame.cycle;
      document.getElementById('frame-position').textContent = 'Frame ' + (frameIndex + 1) + ' / ' + frames.length;
      slider.value = frameIndex;
      ['uv', 'p'].forEach(field => {
        const result = frame.fields[field];
        document.getElementById(field + '-max').textContent = scientific(result.max_abs_error);
        document.getElementById(field + '-l2').textContent = scientific(result.relative_l2_error);
        document.getElementById(field + '-rms').textContent = scientific(result.rms_abs_error);
        document.getElementById(field + '-values').textContent = Number(result.values).toLocaleString('en-US');
      });
      document.getElementById('tol-absolute').textContent = scientific(frame.tolerances.absolute);
      document.getElementById('tol-relative').textContent = scientific(frame.tolerances.relative);
      document.getElementById('tol-l2').textContent = scientific(frame.tolerances.relative_l2);
    }
    function scheduleNext() {
      clearTimeout(timer);
      if (!playing) return;
      const next = (frameIndex + 1) % frames.length;
      const delta = next === 0 ? frames[frames.length - 1].time - frames[frames.length - 2].time : frames[next].time - frames[frameIndex].time;
      // Viewing pace only: 1× displays 0.1 units of saved simulation time per second.
      // Every state and its TIME remain unmodified; no intermediate state is invented.
      timer = setTimeout(() => { frameIndex = next; updateField(); scheduleNext(); }, delta / .1 / Number(speed.value) * 1000);
    }
    function setPlaying(value) {
      playing = value && frames.length > 1;
      playButton.textContent = playing ? 'Pause' : 'Play';
      playButton.setAttribute('aria-pressed', playing ? 'true' : 'false');
      scheduleNext();
    }
    fieldButtons.forEach(button => button.addEventListener('click', () => {
      activeField = button.dataset.field;
      updateField();
    }));
    viewButtons.forEach(button => button.addEventListener('click', () => {
      activeView = button.dataset.view;
      updateField();
    }));
    playButton.disabled = slider.disabled = speed.disabled = frames.length < 2;
    playButton.addEventListener('click', () => setPlaying(!playing));
    slider.addEventListener('input', () => { setPlaying(false); frameIndex = Number(slider.value); updateField(); });
    speed.addEventListener('change', scheduleNext);
    const reducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (frames.length > 1 && !reducedMotion) { frameIndex = 0; updateField(); setPlaying(true); }
  </script>
</body>
</html>
"""
    substitutions = {
        "__AIRFOIL__": escape(airfoil), "__NX__": str(nx), "__NY__": str(ny),
        "__TIME__": actual_time, "__TIME_DISPLAY__": escape(f"{grid.time:.6g}"),
        "__CYCLE__": str(grid.cycle), "__DT__": timestep,
        "__END_TIME__": requested_time, "__COMPARISON_ROWS__": "".join(rows),
        "__ATOL__": escape(_format(tolerances["absolute"])),
        "__RTOL__": escape(_format(tolerances["relative"])),
        "__L2TOL__": escape(_format(tolerances["relative_l2"])),
        "__COMMANDS__": "".join(commands),
        "__FRAME_MAX__": str(len(timeline) - 1), "__FRAME_COUNT__": str(len(timeline)),
        "__TIMELINE_JSON__": json.dumps(timeline, allow_nan=False).replace("<", "\\u003c"),
    }
    for key, value in substitutions.items():
        html = html.replace(key, value)
    page = output_dir / "index.html"
    page.write_text(html, encoding="utf-8")
    return page
