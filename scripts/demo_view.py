"""Render a verified solver result as an offline page using the standard library.

Plots retain the solver's staggered sample positions; they do not infer a solid
mask, interpolate fields, or claim physical validation. No timing is displayed.
"""

from __future__ import annotations

from html import escape
import json
import math
from pathlib import Path, PureWindowsPath
import shlex
from urllib.parse import quote

from compare_vtk import Grid


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


def _field_svg(grid: Grid, field: str, nx: int, ny: int, airfoil: str) -> str:
    dx = grid.coordinates[0][1] - grid.coordinates[0][0]
    dy = grid.coordinates[1][1] - grid.coordinates[1][0]
    if dx <= 0 or dy <= 0:
        raise ValueError("The demonstration requires positive grid spacing")
    domain_x, domain_y = nx * dx, ny * dy
    pitch = nx + 2
    data = grid.fields["uv" if field == "u" else "p"].values
    samples = [data[3 * (j * pitch + i)] if field == "u" else data[j * pitch + i]
               for j in range(1, ny + 1) for i in range(1, nx + 1)]
    if not samples or not all(math.isfinite(value) for value in samples):
        raise ValueError("The demonstration requires complete finite interior data")
    minimum, maximum = min(samples), max(samples)
    limit = max(abs(minimum), abs(maximum)) or 1.0
    left, top, width = 84.0, 84.0, 952.0
    height = width * domain_y / domain_x
    canvas_height = height + 236.0
    scale_x, scale_y = width / domain_x, height / domain_y
    label = "Horizontal velocity u" if field == "u" else "Pressure p"
    position = "x = i·dx, y = (j − ½)·dy" if field == "u" else "x = (i − ½)·dx, y = (j − ½)·dy"
    title = escape(f"{label} · NACA {airfoil} · {nx} × {ny} · TIME {grid.time!r}, CYCLE {grid.cycle}")
    description = escape(
        f"Actual maintained serial-opt solver samples; {nx * ny} interior values. "
        f"Ghost layers are excluded. Sample coordinates: {position}. "
        "No obstacle mask is inferred. The image is a numerical diagnostic, not physical validation."
    )
    metadata = escape(json.dumps({"field": field, "nx": nx, "ny": ny,
                                  "time": grid.time, "cycle": grid.cycle,
                                  "dx": dx, "dy": dy,
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
        f'<text x="{left}" y="33" font-size="20" font-weight="700">{escape(label)}</text>',
        f'<text x="{left}" y="58" font-size="12" fill="#576f7c">NACA {escape(airfoil)} · '
        f'{nx} × {ny} interior samples · actual TIME {grid.time!r} · CYCLE {grid.cycle}</text>',
        f'<rect x="{left}" y="{top}" width="{width}" height="{height:.6f}" fill="#edf1f2"/>',
        '<g clip-path="url(#domain)" shape-rendering="crispEdges">',
    ]
    for j in range(1, ny + 1):
        # SVG y grows downwards; physical y grows upwards.
        svg_y = top + (domain_y - j * dy) * scale_y
        for i in range(1, nx + 1):
            index = (j - 1) * nx + i - 1
            value = samples[index]
            sample_x = i * dx if field == "u" else (i - .5) * dx
            svg_x = left + (sample_x - dx / 2) * scale_x
            hover = escape(f"i={i}, j={j}; x={sample_x:.8g}, y={(j - .5) * dy:.8g}; {field}={value:.12g}")
            pieces.append(
                f'<rect class="sample" data-i="{i}" data-j="{j}" data-value="{value:.17g}" '
                f'x="{svg_x:.6f}" y="{svg_y:.6f}" width="{dx * scale_x:.6f}" '
                f'height="{dy * scale_y:.6f}" fill="{_colour(value, limit)}"><title>{hover}</title></rect>'
            )
    pieces.extend([
        '</g>',
        f'<rect x="{left}" y="{top}" width="{width}" height="{height:.6f}" fill="none" stroke="#acbfc9"/>',
    ])
    baseline = top + height
    for tick in range(5):
        x = left + width * tick / 4
        pieces.append(f'<path d="M{x:.3f} {baseline:.3f}v6" stroke="#7b929f"/>')
        pieces.append(f'<text x="{x:.3f}" y="{baseline + 25:.3f}" text-anchor="middle" font-size="12">{domain_x * tick / 4:.3g}</text>')
    for tick in range(3):
        y = top + height * (1 - tick / 2)
        pieces.append(f'<path d="M{left - 6:.3f} {y:.3f}h6" stroke="#7b929f"/>')
        pieces.append(f'<text x="{left - 13:.3f}" y="{y + 4:.3f}" text-anchor="end" font-size="12">{domain_y * tick / 2:.3g}</text>')
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
        f'Actual interior range: {escape(_format(minimum))} to {escape(_format(maximum))} · solver units</text>',
        f'<text x="{left}" y="{legend_y + 56:.3f}" font-size="12" fill="#576f7c">'
        f'Interior samples only · {escape(position)} · no inferred solid mask</text>',
        '</g></svg>',
    ])
    return "\n".join(pieces) + "\n"


def write_demo_page(output_dir: Path, grid: Grid, comparison: dict,
                    runs: list[dict], settings: dict) -> Path:
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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "velocity.svg").write_text(_field_svg(grid, "u", nx, ny, airfoil), encoding="utf-8")
    (output_dir / "pressure.svg").write_text(_field_svg(grid, "p", nx, ny, airfoil), encoding="utf-8")

    rows = []
    for name, label in (("uv", "Velocity vector (u, v, 0)"), ("p", "Pressure p")):
        result = comparison["fields"][name]
        rows.append(
            f'<tr><th scope="row">{label}<small>{result["values"]:,} values, including ghost layers</small></th>'
            f'<td>{escape(_format(result["max_abs_error"]))}</td>'
            f'<td>{escape(_format(result["relative_l2_error"]))}</td>'
            f'<td>{escape(_format(result["rms_abs_error"]))}</td>'
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
      .switch { width:100%; } .switch button { flex:1; } .command-heading { flex-direction:column; gap:4px; }
      main { padding-left:14px; padding-right:14px; } }
  </style>
</head>
<body>
  <header>
    <div class="eyebrow">Airfoil CFD · local numerical demonstration</div>
    <div class="hero"><div><h1>A real flow field.<br>A checked numerical result.</h1>
      <p class="intro">Explore the final serial-opt solution and its complete-field agreement with the serial solver. Everything on this page works offline.</p></div>
      <div class="success">✓ Numerical checks passed<small>within recorded tolerances</small></div></div>
  </header>
  <main>
    <div class="stats">
      <div class="stat"><span>Airfoil / interior grid</span><strong>NACA __AIRFOIL__</strong><small>__NX__ × __NY__ cells</small></div>
      <div class="stat"><span>Actual simulation TIME</span><strong title="__TIME__">__TIME_DISPLAY__</strong><small>From the final VTR metadata</small></div>
      <div class="stat"><span>Actual CYCLE</span><strong>__CYCLE__</strong><small>__DT__ fixed timestep</small></div>
      <div class="stat"><span>Verification scope</span><strong>2 complete fields</strong><small>Velocity and pressure, including ghosts</small></div>
    </div>
    <section>
      <div class="section-heading"><div><h2>Explore the computed field</h2><p class="subtle">Interior solver samples · solver units · no inferred solid mask</p></div>
        <div class="switch" role="group" aria-label="Displayed field"><button type="button" data-field="u" aria-pressed="true">Velocity u</button><button type="button" data-field="p" aria-pressed="false">Pressure p</button></div></div>
      <figure><img id="field-image" class="field-image" src="velocity.svg" alt="Actual interior horizontal velocity u from the verified final serial-opt solver result">
        <figcaption>Only interior samples are plotted; ghost layers are excluded from these images. Velocity and pressure retain their own staggered sample positions. Values inside the stored obstacle region are not masked or interpreted as fluid.</figcaption></figure>
      <div class="figure-links"><a id="field-download" href="velocity.svg" download>Download displayed SVG</a><a href="pressure.svg">Open pressure SVG</a><a href="velocity.svg">Open velocity SVG</a></div>
      <noscript><p class="subtle">JavaScript is disabled. Open either SVG using the links above to view both fields.</p></noscript>
    </section>
    <section>
      <div class="section-heading"><div><h2>Full-field numerical agreement</h2><p class="subtle">Serial compared with serial-opt using strict VTR validation. The comparison includes every stored value, including ghost layers.</p></div><a href="report.json">Open local report</a></div>
      <div class="table-scroll"><table><thead><tr><th>Compared field</th><th>Maximum absolute error</th><th>Relative L2 error</th><th>RMS absolute error</th><th>Result</th></tr></thead><tbody>__COMPARISON_ROWS__</tbody></table></div>
      <p class="tolerances">Recorded tolerances: absolute __ATOL__; componentwise relative __RTOL__; relative L2 __L2TOL__. Grid coordinates, TIME and CYCLE must also match.</p>
      <div class="scope"><strong>Numerical / functional demonstration</strong>This demonstrates that two implementations agree within tolerance. It is not a performance benchmark or physical validation of the CFD model. No speedup is measured. Requested end time: __END_TIME__; actual saved TIME: __TIME__.</div>
    </section>
    <section><h2>Inspect the run evidence</h2><p class="subtle">Commands omit machine-specific paths. Solver logs and the local report are retained beside this page.</p>__COMMANDS__</section>
    <footer>Airfoil CFD · generated from actual solver output · no network connection or plotting packages required</footer>
  </main>
  <script>
    const image = document.getElementById('field-image');
    const download = document.getElementById('field-download');
    const buttons = document.querySelectorAll('[data-field]');
    buttons.forEach(button => button.addEventListener('click', () => {
      const pressure = button.dataset.field === 'p';
      const file = pressure ? 'pressure.svg' : 'velocity.svg';
      image.src = file;
      image.alt = pressure ? 'Actual interior pressure p from the verified final serial-opt solver result' : 'Actual interior horizontal velocity u from the verified final serial-opt solver result';
      download.href = file;
      buttons.forEach(item => item.setAttribute('aria-pressed', item === button ? 'true' : 'false'));
    }));
  </script>
</body>
</html>
"""
    substitutions = {
        "__AIRFOIL__": escape(airfoil), "__NX__": str(nx), "__NY__": str(ny),
        "__TIME__": actual_time, "__TIME_DISPLAY__": escape(f"{grid.time:.8g}"),
        "__CYCLE__": str(grid.cycle), "__DT__": timestep,
        "__END_TIME__": requested_time, "__COMPARISON_ROWS__": "".join(rows),
        "__ATOL__": escape(_format(tolerances["absolute"])),
        "__RTOL__": escape(_format(tolerances["relative"])),
        "__L2TOL__": escape(_format(tolerances["relative_l2"])),
        "__COMMANDS__": "".join(commands),
    }
    for key, value in substitutions.items():
        html = html.replace(key, value)
    page = output_dir / "index.html"
    page.write_text(html, encoding="utf-8")
    return page
