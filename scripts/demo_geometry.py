"""Reconstruct the serial solver's discrete fluid flags from its grid settings.

This follows serial/setup.c, including its column sampling, floor/ceil indexing
and clipping. It is not a smooth standard-NACA surface reconstruction and does
not infer geometry from velocity or pressure. Returned samples include ghosts
in VTR order: j outermost, i innermost; every ghost sample is non-fluid.
"""

from __future__ import annotations

import math


def fluid_mask(nx: int, ny: int, dx: float, dy: float,
               airfoil: str | int) -> list[bool]:
    """Return flag&C_F for the solver geometry, indexed by j*(nx+2)+i.

    Domain lengths are reconstructed as nx*dx and ny*dy. Pressure samples use
    this mask directly; a horizontal-velocity face is between fluid cells only
    when both mask[index] and mask[index + 1] are true.
    """
    if type(nx) is not int or type(ny) is not int or nx <= 0 or ny <= 0:
        raise ValueError("Grid dimensions must be positive integers")
    try:
        dx, dy = float(dx), float(dy)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Grid spacing must be finite and positive") from error
    xlength, ylength = nx * dx, ny * dy
    if any(not math.isfinite(value) or value <= 0 for value in (dx, dy, xlength, ylength)):
        raise ValueError("Grid spacing and domain lengths must be finite and positive")
    if type(airfoil) is int:
        if not 0 <= airfoil <= 9999:
            raise ValueError("Airfoil must be a four-digit NACA specification")
        specification = f"{airfoil:04d}"
    elif isinstance(airfoil, str):
        specification = airfoil
    else:
        raise ValueError("Airfoil must be a four-digit NACA specification")
    if len(specification) != 4 or not specification.isascii() or not specification.isdigit():
        raise ValueError("Airfoil must contain four ASCII digits")

    naca = int(specification)
    thickness = naca % 100
    camber_location = (naca // 100) % 10
    camber = naca // 1000
    m, p, t = camber / 100.0, camber_location / 10.0, thickness / 100.0
    if m != 0.0 and p == 0.0:
        raise ValueError("A cambered airfoil requires a nonzero camber position")

    pitch = nx + 2
    mask = [False] * (pitch * (ny + 2))
    for j in range(1, ny + 1):
        start = j * pitch + 1
        mask[start:start + nx] = [True] * nx

    for i in range(1, nx + 1):
        x = ((xlength / nx) * i) - 0.5
        if x < 0.0 or x > 1.0:
            continue
        # Keep the source expression order rather than replacing powers or
        # shifting to pressure centres: either can alter floor/ceil boundaries.
        y_t = 5.0 * t * ((0.2969 * math.sqrt(x)) - (0.1260 * x)
                         - (0.3516 * x * x) + (0.2843 * x * x * x)
                         - (0.1015 * x * x * x * x))
        if m == 0.0:
            y_c = dyc_dx = 0.0
        elif x <= p:
            y_c = (m / (p * p)) * ((2.0 * p * x) - (x * x))
            dyc_dx = ((2.0 * m) / (p * p)) * (p - x)
        else:
            y_c = (m / ((1.0 - p) * (1.0 - p))) * (
                (1.0 - (2.0 * p)) + (2.0 * p * x) - (x * x))
            dyc_dx = ((2.0 * m) / ((1.0 - p) * (1.0 - p))) * (p - x)
        theta = math.atan(dyc_dx)
        y_u = y_c + (y_t * math.cos(theta))
        y_l = y_c - (y_t * math.cos(theta))
        first = math.floor((y_l + (ylength / 2.0)) * (ny / ylength))
        stop = math.ceil((y_u + (ylength / 2.0)) * (ny / ylength))
        if first < 1:
            first = 1
        if stop > ny + 1:
            stop = ny + 1
        for j in range(first, stop):
            mask[j * pitch + i] = False
    return mask
