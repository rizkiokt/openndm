#!/usr/bin/env python3
"""IAEA 3D PWR benchmark, quarter core, two groups.

The 2D radial map extruded to 380 cm: a 20 cm bottom reflector, 340 cm of
active core, and a 20 cm top reflector, with vacuum axial boundaries.

The control rods enter from the top and stop with their tips 80 cm above the
bottom of the active core, so they occupy the upper 260 cm of the core
(z = 100 to 360 cm) at the nine rodded positions, and continue through the top
reflector as the rodded reflector material. Reading the 80 cm the other way
round, as an insertion depth measured down from the top of the core, leaves
k_eff roughly 1700 pcm high.

The axial mesh always places the rod tip exactly on a plane boundary. Smearing
a rod tip across a node is worth tens of pcm and shows up as an axial mesh
that refuses to converge monotonically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from common import (
    AXIAL_REFLECTOR,
    CORE_HEIGHT,
    IAEA_BOUNDARIES,
    IAEA_ORDER,
    ROD_TIP_HEIGHT,
    benchmark_settings,
    check_radial_map,
    iaea_library,
    radial_map_to_compositions,
    report,
)

import openndm

#: Published reference eigenvalue for the benchmark as specified.
PUBLISHED_K_EFF = 1.02903

FUEL_2 = IAEA_ORDER.index("fuel_2")
FUEL_2_RODDED = IAEA_ORDER.index("fuel_2_rodded")
REFLECTOR = IAEA_ORDER.index("reflector")
REFLECTOR_RODDED = IAEA_ORDER.index("reflector_rodded")


def axial_mesh(dz_target: float = 20.0) -> tuple[list[float], int]:
    """Axial widths with the rod tip on a plane boundary.

    Returns the widths including both reflectors, and the number of unrodded
    active planes below the rod tip.
    """
    unrodded = ROD_TIP_HEIGHT
    rodded = CORE_HEIGHT - ROD_TIP_HEIGHT
    n_unrodded = max(1, round(unrodded / dz_target))
    n_rodded = max(1, round(rodded / dz_target))
    dz = (
        [AXIAL_REFLECTOR]
        + [unrodded / n_unrodded] * n_unrodded
        + [rodded / n_rodded] * n_rodded
        + [AXIAL_REFLECTOR]
    )
    return dz, n_unrodded


def build(dz_target: float = 20.0, subdivide: int = 1):
    """Extrude the radial map into the three axial zones."""
    check_radial_map()
    radial = radial_map_to_compositions()
    dz, n_unrodded = axial_mesh(dz_target)
    planes = len(dz)
    rodded_position = radial == FUEL_2_RODDED
    outside = radial == openndm.INACTIVE

    core = np.empty((planes, 9, 9), dtype=int)
    # Bottom reflector: no rod reaches this far down.
    core[0] = np.where(outside, openndm.INACTIVE, REFLECTOR)
    # Top reflector: the rods pass through it at the rodded positions.
    core[-1] = np.where(
        outside,
        openndm.INACTIVE,
        np.where(rodded_position, REFLECTOR_RODDED, REFLECTOR),
    )
    for k in range(1, planes - 1):
        # Planes at or below the rod tip see plain fuel 2 at the rod positions.
        below_tip = k <= n_unrodded
        core[k] = np.where(
            rodded_position & below_tip, FUEL_2, radial
        )

    geometry = openndm.Geometry.from_lattice(
        core,
        pitch=(20.0, 20.0, 20.0),
        dz=dz,
        subdivide=(subdivide, subdivide, 1),
        boundaries={**IAEA_BOUNDARIES, "z_min": "vacuum", "z_max": "vacuum"},
        outside="zero_flux",
    )
    return geometry, iaea_library()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dz", type=float, default=20.0, help="target axial node height [cm]"
    )
    parser.add_argument("--subdivide", type=int, default=1)
    parser.add_argument("--statepoint", type=Path, default=None)
    args = parser.parse_args()

    geometry, library = build(args.dz, args.subdivide)
    settings = benchmark_settings()
    nz = geometry.shape[0]
    print(
        f"IAEA-3D: {geometry.n_nodes} nodes, {nz} axial planes, "
        f"rod tips {ROD_TIP_HEIGHT:.0f} cm above the core bottom"
    )

    result = None
    for kernel in ("fdm", "nem", "sanm"):
        model = openndm.Model(geometry, library, settings)
        result = model.solve(kernel=kernel)
        report(f"IAEA-3D dz={args.dz:g}", result, PUBLISHED_K_EFF)

    axial = result.axial_power()
    print("\naxial power profile (SANM):")
    peak = max(axial.max(), 1e-12)
    for k, value in enumerate(axial):
        bar = "#" * round(40 * value / peak)
        print(f"  plane {k:2d}  {value:6.4f}  {bar}")

    if args.statepoint:
        model = openndm.Model(geometry, library, settings)
        openndm.write_statepoint(args.statepoint, model.solve(), model)
        print(f"\nwrote {args.statepoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
