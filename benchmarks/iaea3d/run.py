#!/usr/bin/env python3
"""IAEA 3D PWR benchmark, quarter core, two groups.

The 2D radial map extruded to 380 cm: a 20 cm bottom reflector, 340 cm of
active core with the control bank inserted 80 cm from the top, and a 20 cm
top reflector. Vacuum axial boundaries.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from common import (
    IAEA_BOUNDARIES,
    IAEA_ORDER,
    iaea_library,
    radial_map_to_compositions,
    report,
)

import openndm

#: Published reference eigenvalue for the benchmark as specified.
PUBLISHED_K_EFF = 1.02903

ROD_INSERTION_CM = 80.0
CORE_HEIGHT_CM = 340.0
REFLECTOR_CM = 20.0

FUEL_2 = IAEA_ORDER.index("fuel_2")
FUEL_2_RODDED = IAEA_ORDER.index("fuel_2_rodded")
REFLECTOR = IAEA_ORDER.index("reflector")
REFLECTOR_RODDED = IAEA_ORDER.index("reflector_rodded")


def build(planes: int = 19, subdivide: int = 1):
    """Extrude the radial map, banding the axial mesh into the three zones."""
    radial = radial_map_to_compositions()
    dz_reflector = REFLECTOR_CM
    active_planes = planes - 2
    dz_active = CORE_HEIGHT_CM / active_planes
    dz = [dz_reflector] + [dz_active] * active_planes + [dz_reflector]

    core = np.repeat(radial[np.newaxis, :, :], planes, axis=0)
    # Bottom and top reflector planes: everything becomes reflector, keeping
    # the rodded reflector where the bank passes through the top plane.
    core[0] = np.where(core[0] == openndm.INACTIVE, openndm.INACTIVE, REFLECTOR)
    core[-1] = np.where(
        core[-1] == openndm.INACTIVE,
        openndm.INACTIVE,
        np.where(radial == FUEL_2_RODDED, REFLECTOR_RODDED, REFLECTOR),
    )
    # The bank is withdrawn from the lower part of the core: the rodded map
    # positions revert to plain fuel below the insertion depth.
    height = np.cumsum(dz)
    top_of_core = REFLECTOR_CM + CORE_HEIGHT_CM
    for k in range(1, planes - 1):
        if height[k] <= top_of_core - ROD_INSERTION_CM:
            core[k] = np.where(core[k] == FUEL_2_RODDED, FUEL_2, core[k])

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
    parser.add_argument("--planes", type=int, default=19)
    parser.add_argument("--subdivide", type=int, default=1)
    args = parser.parse_args()

    geometry, library = build(args.planes, args.subdivide)
    settings = openndm.Settings(verbosity=0)
    print(f"IAEA-3D: {geometry.n_nodes} nodes, {args.planes} axial planes")

    result = None
    for kernel in ("fdm", "nem", "sanm"):
        model = openndm.Model(geometry, library, settings)
        result = model.solve(kernel=kernel)
        report(f"IAEA-3D planes={args.planes}", result, PUBLISHED_K_EFF)

    axial = result.axial_power()
    print("\naxial power profile (SANM):")
    for k, value in enumerate(axial):
        bar = "#" * round(40 * value / max(axial.max(), 1e-12))
        print(f"  plane {k:2d}  {value:6.4f}  {bar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
