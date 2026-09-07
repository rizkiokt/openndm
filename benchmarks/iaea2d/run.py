#!/usr/bin/env python3
"""IAEA 2D PWR benchmark, quarter core, two groups.

The canonical coarse-mesh nodal verification problem: 9x9 assemblies of 20 cm
pitch, reflective on the two symmetry faces and zero flux on the outer
reflector boundary.

Run with ``python run.py``; pass ``--subdivide N`` to refine radially.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from common import (
    IAEA_BOUNDARIES,
    iaea_library,
    radial_map_to_compositions,
    report,
)

import openndm

#: Published fine-mesh eigenvalue for the benchmark as specified.
PUBLISHED_K_EFF = 1.02959


def build(subdivide: int = 1) -> tuple[openndm.Geometry, openndm.XSLibrary]:
    radial = radial_map_to_compositions()
    core = radial[np.newaxis, :, :]
    geometry = openndm.Geometry.from_lattice(
        core,
        pitch=(20.0, 20.0, 20.0),
        subdivide=(subdivide, subdivide, 1),
        boundaries={
            **IAEA_BOUNDARIES,
            "z_min": "reflective",
            "z_max": "reflective",
        },
        outside="zero_flux",
    )
    return geometry, iaea_library()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subdivide", type=int, default=1)
    parser.add_argument("--statepoint", type=Path, default=None)
    args = parser.parse_args()

    geometry, library = build(args.subdivide)
    settings = openndm.Settings(verbosity=0)
    print(f"IAEA-2D: {geometry.n_nodes} nodes, {args.subdivide} node(s)/assembly")

    result = None
    for kernel in ("fdm", "nem", "sanm"):
        model = openndm.Model(geometry, library, settings)
        result = model.solve(kernel=kernel)
        report(f"IAEA-2D sub={args.subdivide}", result, PUBLISHED_K_EFF)

    radial = result.radial_power()
    print("\nradial assembly power (SANM):")
    for row in radial[::-1]:
        print("  " + " ".join("  .  " if v == 0 else f"{v:5.3f}" for v in row))

    if args.statepoint:
        model = openndm.Model(geometry, library, settings)
        openndm.write_statepoint(args.statepoint, model.solve(), model)
        print(f"\nwrote {args.statepoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
