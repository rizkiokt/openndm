#!/usr/bin/env python3
"""C-2: assembly discontinuity factors from a single reflected assembly.

The second of the OpenMC-coupling validation cases in the specification. A
single assembly with reflective boundaries is an infinite lattice, so the
homogeneous solution of the homogenised assembly is flat and equal to its
volume average. The discontinuity factor is then just the ratio of the
heterogeneous surface flux to that average, which makes two things checkable
without any reference calculation:

* a **homogeneous** assembly must return factors of exactly 1.0, which
  verifies the slab geometry, the width normalisation and the group ordering
  in :func:`openndm.gc.compute_adf`;
* a **heterogeneous** assembly must return factors above 1.0 in the thermal
  group, because the surface of a pin lattice sits in the water gap where the
  thermal flux peaks, and near 1.0 in the fast group where it does not.

Run with::

    OPENMC_CROSS_SECTIONS=/path/to/cross_sections.xml python c2_assembly_adf.py
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import openmc
import openmc.mgxs

from openndm.gc import add_adf_tallies, compute_adf

GROUP_EDGES = [0.0, 0.625, 20.0e6]
PITCH = 1.26
LATTICE_SHAPE = (5, 5)
SLAB_FRACTION = 0.05


#: Number densities in atoms/barn-cm.
FUEL_NUCLIDES = {"U235": 8.0e-4, "U238": 2.2e-2, "O16": 4.6e-2}
WATER_NUCLIDES = {"H1": 4.9e-2, "O16": 2.4e-2}
PIN_RADIUS = 0.41


def _materials():
    """Fuel, water, and their exact volume-homogenised mixture.

    The mixture is formed by hand rather than with
    ``Material.mix_materials``, which refuses to mix materials carrying an
    S(alpha,beta) table. Volume weighting the number densities is what that
    call would have done anyway, and it keeps the thermal scattering law on
    the hydrogen where it belongs.
    """
    fuel = openmc.Material(name="UO2")
    for nuclide, density in FUEL_NUCLIDES.items():
        fuel.add_nuclide(nuclide, density)
    fuel.set_density("sum")

    water = openmc.Material(name="water")
    for nuclide, density in WATER_NUCLIDES.items():
        water.add_nuclide(nuclide, density)
    water.add_s_alpha_beta("c_H_in_H2O")
    water.set_density("sum")

    fraction = np.pi * PIN_RADIUS ** 2 / (PITCH * PITCH)
    mixed: dict[str, float] = {}
    for nuclide, density in FUEL_NUCLIDES.items():
        mixed[nuclide] = mixed.get(nuclide, 0.0) + density * fraction
    for nuclide, density in WATER_NUCLIDES.items():
        mixed[nuclide] = mixed.get(nuclide, 0.0) + density * (1.0 - fraction)

    blend = openmc.Material(name="homogenised")
    for nuclide, density in mixed.items():
        blend.add_nuclide(nuclide, density)
    blend.add_s_alpha_beta("c_H_in_H2O")
    blend.set_density("sum")
    return fuel, water, blend


def build_model(heterogeneous: bool, particles: int, batches: int):
    fuel, water, blend = _materials()

    if heterogeneous:
        surface = openmc.ZCylinder(r=PIN_RADIUS)
        pin = openmc.Cell(fill=fuel, region=-surface)
        moderator = openmc.Cell(fill=water, region=+surface)
        cell_universe = openmc.Universe(cells=[pin, moderator])
        materials = [fuel, water]
    else:
        cell_universe = openmc.Universe(cells=[openmc.Cell(fill=blend)])
        materials = [blend]

    lattice = openmc.RectLattice()
    lattice.pitch = (PITCH, PITCH)
    extent = (LATTICE_SHAPE[0] * PITCH, LATTICE_SHAPE[1] * PITCH)
    lattice.lower_left = (-0.5 * extent[0], -0.5 * extent[1])
    lattice.universes = [
        [cell_universe] * LATTICE_SHAPE[0] for _ in range(LATTICE_SHAPE[1])
    ]

    box = openmc.model.RectangularParallelepiped(
        -0.5 * extent[0], 0.5 * extent[0],
        -0.5 * extent[1], 0.5 * extent[1],
        -0.5, 0.5,
        boundary_type="reflective",
    )
    root = openmc.Cell(fill=lattice, region=-box)
    geometry = openmc.Geometry([root])

    settings = openmc.Settings()
    settings.run_mode = "eigenvalue"
    settings.particles = particles
    settings.batches = batches
    settings.inactive = max(10, batches // 5)
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Box(
            (-0.5 * extent[0], -0.5 * extent[1], -0.5),
            (0.5 * extent[0], 0.5 * extent[1], 0.5),
        )
    )
    settings.output = {"tallies": False}
    model = openmc.Model(
        geometry=geometry, settings=settings, materials=materials
    )
    model.tallies = openmc.Tallies()
    return model, lattice


def run_case(label, heterogeneous, particles, batches, slab_fraction):
    model, lattice = build_model(heterogeneous, particles, batches)
    groups = openmc.mgxs.EnergyGroups(GROUP_EDGES)
    add_adf_tallies(model, lattice, groups, slab_fraction=slab_fraction)

    workdir = Path(tempfile.mkdtemp(prefix="openndm_c2_"))
    try:
        statepoint = model.run(cwd=str(workdir), output=False)
        with openmc.StatePoint(statepoint) as sp:
            keff = sp.keff
            result = compute_adf(
                sp, slab_fraction=slab_fraction, warn_sigma=1.0
            )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print(f"\n{label}")
    print(f"  k_inf = {keff.nominal_value:.6f} +/- {1e5 * keff.std_dev:.1f} pcm")
    names = ("x_min", "x_max", "y_min", "y_max")
    print(f"  {'face':7s} {'fast':>18s} {'thermal':>18s}")
    for face in range(4):
        cells = " ".join(
            f"{result.values[face, g]:8.5f} +/-{result.std_dev[face, g]:7.5f}"
            for g in range(2)
        )
        print(f"  {names[face]:7s} {cells}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=40000)
    parser.add_argument("--batches", type=int, default=120)
    parser.add_argument("--slab-fraction", type=float, default=SLAB_FRACTION)
    args = parser.parse_args()

    print("C-2 assembly discontinuity factors")
    homogeneous = run_case(
        "homogeneous assembly: every factor must be 1.0",
        False, args.particles, args.batches, args.slab_fraction,
    )
    heterogeneous = run_case(
        "heterogeneous pin lattice: thermal factors must exceed 1.0",
        True, args.particles, args.batches, args.slab_fraction,
    )

    radial = slice(0, 4)
    worst = float(np.abs(homogeneous.values[radial] - 1.0).max())
    sigma = float(homogeneous.std_dev[radial].max())
    print(f"\nhomogeneous: worst deviation from 1.0 = {worst:.5f} "
          f"({worst / max(sigma, 1e-12):.2f} sigma)")
    # Group 0 is the fast group, matching from_mgxs_library and the solver.
    fast = heterogeneous.values[radial, 0]
    thermal = heterogeneous.values[radial, 1]
    print(f"heterogeneous: thermal {thermal.min():.4f}..{thermal.max():.4f}, "
          f"fast {fast.min():.4f}..{fast.max():.4f}")

    ok = worst < max(3.0 * sigma, 0.01)
    if not thermal.min() > 1.0:
        print("FAIL: the surface of a pin lattice is water, so the thermal "
              "factor must exceed one. Check the group ordering.")
        ok = False
    if not fast.max() < 1.0:
        print("FAIL: the fast flux peaks in the fuel, not at the surface, so "
              "the fast factor must fall below one.")
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
