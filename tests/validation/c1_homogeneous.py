#!/usr/bin/env python3
"""C-1: homogeneous consistency between OpenMC and OpenNDM.

The first of the OpenMC-coupling validation cases in the specification. A
uniform infinite medium is run in OpenMC with continuous-energy physics; the
multi-group cross sections tallied from that same run are handed to OpenNDM,
which solves a single fully reflected node. The two eigenvalues must agree.

This isolates the cross section translation path and nothing else. There is no
spatial discretisation, no leakage, no homogenisation error and no
discontinuity factor: any disagreement beyond the Monte Carlo uncertainty is a
defect in the translation.

Run with::

    OPENMC_CROSS_SECTIONS=/path/to/cross_sections.xml python c1_homogeneous.py
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import openmc
import openmc.mgxs

import openndm
from openndm.gc import from_mgxs_library

#: Two-group structure with the usual 0.625 eV thermal cutoff.
GROUP_EDGES_2 = [0.0, 0.625, 20.0e6]
#: Eight-group structure, for the multi-group path.
GROUP_EDGES_8 = [0.0, 0.058, 0.14, 0.28, 0.625, 4.0, 5.53e3, 8.21e5, 20.0e6]


def build_model(temperature: float = 900.0, particles: int = 20000,
                batches: int = 60, inactive: int = 15) -> openmc.Model:
    """A homogeneous UO2/water mixture in a fully reflected cube."""
    fuel = openmc.Material(name="homogenised UO2 and water")
    fuel.add_nuclide("U235", 8.0e-4)
    fuel.add_nuclide("U238", 1.9e-2)
    fuel.add_nuclide("O16", 4.6e-2)
    fuel.add_nuclide("H1", 4.0e-2)
    fuel.set_density("sum")
    fuel.temperature = temperature

    box = openmc.model.RectangularParallelepiped(
        -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, boundary_type="reflective"
    )
    cell = openmc.Cell(fill=fuel, region=-box)
    geometry = openmc.Geometry([cell])

    settings = openmc.Settings()
    settings.run_mode = "eigenvalue"
    settings.particles = particles
    settings.batches = batches
    settings.inactive = inactive
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Box((-10, -10, -10), (10, 10, 10))
    )
    settings.output = {"tallies": False}
    return openmc.Model(geometry=geometry, settings=settings, materials=[fuel])


def build_mgxs_library(model: openmc.Model, edges) -> openmc.mgxs.Library:
    """Instrument the model with everything OpenNDM ingests."""
    groups = openmc.mgxs.EnergyGroups(edges)
    library = openmc.mgxs.Library(model.geometry)
    library.energy_groups = groups
    library.domain_type = "material"
    library.domains = model.materials
    library.mgxs_types = [
        "total",
        "absorption",
        "nu-fission",
        "kappa-fission",
        "chi",
        "transport",
        "diffusion-coefficient",
        "inverse-velocity",
        # Both scattering matrices: their row sums differ by the (n,xn)
        # multiplicity, which OpenNDM subtracts from absorption so the extra
        # neutrons appear in the balance. Tallying only one is worth 230 pcm.
        "consistent nu-scatter matrix",
        "consistent scatter matrix",
    ]
    library.by_nuclide = False
    library.build_library()
    # Renamed in OpenMC 0.15; support both so this runs on stable and develop.
    add = getattr(library, "add_to_tallies", None) or library.add_to_tallies_file
    add(model.tallies, merge=True)
    return library


def solve_with_openndm(library: openmc.mgxs.Library, **kwargs) -> openndm.Result:
    """Translate and solve a single fully reflected node."""
    xslib = from_mgxs_library(library, **kwargs)
    geometry = openndm.Geometry.from_lattice(
        np.zeros((1, 1, 1), dtype=int),
        pitch=20.0,
        boundaries=dict.fromkeys(
            ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"], "reflective"
        ),
    )
    settings = openndm.Settings(
        verbosity=0,
        k_tolerance=1.0e-12,
        fission_source_tolerance=1.0e-11,
        inner_tolerance=1.0e-10,
        max_outer=5000,
    )
    return xslib, openndm.Model(geometry, xslib, settings).solve()


def report(label, k_openndm, k_openmc, sigma):
    difference = k_openndm - k_openmc.nominal_value
    sigmas = abs(difference) / sigma if sigma > 0 else float("inf")
    verdict = "PASS" if sigmas <= 1.0 else ("marginal" if sigmas <= 3.0 else "FAIL")
    print(
        f"  {label:<34s} k_ndm={k_openndm:.6f}  "
        f"diff={1e5 * difference:+8.1f} pcm  = {sigmas:5.2f} sigma   {verdict}"
    )
    return sigmas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=2, choices=(2, 8))
    parser.add_argument("--particles", type=int, default=20000)
    parser.add_argument("--batches", type=int, default=60)
    parser.add_argument("--keep", action="store_true", help="keep the run directory")
    args = parser.parse_args()

    edges = GROUP_EDGES_2 if args.groups == 2 else GROUP_EDGES_8
    workdir = Path(tempfile.mkdtemp(prefix="openndm_c1_"))
    try:
        model = build_model(particles=args.particles, batches=args.batches)
        model.tallies = openmc.Tallies()
        library = build_mgxs_library(model, edges)

        statepoint_path = model.run(cwd=str(workdir), output=False)
        with openmc.StatePoint(statepoint_path) as sp:
            library.load_from_statepoint(sp)
            k_openmc = sp.keff

        sigma = float(k_openmc.std_dev)
        print(f"\nC-1 homogeneous consistency, {args.groups} groups")
        print(f"  OpenMC continuous energy   k_inf = {k_openmc.nominal_value:.6f}"
              f" +/- {1e5 * sigma:.1f} pcm\n")

        worst = 0.0
        for label, kwargs in (
            ("D from diffusion-coefficient", {"prefer": "diffusion-coefficient"}),
            ("D from transport", {"prefer": "transport"}),
            (
                "multiplicity correction disabled",
                {"scattering_multiplicity": "ignore"},
            ),
        ):
            xslib, result = solve_with_openndm(library, **kwargs)
            worst = max(worst, report(label, result.k_eff, k_openmc, sigma))

        # k_inf depends only on absorption, production and the flux spectrum, so
        # the diffusion coefficient must not move it at all. If it does, the
        # geometry is not actually infinite.
        print()
        comp = xslib.composition(0)

        def show(name, values):
            text = np.array2string(np.asarray(values), precision=5)
            print(f"  {name:<11s} {text}")

        print(f"  groups={xslib.n_groups}  uncertainties carried="
              f"{xslib.has_uncertainty}")
        for name in ("absorption", "nu_fission", "chi", "D"):
            show(name, getattr(comp, name))
        scatter = np.asarray(comp.scatter).reshape(xslib.n_groups, xslib.n_groups)
        print(f"  scatter[from][to]\n{np.array2string(scatter, precision=5)}")

        # The analytic infinite-medium eigenvalue from the translated data.
        flux = np.asarray(result.flux).ravel()
        k_balance = float(
            np.dot(np.asarray(comp.nu_fission), flux)
            / np.dot(np.asarray(comp.absorption), flux)
        )
        print(f"\n  balance check  sum(nuSf phi)/sum(Sa phi) = {k_balance:.6f}"
              f"  ({1e5 * (k_balance - result.k_eff):+.2f} pcm from the solve)")

        return 0 if worst <= 1.0 else 1
    finally:
        if args.keep:
            print(f"\nrun directory kept at {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
