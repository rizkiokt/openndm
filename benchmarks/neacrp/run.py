#!/usr/bin/env python3
"""NEACRP PWR rod ejection benchmark, initial steady state.

Six cases over three geometries: A and B are quarter cores with rotational
symmetry, C is a half core. Each is a critical boron search at hot zero power
(A1, B1, C1) or at full power (A2, B2, C2).

This deck runs the **hot zero power** cases. At 1e-4 % of nominal the coolant
carries no heat, so fuel and moderator sit at the inlet temperature and the
thermal-hydraulic coupling has nothing to do. That isolates the geometry, the
cross section expansion, the rod banks and the boron search, which is the
half of the problem worth getting right before feedback is added.

The data is parsed from the KOMODO sample decks rather than transcribed; see
``neacrp_data.py`` for provenance and ``neacrp_build.py`` for the three
interpretations that turn it into a solvable model. The reference solution is
in ``neacrp_reference.py`` and comes from a different document again.

Run with ``python run.py``; pass ``--case A1`` for one case.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import neacrp_build as build
import neacrp_data as deck
import neacrp_reference as reference

import openndm

HZP_CASES = ("A1", "B1", "C1")

SETTINGS = openndm.Settings(
    verbosity=0,
    kernel="sanm",
    k_tolerance=1.0e-9,
    fission_source_tolerance=1.0e-8,
    max_outer=2000,
)


def solve_case(case, settings=SETTINGS):
    """Search the critical boron of one hot zero power case.

    Returns
    -------
    (BoronSearchResult, Geometry)
    """
    fuel, moderator, density = build.hzp_state()
    geometry = build.build_geometry(case)
    library = build.build_library(case, deck.BORON_REFERENCE, fuel, moderator,
                                  density)
    build.build_rods(case, geometry, library)
    model = openndm.Model(geometry, library, settings)

    def apply_boron(target, ppm):
        build.build_library(case, ppm, fuel, moderator, density, out=target)

    search = model.search_boron(
        apply_boron,
        target_k=1.0,
        guess=deck.BORON_REFERENCE,
        bracket=(0.0, 3000.0),
        tolerance=1.0e-7,
    )
    return search, geometry


def volume_weighted_peaking(result, geometry):
    """Peak node power over the volume-weighted core average.

    ``Result.f_q`` normalises to an arithmetic mean over powered nodes, which
    is the same thing only on a uniform mesh. This deck's axial layers run
    from 7.7 cm to 30 cm, so the two differ by 27 percent and the arithmetic
    one is not the core average any peaking factor means.
    """
    power = result.power
    volume = geometry.volumes
    powered = power > 0.0
    average = float(
        (power[powered] * volume[powered]).sum() / volume[powered].sum()
    )
    return float(power.max()) / average


def report(case, search, geometry):
    """Print one case against the reference, in pcm and in percent."""
    expected = reference.INITIAL_STEADY_STATE[case]
    result = search.result
    boron_error = search.boron - expected["boron_ppm"]

    print(f"\n{case}  ({deck.CASES[case]['geometry'].lower()} core, "
          f"{result.flux.shape[0]} nodes)")
    print(f"  {'quantity':<22}{'OpenNDM':>12}{'reference':>12}{'difference':>14}")
    print(f"  {'critical boron, ppm':<22}{search.boron:12.1f}"
          f"{expected['boron_ppm']:12.1f}{boron_error:+13.1f} ")
    peaking = volume_weighted_peaking(result, geometry)
    print(f"  {'F_Q (volume mean)':<22}{peaking:12.3f}{expected['f_q']:12.3f}"
          f"{100.0 * (peaking / expected['f_q'] - 1.0):+13.2f}%")
    print(f"  {'F_Q (Result.f_q)':<22}{result.f_q:12.3f}{expected['f_q']:12.3f}"
          f"{100.0 * (result.f_q / expected['f_q'] - 1.0):+13.2f}%")
    print(f"  {'F_xy (as F_dH)':<22}{result.f_dh:12.3f}{expected['f_xy']:12.3f}"
          f"{100.0 * (result.f_dh / expected['f_xy'] - 1.0):+13.2f}%")
    print(f"  {'k_eff at that boron':<22}{result.k_eff:12.6f}"
          f"{1.0:>12.6f}{1.0e5 * (result.k_eff - 1.0):+13.2f} pcm")
    print(f"  search took {search.iterations} evaluations")
    return boron_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=HZP_CASES, help="run one case")
    args = parser.parse_args()

    cases = [args.case] if args.case else list(HZP_CASES)
    print("NEACRP PWR rod ejection benchmark, initial steady state at HZP")
    print("Reference: NEA/NSC/DOC(93)25 Table 3.1, PANTHER at 4x16 nodes "
          "per assembly")

    errors = {}
    for case in cases:
        errors[case] = report(case, *solve_case(case))

    print("\ncritical boron error, ppm")
    for case, error in errors.items():
        print(f"  {case}  {error:+8.1f}")
    print("\nResult.f_q normalises to an arithmetic mean over powered nodes,")
    print("which is the core average only on a uniform mesh. This deck's axial")
    print("layers run 7.7 cm to 30 cm, so both rows are shown.")
    print("\nF_xy is compared against the radial peaking OpenNDM reports as")
    print("F_dH. NEACRP-L-335 Section 4 defines B3, the maximum power peaking")
    print("factor, which is F_Q; it does not define F_xy, so that row states")
    print("an inferred correspondence rather than a verified one.")


if __name__ == "__main__":
    main()
