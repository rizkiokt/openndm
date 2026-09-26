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
import numpy as np

import openndm

HZP_CASES = ("A1", "B1", "C1")
FP_CASES = ("A2", "B2", "C2")

GUIDE_TUBE_RADIUS = 12.259e-3 / 2.0
"""Guide tube outer radius, m. NEACRP-L-335 Table 2.7, 12.259 mm."""

GAP_CONDUCTANCE = 1.0e4
"""Pellet-to-cladding conductance, W/(m^2 K). NEACRP-L-335 Section 2.10."""

RELAXATION = 0.4
"""Under-relaxation on the power handed to the thermal solver.

At full power the hot assemblies reach saturation, and void feedback on the
coolant density is steep enough that an undamped loop oscillates rather than
converging. This is the remedy the coupling documents, and it changes the
path to the fixed point rather than the fixed point.
"""

FILM_COEFFICIENT = 3.0e4
"""Cladding-to-coolant coefficient, W/(m^2 K).

The specification does not give one. Section 2.10 fixes only the gap, and the
draft says in as many words that the treatment of heat transfer to the
coolant is left to each participant. This is a choice, not a citation.
"""

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


def thermal_model(case, fine):
    """The assembly-wise channel and pin conduction for one case."""
    card = deck.CASES[case]
    pins = openndm.PinGeometry(
        fuel_radius=card["fuel_radius"],
        gap_thickness=card["gap_thickness"],
        clad_thickness=card["clad_thickness"],
        pin_pitch=card["pin_pitch"],
        n_pins=card["n_pins"],
        n_guide_tubes=card["n_guide_tubes"],
        guide_tube_radius=GUIDE_TUBE_RADIUS,
    )
    conduction = openndm.PinConduction(
        pins,
        fuel_conductivity=openndm.neacrp_fuel_conductivity,
        clad_conductivity=openndm.neacrp_clad_conductivity,
        gap_conductance=GAP_CONDUCTANCE,
        film_coefficient=FILM_COEFFICIENT,
        doppler_weight=build.DOPPLER_WEIGHT,
    )
    coarse = build.coarse_geometry(case)
    channel = openndm.ChannelModel(
        coarse,
        pins,
        mass_flow=card["mass_flow"],
        inlet_temperature=card["inlet_temperature"],
        pressure=build.PRESSURE,
        direct_heating=card["direct_heating"],
        conduction=conduction,
        two_phase=True,
    )
    return build.AssemblyChannels(case, fine, coarse, channel)


def solve_coupled_case(case, settings=SETTINGS):
    """Search the critical boron of one full power case, with feedback.

    Returns
    -------
    (BoronSearchResult, Geometry, AssemblyChannels)
    """
    card = deck.CASES[case]
    geometry, base = build.per_node_geometry(case)
    fuel, moderator, density = build.hzp_state()
    library = build.build_node_library(
        case,
        base,
        deck.BORON_REFERENCE,
        np.full(base.size, fuel),
        np.full(base.size, moderator),
        np.full(base.size, density),
    )
    build.build_node_rods(case, geometry, library)
    model = openndm.Model(geometry, library, settings)
    thermal = thermal_model(case, geometry)
    state = {"ppm": deck.BORON_REFERENCE}

    def apply_state(temperatures, densities):
        build.build_node_library(
            case,
            base,
            state["ppm"],
            temperatures["doppler_temperature"],
            temperatures["moderator_temperature"],
            densities["moderator_density"] / 1000.0,
            out=model.library,
        )

    def apply_boron(target, ppm):
        state["ppm"] = ppm

    def evaluate_state():
        return model.solve_coupled(
            thermal,
            apply_state,
            total_power=card["power"],
            percent=card["percent"],
            tolerance=1.0e-6,
            max_iterations=120,
            relaxation=RELAXATION,
        ).result

    search = model.search_boron(
        apply_boron,
        target_k=1.0,
        guess=deck.BORON_REFERENCE,
        bracket=(0.0, 3000.0),
        tolerance=1.0e-7,
        evaluate_state=evaluate_state,
    )
    return search, geometry, thermal


def report_coupled(case, search, geometry, thermal):
    """Print one full power case, adding the temperatures to the comparison."""
    expected = reference.INITIAL_STEADY_STATE[case]
    result = search.result
    doppler = thermal.average_doppler(case) - 273.15
    centre = float(np.max(thermal.channel.pin_state.centre)) - 273.15

    print(f"\n{case}  ({deck.CASES[case]['geometry'].lower()} core, "
          f"{geometry.n_nodes} nodes, coupled)")
    print(f"  {'quantity':<22}{'OpenNDM':>12}{'reference':>12}{'difference':>14}")
    print(f"  {'critical boron, ppm':<22}{search.boron:12.1f}"
          f"{expected['boron_ppm']:12.1f}"
          f"{search.boron - expected['boron_ppm']:+13.1f} ")
    peaking = volume_weighted_peaking(result, geometry)
    print(f"  {'F_Q (volume mean)':<22}{peaking:12.3f}{expected['f_q']:12.3f}"
          f"{100.0 * (peaking / expected['f_q'] - 1.0):+13.2f}%")
    print(f"  {'F_xy (as F_dH)':<22}{result.f_dh:12.3f}{expected['f_xy']:12.3f}"
          f"{100.0 * (result.f_dh / expected['f_xy'] - 1.0):+13.2f}%")
    print(f"  {'T_Doppler, C':<22}{doppler:12.1f}{expected['t_doppler']:12.1f}"
          f"{doppler - expected['t_doppler']:+13.1f} ")
    print(f"  {'T_centre peak, C':<22}{centre:12.1f}{expected['t_centre']:12.1f}"
          f"{centre - expected['t_centre']:+13.1f} ")
    void = thermal.channel.void_fraction
    if void.max() > 0.0:
        boiling = int((void > 0.0).sum())
        print(f"  {boiling} of {void.size} channel nodes reach saturation, "
              f"peak void {void.max():.3f}")
    else:
        print("  no channel node reaches saturation")
    print(f"  search took {search.iterations} coupled evaluations")
    return search.boron - expected["boron_ppm"]


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
    parser.add_argument(
        "--case", choices=HZP_CASES + FP_CASES, help="run one case"
    )
    parser.add_argument(
        "--full-power", action="store_true", help="run A2, B2 and C2"
    )
    args = parser.parse_args()

    if args.case:
        cases = [args.case]
    elif args.full_power:
        cases = list(FP_CASES)
    else:
        cases = list(HZP_CASES)
    print("NEACRP PWR rod ejection benchmark, initial steady state")
    print("Reference: NEA/NSC/DOC(93)25 Table 3.1, PANTHER at 4x16 nodes "
          "per assembly")

    errors = {}
    for case in cases:
        if case in FP_CASES:
            errors[case] = report_coupled(case, *solve_coupled_case(case))
        else:
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
