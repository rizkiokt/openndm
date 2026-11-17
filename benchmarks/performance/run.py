"""Performance acceptance cases (NFR-PERF-1 to NFR-PERF-7).

The specification states numeric targets; until this deck existed none of them
had been run. A target without a machine is not a claim, so the hardware is
named in the output and belongs with any number taken from it.

Run with ``python run.py``. ``--quick`` skips the thread sweep.
"""

from __future__ import annotations

import argparse
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (
    IAEA_MATERIALS,
    IAEA_ORDER,
    radial_map_to_compositions,
)

import openndm

AXIAL_PLANES = 24
"""Axial planes of the NFR-PERF-1 problem, per the specification."""

CORE_HEIGHT = 380.0
"""Height of the NFR-PERF-1 core, cm."""

FAST_GROUPS = 7
"""Number of fast groups in the synthetic eight-group library."""

REPEATS = 3
"""Repeats of each timed case.

The median is reported, because a single run on a laptop catches whatever
else the scheduler was doing.
"""

TARGETS = {
    "FR-OPT-3": ">= 2x vs cold",
    "NFR-PERF-1": "< 1 s",
    "NFR-PERF-2": "< 10 s",
    "NFR-PERF-3": "< 5 s",
    "NFR-PERF-4": "< 5 min",
    "NFR-PERF-5": "< 500 MB",
    "NFR-PERF-6": "< 0.3 s",
    "NFR-PERF-7": ">= 60% on 8 threads",
}


def describe_machine() -> str:
    """Everything needed to make a timing on this box reproducible elsewhere."""
    model = "unknown"
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            model = line.split(":", 1)[1].strip()
            break
    governor = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    scaling = governor.read_text().strip() if governor.exists() else "unknown"
    lines = [
        f"  CPU            {model}",
        f"  cores/threads  {physical_cores()} physical, {logical_cores()} logical",
        f"  governor       {scaling}",
        f"  platform       {platform.platform()}",
        f"  python         {platform.python_version()}",
        f"  openndm        {openndm.__version__}",
        f"  build          {build_description()}",
    ]
    return "\n".join(lines)


def build_description() -> str:
    """Build type and OpenMP state, read from the CMake cache that produced it."""
    root = Path(__file__).resolve().parents[2]
    for cache in sorted(root.glob("build/*/CMakeCache.txt")):
        text = cache.read_text()
        build_type = _cache_value(text, "CMAKE_BUILD_TYPE")
        openmp = _cache_value(text, "OPENNDM_USE_OPENMP")
        compiler = Path(_cache_value(text, "CMAKE_CXX_COMPILER") or "?").name
        return f"{build_type}, OpenMP={openmp}, {compiler}"
    return "unknown (no CMake cache found)"


def _cache_value(text: str, key: str) -> str:
    for line in text.splitlines():
        if line.startswith(f"{key}:"):
            return line.split("=", 1)[1]
    return ""


def physical_cores() -> int:
    try:
        out = subprocess.run(
            ["lscpu", "-p=Core,Socket"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return 0
    rows = {line for line in out.splitlines() if not line.startswith("#")}
    return len(rows)


def logical_cores() -> int:
    import os

    return os.cpu_count() or 0


def eight_group_library() -> openndm.XSLibrary:
    """Synthetic eight-group library, for sizing the solve and nothing else.

    NFR-PERF-2 asks for the NFR-PERF-1 problem at eight groups, and no
    eight-group data exists for this core. The IAEA thermal group is kept
    exactly and its fast group is resolved into seven, each carrying the same
    absorption and scattering seven times faster, so a neutron makes the same
    journey through a longer chain. The eigenvalue that comes out is close to
    the two-group one but means nothing on its own: what this library is for
    is an 8x8 scattering matrix and eight group sweeps per outer.
    """
    groups = FAST_GROUPS + 1
    library = openndm.XSLibrary(n_groups=groups, n_compositions=len(IAEA_ORDER))
    for index, name in enumerate(IAEA_ORDER):
        d1, d2, a1, a2, f2, s12 = IAEA_MATERIALS[name]
        scatter = np.zeros((groups, groups))
        for g in range(FAST_GROUPS):
            scatter[g][g + 1] = FAST_GROUPS * s12
        chi = np.zeros(groups)
        nu_fission = np.zeros(groups)
        if f2 > 0.0:
            chi[0] = 1.0
            nu_fission[-1] = f2
        library.set_composition(
            index,
            D=[d1] * FAST_GROUPS + [d2],
            absorption=[a1] * FAST_GROUPS + [a2],
            nu_fission=nu_fission,
            kappa_fission=nu_fission,
            chi=chi,
            scatter=scatter,
        )
    library.finalize(warn=False)
    return library


def two_group_library() -> openndm.XSLibrary:
    from common import iaea_library

    return iaea_library()


def core_geometry(subdivide: int = 1, planes: int = AXIAL_PLANES):
    """IAEA quarter core extruded to the NFR-PERF-1 axial mesh."""
    radial = radial_map_to_compositions()
    core = np.repeat(radial[np.newaxis, :, :], planes, axis=0)
    return openndm.Geometry.from_lattice(
        core,
        pitch=(20.0, 20.0, CORE_HEIGHT / planes),
        subdivide=(subdivide, subdivide, 1),
        boundaries={
            "x_min": "reflective",
            "y_min": "reflective",
            "x_max": "zero_flux",
            "y_max": "zero_flux",
            "z_min": "zero_flux",
            "z_max": "zero_flux",
        },
        outside="zero_flux",
    )


def timed_solve(geometry, library, *, threads=1, warm=False, repeats=REPEATS):
    """Median solve time in seconds over ``repeats`` runs, and the last result."""
    settings = openndm.Settings(verbosity=0, threads=threads, warm_start=warm)
    model = openndm.Model(geometry, library, settings)
    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = model.solve()
        times.append(time.perf_counter() - start)
    return float(np.median(times)), result


def peak_memory_mb() -> float:
    """Peak resident size of this process in MB, including the interpreter."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def line(case: str, measured: str, verdict: str) -> None:
    print(f"  {case:<12s} {TARGETS[case]:<22s} {measured:<24s} {verdict}")


def verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def run_perf1():
    geometry = core_geometry()
    seconds, result = timed_solve(geometry, two_group_library())
    nodes = geometry.n_nodes
    line(
        "NFR-PERF-1",
        f"{seconds * 1e3:.1f} ms ({nodes} nodes, 2 g)",
        verdict(seconds < 1.0),
    )
    return result


def run_perf2():
    geometry = core_geometry(subdivide=2)
    seconds, result = timed_solve(geometry, eight_group_library())
    nodes = geometry.n_nodes
    line(
        "NFR-PERF-2",
        f"{seconds:.2f} s ({nodes} nodes, 8 g)",
        verdict(seconds < 10.0),
    )
    peak = peak_memory_mb()
    line("NFR-PERF-5", f"{peak:.0f} MB peak RSS", verdict(peak < 500.0))
    return result


INLET_TEMPERATURE = 560.0
"""Coolant temperature entering every channel of the NFR-PERF-3 case, K."""

FUEL_TEMPERATURES = (INLET_TEMPERATURE, 1500.0)
"""Fuel temperature branch points of the Doppler library, K."""

DOPPLER_SWING = 0.05
"""Fractional absorption rise across :data:`FUEL_TEMPERATURES`.

The IAEA data is a single state point, so a coupled solve has nothing to
branch on and the Picard loop converges in one iteration, which times
nothing. Absorption rising 5% over 940 K is worth a few thousand pcm on this
core, the order a PWR carries. The eigenvalue it lands on is not an IAEA
number.
"""

TOTAL_POWER = 1.0e9
"""Thermal power of the modelled quarter core at full power, W.

A gigawatt over the 58 fuel positions of the map is about 17 MW each, which
against :data:`CHANNEL_MASS_FLOW` is about 35 K of rise on an average
channel: where a PWR sits. The specification fixes neither number, so the
pair sizes the thermal-hydraulic half of the problem rather than describing a
reactor.
"""

CHANNEL_MASS_FLOW = 90.0
"""Coolant mass flow per assembly channel, kg/s."""

ASSEMBLY_PINS = openndm.PinGeometry(
    fuel_radius=4.11950e-3,
    gap_thickness=6.8e-5,
    clad_thickness=5.71e-4,
    pin_pitch=1.2655e-2,
    n_pins=264,
    n_guide_tubes=25,
)
"""One channel's worth of a 17x17 PWR fuel assembly, m."""

FUEL_CONDUCTIVITY = 3.0
"""Pellet thermal conductivity, W/(m K)."""

CLAD_CONDUCTIVITY = 15.0
"""Cladding thermal conductivity, W/(m K)."""

GAP_CONDUCTANCE = 1.0e4
"""Pellet-to-cladding conductance, W/(m^2 K)."""

FILM_COEFFICIENT = 3.0e4
"""Cladding-to-coolant coefficient, W/(m^2 K)."""


def doppler_library(n_compositions: int) -> openndm.XSLibrary:
    """The IAEA two-group data on one fuel temperature axis.

    Every composition carries the same :data:`DOPPLER_SWING`, so the shape of
    the feedback comes from the temperature distribution rather than from the
    data.
    """
    library = openndm.XSLibrary(n_groups=2, n_compositions=n_compositions)
    library.set_axes([("fuel_temperature", list(FUEL_TEMPERATURES))])
    for state, factor in enumerate((1.0, 1.0 + DOPPLER_SWING)):
        for index, name in enumerate(IAEA_ORDER[:n_compositions]):
            d1, d2, a1, a2, f2, s12 = IAEA_MATERIALS[name]
            library.set_composition(
                index,
                state=state,
                D=[d1, d2],
                absorption=[a1 * factor, a2 * factor],
                nu_fission=[0.0, f2],
                kappa_fission=[0.0, f2],
                chi=[1.0, 0.0],
                scatter=[[0.0, s12], [0.0, 0.0]],
            )
    library.finalize(warn=False)
    return library


def assembly_channel(geometry) -> openndm.ChannelModel:
    """One closed channel per assembly column, with radial pin conduction."""
    conduction = openndm.PinConduction(
        ASSEMBLY_PINS,
        fuel_conductivity=FUEL_CONDUCTIVITY,
        clad_conductivity=CLAD_CONDUCTIVITY,
        gap_conductance=GAP_CONDUCTANCE,
        film_coefficient=FILM_COEFFICIENT,
    )
    return openndm.ChannelModel(
        geometry,
        ASSEMBLY_PINS,
        mass_flow=CHANNEL_MASS_FLOW,
        inlet_temperature=INLET_TEMPERATURE,
        conduction=conduction,
    )


def coupled_case(geometry):
    """A model, its channel and the ``apply_state`` that joins the two.

    Cross sections live per composition, so the feedback is resolved onto the
    four the core map carries rather than onto every node. That is as far as
    single-state IAEA data reaches.
    """
    branch = doppler_library(geometry.n_compositions)
    model = openndm.Model(
        geometry,
        branch.interpolate(fuel_temperature=INLET_TEMPERATURE),
        openndm.Settings(verbosity=0, threads=1, warm_start=True),
    )
    mapping = openndm.CompositionMapping(geometry)

    def apply_state(temperatures, densities):
        fuel = mapping.average(temperatures["doppler_temperature"])
        branch.interpolate_by_composition(
            {"fuel_temperature": fuel}, out=model.library
        )

    return model, assembly_channel(geometry), apply_state


def timed_coupled_solve(geometry, repeats=REPEATS):
    """Median coupled solve time in seconds, and the last result.

    Each repeat builds its own model, so every one starts from the cold
    library rather than from the fixed point the one before it found.
    """
    times = []
    coupled = None
    for _ in range(repeats):
        model, channel, apply_state = coupled_case(geometry)
        start = time.perf_counter()
        coupled = model.solve_coupled(channel, apply_state, total_power=TOTAL_POWER)
        times.append(time.perf_counter() - start)
    return float(np.median(times)), coupled


def run_perf3():
    geometry = core_geometry()
    seconds, coupled = timed_coupled_solve(geometry)
    line(
        "NFR-PERF-3",
        f"{seconds:.2f} s ({coupled.iterations} Picard)",
        verdict(seconds < 5.0),
    )
    peak_fuel = float(np.max(coupled.temperatures["doppler_temperature"]))
    peak_coolant = float(np.max(coupled.temperatures["moderator_temperature"]))
    print(
        f"    k_eff {coupled.k_eff:.6f}, peak Doppler temperature "
        f"{peak_fuel:.0f} K, peak coolant {peak_coolant:.0f} K"
    )
    return coupled


def shuffled_solve(warm_start: bool):
    """Time a re-solve after four assemblies have been swapped.

    ``refresh()`` is required: without it the solver keeps the cross sections
    of the old loading pattern and returns an answer 366 pcm out, which looks
    entirely plausible.
    """
    swaps = [(0, 40), (1, 41), (2, 42), (3, 43)]
    model = openndm.Model(
        core_geometry(),
        two_group_library(),
        openndm.Settings(verbosity=0, threads=1, warm_start=warm_start),
    )
    model.solve()
    times = []
    outers = 0
    for _ in range(REPEATS):
        for a, b in swaps:
            model.swap_assemblies(a, b)
        model.refresh()
        start = time.perf_counter()
        result = model.solve()
        times.append(time.perf_counter() - start)
        outers = result.outer_iterations
    return float(np.median(times)), outers


def run_perf6():
    warm, warm_outers = shuffled_solve(warm_start=True)
    cold, cold_outers = shuffled_solve(warm_start=False)
    line(
        "NFR-PERF-6",
        f"{warm * 1e3:.1f} ms ({warm_outers} outers)",
        verdict(warm < 0.3),
    )
    line(
        "FR-OPT-3",
        f"{cold / warm:.2f}x vs cold ({cold_outers} outers)",
        verdict(cold / warm >= 2.0),
    )
    return cold / warm


SCALING_SUBDIVIDE = 3
"""Radial subdivision of the thread-scaling case.

NFR-PERF-7 applies above 1e5 nodes x groups and the NFR-PERF-2 problem is
only 57600, so the requirement has to be tested on a larger one than it.
"""


def amdahl_serial_fraction(speedup: float, threads: int) -> float:
    """Serial fraction implied by a speedup, from Amdahl's law."""
    if threads <= 1 or speedup <= 0.0:
        return float("nan")
    return (threads / speedup - 1.0) / (threads - 1.0)


def run_perf7(max_threads: int):
    geometry = core_geometry(subdivide=SCALING_SUBDIVIDE)
    library = eight_group_library()
    size = geometry.n_nodes * library.n_groups
    print(f"\n  thread scaling on {geometry.n_nodes} nodes x 8 groups = {size}")
    if size <= 1.0e5:
        print("  (below the 1e5 nodes x groups NFR-PERF-7 applies to)")

    counts = [n for n in (1, 2, 4, 8) if n <= max_threads]
    baseline = None
    efficiency = {}
    eigenvalues = set()
    for n in counts:
        seconds, result = timed_solve(geometry, library, threads=n, repeats=2)
        eigenvalues.add(result.k_eff)
        if baseline is None:
            baseline = seconds
        speedup = baseline / seconds
        efficiency[n] = speedup / n
        print(
            f"    {n:2d} threads  {seconds:7.3f} s  speedup {speedup:4.2f}x  "
            f"efficiency {100 * efficiency[n]:5.1f}%"
        )
    print(
        f"    eigenvalue identical on every thread count: "
        f"{len(eigenvalues) == 1} (FR-OPT-4)"
    )
    if 8 in efficiency:
        serial = amdahl_serial_fraction(8.0 * efficiency[8], 8)
        print(
            f"    implied serial fraction {100 * serial:.0f}%, which is the "
            f"cost FR-SOL-7 names"
        )
        line(
            "NFR-PERF-7",
            f"{100 * efficiency[8]:.1f}% on 8 threads",
            verdict(efficiency[8] >= 0.60),
        )
    else:
        line("NFR-PERF-7", "not run, fewer than 8 cores", "SKIP")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="skip the thread sweep")
    parser.add_argument("--max-threads", type=int, default=8)
    args = parser.parse_args()

    print("Performance acceptance cases, NFR-PERF-1 to NFR-PERF-7\n")
    print(describe_machine())
    print(f"\n  median of {REPEATS} runs, library default convergence settings\n")
    print(f"  {'case':<12s} {'target':<22s} {'measured':<24s} verdict")
    print(f"  {'-' * 12} {'-' * 22} {'-' * 24} -------")

    run_perf1()
    run_perf2()
    run_perf3()
    run_perf6()

    line("NFR-PERF-4", "not runnable", "BLOCKED")

    if not args.quick:
        run_perf7(args.max_threads)

    print(
        "\n  NFR-PERF-3 couples the NFR-PERF-1 problem to one channel per\n"
        "  assembly, with Doppler feedback the IAEA data does not carry and this\n"
        "  deck invents. The time is the case the requirement names; the\n"
        "  eigenvalue is not an IAEA number. Warm started, so each Picard\n"
        "  iteration resumes from the flux of the one before.\n"
        "\n"
        "  NFR-PERF-4 names adaptive time stepping, which is not implemented; a\n"
        "  fixed-step rod ejection can be timed but it is not the case the\n"
        "  specification states.\n"
        "\n"
        "  FR-OPT-3 falls short for a reason worth knowing. refresh() keeps the\n"
        "  flux, so the warm start begins an order of magnitude closer than a\n"
        "  cold one. But the outer error falls only by about 0.65 per iteration,\n"
        "  so that decade saves a few outers out of twenty. A 2x speedup needs\n"
        "  faster outer convergence rather than a better initial guess."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
