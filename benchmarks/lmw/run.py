#!/usr/bin/env python3
"""LMW operational transient, quarter core, two groups, six precursor groups.

A 6x6 quarter core of 20 cm assemblies over three compositions, 200 cm tall
with 20 cm axial reflectors. Two control rod banks move against each other
over 60 seconds: bank 2 withdraws from 100 steps at t = 0, and bank 1 inserts
from 180 steps at t = 7.5 s, both at 3 steps per second. One step is 1 cm and
the axial mesh is 5 cm, so the tip sits inside a node for four steps out of
five and the cusping correction is active almost throughout.

This is the first deck that exercises rod banks, cusping, precursors and
theta-weighted integration together.

**The deck states the scenario and not its answer.** The published LMW
reference is a power-versus-time curve, which is not in the file the data was
parsed from, so nothing here is a comparison against a reference: the power
history is reported and the verification is internal. Writing a reference
curve down from memory and calling the agreement verification is the failure
this project already made once, recorded in ``benchmarks/README.md``.

What the internal checks establish is that the mesh is converged and the time
step is not. Eight times the nodes moves the peak by 0.06%, while halving the
deck's 0.25 s step moves it by 1.7% and is still falling first order. The
scheme is second order at theta = 0.5, but only for steps that resolve the
prompt time constant of about a millisecond; three hundred times that, which
is what an operational transient runs at, is the stiff regime where the
theta method loses an order. See ``benchmarks/README.md`` for the numbers.

Run with ``python run.py``; ``--dt`` and ``--subdivide`` refine, and
``--rod-timing end`` places the banks at the end of each interval rather than
its midpoint. The midpoint is the quadrature point the step's single frozen
operator wants, but the order reduction above dominates the difference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from common import benchmark_settings
from lmw_data import (
    BANK_MAP,
    BETA,
    BOUNDARY_CODES,
    COMPOSITIONS,
    DECAY_CONSTANT,
    DX,
    DY,
    DZ,
    INITIAL_POSITION,
    MAX_STEPS,
    MOTION,
    N_GROUPS,
    PLANAR_TYPES,
    PLANE_ASSIGNMENT,
    ROD_DELTA,
    STEP_SIZE,
    THETA,
    TIME_STEP,
    TOTAL_TIME,
    ZERO_POSITION,
)

import openndm

_BC = {0: "zero_flux", 1: "vacuum", 2: "reflective"}
"""KOMODO boundary codes to OpenNDM names."""

BOUNDARIES = dict(
    zip(
        ("x_max", "x_min", "y_max", "y_min", "z_min", "z_max"),
        (_BC[code] for code in BOUNDARY_CODES),
        strict=True,
    )
)
"""The deck gives boundaries as east, west, north, south, bottom, top.

``1 2 1 2 1 1`` makes west and south the symmetry cuts, which is what the
half-width first assembly on each of those axes is for.
"""

BANK_NAMES = tuple(INITIAL_POSITION)
"""Bank names in the order the deck numbers them."""


def node_map() -> np.ndarray:
    """Composition of every node, ``(nz, ny, nx)``, as the deck numbers them."""
    return PLANAR_TYPES[PLANE_ASSIGNMENT]


def rod_bases() -> list[int]:
    """Compositions the banks can reach, as zero-based library indices."""
    reached = node_map()[:, BANK_MAP > 0]
    return sorted({int(c) - 1 for c in np.unique(reached)})


def _rodded_fields(base: int) -> dict:
    """Group constants of one composition with the deck's rod increments."""
    fields = {key: list(value) for key, value in COMPOSITIONS[base].items()}
    delta = ROD_DELTA[base]
    for key in ("absorption", "nu_fission", "kappa_fission"):
        fields[key] = [a + b for a, b in zip(fields[key], delta[key], strict=True)]
    if any(delta["transport"]):
        fields["transport"] = [
            1.0 / (3.0 * d) + t
            for d, t in zip(fields.pop("D"), delta["transport"], strict=True)
        ]
    return fields


def lmw_library():
    """Build the library and the composition slots the banks write into.

    Returns
    -------
    library : openndm.XSLibrary
    rodded : dict
        Rodded counterpart of every composition a bank can reach.
    cusp : dict
        Spare slot per bank per reachable composition, holding the mixture of
        the node the tip sits inside.
    """
    bases = rod_bases()
    n = len(COMPOSITIONS)
    rodded = {base: n + offset for offset, base in enumerate(bases)}
    n += len(bases)
    cusp = {}
    for name in BANK_NAMES:
        cusp[name] = {base: n + offset for offset, base in enumerate(bases)}
        n += len(bases)

    library = openndm.XSLibrary(N_GROUPS, n)
    for index, composition in enumerate(COMPOSITIONS):
        library.set_composition(index, **composition)
    for base, slot in rodded.items():
        library.set_composition(slot, **_rodded_fields(base))
    for slots in cusp.values():
        for base, slot in slots.items():
            library.set_composition(slot, **COMPOSITIONS[base])
    library.set_delayed(BETA, DECAY_CONSTANT)
    library.finalize(warn=False)
    return library, rodded, cusp


def build(subdivide: int = 1):
    """Assemble the geometry, library and banks, with every bank withdrawn."""
    core = node_map()
    lattice = np.where(core == 0, openndm.INACTIVE, core - 1)
    geometry = openndm.Geometry.from_lattice(
        lattice,
        pitch=(DX, DY, DZ),
        subdivide=(subdivide, subdivide, subdivide),
        boundaries=BOUNDARIES,
        outside="vacuum",
    )
    library, rodded, cusp = lmw_library()
    columns = np.repeat(np.repeat(BANK_MAP, subdivide, axis=0), subdivide, axis=1)
    banks = [
        openndm.ControlRodBank(
            name=name,
            columns=columns == number + 1,
            rodded=rodded,
            step_size=STEP_SIZE,
            zero_position=ZERO_POSITION,
            max_steps=MAX_STEPS,
            cusp=cusp[name],
        )
        for number, name in enumerate(BANK_NAMES)
    ]
    rods = openndm.ControlRods(geometry, banks, library=library)
    return geometry, library, rods


def position(name: str, time: float) -> float:
    """Position of one bank in steps at ``time``, following the deck's ramp."""
    start = INITIAL_POSITION[name]
    motion = MOTION[name]
    if time <= motion["start"]:
        return start
    travelled = motion["speed"] * (time - motion["start"])
    if motion["final"] >= start:
        return min(motion["final"], start + travelled)
    return max(motion["final"], start - travelled)


def positions_at(time: float) -> dict[str, float]:
    return {name: position(name, time) for name in BANK_NAMES}


def run(
    model,
    rods,
    *,
    dt: float = TIME_STEP,
    total: float = TOTAL_TIME,
    rod_timing: str = "midpoint",
    record_every: float | None = None,
):
    """Run the transient and return the steady state and the power history.

    Parameters
    ----------
    model : openndm.Model
    rods : openndm.ControlRods
    dt : float
        Time step, in seconds.
    total : float
        Duration, in seconds.
    rod_timing : {'midpoint', 'end'}
        Where in each interval the banks are placed. The step holds one
        operator over the whole interval, so the midpoint is the quadrature
        point a ramp asks for; see ``docs/theory.md``.
    record_every : float, optional
        Sampling interval for the returned history. Every step by default.

    Returns
    -------
    k_eff : float
        Steady-state eigenvalue at the initial bank positions.
    history : ndarray, shape (n_samples, 2)
        Time in seconds and power relative to the steady state.
    """
    rods.insert(positions_at(0.0))
    model.refresh()
    steady = rods.converge_cusping(model)
    transient = model.start_transient(theta=THETA)
    reference_power = transient.step(1.0e-12).total_power

    history = [(0.0, 1.0)]
    next_sample = record_every if record_every else 0.0
    elapsed = 0.0
    while elapsed < total - 1.0e-9:
        span = min(dt, total - elapsed)
        when = elapsed + (0.5 * span if rod_timing == "midpoint" else span)
        rods.insert(positions_at(when))
        rods.reweight(transient.flux)
        step = transient.step(span)
        elapsed += span
        if record_every is None or elapsed >= next_sample - 1.0e-9:
            history.append((elapsed, step.total_power / reference_power))
            if record_every:
                next_sample += record_every
    return steady.k_eff, np.array(history)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subdivide", type=int, default=1)
    parser.add_argument("--dt", type=float, default=TIME_STEP)
    parser.add_argument("--total", type=float, default=TOTAL_TIME)
    parser.add_argument("--kernel", default="sanm")
    parser.add_argument("--rod-timing", choices=("midpoint", "end"), default="midpoint")
    args = parser.parse_args()

    geometry, library, rods = build(args.subdivide)
    settings = benchmark_settings(kernel=args.kernel)
    model = openndm.Model(geometry, library, settings)
    print(
        f"LMW: {geometry.n_nodes} nodes, kernel={args.kernel}, "
        f"dt={args.dt} s, theta={THETA}, rods at interval {args.rod_timing}"
    )
    k_eff, history = run(
        model,
        rods,
        dt=args.dt,
        total=args.total,
        rod_timing=args.rod_timing,
        record_every=5.0,
    )
    print(f"steady state k_eff = {k_eff:.6f}")
    print(f"{'t (s)':>8s}  {'P/P0':>10s}  {'bank 1':>8s}  {'bank 2':>8s}")
    for time, power in history:
        at = positions_at(time)
        print(
            f"{time:8.2f}  {power:10.6f}  "
            f"{at['bank_1']:8.2f}  {at['bank_2']:8.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
