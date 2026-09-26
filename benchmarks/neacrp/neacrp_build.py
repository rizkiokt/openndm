"""Turn the parsed NEACRP deck into a geometry, a library and rod banks.

The deck stores what KOMODO stores. Everything that turns it into something
OpenNDM can solve lives here, so ``neacrp_data`` stays a faithful record of
the source and this file carries the interpretation.

Three interpretations are worth knowing about before reading the code.

**The expansion is on the transport cross section.** ``D = 1 / (3 sigma_tr)``
is nonlinear, so the four derivative terms are summed on sigma_tr and D is
formed afterwards.

**Rod increments do not add fission to a reflector.** The increments are the
same for every composition, including the three that carry no fuel, and
applying them there would make nu-fission negative. NEACRP-L-335's covering
letter, Finnemann to benchmark participants, 18 February 1992, says the
increments "are valid in the core and in the reflector with the exception
that the increments of the fission cross sections of the latter are put to
zero".

**The printed maps run north to south.** KOMODO prints a radial map with
its first row at the top of the figure, which is the reflective symmetry
face, while OpenNDM indexes y upward from ``y_min``. The rows are therefore
reversed on the way in. The check that says so: the deck puts its half-width
assembly at the *last* y entry and its reflective face on ``y_max``, and the
printed maps put the core centre in the *first* row. Only the flip makes
those two agree, and without it the half-width assembly lands against the
zero-flux face instead of the symmetry line.

**The node mesh is uniform although the assembly mesh is not.** A half-width
assembly on a symmetry face is divided once and a full assembly twice, so
every node comes out 10.803 cm across. ``Geometry.subdivide`` splits every
cell by the same factor and cannot express that, so the lattice is expanded
here and handed over at node level.
"""

from __future__ import annotations

import neacrp_data as deck
import numpy as np

import openndm

PRESSURE = 15.5e6
"""Core pressure, Pa. NEACRP-L-335 Section 2.11, 155 bar."""

HZP_TEMPERATURE = 559.15
"""Hot zero power temperature, K. NEACRP-L-335 Table 2.8, 286 C."""

DOPPLER_WEIGHT = 0.7
"""Weight on the pellet surface. NEACRP-L-335 Section 2.5, a = 0.7."""

N_BASE = deck.N_COMPOSITIONS
"""Compositions the deck defines, before rodded copies are added."""

REFLECTOR = tuple(
    index
    for index in range(N_BASE)
    if all(group[2] == 0.0 for group in deck.BASE[index])
)
"""Compositions carrying no fuel, found from the data rather than listed."""


def expand(values, divisions):
    """Repeat each entry ``divisions[i]`` times, splitting widths evenly."""
    out = []
    for value, count in zip(values, divisions, strict=True):
        out.extend([value / count] * count)
    return out


def repeat(indices, divisions):
    """Repeat each index ``divisions[i]`` times, leaving the values alone."""
    out = []
    for value, count in zip(indices, divisions, strict=True):
        out.extend([value] * count)
    return out


def lattice(geometry):
    """Node-level composition map and widths for one deck geometry.

    Returns
    -------
    composition : ndarray of int, shape (nz, ny, nx)
        Zero-based, with :data:`openndm.INACTIVE` outside the core.
    dx, dy, dz : list of float
        Node widths in cm.
    """
    planes = [np.asarray(plane, dtype=int) for plane in geometry["planar"]]
    stacked = np.stack([planes[index - 1] for index in geometry["assignment"]])
    stacked = stacked[:, ::-1, :]

    rows = np.repeat(stacked, geometry["y_div"], axis=1)
    columns = np.repeat(rows, geometry["x_div"], axis=2)
    composition = np.where(columns == 0, openndm.INACTIVE, columns - 1)

    return (
        composition,
        expand(geometry["dx"], geometry["x_div"]),
        expand(geometry["dy"], geometry["y_div"]),
        expand(geometry["dz"], geometry["z_div"]),
    )


def boundaries(geometry):
    """KOMODO's east, west, north, south, bottom, top as OpenNDM names."""
    names = ("x_max", "x_min", "y_max", "y_min", "z_min", "z_max")
    codes = {0: "zero_flux", 1: "vacuum", 2: "reflective"}
    return dict(
        zip(names, (codes[c] for c in geometry["boundaries"]), strict=True)
    )


def geometry_of(case):
    """The deck geometry a case runs on."""
    return getattr(deck, f"{deck.CASES[case]['geometry']}_GEOMETRY")


def build_geometry(case):
    """A Geometry with every bank withdrawn, as ControlRods expects."""
    data = geometry_of(case)
    composition, dx, dy, dz = lattice(data)
    return openndm.Geometry.from_lattice(
        composition,
        pitch=(dx[0], dy[0], dz[0]),
        dx=dx,
        dy=dy,
        dz=dz,
        boundaries=boundaries(data),
        outside="zero_flux",
    )


def expanded(boron, fuel_temperature, moderator_temperature, coolant_density):
    """Cross sections at one state, per composition and group.

    Every argument is a scalar, so this is the uniform state a hot zero
    power case sits at. Columns follow :data:`neacrp_data.BASE_FIELDS`.

    Parameters
    ----------
    boron : float
        Concentration in ppm.
    fuel_temperature, moderator_temperature : float
        In K. Fuel temperature enters through its square root.
    coolant_density : float
        In g/cm^3.
    """
    base = np.asarray(deck.BASE, dtype=float)
    out = base.copy()

    terms = (
        (deck.BORON_DELTA, boron - deck.BORON_REFERENCE),
        (
            deck.FUEL_TEMPERATURE_DELTA,
            np.sqrt(fuel_temperature) - np.sqrt(deck.FUEL_TEMPERATURE_REFERENCE),
        ),
        (
            deck.MODERATOR_TEMPERATURE_DELTA,
            moderator_temperature - deck.MODERATOR_TEMPERATURE_REFERENCE,
        ),
        (
            deck.COOLANT_DENSITY_DELTA,
            coolant_density - deck.COOLANT_DENSITY_REFERENCE,
        ),
    )
    for table, change in terms:
        delta = np.asarray(table, dtype=float)
        out[:, :, :4] += delta[:, :, :4] * change
        out[:, :, 5:] += delta[:, :, 4:] * change
    return out


def rodded(values, increment):
    """Add a bank's increment, keeping fission out of the reflector."""
    out = np.asarray(values, dtype=float).copy()
    delta = np.asarray(increment, dtype=float)
    out[:, :, :4] += delta[:, :, :4]
    out[:, :, 5:] += delta[:, :, 4:]
    for index in REFLECTOR:
        out[index, :, 2:4] = np.asarray(deck.BASE, dtype=float)[index, :, 2:4]
    return out


def write(library, index, row):
    """Write one composition from a row of :data:`neacrp_data.BASE_FIELDS`."""
    transport = row[:, 0]
    scatter = np.zeros((deck.N_GROUPS, deck.N_GROUPS))
    scatter[:, 0] = row[:, 5]
    scatter[:, 1] = row[:, 6]
    library.set_composition(
        index,
        D=1.0 / (3.0 * transport),
        absorption=row[:, 1],
        nu_fission=row[:, 2],
        kappa_fission=row[:, 3],
        chi=row[:, 4],
        scatter=scatter,
    )


def build_library(case, boron, fuel_temperature, moderator_temperature,
                  coolant_density, out=None):
    """A finalized library of unrodded compositions followed by rodded ones.

    Composition ``i`` is the deck's composition ``i + 1`` withdrawn, and
    ``i + N_BASE`` is the same composition with the bank's increment.

    ``out`` rewrites an existing library in place, which a boron search
    needs: the solver holds a reference to its library, so replacing the
    object would strand it.
    """
    values = expanded(
        boron, fuel_temperature, moderator_temperature, coolant_density
    )
    with_rods = rodded(values, deck.CASES[case]["rod_delta"])

    library = openndm.XSLibrary(deck.N_GROUPS, 2 * N_BASE) if out is None else out
    for index in range(N_BASE):
        write(library, index, values[index])
        write(library, index + N_BASE, with_rods[index])
    library.finalize(warn=False)
    return library


def hzp_state():
    """The uniform hot zero power state, as (T_fuel, T_moderator, density).

    Nothing is heating, so fuel and moderator sit at the inlet temperature
    and the coolant density is whatever IF97 gives there. NEACRP-L-335
    Table 2.8 for the temperature, Section 2.11 for the pressure.
    """
    water = openndm.IF97Water()
    density = float(water.density(PRESSURE, HZP_TEMPERATURE)) / 1000.0
    return HZP_TEMPERATURE, HZP_TEMPERATURE, density


def build_rods(case, geometry, library=None):
    """Control rod banks for a case, every bank at its deck position."""
    card = deck.CASES[case]
    data = geometry_of(case)
    printed = np.asarray(card["bank_map"], dtype=int)[::-1, :]
    bank_map = np.repeat(
        np.repeat(printed, data["y_div"], axis=0), data["x_div"], axis=1
    )
    substitution = {index: index + N_BASE for index in range(N_BASE)}

    banks = [
        openndm.ControlRodBank(
            name=f"bank{number}",
            columns=bank_map == number,
            rodded=substitution,
            step_size=card["step_size"],
            zero_position=card["zero_position"],
            max_steps=card["max_steps"],
        )
        for number in range(1, card["n_banks"] + 1)
    ]
    rods = openndm.ControlRods(geometry, banks, library=library)
    rods.insert(
        {f"bank{n + 1}": position for n, position in enumerate(card["positions"])}
    )
    return rods
