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


def per_node_geometry(case):
    """A geometry where every node carries its own composition index.

    The deck's cross sections are a derivative expansion evaluated at the
    local state, and at full power that state varies node to node. OpenNDM
    ties cross sections to compositions, so resolving the distribution means
    one composition per node: node ``i`` is composition ``i`` withdrawn and
    ``i + n_nodes`` rodded.

    Returns
    -------
    geometry : Geometry
    base : ndarray of int, shape (n_nodes,)
        The deck composition each node started as, which is what the
        expansion and the rod increment are read from.
    """
    geometry = build_geometry(case)
    base = np.asarray(geometry.compositions, dtype=int).copy()
    for node in range(geometry.n_nodes):
        geometry.set_composition(node, node)
    return geometry, base


def expanded_nodes(base, boron, fuel_temperature, moderator_temperature,
                   coolant_density):
    """Cross sections per node, each at its own state.

    Parameters
    ----------
    base : ndarray of int, shape (n_nodes,)
        Deck composition per node.
    boron : float
        Concentration in ppm, uniform over the core.
    fuel_temperature, moderator_temperature, coolant_density : ndarray
        Per node, in K, K and g/cm^3.

    Returns
    -------
    ndarray, shape (n_nodes, n_groups, 7)
    """
    out = np.asarray(deck.BASE, dtype=float)[base].copy()
    shape = (-1, 1, 1)
    terms = (
        (deck.BORON_DELTA, np.full(base.shape, boron - deck.BORON_REFERENCE)),
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
        delta = np.asarray(table, dtype=float)[base]
        scaled = np.asarray(change, dtype=float).reshape(shape)
        out[:, :, :4] += delta[:, :, :4] * scaled
        out[:, :, 5:] += delta[:, :, 4:] * scaled
    return out


def rodded_nodes(values, base, increment):
    """Add the rod increment per node, keeping fission out of the reflector."""
    out = np.asarray(values, dtype=float).copy()
    delta = np.asarray(increment, dtype=float)[base]
    out[:, :, :4] += delta[:, :, :4]
    out[:, :, 5:] += delta[:, :, 4:]
    reflector = np.isin(base, REFLECTOR)
    out[reflector, :, 2:4] = values[reflector, :, 2:4]
    return out


def build_node_library(case, base, boron, fuel_temperature,
                       moderator_temperature, coolant_density, out=None):
    """A library of one composition per node, unrodded then rodded."""
    values = expanded_nodes(
        base, boron, fuel_temperature, moderator_temperature, coolant_density
    )
    with_rods = rodded_nodes(values, base, deck.CASES[case]["rod_delta"])

    n_nodes = base.size
    library = (
        openndm.XSLibrary(deck.N_GROUPS, 2 * n_nodes) if out is None else out
    )
    for node in range(n_nodes):
        write(library, node, values[node])
        write(library, node + n_nodes, with_rods[node])
    library.finalize(warn=False)
    return library


def build_node_rods(case, geometry, library=None):
    """Rod banks for a per-node geometry, substituting node to node."""
    card = deck.CASES[case]
    data = geometry_of(case)
    printed = np.asarray(card["bank_map"], dtype=int)[::-1, :]
    bank_map = np.repeat(
        np.repeat(printed, data["y_div"], axis=0), data["x_div"], axis=1
    )
    n_nodes = geometry.n_nodes
    substitution = {node: node + n_nodes for node in range(n_nodes)}

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


def assembly_columns(case):
    """Which assembly each node column belongs to, as (rows, columns).

    NEACRP-L-335 Section 2.8 makes each fuel assembly one channel region, so
    the thermal-hydraulics runs on assemblies while the neutronics runs on
    the 2x2 nodes inside them.
    """
    data = geometry_of(case)
    rows = np.repeat(np.arange(len(data["y_div"])), data["y_div"])
    columns = np.repeat(np.arange(len(data["x_div"])), data["x_div"])
    return rows, columns


def coarse_geometry(case):
    """An assembly-level geometry, one node per assembly per axial layer.

    The thermal-hydraulics runs on this and the neutronics on the subdivided
    mesh, which is what NEACRP-L-335 Section 2.8 asks for: each fuel assembly
    is one channel region.
    """
    data = geometry_of(case)
    planes = [np.asarray(plane, dtype=int) for plane in data["planar"]]
    stacked = np.stack([planes[index - 1] for index in data["assignment"]])
    stacked = stacked[:, ::-1, :]
    composition = np.where(stacked == 0, openndm.INACTIVE, stacked - 1)
    return openndm.Geometry.from_lattice(
        composition,
        pitch=(data["dx"][0], data["dy"][0], data["dz"][0]),
        dx=data["dx"],
        dy=data["dy"],
        dz=data["dz"],
        boundaries=boundaries(data),
        outside="zero_flux",
    )


def whole_assembly_factor(case, coarse):
    """Scale each channel's power up to a whole assembly's worth.

    An assembly on a symmetry face is modelled as a half or a quarter of
    itself, so it carries that fraction of the power. The channel model
    gives every channel one assembly's mass flow and one assembly's pins,
    which is right for a whole assembly and wrong for a fraction of one.

    Both the power and the flow scale together, so handing the channel the
    whole assembly's power against the whole assembly's flow gives exactly
    the enthalpy rise and linear heat rate that fraction really sees. The
    alternative, scaling the flow and the pin count down instead, cannot be
    expressed: a quarter of 25 guide tubes is not a whole number.
    """
    data = geometry_of(case)
    full = max(data["dx"]) * max(data["dy"])
    nz, ny, nx = coarse.shape
    area = np.outer(data["dy"], data["dx"])
    mapping = np.asarray(coarse.lattice_to_node).reshape(coarse.shape)
    factor = np.ones(coarse.n_nodes, dtype=float)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                node = mapping[k, j, i]
                if node >= 0:
                    factor[node] = full / area[j, i]
    return factor


class AssemblyChannels:
    """A per-assembly thermal solver driven by per-node power.

    ``ChannelModel`` makes one channel per radial column of whatever geometry
    it is given, so it is given the assembly-level mesh. This adapter sums
    the node powers into their assembly on the way in and broadcasts the
    assembly's state back to its nodes on the way out, which keeps the
    thermal-hydraulics on the channel region the specification defines while
    the neutronics stays on the subdivided mesh.
    """

    def __init__(self, case, fine, coarse, channel):
        self.channel = channel
        self.channel_geometry = coarse
        data = geometry_of(case)
        rows = np.repeat(np.arange(len(data["y_div"])), data["y_div"])
        columns = np.repeat(np.arange(len(data["x_div"])), data["x_div"])

        fine_map = np.asarray(fine.lattice_to_node).reshape(fine.shape)
        coarse_map = np.asarray(coarse.lattice_to_node).reshape(coarse.shape)
        target = np.full(fine.n_nodes, -1, dtype=int)
        for k in range(fine.shape[0]):
            for j in range(fine.shape[1]):
                for i in range(fine.shape[2]):
                    node = fine_map[k, j, i]
                    if node >= 0:
                        target[node] = coarse_map[k, rows[j], columns[i]]
        if np.any(target < 0):
            raise ValueError("a node has no assembly to belong to")
        self._target = target
        self._n_coarse = coarse.n_nodes
        self._whole_assembly = whole_assembly_factor(case, coarse)

    def set_heat_source(self, q):
        gathered = np.bincount(
            self._target, weights=np.asarray(q, dtype=float),
            minlength=self._n_coarse,
        )
        self.channel.set_heat_source(gathered * self._whole_assembly)

    def solve(self):
        self.channel.solve()

    def get_temperatures(self):
        return {
            name: values[self._target]
            for name, values in self.channel.get_temperatures().items()
        }

    def get_densities(self):
        return {
            name: values[self._target]
            for name, values in self.channel.get_densities().items()
        }

    def average_doppler(self, case):
        """Volume-average Doppler temperature over the fuel, K.

        Over the fuel, not over the mesh. A reflector node has no fuel and
        sits at the coolant temperature, so including it in the average
        drags the figure down by well over a hundred degrees without saying
        anything about the fuel.
        """
        geometry = self.channel_geometry
        fuelled = ~np.isin(
            np.asarray(geometry.compositions, dtype=int), REFLECTOR
        )
        doppler = self.channel.get_temperatures()["doppler_temperature"]
        volume = np.asarray(geometry.volumes, dtype=float)
        return float(
            (doppler[fuelled] * volume[fuelled]).sum() / volume[fuelled].sum()
        )
