"""Core maps read from an ``openmc.RectLattice`` (FR-GEO-7).

One lattice gives both the domain order of the MGXS library and the
composition map of the nodal geometry, so the two cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..exceptions import InputError
from ..geometry import INACTIVE
from .mgxs import require_openmc

__all__ = ["composition_map", "lattice_universes"]


def _universe_grid(lattice) -> np.ndarray:
    """Universe at each lattice position, indexed ``(k, j, i)``.

    ``j = 0`` is the lowest y. ``lattice.universes`` stores the highest row
    first, so it is read through ``get_universe``, which takes the index
    counted from the lower-left corner.
    """
    openmc = require_openmc()
    if not isinstance(lattice, openmc.RectLattice):
        raise InputError(
            f"expected an openmc.RectLattice, got {type(lattice).__name__}"
        )
    nx, ny, nz = (*lattice.shape, 1)[:3]
    grid = np.empty((nz, ny, nx), dtype=object)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                index = (i, j, k) if lattice.ndim == 3 else (i, j)
                grid[k, j, i] = lattice.get_universe(index)
    return grid


def _outer_id(lattice) -> int | None:
    return None if lattice.outer is None else lattice.outer.id


def lattice_universes(lattice) -> list:
    """Return the universes a lattice places in the core, in composition order.

    Order is first appearance scanning from the lowest ``(x, y, z)`` corner, x
    fastest. The lattice's ``outer`` universe marks out-of-core positions and
    is left out.

    Parameters
    ----------
    lattice : openmc.RectLattice
        Two- or three-dimensional core lattice.

    Returns
    -------
    list of openmc.Universe
        Use it as ``mgxs_lib.domains`` with ``domain_type = 'universe'``, so
        composition ``n`` of the library is ``universes[n]``.
    """
    outer_id = _outer_id(lattice)
    by_id = {}
    for universe in _universe_grid(lattice).ravel():
        if universe.id != outer_id:
            by_id.setdefault(universe.id, universe)
    return list(by_id.values())


def composition_map(lattice, domains: Sequence | None = None) -> np.ndarray:
    """Return the composition index at each position of a lattice.

    Parameters
    ----------
    lattice : openmc.RectLattice
        Two- or three-dimensional core lattice.
    domains : sequence of openmc.Universe or int, optional
        Universes, or their ids, in composition order: the sequence the MGXS
        library was built on. Matched by id, so universes reloaded from a
        statepoint work. Defaults to :func:`lattice_universes`.

    Returns
    -------
    numpy.ndarray of int, shape (nz, ny, nx)
        Composition per position, ``INACTIVE`` where the lattice holds its
        ``outer`` universe. ``nz`` is 1 for a two-dimensional lattice, and row
        0 is the lowest y.

    Raises
    ------
    InputError
        If ``lattice`` is not an ``openmc.RectLattice``, or it holds a
        universe that is neither in ``domains`` nor its ``outer``.
    """
    universes = _universe_grid(lattice)
    if domains is None:
        domains = lattice_universes(lattice)
    composition_of = {
        getattr(domain, "id", domain): index for index, domain in enumerate(domains)
    }
    outer_id = _outer_id(lattice)

    composition = np.empty(universes.shape, dtype=np.int32)
    for position, universe in np.ndenumerate(universes):
        if universe.id == outer_id:
            composition[position] = INACTIVE
        elif universe.id in composition_of:
            composition[position] = composition_of[universe.id]
        else:
            k, j, i = position
            raise InputError(
                f"universe {universe.id} at lattice position (i={i}, j={j}, "
                f"k={k}) is not among the domains and is not the outer universe"
            )
    return composition
