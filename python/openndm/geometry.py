"""Geometry construction (FR-GEO)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from . import _core

__all__ = ["INACTIVE", "Geometry"]

INACTIVE = -1
"""Marker for a lattice position that lies outside the core (FR-GEO-2)."""

_BC = {
    "zero_flux": _core.BoundaryType.zero_flux,
    "vacuum": _core.BoundaryType.vacuum,
    "reflective": _core.BoundaryType.reflective,
    "albedo": _core.BoundaryType.albedo,
}
_FACES = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
_AXIS_NAMES = ("x", "y", "axial")
_AXIS_SLICES = ("column", "row", "plane")


def _boundary(value):
    if isinstance(value, str):
        try:
            return _BC[value.lower()]
        except KeyError:
            raise ValueError(
                f"unknown boundary condition {value!r}; "
                f"choose from {sorted(_BC)}"
            ) from None
    return value


class Geometry:
    """A 3D node/surface graph, built from a structured core map.

    The Python object is a thin wrapper: solver kernels see only the abstract
    node/surface connectivity, never ``(i, j, k)`` indices, which is what lets
    a hexagonal builder be added later without touching them (FR-GEO-6).
    """

    def __init__(self, core: _core.Geometry, *, shape=None, dx=None, dy=None,
                 dz=None):
        self._g = core
        self._shape = shape or core.lattice_shape
        self._widths = [
            None if w is None else np.asarray(w, dtype=float)
            for w in (dx, dy, dz)
        ]

    @classmethod
    def from_lattice(
        cls,
        composition: np.ndarray,
        pitch: float | Sequence[float],
        *,
        dz: Sequence[float] | None = None,
        dx: Sequence[float] | None = None,
        dy: Sequence[float] | None = None,
        boundaries: Mapping[str, str] | None = None,
        albedo: Mapping[str, Sequence[float]] | None = None,
        outside: str = "vacuum",
        outside_albedo: Sequence[float] | None = None,
        subdivide: int | Sequence[int] = 1,
        rotation: np.ndarray | None = None,
    ) -> Geometry:
        """Build a Cartesian geometry from a ``(nz, ny, nx)`` composition map.

        Parameters
        ----------
        composition : array_like of int, shape (nz, ny, nx)
            Composition index per lattice position, with ``INACTIVE`` (-1)
            marking a position outside the core (FR-GEO-2). A 2D
            ``(ny, nx)`` array is promoted to a single axial plane.
        pitch : float or (dx, dy, dz)
            Uniform node size. Ignored along an axis for which an explicit
            width array is given.
        dx, dy, dz : sequence of float, optional
            Explicit non-uniform node widths (FR-GEO-1), one per *lattice*
            cell, before subdivision. ``subdivide`` splits each of them into
            equal parts, so the extent along the axis does not depend on how
            finely it is meshed.
        boundaries : mapping, optional
            Condition per outer face, keyed ``'x_min'``, ``'x_max'``,
            ``'y_min'``, ``'y_max'``, ``'z_min'``, ``'z_max'`` and valued
            ``'zero_flux'``, ``'vacuum'``, ``'reflective'`` or ``'albedo'``
            (FR-GEO-5). Anything unset defaults to ``'vacuum'``.
        albedo : mapping, optional
            Per-face albedo :math:`\\beta_g = J^-/J^+`, one value per group.
            Required for a face set to ``'albedo'``.
        outside : str
            Condition applied where an *interior* face looks at an inactive
            position. This is deliberately separate from ``boundaries``: on a
            quarter-core map the mesh edges carry the symmetry conditions, and
            an out-of-core position must not inherit them.
        subdivide : int or (nx, ny, nz)
            Split each lattice cell into this many nodes per direction
            (FR-GEO-4). Discontinuity factors follow the composition, so a
            subdivided assembly keeps the ADFs of its parent.
        rotation : array_like of int, shape (nz, ny, nx), optional
            Quarter turns counter-clockwise about +z applied to the assembly
            at each lattice position, 0 to 3 (FR-OPT-7). Only the
            discontinuity factors turn with it; a rotation is meaningless
            without them and costs nothing when they are all one.

            Rotation belongs to the position and not to the composition,
            because one assembly type is loaded at many positions in
            different orientations. This follows KOMODO's ``%ADF`` ``ROT``
            convention, so a deck translates without re-deriving anything.

        Returns
        -------
        Geometry

        Examples
        --------
        >>> import numpy as np
        >>> core = np.zeros((4, 3, 3), dtype=int)
        >>> g = Geometry.from_lattice(core, pitch=20.0)
        >>> g.n_nodes
        36
        """
        comp = np.asarray(composition, dtype=np.int32)
        if comp.ndim == 2:
            comp = comp[np.newaxis, :, :]
        if comp.ndim != 3:
            raise ValueError(
                f"composition map must be 2D or 3D, got {comp.ndim}D"
            )

        original_shape = comp.shape
        sub = np.broadcast_to(np.asarray(subdivide, dtype=int), (3,))
        if np.any(sub < 1):
            raise ValueError("subdivide must be at least 1 in every direction")
        if np.any(sub > 1):
            comp = np.repeat(comp, sub[2], axis=0)
            comp = np.repeat(comp, sub[1], axis=1)
            comp = np.repeat(comp, sub[0], axis=2)

        nz, ny, nx = comp.shape
        p = np.broadcast_to(np.asarray(pitch, dtype=float), (3,))
        widths = []
        for axis, (n, explicit, sub_n) in enumerate(
            zip((nx, ny, nz), (dx, dy, dz), sub, strict=True)
        ):
            if explicit is None:
                widths.append(np.full(n, p[axis] / sub_n, dtype=float))
            else:
                shared_width = np.asarray(explicit, dtype=float) / sub_n
                w = np.repeat(shared_width, sub_n)
                if w.size != n:
                    raise ValueError(
                        f"axis {axis}: {w.size} widths for {n} nodes"
                    )
                widths.append(w)

        spec = _core.CartesianSpec()
        spec.dx, spec.dy, spec.dz = (list(w) for w in widths)
        spec.composition = [int(v) for v in comp.ravel(order="C")]

        if rotation is not None:
            turns = np.asarray(rotation, dtype=np.int32)
            if turns.ndim == 2:
                turns = turns[np.newaxis, :, :]
            if turns.shape != original_shape:
                raise ValueError(
                    f"rotation map must have shape {original_shape} to match "
                    f"the composition map, got {turns.shape}"
                )
            if np.any(turns < 0) | np.any(turns > 3):
                raise ValueError(
                    "rotation must be 0, 1, 2 or 3 quarter turns "
                    "counter-clockwise"
                )
            if np.any(sub > 1):
                turns = np.repeat(turns, sub[2], axis=0)
                turns = np.repeat(turns, sub[1], axis=1)
                turns = np.repeat(turns, sub[0], axis=2)
            spec.rotation = [int(v) for v in turns.ravel(order="C")]

        bc = dict.fromkeys(_FACES, "vacuum")
        bc.update({k.lower(): v for k, v in (boundaries or {}).items()})
        unknown = set(bc) - set(_FACES)
        if unknown:
            raise ValueError(
                f"unknown boundary face(s) {sorted(unknown)}; "
                f"expected {list(_FACES)}"
            )
        spec.bc = [_boundary(bc[f]) for f in _FACES]

        alb = {k.lower(): list(map(float, v)) for k, v in (albedo or {}).items()}
        spec.albedo = [alb.get(f, []) for f in _FACES]
        spec.inactive_bc = _boundary(outside)
        spec.inactive_albedo = list(map(float, outside_albedo or []))

        return cls(
            _core.Geometry.from_cartesian(spec),
            shape=(nz, ny, nx),
            dx=widths[0],
            dy=widths[1],
            dz=widths[2],
        )

    @property
    def n_nodes(self) -> int:
        return self._g.n_nodes

    @property
    def n_surfaces(self) -> int:
        return self._g.n_surfaces

    @property
    def n_compositions(self) -> int:
        return self._g.n_compositions

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape ``(nz, ny, nx)`` of the originating lattice."""
        return tuple(self._shape)

    @property
    def volumes(self) -> np.ndarray:
        """Node volumes in cm^3, shape ``(n_nodes,)``."""
        return self._g.volumes

    @property
    def total_volume(self) -> float:
        return self._g.total_volume

    @property
    def compositions(self) -> np.ndarray:
        """Composition index per node, shape ``(n_nodes,)``."""
        return self._g.compositions

    @property
    def rotations(self) -> np.ndarray:
        """Quarter turns counter-clockwise per node, shape ``(n_nodes,)``."""
        return self._g.rotations

    @property
    def dx(self) -> np.ndarray:
        """Node widths along x in cm, one per column of ``shape[2]``."""
        return self._width(0)

    @property
    def dy(self) -> np.ndarray:
        """Node widths along y in cm, one per row of ``shape[1]``."""
        return self._width(1)

    @property
    def dz(self) -> np.ndarray:
        """Axial node widths in cm, one per plane of ``shape[0]``.

        A control rod bank at a continuous position needs the plane
        boundaries, and a caller reconstructing them from its own input gets
        them wrong as soon as ``subdivide`` is used: the widths here are the
        post-subdivision ones.
        """
        return self._width(2)

    @property
    def lattice_to_node(self) -> np.ndarray:
        """Node index per flat lattice position, -1 where inactive."""
        return self._g.lattice_to_node

    def expand(self, node_values: np.ndarray, fill: float = np.nan) -> np.ndarray:
        """Scatter a per-node array back onto the ``(nz, ny, nx)`` lattice.

        Inactive positions take ``fill``.
        """
        values = np.asarray(node_values, dtype=float)
        if values.shape[0] != self.n_nodes:
            raise ValueError(
                f"expected {self.n_nodes} node values, got {values.shape[0]}"
            )
        mapping = self.lattice_to_node
        out = np.full(mapping.shape + values.shape[1:], fill, dtype=float)
        active = mapping >= 0
        out[active] = values[mapping[active]]
        return out.reshape(self.shape + values.shape[1:])

    def _width(self, axis: int) -> np.ndarray:
        if self._widths[axis] is None:
            self._widths[axis] = self._derive_widths(axis)
        return self._widths[axis]

    def _derive_widths(self, axis: int) -> np.ndarray:
        """Recover one axis's node widths from the node graph.

        Only reached for a Geometry built straight from a ``_core.Geometry``
        rather than through :meth:`from_lattice`, which records them.
        """
        widths = np.zeros(self.shape[2 - axis], dtype=float)
        for node in self._g.nodes:
            widths[node.ijk[axis]] = node.width[axis]
        if np.any(widths <= 0.0):
            raise ValueError(
                f"could not recover the {_AXIS_NAMES[axis]} mesh: "
                f"{_AXIS_SLICES[axis]}(s) "
                f"{np.flatnonzero(widths <= 0.0).tolist()} have no active node"
            )
        return widths

    def set_composition(self, node: int, composition: int) -> None:
        """Reassign one node's composition in place (FR-OPT-7)."""
        self._g.set_composition(int(node), int(composition))

    def set_rotation(self, node: int, quarter_turns: int) -> None:
        """Rotate one node's assembly in place, counter-clockwise (FR-OPT-7).

        Parameters
        ----------
        node : int
        quarter_turns : int
            0 to 3. Only the discontinuity factors turn with it.
        """
        self._g.set_rotation(int(node), int(quarter_turns))

    def __repr__(self) -> str:
        nz, ny, nx = self.shape
        return (
            f"<Geometry {nx}x{ny}x{nz} lattice, {self.n_nodes} nodes, "
            f"{self.n_surfaces} surfaces>"
        )
