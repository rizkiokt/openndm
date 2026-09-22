"""VTK export for 3D visualisation (FR-OUT-6).

Writes the serial XML unstructured grid ParaView reads, one hexahedral cell
per active node on the real Cartesian mesh. An out-of-core position gets no
cell at all rather than a cell carrying zero, because in a plot those two look
identical and the first one makes a wrong core map look plausible.

The format is plain XML and is written directly, so the package keeps
importing without a VTK installation, the way FR-OMC-14 requires of OpenMC.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .exceptions import InputError

__all__ = ["write_vtk"]

_HEXAHEDRON = 12

_CORNERS = (
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
)


def write_vtk(path, result, model, *, extra: dict | None = None) -> Path:
    """Write a solve result as an unstructured grid for ParaView.

    Parameters
    ----------
    path : path-like
        Destination file. ``.vtu`` is the conventional suffix for this
        format and is not enforced.
    result : Result
        The object returned by :meth:`openndm.Model.solve`.
    model : Model
        Provides the geometry the cells are built from.
    extra : mapping of str to array_like, optional
        Further per-node fields to write alongside, each of shape
        ``(n_nodes,)``. Use it for anything the solver does not produce, such
        as a temperature distribution from a coupled run.

    Returns
    -------
    pathlib.Path

    Raises
    ------
    InputError
        If an ``extra`` field is not one value per node, or shadows a field
        this function already writes.

    Notes
    -----
    Cell data is ``power``, ``flux_g1`` to ``flux_gG`` and ``composition``,
    in node order. Groups are numbered from one, as a reactor physicist reads
    them off a legend, rather than from zero as the arrays are indexed.

    The eigenvalue is written as field data, so ParaView shows it without it
    having to be painted onto every cell.

    Examples
    --------
    >>> openndm.write_vtk("core.vtu", result, model)  # doctest: +SKIP
    """
    geometry = model.geometry
    nodes = _active_nodes(geometry)
    fields = _cell_fields(result, geometry, nodes, extra)

    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        _write_grid(f, geometry, nodes, fields, float(result.k_eff))
    return path


def _active_nodes(geometry) -> np.ndarray:
    """Node index of every active lattice position, in ``(z, y, x)`` order."""
    mapping = geometry.lattice_to_node.reshape(geometry.shape)
    return mapping[mapping >= 0]


def _cell_fields(result, geometry, nodes, extra) -> dict[str, np.ndarray]:
    """Every array to be written as cell data, already in cell order."""
    flux = np.asarray(result.flux)
    fields: dict[str, np.ndarray] = {"power": np.asarray(result.power)[nodes]}
    for group in range(flux.shape[1]):
        fields[f"flux_g{group + 1}"] = flux[nodes, group]
    fields["composition"] = np.asarray(geometry.compositions)[nodes]

    for name, values in (extra or {}).items():
        if name in fields:
            raise InputError(
                f"extra field {name!r} shadows one this writer already "
                f"produces; choose another name"
            )
        array = np.asarray(values, dtype=float)
        if array.shape != (geometry.n_nodes,):
            raise InputError(
                f"extra field {name!r} must hold one value per node, "
                f"expected shape ({geometry.n_nodes},), got {array.shape}"
            )
        fields[str(name)] = array[nodes]
    return fields


def _edges(widths: np.ndarray) -> np.ndarray:
    """Cell boundaries along one axis, starting at zero."""
    return np.concatenate([[0.0], np.cumsum(widths)])


def _points(geometry) -> np.ndarray:
    """Corner coordinates of the full lattice, x fastest, shape (n, 3)."""
    x, y, z = (_edges(w) for w in (geometry.dx, geometry.dy, geometry.dz))
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def _connectivity(geometry) -> np.ndarray:
    """The eight corner point indices of each active cell, shape (n, 8)."""
    _, ny, nx = geometry.shape
    mapping = geometry.lattice_to_node.reshape(geometry.shape)
    planes, rows, columns = np.nonzero(mapping >= 0)
    corners = [
        (columns + di) + (rows + dj) * (nx + 1) + (planes + dk) * (nx + 1) * (ny + 1)
        for di, dj, dk in _CORNERS
    ]
    return np.column_stack(corners)


def _numbers(values, fmt: str) -> str:
    """Flatten an array into the whitespace-separated text a DataArray holds."""
    return " ".join(format(v, fmt) for v in np.asarray(values).ravel())


def _write_array(f, name, values, vtk_type: str, *, components: int = 1) -> None:
    """Write one ``DataArray`` element.

    Floats carry 17 significant digits, which is what makes a written file
    read back as the array it came from rather than close to it.
    """
    fmt = ".17g" if vtk_type == "Float64" else "d"
    named = "" if name is None else f' Name="{name}"'
    extra = "" if components == 1 else f' NumberOfComponents="{components}"'
    f.write(
        f'        <DataArray type="{vtk_type}"{named}{extra} format="ascii">\n'
        f"          {_numbers(values, fmt)}\n"
        f"        </DataArray>\n"
    )


def _write_grid(f, geometry, nodes, fields, k_eff: float) -> None:
    """Write the whole document, which is small enough to hold in one pass."""
    points = _points(geometry)
    cells = _connectivity(geometry)
    offsets = np.arange(1, len(cells) + 1) * 8

    f.write('<?xml version="1.0"?>\n')
    f.write(
        '<VTKFile type="UnstructuredGrid" version="1.0" '
        'byte_order="LittleEndian">\n'
    )
    f.write("  <UnstructuredGrid>\n")
    f.write("    <FieldData>\n")
    f.write(
        f'      <DataArray type="Float64" Name="k_eff" NumberOfTuples="1" '
        f'format="ascii">{k_eff:.17g}</DataArray>\n'
    )
    f.write("    </FieldData>\n")
    f.write(
        f'    <Piece NumberOfPoints="{len(points)}" '
        f'NumberOfCells="{len(cells)}">\n'
    )

    f.write("      <Points>\n")
    _write_array(f, None, points, "Float64", components=3)
    f.write("      </Points>\n")

    f.write("      <Cells>\n")
    _write_array(f, "connectivity", cells, "Int64")
    _write_array(f, "offsets", offsets, "Int64")
    _write_array(f, "types", np.full(len(cells), _HEXAHEDRON), "UInt8")
    f.write("      </Cells>\n")

    f.write('      <CellData Scalars="power">\n')
    for name, values in fields.items():
        vtk_type = "Int32" if np.issubdtype(values.dtype, np.integer) else "Float64"
        _write_array(f, name, values, vtk_type)
    f.write("      </CellData>\n")

    f.write("    </Piece>\n")
    f.write("  </UnstructuredGrid>\n")
    f.write("</VTKFile>\n")
