"""VTK export for 3D visualisation (FR-OUT-6)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

import openndm

from conftest import IAEA_MAP, iaea_library

HEXAHEDRON = 12


def read_vtu(path):
    """Parse a written grid back into points, cells and cell data."""
    root = ET.parse(path).getroot()
    piece = root.find("./UnstructuredGrid/Piece")

    def numbers(element, dtype=float):
        return np.fromstring(element.text, sep=" ", dtype=dtype)

    points = numbers(piece.find("./Points/DataArray")).reshape(-1, 3)
    cells = {
        array.get("Name"): numbers(array, np.int64)
        for array in piece.findall("./Cells/DataArray")
    }
    data = {
        array.get("Name"): numbers(array)
        for array in piece.findall("./CellData/DataArray")
    }
    field = {
        array.get("Name"): numbers(array)
        for array in root.findall("./UnstructuredGrid/FieldData/DataArray")
    }
    return {
        "points": points,
        "connectivity": cells["connectivity"].reshape(-1, 8),
        "offsets": cells["offsets"],
        "types": cells["types"],
        "data": data,
        "field": field,
        "n_points": int(piece.get("NumberOfPoints")),
        "n_cells": int(piece.get("NumberOfCells")),
    }


def quarter_core(dz, subdivide=1):
    """A model on the IAEA radial map, which has out-of-core positions."""
    core = np.where(IAEA_MAP == 0, openndm.INACTIVE, IAEA_MAP - 1)
    core = np.repeat(core[np.newaxis, :, :], len(dz), axis=0)
    geometry = openndm.Geometry.from_lattice(
        core,
        pitch=20.0,
        dz=dz,
        subdivide=(1, 1, subdivide),
        boundaries={
            "x_min": "reflective",
            "y_min": "reflective",
            "x_max": "zero_flux",
            "y_max": "zero_flux",
            "z_min": "reflective",
            "z_max": "reflective",
        },
        outside="zero_flux",
    )
    return openndm.Model(geometry, iaea_library(), openndm.Settings(verbosity=0))


@pytest.fixture(scope="module")
def solved():
    model = quarter_core(dz=[20.0, 60.0, 20.0])
    return model.solve(), model


def test_the_cell_count_is_the_active_node_count(tmp_path, solved):
    result, model = solved
    grid = read_vtu(openndm.write_vtk(tmp_path / "core.vtu", result, model))

    lattice_positions = int(np.prod(model.geometry.shape))
    assert grid["n_cells"] == model.geometry.n_nodes
    assert grid["n_cells"] < lattice_positions
    assert len(grid["connectivity"]) == grid["n_cells"]


def test_every_cell_is_a_hexahedron(tmp_path, solved):
    result, model = solved
    grid = read_vtu(openndm.write_vtk(tmp_path / "core.vtu", result, model))

    assert np.all(grid["types"] == HEXAHEDRON)
    np.testing.assert_array_equal(
        grid["offsets"], np.arange(1, grid["n_cells"] + 1) * 8
    )


def test_the_cell_data_round_trips_the_arrays_it_came_from(tmp_path, solved):
    result, model = solved
    grid = read_vtu(openndm.write_vtk(tmp_path / "core.vtu", result, model))

    mapping = model.geometry.lattice_to_node.reshape(model.geometry.shape)
    nodes = mapping[mapping >= 0]

    np.testing.assert_array_equal(grid["data"]["power"], result.power[nodes])
    np.testing.assert_array_equal(
        grid["data"]["composition"], model.geometry.compositions[nodes]
    )
    for group in range(model.library.n_groups):
        np.testing.assert_array_equal(
            grid["data"][f"flux_g{group + 1}"], result.flux[nodes, group]
        )
    assert grid["field"]["k_eff"][0] == result.k_eff


def test_groups_are_numbered_from_one(tmp_path, solved):
    result, model = solved
    grid = read_vtu(openndm.write_vtk(tmp_path / "core.vtu", result, model))

    assert "flux_g1" in grid["data"]
    assert "flux_g2" in grid["data"]
    assert "flux_g0" not in grid["data"]
    assert f"flux_g{model.library.n_groups + 1}" not in grid["data"]


def test_an_out_of_core_position_gets_no_cell_rather_than_a_zero(tmp_path, solved):
    result, model = solved
    grid = read_vtu(openndm.write_vtk(tmp_path / "core.vtu", result, model))

    planes = model.geometry.shape[0]
    outside = int(np.count_nonzero(IAEA_MAP == 0)) * planes
    assert grid["n_cells"] == int(np.prod(model.geometry.shape)) - outside
    assert np.all(grid["data"]["composition"] != openndm.INACTIVE)


def test_a_powerless_reflector_node_is_still_written(tmp_path, solved):
    result, model = solved
    grid = read_vtu(openndm.write_vtk(tmp_path / "core.vtu", result, model))

    zero_power = grid["data"]["power"] == 0.0
    assert zero_power.any(), "the IAEA map has reflector nodes, which make no power"
    assert np.all(grid["data"]["composition"][zero_power] >= 0), (
        "a reflector is in the core and gets a cell; only an out-of-core "
        "position is absent"
    )


def test_the_cells_sit_on_the_real_non_uniform_mesh(tmp_path):
    model = quarter_core(dz=[20.0, 60.0, 20.0])
    result = model.solve()
    grid = read_vtu(openndm.write_vtk(tmp_path / "core.vtu", result, model))

    corners = grid["points"][grid["connectivity"]]
    heights = corners[:, 4:, 2].min(axis=1) - corners[:, :4, 2].max(axis=1)
    assert sorted(set(np.round(heights, 9))) == [20.0, 60.0]

    widths = corners[:, 1, 0] - corners[:, 0, 0]
    np.testing.assert_allclose(widths, 20.0)

    assert grid["points"][:, 2].max() == pytest.approx(100.0)


def test_the_cell_volumes_match_the_node_volumes(tmp_path, solved):
    result, model = solved
    grid = read_vtu(openndm.write_vtk(tmp_path / "core.vtu", result, model))

    corners = grid["points"][grid["connectivity"]]
    extent = corners.max(axis=1) - corners.min(axis=1)
    volumes = np.prod(extent, axis=1)

    mapping = model.geometry.lattice_to_node.reshape(model.geometry.shape)
    nodes = mapping[mapping >= 0]
    np.testing.assert_allclose(volumes, model.geometry.volumes[nodes], rtol=1e-12)


def test_the_first_face_winds_toward_the_second(tmp_path, solved):
    """VTK's right-hand rule: the 0-1-2-3 normal must point at face 4-5-6-7.

    Wound the other way every cell is inverted, which a reader reports as a
    negative volume and a plot shows as nothing at all.
    """
    result, model = solved
    grid = read_vtu(openndm.write_vtk(tmp_path / "core.vtu", result, model))

    corners = grid["points"][grid["connectivity"]]
    normal = np.cross(
        corners[:, 1] - corners[:, 0], corners[:, 3] - corners[:, 0]
    )
    toward_top = corners[:, 4] - corners[:, 0]
    assert np.all(np.einsum("ij,ij->i", normal, toward_top) > 0.0)


def test_the_point_grid_spans_the_whole_lattice(tmp_path, solved):
    result, model = solved
    grid = read_vtu(openndm.write_vtk(tmp_path / "core.vtu", result, model))

    nz, ny, nx = model.geometry.shape
    assert grid["n_points"] == (nx + 1) * (ny + 1) * (nz + 1)
    assert len(grid["points"]) == grid["n_points"]


def test_subdivision_is_visible_in_the_written_mesh(tmp_path):
    model = quarter_core(dz=[100.0], subdivide=4)
    result = model.solve()
    grid = read_vtu(openndm.write_vtk(tmp_path / "core.vtu", result, model))

    assert grid["n_cells"] == model.geometry.n_nodes
    corners = grid["points"][grid["connectivity"]]
    heights = corners[:, 4:, 2].min(axis=1) - corners[:, :4, 2].max(axis=1)
    np.testing.assert_allclose(heights, 25.0)


def test_an_extra_field_is_written_alongside(tmp_path, solved):
    result, model = solved
    temperature = 560.0 + np.arange(model.geometry.n_nodes, dtype=float)

    written = openndm.write_vtk(
        tmp_path / "core.vtu", result, model, extra={"fuel_temperature": temperature}
    )
    grid = read_vtu(written)

    mapping = model.geometry.lattice_to_node.reshape(model.geometry.shape)
    np.testing.assert_array_equal(
        grid["data"]["fuel_temperature"], temperature[mapping[mapping >= 0]]
    )


def test_an_extra_field_of_the_wrong_length_is_an_error(tmp_path, solved):
    result, model = solved
    with pytest.raises(openndm.InputError, match="one value per node"):
        openndm.write_vtk(
            tmp_path / "core.vtu", result, model, extra={"t": np.zeros(3)}
        )


def test_an_extra_field_may_not_shadow_a_written_one(tmp_path, solved):
    result, model = solved
    with pytest.raises(openndm.InputError, match="shadows"):
        openndm.write_vtk(
            tmp_path / "core.vtu",
            result,
            model,
            extra={"power": np.zeros(model.geometry.n_nodes)},
        )


def test_the_path_is_returned(tmp_path, solved):
    result, model = solved
    path = tmp_path / "core.vtu"
    assert openndm.write_vtk(path, result, model) == path
    assert path.stat().st_size > 0
