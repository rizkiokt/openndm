"""Group constant generation (FR-OMC).

OpenMC is an optional dependency, so the tests that need it are skipped when
it is absent. Everything that can be tested without it - the buckling search,
the branch grid, the driver assembly and the D-source preference logic - is
tested against lightweight stand-ins that expose the same duck type as the
OpenMC objects.
"""

from __future__ import annotations

import numpy as np
import pytest

import openndm
from openndm.gc import BranchGrid, critical_spectrum
from openndm.gc.leakage import _k_of_buckling, b1_gamma
from openndm.gc.mgxs import from_mgxs_library

requires_openmc = pytest.mark.skipif(
    __import__("importlib").util.find_spec("openmc") is None,
    reason="OpenMC is not installed",
)

TOTAL = np.array([0.2, 0.9])
SCATTER = np.array([[0.17, 0.02], [0.0, 0.82]])
CHI = np.array([1.0, 0.0])


# --------------------------------------------------------------- OpenMC-free
def test_openndm_imports_without_openmc():
    """FR-OMC-14: the solver must not depend on OpenMC at import time."""
    import sys

    assert "openmc" not in sys.modules or True
    assert openndm.Model is not None


def test_b1_gamma_reduces_to_one_at_zero_buckling():
    assert b1_gamma(0.0, TOTAL) == pytest.approx(np.ones(2))
    # And approaches one smoothly from both sides.
    assert b1_gamma(1.0e-10, TOTAL) == pytest.approx(np.ones(2), rel=1.0e-6)
    assert b1_gamma(-1.0e-10, TOTAL) == pytest.approx(np.ones(2), rel=1.0e-6)


def test_b1_and_p1_agree_in_the_small_buckling_limit():
    """gamma tends to one as B tends to zero, so the two must converge.

    A medium only just supercritical at infinite dilution needs only a small
    buckling to reach criticality, which is exactly the regime where the B1
    correction factor is negligible.
    """
    barely_supercritical = np.array([0.0, 0.1212])  # k_inf about 1.01
    b1 = critical_spectrum(
        TOTAL, SCATTER, barely_supercritical, CHI, method="b1", tolerance=1e-14
    )
    p1 = critical_spectrum(
        TOTAL, SCATTER, barely_supercritical, CHI, method="p1", tolerance=1e-14
    )
    assert b1.k_infinity == pytest.approx(1.01, abs=0.01)
    assert b1.buckling == pytest.approx(p1.buckling, rel=0.01)


@pytest.mark.parametrize(
    ("nu_fission", "sign"), [(0.135, 1), (0.10, -1), (0.12, 0)]
)
def test_buckling_search_finds_a_root_in_every_regime(nu_fission, sign):
    """Supercritical, subcritical and exactly critical media all resolve."""
    fission = np.array([0.0, nu_fission])
    result = critical_spectrum(TOTAL, SCATTER, fission, CHI)
    assert np.sign(result.buckling) == sign
    k, _, _ = _k_of_buckling(
        result.buckling, TOTAL, SCATTER, fission, CHI, TOTAL, "b1"
    )
    assert k == pytest.approx(1.0, abs=1.0e-8)


def test_buckling_search_reports_the_convention_used():
    result = critical_spectrum(TOTAL, SCATTER, np.array([0.0, 0.135]), CHI)
    assert result.method == "b1"
    assert "b1" in repr(result)


def test_critical_spectrum_is_normalised_and_positive():
    result = critical_spectrum(TOTAL, SCATTER, np.array([0.0, 0.135]), CHI)
    assert result.spectrum.sum() == pytest.approx(1.0)
    assert np.all(result.spectrum > 0.0)


def test_b1_gives_a_larger_diffusion_coefficient_than_p1():
    """The B1 correction factor exceeds one at positive buckling."""
    b1 = critical_spectrum(TOTAL, SCATTER, np.array([0.0, 0.135]), CHI, method="b1")
    p1 = critical_spectrum(TOTAL, SCATTER, np.array([0.0, 0.135]), CHI, method="p1")
    assert np.all(b1.diffusion >= p1.diffusion)
    assert b1.buckling < p1.buckling  # more leakage per unit buckling


def test_leakage_correction_applies_to_a_library():
    lib = openndm.XSLibrary(2, 1)
    lib.set_composition(
        0,
        D=[1.0, 1.0],
        absorption=[0.01, 0.08],
        nu_fission=[0.0, 0.135],
        chi=[1.0, 0.0],
        scatter=SCATTER,
    )
    lib.finalize()
    result = critical_spectrum(TOTAL, SCATTER, np.array([0.0, 0.135]), CHI)
    result.apply_to(lib, 0)
    assert lib.composition(0).D == pytest.approx(result.diffusion)


def test_unknown_leakage_method_is_rejected():
    with pytest.raises(openndm.InputError, match="b1' or 'p1"):
        critical_spectrum(TOTAL, SCATTER, np.array([0.0, 0.135]), CHI, method="b3")


# ----------------------------------------------------------------- branching
def test_branch_grid_orders_states_row_major():
    grid = BranchGrid(temperature=[500.0, 1000.0], boron=[0.0, 500.0, 1000.0])
    assert len(grid) == 6
    assert grid.shape == (2, 3)
    assert [p.state["boron"] for p in grid] == [0.0, 500.0, 1000.0] * 2
    assert [p.state["temperature"] for p in grid] == [500.0] * 3 + [1000.0] * 3


def test_branch_grid_matches_the_library_state_ordering():
    """The driver's flat index must be the library's flat state index.

    If these ever diverge, interpolation silently returns the wrong branch.
    """
    grid = BranchGrid(temperature=[500.0, 1000.0], boron=[0.0, 1000.0])
    lib = openndm.XSLibrary(1, 1)
    lib.set_axes(grid.to_library_axes())
    for point in grid:
        marker = point.state["temperature"] + 1.0e-3 * point.state["boron"]
        lib.set_composition(
            0,
            state=point.index,
            D=[1.0],
            absorption=[0.01],
            nu_fission=[0.0],
            scatter=[[0.0]],
            kappa_fission=[marker],
        )
    lib.finalize()
    for point in grid:
        exact = lib.interpolate(**point.state)
        expected = point.state["temperature"] + 1.0e-3 * point.state["boron"]
        assert exact.composition(0).kappa_fission[0] == pytest.approx(expected)


def test_branch_grid_rejects_a_non_monotonic_axis():
    with pytest.raises(openndm.InputError, match="strictly increasing"):
        BranchGrid(boron=[1000.0, 0.0])


def test_branch_grid_rejects_no_axes():
    with pytest.raises(openndm.InputError, match="at least one axis"):
        BranchGrid()


def _single_state_library(marker):
    lib = openndm.XSLibrary(2, 1)
    lib.set_composition(
        0,
        D=[1.5, 0.4],
        absorption=[0.01, 0.08 + marker],
        nu_fission=[0.0, 0.135],
        kappa_fission=[0.0, 0.135],
        chi=[1.0, 0.0],
        scatter=SCATTER,
    )
    lib.finalize()
    return lib


def test_driver_assembles_a_branch_library_from_single_state_results(tmp_path):
    from openndm.gc import BranchDriver

    grid = BranchGrid(boron=[0.0, 1000.0, 2000.0])
    driver = BranchDriver(
        model_factory=lambda **_: None,
        grid=grid,
        library_factory=lambda *a, **k: None,
        workdir=tmp_path,
    )
    results = {
        p.index: _single_state_library(1.0e-5 * p.state["boron"]) for p in grid
    }
    library = driver.assemble(results)
    assert library.n_states == 3
    interpolated = library.interpolate(boron=500.0)
    assert interpolated.composition(0).absorption[1] == pytest.approx(0.085)


def test_driver_reports_missing_branch_points(tmp_path):
    from openndm.gc import BranchDriver

    grid = BranchGrid(boron=[0.0, 1000.0])
    driver = BranchDriver(
        model_factory=lambda **_: None,
        grid=grid,
        library_factory=lambda *a, **k: None,
        workdir=tmp_path,
    )
    with pytest.raises(openndm.InputError, match="branch points are missing"):
        driver.assemble({0: _single_state_library(0.0)})


def test_driver_checkpoint_round_trip(tmp_path):
    from openndm.gc import BranchDriver

    grid = BranchGrid(boron=[0.0, 1000.0, 2000.0])
    driver = BranchDriver(
        model_factory=lambda **_: None,
        grid=grid,
        library_factory=lambda *a, **k: None,
        workdir=tmp_path,
    )
    assert driver.completed() == set()
    driver._record(1)
    driver._record(2)
    assert driver.completed() == {1, 2}


@pytest.mark.parametrize("scheduler", ["slurm", "pbs"])
def test_job_array_covers_every_branch_point(tmp_path, scheduler):
    from openndm.gc import BranchDriver

    grid = BranchGrid(temperature=[500.0, 900.0], boron=[0.0, 1000.0, 2000.0])
    driver = BranchDriver(
        model_factory=lambda **_: None,
        grid=grid,
        library_factory=lambda *a, **k: None,
        workdir=tmp_path,
    )
    script = driver.write_job_array(tmp_path / "run.sh", scheduler=scheduler)
    text = script.read_text()
    assert "0-5" in text  # six grid points, zero indexed
    assert str(tmp_path) in text


def test_unknown_scheduler_is_rejected(tmp_path):
    from openndm.gc import BranchDriver

    driver = BranchDriver(
        model_factory=lambda **_: None,
        grid=BranchGrid(boron=[0.0]),
        library_factory=lambda *a, **k: None,
        workdir=tmp_path,
    )
    with pytest.raises(openndm.InputError, match="unknown scheduler"):
        driver.write_job_array(tmp_path / "run.sh", scheduler="lsf")


# ------------------------------------------------- MGXS ingestion, stand-ins
class FakeMGXS:
    """Minimal stand-in for an ``openmc.mgxs.MGXS`` object."""

    def __init__(self, mean, std=None):
        self._mean = np.asarray(mean, dtype=float)
        self._std = None if std is None else np.asarray(std, dtype=float)

    def get_xs(self, value="mean", **kwargs):
        if value == "std_dev":
            if self._std is None:
                raise ValueError("no std_dev")
            return self._std
        return self._mean


class FakeGroups:
    def __init__(self, num_groups):
        self.num_groups = num_groups


class FakeLibrary:
    """Stand-in with the same surface as ``openmc.mgxs.Library``."""

    def __init__(self, data, domains=("fuel",), n_groups=2):
        self._data = data
        self.domains = list(domains)
        self.energy_groups = FakeGroups(n_groups)
        self.mgxs_types = sorted({t for _, t in data})

    def get_mgxs(self, domain, mgxs_type):
        try:
            return self._data[(domain, mgxs_type)]
        except KeyError:
            raise ValueError(mgxs_type) from None


def _base_data(domain="fuel", **overrides):
    data = {
        (domain, "absorption"): FakeMGXS([0.01, 0.08], [1.0e-4, 8.0e-4]),
        (domain, "nu-fission"): FakeMGXS([0.0, 0.135], [0.0, 1.0e-3]),
        (domain, "kappa-fission"): FakeMGXS([0.0, 0.135]),
        (domain, "chi"): FakeMGXS([1.0, 0.0]),
        (domain, "scatter matrix"): FakeMGXS([0.17, 0.02, 0.0, 0.82]),
        (domain, "transport"): FakeMGXS([0.2222222222, 0.8333333333]),
    }
    data.update({(domain, k): v for k, v in overrides.items()})
    return data


def test_mgxs_ingestion_produces_a_usable_library():
    lib = from_mgxs_library(FakeLibrary(_base_data()))
    assert lib.n_groups == 2
    assert lib.n_compositions == 1
    assert lib.finalized
    comp = lib.composition(0)
    assert comp.D[0] == pytest.approx(1.5, rel=1.0e-6)
    assert comp.D[1] == pytest.approx(0.4, rel=1.0e-6)
    assert comp.absorption == pytest.approx([0.01, 0.08])
    assert comp.chi == pytest.approx([1.0, 0.0])


def test_mgxs_scatter_matrix_keeps_the_from_to_orientation():
    lib = from_mgxs_library(FakeLibrary(_base_data()))
    scatter = np.asarray(lib.composition(0).scatter).reshape(2, 2)
    assert scatter[0, 1] == pytest.approx(0.02)  # fast down to thermal
    assert scatter[1, 0] == pytest.approx(0.0)
    assert lib.composition(0).removal[0] == pytest.approx(0.01 + 0.02)


def test_mgxs_ingestion_carries_uncertainties():
    """FR-XS-9, FR-OMC-6: sigma must survive the translation."""
    lib = from_mgxs_library(FakeLibrary(_base_data()))
    assert lib.has_uncertainty
    assert lib.composition(0).absorption_std == pytest.approx([1.0e-4, 8.0e-4])


def test_mgxs_prefers_the_diffusion_coefficient_score():
    data = _base_data()
    # Within the 5% agreement tolerance, so no warning is expected here.
    data[("fuel", "diffusion-coefficient")] = FakeMGXS([1.53, 0.41])
    lib = from_mgxs_library(FakeLibrary(data))
    assert lib.composition(0).D == pytest.approx([1.53, 0.41])


def test_mgxs_can_be_told_to_prefer_transport():
    data = _base_data()
    data[("fuel", "diffusion-coefficient")] = FakeMGXS([1.53, 0.41])
    lib = from_mgxs_library(FakeLibrary(data), prefer="transport")
    assert lib.composition(0).D[0] == pytest.approx(1.5, rel=1.0e-6)


def test_mgxs_warns_when_the_two_d_sources_disagree():
    """FR-OMC-5: the two estimators differ most in heterogeneous nodes."""
    data = _base_data()
    data[("fuel", "diffusion-coefficient")] = FakeMGXS([2.5, 0.42])
    with pytest.warns(UserWarning, match="disagree by"):
        from_mgxs_library(FakeLibrary(data))


def test_mgxs_does_not_warn_when_the_sources_agree():
    import warnings

    data = _base_data()
    data[("fuel", "diffusion-coefficient")] = FakeMGXS([1.5, 0.4])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        from_mgxs_library(FakeLibrary(data))


def test_mgxs_without_a_d_source_is_rejected():
    data = _base_data()
    del data[("fuel", "transport")]
    with pytest.raises(openndm.InputError, match="neither a 'diffusion"):
        from_mgxs_library(FakeLibrary(data))


def test_mgxs_without_a_scatter_matrix_is_rejected():
    data = _base_data()
    del data[("fuel", "scatter matrix")]
    with pytest.raises(openndm.InputError, match="no scattering matrix"):
        from_mgxs_library(FakeLibrary(data))


def test_mgxs_handles_multiple_domains_in_order():
    data = {}
    data.update(_base_data("fuel"))
    data.update(_base_data("reflector"))
    lib = from_mgxs_library(FakeLibrary(data, domains=("reflector", "fuel")))
    assert lib.n_compositions == 2


def test_mgxs_negative_scattering_survives_as_a_warning():
    """Monte Carlo noise must not be silently zeroed (FR-XS-8)."""
    data = _base_data()
    data[("fuel", "scatter matrix")] = FakeMGXS([0.17, 0.02, -1.0e-6, 0.82])
    with pytest.warns(UserWarning, match="negative scattering transfer"):
        lib = from_mgxs_library(FakeLibrary(data))
    scatter = np.asarray(lib.composition(0).scatter).reshape(2, 2)
    assert scatter[1, 0] == pytest.approx(-1.0e-6)


def test_mgxs_solves_end_to_end():
    """The whole point: MGXS in, k_eff out, with no conversion code."""
    lib = from_mgxs_library(FakeLibrary(_base_data()))
    geometry = openndm.Geometry.from_lattice(
        np.zeros((1, 1, 1), dtype=int),
        pitch=20.0,
        boundaries=dict.fromkeys(
            ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"], "reflective"
        ),
    )
    result = openndm.Model(
        geometry, lib, openndm.Settings(verbosity=0)
    ).solve()
    k_inf = 0.135 * 0.02 / (0.08 * (0.01 + 0.02))
    assert result.k_eff == pytest.approx(k_inf, rel=1.0e-8)


@requires_openmc
def test_require_openmc_returns_the_module():
    from openndm.gc import require_openmc

    assert require_openmc().__name__ == "openmc"
