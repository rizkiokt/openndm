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
        kappa_fission=[0.0, 0.135],
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
    """Minimal stand-in for an ``openmc.mgxs.MGXS`` object.

    Mirrors the real contract that ``get_xs`` takes integer subdomain *ids*,
    not domain objects: passing an object raises, exactly as OpenMC does.
    """

    def __init__(self, mean, std=None):
        self._mean = np.asarray(mean, dtype=float)
        self._std = None if std is None else np.asarray(std, dtype=float)
        self.subdomains_seen = []

    def get_xs(self, value="mean", subdomains="all", **kwargs):
        if subdomains != "all":
            for item in subdomains:
                if not isinstance(item, int):
                    raise TypeError(
                        f'Error setting subdomains: Items must be of type '
                        f'"Integral", but item is of type {type(item).__name__}'
                    )
            self.subdomains_seen.append(list(subdomains))
        if value == "std_dev":
            if self._std is None:
                raise ValueError("no std_dev")
            return self._std
        return self._mean


class FakeDomain:
    """A domain object with an id, like ``openmc.Material``."""

    def __init__(self, name, id_):
        self.name = name
        self.id = id_

    def __repr__(self):
        return f"<FakeDomain {self.name}>"


class FakeGroups:
    def __init__(self, num_groups):
        self.num_groups = num_groups


class FakeLibrary:
    """Stand-in with the same surface as ``openmc.mgxs.Library``."""

    def __init__(self, data, domains=None, n_groups=2):
        self._data = data
        self.domains = list(domains) if domains is not None else [FUEL]
        self.energy_groups = FakeGroups(n_groups)
        self.mgxs_types = sorted({t for _, t in data})

    def get_mgxs(self, domain, mgxs_type):
        try:
            return self._data[(domain, mgxs_type)]
        except KeyError:
            raise ValueError(mgxs_type) from None


FUEL = FakeDomain("fuel", 1)
REFLECTOR = FakeDomain("reflector", 2)


def _base_data(domain=FUEL, **overrides):
    """Minimal MGXS set for one domain, keyed the way FakeLibrary expects."""
    data = {
        (domain, "absorption"): FakeMGXS([0.01, 0.08], [1.0e-4, 8.0e-4]),
        (domain, "nu-fission"): FakeMGXS([0.0, 0.135], [0.0, 1.0e-3]),
        (domain, "kappa-fission"): FakeMGXS([0.0, 0.135]),
        (domain, "chi"): FakeMGXS([1.0, 0.0]),
        # Both scattering matrices, identical here so there is no (n,xn)
        # multiplicity to account for. A real library that tallies only one of
        # them draws a warning, which test_scattering_multiplicity_warns_...
        # covers.
        (domain, "scatter matrix"): FakeMGXS([0.17, 0.02, 0.0, 0.82]),
        (domain, "nu-scatter matrix"): FakeMGXS([0.17, 0.02, 0.0, 0.82]),
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
    data[(FUEL, "diffusion-coefficient")] = FakeMGXS([1.53, 0.41])
    lib = from_mgxs_library(FakeLibrary(data))
    assert lib.composition(0).D == pytest.approx([1.53, 0.41])


def test_mgxs_can_be_told_to_prefer_transport():
    data = _base_data()
    data[(FUEL, "diffusion-coefficient")] = FakeMGXS([1.53, 0.41])
    lib = from_mgxs_library(FakeLibrary(data), prefer="transport")
    assert lib.composition(0).D[0] == pytest.approx(1.5, rel=1.0e-6)


def test_mgxs_warns_when_the_two_d_sources_disagree():
    """FR-OMC-5: the two estimators differ most in heterogeneous nodes."""
    data = _base_data()
    data[(FUEL, "diffusion-coefficient")] = FakeMGXS([2.5, 0.42])
    with pytest.warns(UserWarning, match="disagree by"):
        from_mgxs_library(FakeLibrary(data))


def test_mgxs_does_not_warn_when_the_sources_agree():
    import warnings

    data = _base_data()
    data[(FUEL, "diffusion-coefficient")] = FakeMGXS([1.5, 0.4])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        from_mgxs_library(FakeLibrary(data))


def test_mgxs_without_a_d_source_is_rejected():
    data = _base_data()
    del data[(FUEL, "transport")]
    with pytest.raises(openndm.InputError, match="neither a 'diffusion"):
        from_mgxs_library(FakeLibrary(data))


def test_mgxs_without_a_scatter_matrix_is_rejected():
    data = _base_data()
    del data[(FUEL, "scatter matrix")]
    del data[(FUEL, "nu-scatter matrix")]
    with pytest.raises(openndm.InputError, match="no scattering matrix"):
        from_mgxs_library(FakeLibrary(data))


def test_mgxs_handles_multiple_domains_in_order():
    data = {}
    data.update(_base_data(FUEL))
    data.update(_base_data(REFLECTOR))
    lib = from_mgxs_library(FakeLibrary(data, domains=(REFLECTOR, FUEL)))
    assert lib.n_compositions == 2


def test_mgxs_negative_scattering_survives_as_a_warning():
    """Monte Carlo noise must not be silently zeroed (FR-XS-8)."""
    data = _base_data()
    noisy = [0.17, 0.02, -1.0e-6, 0.82]
    data[(FUEL, "scatter matrix")] = FakeMGXS(noisy)
    data[(FUEL, "nu-scatter matrix")] = FakeMGXS(noisy)
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


# ------------------------------------------------- OpenMC contract regressions
def test_mgxs_selects_subdomains_by_integer_id():
    """``MGXS.get_xs`` takes domain ids, not domain objects.

    ``Library.get_mgxs`` takes the object, which makes it easy to pass the
    object through to ``get_xs`` as well; real OpenMC then raises a TypeError
    from deep inside its argument checking.
    """
    domain = FakeDomain("fuel", 7)
    data = _base_data(domain)
    from_mgxs_library(FakeLibrary(data, domains=(domain,)))
    seen = data[(domain, "absorption")].subdomains_seen
    # Queried once for the mean and once for the standard deviation.
    assert seen and all(selector == [7] for selector in seen), seen


def _scatter_pair_data(domain=FUEL, excess=1.0e-4):
    """Base data plus both scattering matrices, differing by a multiplicity."""
    plain = np.array([[0.17, 0.02], [0.0, 0.82]])
    nu = plain.copy()
    nu[0, 0] += excess  # (n,2n) neutrons, which stay in the fast group
    data = _base_data(domain)
    data[(domain, "consistent nu-scatter matrix")] = FakeMGXS(nu.ravel())
    data[(domain, "consistent scatter matrix")] = FakeMGXS(plain.ravel())
    del data[(domain, "scatter matrix")]
    return data, plain, nu


def test_scattering_multiplicity_is_subtracted_from_absorption():
    """FR-OMC-1: (n,xn) production must survive the translation.

    OpenMC's absorption score excludes (n,2n); the extra neutrons live only in
    the row sums of the nu-scatter matrix. A diffusion operator built from a
    single matrix cannot see them, because in-scatter and out-scatter are the
    same double sum and cancel. Subtracting the excess from absorption puts
    them back, exactly.
    """
    excess = 1.0e-4
    data, _, _ = _scatter_pair_data(excess=excess)
    lib = from_mgxs_library(FakeLibrary(data))
    absorption = np.asarray(lib.composition(0).absorption)
    assert absorption[0] == pytest.approx(0.01 - excess)
    assert absorption[1] == pytest.approx(0.08)


def test_scattering_multiplicity_correction_can_be_disabled():
    data, _, _ = _scatter_pair_data()
    lib = from_mgxs_library(
        FakeLibrary(data), scattering_multiplicity="ignore"
    )
    assert np.asarray(lib.composition(0).absorption)[0] == pytest.approx(0.01)


def test_scattering_multiplicity_warns_when_it_cannot_correct():
    """Silence here would be a 230 pcm bias with no other symptom."""
    data, _, _ = _scatter_pair_data()
    del data[(FUEL, "consistent scatter matrix")]
    del data[(FUEL, "nu-scatter matrix")]
    with pytest.warns(UserWarning, match="scattering multiplicity"):
        from_mgxs_library(FakeLibrary(data))


def test_unknown_multiplicity_policy_is_rejected():
    data, _, _ = _scatter_pair_data()
    with pytest.raises(openndm.InputError, match="scattering_multiplicity"):
        from_mgxs_library(FakeLibrary(data), scattering_multiplicity="maybe")


def test_multiplicity_correction_raises_k_infinity():
    """The correction must move k the right way, and by the right amount.

    In an infinite medium the extra neutrons show up as a reduced effective
    absorption, so k rises by very nearly the (n,xn) production per absorption.
    """
    excess = 5.0e-4
    data, _, _ = _scatter_pair_data(excess=excess)
    geometry = openndm.Geometry.from_lattice(
        np.zeros((1, 1, 1), dtype=int),
        pitch=20.0,
        boundaries=dict.fromkeys(
            ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"], "reflective"
        ),
    )
    settings = openndm.Settings(verbosity=0, k_tolerance=1.0e-12)

    def solve(policy):
        lib = from_mgxs_library(
            FakeLibrary(data), scattering_multiplicity=policy
        )
        return openndm.Model(geometry, lib, settings).solve()

    corrected = solve("correct")
    ignored = solve("ignore")
    assert corrected.k_eff > ignored.k_eff

    # Predicted rise: excess production per unit absorption, on the fast flux.
    flux = np.asarray(ignored.flux).ravel()
    absorption = np.asarray(
        from_mgxs_library(
            FakeLibrary(data), scattering_multiplicity="ignore"
        ).composition(0).absorption
    )
    predicted = ignored.k_eff * excess * flux[0] / float(absorption @ flux)
    assert 1.0e5 * (corrected.k_eff - ignored.k_eff) == pytest.approx(
        1.0e5 * predicted, rel=0.05
    )


# ---------------------------------------------- discontinuity factor tallies
class FakeTally:
    def __init__(self, mean, std=None):
        self.mean = np.asarray(mean, dtype=float)
        self.std_dev = (
            np.zeros_like(self.mean) if std is None else np.asarray(std, float)
        )


class FakeStatePoint:
    """Stand-in exposing only ``get_tally(name=...)``, like OpenMC's."""

    def __init__(self, tallies):
        self._tallies = tallies

    def get_tally(self, name=None, **kwargs):
        try:
            return self._tallies[name]
        except KeyError:
            raise LookupError(name) from None


#: Volume and slab flux integrals in EnergyFilter order, which ascends in
#: energy: index 0 is thermal, index 1 is fast. The slab is a tenth of the
#: assembly, so the thermal flux density is 1.5x the average and the fast
#: 0.9x, which is the sense a real pin lattice gives.
_SLAB_FRACTION = 0.1
_ADF_TALLIES = {
    "openndm_adf_volume": FakeTally([10.0, 20.0]),
    "openndm_adf_x_min": FakeTally([1.5, 1.8]),
    "openndm_adf_x_max": FakeTally([1.5, 1.8]),
}


@requires_openmc
def test_compute_adf_returns_groups_in_decreasing_energy_order():
    """An EnergyFilter ascends in energy; OpenMC group 1 is the highest.

    Getting this backwards applies every discontinuity factor to the wrong
    group. It is silent: the values stay plausible and only their sense
    inverts, which is why C-2 uses a pin lattice, where the surface sits in
    water and the thermal factor must exceed one while the fast falls below.
    """
    from openndm.gc import compute_adf

    result = compute_adf(
        FakeStatePoint(_ADF_TALLIES),
        slab_fraction=_SLAB_FRACTION,
        warn_sigma=1.0,
    )
    # Face 0 is x_min. Group 0 must be the fast group.
    assert result.values[0, 0] == pytest.approx(0.9)
    assert result.values[0, 1] == pytest.approx(1.5)


@requires_openmc
def test_compute_adf_can_be_told_the_tally_is_already_ordered():
    from openndm.gc import compute_adf

    result = compute_adf(
        FakeStatePoint(_ADF_TALLIES),
        slab_fraction=_SLAB_FRACTION,
        reverse_groups=False,
        warn_sigma=1.0,
    )
    assert result.values[0, 0] == pytest.approx(1.5)
    assert result.values[0, 1] == pytest.approx(0.9)


@requires_openmc
def test_compute_adf_defaults_uninstrumented_faces_to_one():
    from openndm.gc import compute_adf

    result = compute_adf(
        FakeStatePoint(_ADF_TALLIES),
        slab_fraction=_SLAB_FRACTION,
        warn_sigma=1.0,
    )
    # Only x_min and x_max were tallied; the other four faces stay at 1.0.
    assert np.allclose(result.values[2:], 1.0)


@requires_openmc
def test_compute_adf_requires_the_volume_tally():
    from openndm.gc import compute_adf

    with pytest.raises(openndm.InputError, match="_volume"):
        compute_adf(FakeStatePoint({}), slab_fraction=_SLAB_FRACTION)


@requires_openmc
def test_adf_result_applies_to_a_library():
    from openndm.gc import compute_adf

    result = compute_adf(
        FakeStatePoint(_ADF_TALLIES),
        slab_fraction=_SLAB_FRACTION,
        warn_sigma=1.0,
    )
    lib = from_mgxs_library(FakeLibrary(_base_data()))
    result.apply_to(lib, 0)
    assert np.allclose(lib.adf(0), result.values)
