"""Analytic square-root temperature feedback (FR-XS-7)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import openndm

from conftest import ONE_GROUP, analytic_k, cuboid, one_group_library

GAMMA = 2.5e-3
T0 = 300.0
SIDE = 100.0


def doppler_library():
    library = one_group_library()
    return library


def feedback(library, **overrides):
    kwargs = {
        "gamma": GAMMA,
        "reference_temperature": T0,
    }
    kwargs.update(overrides)
    return openndm.DopplerFeedback(library, [0], **kwargs)


def test_factor_is_the_square_root_law():
    model = feedback(doppler_library())
    for temperature in (0.0, 300.0, 900.0, 2000.0):
        expected = 1.0 + GAMMA * (math.sqrt(temperature) - math.sqrt(T0))
        assert model.factor(temperature) == pytest.approx(expected, rel=1e-15)
    assert model.factor(T0) == pytest.approx(1.0, abs=1e-15)


def test_the_reference_temperature_restores_the_library_exactly():
    library = doppler_library()
    before = np.asarray(library.composition(0).absorption, dtype=float).copy()
    model = feedback(library)
    model.apply(1200.0)
    assert not np.allclose(library.composition(0).absorption, before)
    model.apply(T0)
    assert np.array_equal(library.composition(0).absorption, before), (
        "returning to the reference temperature must restore the base data"
    )


def test_temperatures_are_absolute_not_incremental():
    library_a = doppler_library()
    model_a = feedback(library_a)
    model_a.apply(800.0)
    model_a.apply(1500.0)

    library_b = doppler_library()
    feedback(library_b).apply(1500.0)

    assert np.array_equal(
        library_a.composition(0).absorption, library_b.composition(0).absorption
    )


def test_the_eigenvalue_matches_the_analytic_value_at_temperature(tight):
    """k = nuSf / (Sigma_a(T) + D B^2), with Sigma_a scaled by the law."""
    library = doppler_library()
    model = feedback(library)
    geometry = cuboid(32, SIDE)
    reactor = openndm.Model(geometry, library, tight)

    for temperature in (T0, 900.0, 1800.0):
        model.apply(temperature)
        reactor.refresh()
        expected = analytic_k(
            side=SIDE,
            D=ONE_GROUP["D"],
            absorption=ONE_GROUP["absorption"] * model.factor(temperature),
            nu_fission=ONE_GROUP["nu_fission"],
        )
        error_pcm = 1.0e5 * (reactor.solve().k_eff - expected)
        assert abs(error_pcm) < 5.0, (
            f"at {temperature} K the eigenvalue is {error_pcm:+.2f} pcm from "
            f"the analytic value"
        )


def test_doppler_feedback_is_negative(tight):
    """Heating the fuel must remove reactivity."""
    library = doppler_library()
    model = feedback(library)
    reactor = openndm.Model(cuboid(16, SIDE), library, tight)

    model.apply(T0)
    reactor.refresh()
    cold = reactor.solve().k_eff
    model.apply(1500.0)
    reactor.refresh()
    hot = reactor.solve().k_eff

    worth_pcm = 1.0e5 * (1.0 / hot - 1.0 / cold)
    assert worth_pcm > 100.0, (
        f"heating 300 K -> 1500 K gave {worth_pcm:+.1f} pcm; Doppler must be negative"
    )


def test_only_the_named_groups_are_scaled():
    library = openndm.XSLibrary(2, 1)
    library.set_composition(
        0,
        D=[1.5, 0.4],
        absorption=[0.01, 0.085],
        nu_fission=[0.0, 0.135],
        kappa_fission=[0.0, 0.135],
        chi=[1.0, 0.0],
        scatter=[[0.0, 0.02], [0.0, 0.0]],
    )
    library.finalize(warn=False)
    model = openndm.DopplerFeedback(
        library, [0], gamma=GAMMA, reference_temperature=T0, groups=[1]
    )
    model.apply(1200.0)
    absorption = np.asarray(library.composition(0).absorption, dtype=float)
    assert absorption[0] == pytest.approx(0.01, rel=1e-15), "fast group moved"
    assert absorption[1] == pytest.approx(0.085 * model.factor(1200.0))


def test_other_compositions_are_untouched():
    library = openndm.XSLibrary(1, 2)
    for index in (0, 1):
        library.set_composition(
            index,
            D=[1.0],
            absorption=[0.08],
            nu_fission=[0.1],
            kappa_fission=[0.1],
            chi=[1.0],
            scatter=[[0.0]],
        )
    library.finalize(warn=False)
    openndm.DopplerFeedback(library, [0], gamma=GAMMA, reference_temperature=T0).apply(
        1500.0
    )
    assert library.composition(1).absorption[0] == pytest.approx(0.08)
    assert library.composition(0).absorption[0] != pytest.approx(0.08)


def test_every_field_survives_the_round_trip():
    """set_composition writes a whole composition, so the rest must be kept."""
    library = openndm.XSLibrary(2, 1)
    library.set_composition(
        0,
        D=[1.5, 0.4],
        absorption=[0.01, 0.085],
        nu_fission=[0.0, 0.135],
        kappa_fission=[0.0, 0.130],
        chi=[1.0, 0.0],
        scatter=[[0.0, 0.02], [0.001, 0.0]],
        inv_velocity=[1.0e-7, 1.0e-5],
    )
    library.finalize(warn=False)
    openndm.DopplerFeedback(library, [0], gamma=GAMMA, reference_temperature=T0).apply(
        1000.0
    )

    composition = library.composition(0)
    assert list(composition.D) == [1.5, 0.4]
    assert list(composition.nu_fission) == [0.0, 0.135]
    assert list(composition.kappa_fission) == [0.0, 0.130]
    assert list(composition.chi) == [1.0, 0.0]
    assert list(composition.scatter) == [0.0, 0.02, 0.001, 0.0]
    assert list(composition.inv_velocity) == [1.0e-7, 1.0e-5]


def test_apply_leaves_the_library_solvable(tight):
    library = doppler_library()
    reactor = openndm.Model(cuboid(8, SIDE), library, tight)
    feedback(library).apply(1000.0)
    assert library.finalized, "apply must re-finalize"
    reactor.refresh()
    assert reactor.solve().converged


def test_a_per_composition_temperature_distribution():
    library = openndm.XSLibrary(1, 3)
    for index in range(3):
        library.set_composition(
            index,
            D=[1.0],
            absorption=[0.08],
            nu_fission=[0.1],
            kappa_fission=[0.1],
            chi=[1.0],
            scatter=[[0.0]],
        )
    library.finalize(warn=False)
    model = openndm.DopplerFeedback(
        library, [0, 1, 2], gamma=GAMMA, reference_temperature=T0
    )
    model.apply([300.0, 900.0, 1500.0])
    for index, temperature in enumerate((300.0, 900.0, 1500.0)):
        assert library.composition(index).absorption[0] == pytest.approx(
            0.08 * model.factor(temperature)
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"field": "chi"}, "cannot apply feedback to"),
        ({"reference_temperature": 0.0}, "above absolute zero"),
        ({"groups": [3]}, "outside the library"),
    ],
)
def test_construction_is_validated(kwargs, match):
    with pytest.raises(openndm.InputError, match=match):
        feedback(doppler_library(), **kwargs)


def test_unknown_composition_is_rejected():
    with pytest.raises(openndm.InputError, match="outside the library"):
        openndm.DopplerFeedback(
            doppler_library(), [7], gamma=GAMMA, reference_temperature=T0
        )
    with pytest.raises(openndm.InputError, match="no compositions given"):
        openndm.DopplerFeedback(
            doppler_library(), [], gamma=GAMMA, reference_temperature=T0
        )


def test_application_is_validated():
    model = feedback(doppler_library())
    with pytest.raises(openndm.InputError, match="must not be negative"):
        model.apply(-1.0)
    with pytest.raises(openndm.InputError, match="expected 1 or 1 temperatures"):
        model.apply([300.0, 400.0])


def test_a_negative_cross_section_is_refused():
    """Silence here would be a physically impossible library."""
    model = feedback(doppler_library(), gamma=-0.5)
    with pytest.raises(openndm.InputError, match="negative absorption"):
        model.apply(2000.0)
