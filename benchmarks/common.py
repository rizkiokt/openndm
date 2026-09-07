"""Shared data for the benchmark decks.

Group constants live here rather than being duplicated per deck, so a
correction to a cross section cannot silently apply to only one of them.
"""

from __future__ import annotations

import numpy as np

import openndm

#: IAEA PWR benchmark macroscopic data (ANL-7416, Problem 11-A2).
#: Columns: D1, D2, Sigma_a1, Sigma_a2, nuSigma_f2, Sigma_s 1->2.
IAEA_MATERIALS = {
    "fuel_1": (1.5, 0.4, 0.010, 0.080, 0.135, 0.020),
    "fuel_2": (1.5, 0.4, 0.010, 0.085, 0.135, 0.020),
    "fuel_2_rodded": (1.5, 0.4, 0.010, 0.130, 0.135, 0.020),
    "reflector": (2.0, 0.3, 0.000, 0.010, 0.000, 0.040),
    "reflector_rodded": (2.0, 0.3, 0.000, 0.055, 0.000, 0.040),
}
IAEA_ORDER = list(IAEA_MATERIALS)

#: Quarter-core radial map, 20 cm assembly pitch. Row 0 and column 0 lie on
#: the symmetry lines; 0 marks a position outside the core.
IAEA_RADIAL_MAP = np.array(
    [
        [3, 2, 2, 2, 3, 2, 2, 1, 4],
        [2, 2, 2, 2, 2, 2, 2, 1, 4],
        [2, 2, 2, 2, 2, 2, 2, 1, 4],
        [2, 2, 2, 2, 2, 2, 2, 1, 4],
        [3, 2, 2, 2, 3, 2, 1, 1, 4],
        [2, 2, 2, 2, 2, 2, 1, 4, 4],
        [2, 2, 2, 2, 1, 1, 4, 4, 0],
        [1, 1, 1, 1, 1, 4, 4, 0, 0],
        [4, 4, 4, 4, 4, 0, 0, 0, 0],
    ]
)

IAEA_BOUNDARIES = {
    "x_min": "reflective",
    "y_min": "reflective",
    "x_max": "zero_flux",
    "y_max": "zero_flux",
}


def iaea_library() -> openndm.XSLibrary:
    """Two-group IAEA library, compositions in :data:`IAEA_ORDER`."""
    lib = openndm.XSLibrary(n_groups=2, n_compositions=len(IAEA_ORDER))
    for index, name in enumerate(IAEA_ORDER):
        d1, d2, a1, a2, f2, s12 = IAEA_MATERIALS[name]
        lib.set_composition(
            index,
            D=[d1, d2],
            absorption=[a1, a2],
            nu_fission=[0.0, f2],
            kappa_fission=[0.0, f2],
            chi=[1.0, 0.0] if f2 > 0.0 else [0.0, 0.0],
            scatter=[[0.0, s12], [0.0, 0.0]],
        )
    lib.finalize()
    return lib


def radial_map_to_compositions() -> np.ndarray:
    """Convert the 1-based benchmark map to 0-based composition indices."""
    return np.where(
        IAEA_RADIAL_MAP == 0, openndm.INACTIVE, IAEA_RADIAL_MAP - 1
    )


def report(name: str, result, reference: float | None = None) -> None:
    """Print a one-line summary in a form that is easy to diff between runs."""
    line = (
        f"{name:<26s} kernel={result.kernel:<5s} "
        f"k_eff={result.k_eff:.6f}  F_q={result.f_q:.4f}  "
        f"F_dH={result.f_dh:.4f}  outers={result.outer_iterations:4d}  "
        f"{result.runtime * 1e3:7.1f} ms"
    )
    if reference is not None:
        line += f"  ref={reference:.6f} ({1e5 * (result.k_eff - reference):+7.1f} pcm)"
    print(line)
