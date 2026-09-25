"""Shared data for the benchmark decks.

Group constants live here rather than being duplicated per deck, so a
correction to a cross section cannot silently apply to only one of them.
"""

from __future__ import annotations

import numpy as np

import openndm

IAEA_MATERIALS = {
    "fuel_1": (1.5, 0.4, 0.010, 0.080, 0.135, 0.020),
    "fuel_2": (1.5, 0.4, 0.010, 0.085, 0.135, 0.020),
    "fuel_2_rodded": (1.5, 0.4, 0.010, 0.130, 0.135, 0.020),
    "reflector": (2.0, 0.3, 0.000, 0.010, 0.000, 0.040),
    "reflector_rodded": (2.0, 0.3, 0.000, 0.055, 0.000, 0.040),
}
"""IAEA PWR benchmark macroscopic data, ANL-7416 Problem 11-A2.

Columns are D1, D2 in cm, then Sigma_a1, Sigma_a2, nuSigma_f2 and the
1->2 scatter, all cm^-1.
"""

IAEA_ORDER = list(IAEA_MATERIALS)

IAEA_RADIAL_MAP = np.array(
    [
        [3, 2, 2, 2, 3, 2, 2, 1, 4],
        [2, 2, 2, 2, 2, 2, 2, 1, 4],
        [2, 2, 2, 2, 2, 2, 2, 1, 4],
        [2, 2, 2, 2, 2, 2, 2, 1, 4],
        [3, 2, 2, 2, 3, 2, 1, 1, 4],
        [2, 2, 2, 2, 2, 1, 1, 4, 4],
        [2, 2, 2, 2, 1, 1, 4, 4, 0],
        [1, 1, 1, 1, 1, 4, 4, 0, 0],
        [4, 4, 4, 4, 4, 4, 0, 0, 0],
    ]
)
"""Quarter-core radial map, 20 cm assembly pitch.

Row 0 and column 0 lie on the symmetry lines, 0 marks a position outside the
core, and the remaining entries are the 1-based material numbers of
:data:`IAEA_MATERIALS`.
"""


def check_radial_map(core_map: np.ndarray | None = None) -> None:
    """Assert the structural invariants of the quarter-core map.

    The map must be symmetric about the diagonal and the peripheral fuel-1
    zone must be an edge-connected band one cell thick following the core
    outline. A single dropped cell in either is worth about 80 pcm and has no
    other visible symptom.

    Raises
    ------
    ValueError
        If the map is not symmetric about the diagonal, or if the peripheral
        fuel-1 band is not edge-connected.
    """
    m = IAEA_RADIAL_MAP if core_map is None else np.asarray(core_map)
    if not np.array_equal(m, m.T):
        bad = [tuple(int(v) for v in p) for p in np.argwhere(m != m.T)]
        raise ValueError(f"core map is not symmetric about the diagonal: {bad}")

    ring = {tuple(int(v) for v in p) for p in np.argwhere(m == 1)}
    if not ring:
        raise ValueError("core map has no peripheral fuel-1 zone")
    start = next(iter(ring))
    seen, stack = set(), [start]
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (cell[0] + dj, cell[1] + di)
            if nb in ring and nb not in seen:
                stack.append(nb)
    if len(seen) != len(ring):
        raise ValueError(
            f"the peripheral fuel-1 band is not edge-connected: "
            f"{len(ring) - len(seen)} of {len(ring)} cells are detached"
        )

IAEA_BOUNDARIES = {
    "x_min": "reflective",
    "y_min": "reflective",
    "x_max": "zero_flux",
    "y_max": "zero_flux",
}

AXIAL_REFLECTOR = 20.0
"""Thickness of the bottom and of the top axial reflector, cm."""

CORE_HEIGHT = 340.0
"""Height of the active core, cm: z from 20 to 360."""

ROD_TIP_HEIGHT = 80.0
"""Height of the rod tips above the bottom of the active core, cm.

The rods enter from the top and stop here, so they occupy the upper
``CORE_HEIGHT - ROD_TIP_HEIGHT`` = 260 cm of the core. Read the other way
round, as 80 cm of insertion measured down from the top, k_eff comes out
about 1700 pcm high.
"""


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


def benchmark_settings(**overrides) -> openndm.Settings:
    """Iteration-converged settings for a benchmark deck.

    Tighter than the library defaults, so that the reported eigenvalue is
    limited by the spatial discretisation rather than by the iteration. See
    ``benchmarks/README.md``.
    """
    options = {
        "verbosity": 0,
        "k_tolerance": 1.0e-11,
        "fission_source_tolerance": 1.0e-10,
        "inner_tolerance": 1.0e-9,
        "max_inner": 400,
        "max_outer": 5000,
    }
    options.update(overrides)
    return openndm.Settings(**options)


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
