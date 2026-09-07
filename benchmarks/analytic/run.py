#!/usr/bin/env python3
"""V-1: bare homogeneous cuboid against the analytic buckling solution.

The only benchmark here whose reference is exact rather than published, and
therefore the one that pins down the spatial discretisation without any
appeal to another code.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from common import benchmark_settings

import openndm

D, SIGMA_A, NU_SIGMA_F, SIDE = 1.0, 0.08, 0.1, 100.0


def analytic() -> float:
    buckling = 3.0 * (np.pi / SIDE) ** 2
    return NU_SIGMA_F / (SIGMA_A + D * buckling)


def main() -> int:
    library = openndm.XSLibrary(1, 1)
    library.set_composition(
        0,
        D=[D],
        absorption=[SIGMA_A],
        nu_fission=[NU_SIGMA_F],
        kappa_fission=[NU_SIGMA_F],
        chi=[1.0],
        scatter=[[0.0]],
    )
    library.finalize()

    reference = analytic()
    settings = benchmark_settings()
    print(f"analytic k_eff = {reference:.9f}\n")
    print(f"{'nodes/side':>10}  {'FDM':>22}  {'NEM':>22}  {'SANM':>22}")
    for n in (4, 8, 16, 32, 64):
        geometry = openndm.Geometry.from_lattice(
            np.zeros((n, n, n), dtype=int),
            pitch=SIDE / n,
            boundaries=dict.fromkeys(
                ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"],
                "zero_flux",
            ),
        )
        cells = []
        for kernel in ("fdm", "nem", "sanm"):
            k = openndm.Model(geometry, library, settings).solve(kernel=kernel).k_eff
            cells.append(f"{k:.8f} ({1e5 * (k - reference):+7.2f} pcm)")
        print(f"{n:>10}  " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
