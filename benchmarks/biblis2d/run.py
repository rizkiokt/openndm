#!/usr/bin/env python3
"""BIBLIS 2D PWR benchmark, quarter core, two groups.

A 9x9 quarter core of 23.1226 cm assemblies over eight compositions, with
half-width assemblies on the two symmetry faces. The deck's assembly
divisions make the node mesh uniform at 11.5613 cm, so it runs at 17x17.

The data is parsed from the KOMODO sample deck rather than transcribed; see
``biblis_data.py`` for the provenance. An earlier attempt at this benchmark was
written from memory and **not shipped**, because the map it produced used
five of the eight compositions and put fuel where the reflector belongs.
The reference it was written against was wrong too: 1.02513 against the
1.02511 the source deck states. Fitting a map to a published eigenvalue
would have reproduced a wrong number with a wrong core and looked like a
pass.

Run with ``python run.py``; pass ``--subdivide N`` to refine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from biblis_data import COMPOSITIONS, DZ, NODE_MAP, NODE_WIDTH, PUBLISHED_K_EFF
from common import benchmark_settings, report

import openndm

_BC = {0: "zero_flux", 1: "vacuum", 2: "reflective"}
"""KOMODO boundary codes to OpenNDM names."""

BOUNDARIES = {
    "x_max": _BC[1],
    "x_min": _BC[2],
    "y_max": _BC[2],
    "y_min": _BC[1],
    "z_min": _BC[2],
    "z_max": _BC[2],
}
"""The source deck's (east, west, north, south, bottom, top) = 1 2 2 1 2 2.

East and south are the outer faces, west and north the symmetry cuts, and the
two axial faces are reflective, which makes the problem 2D.
"""


def biblis_library() -> openndm.XSLibrary:
    """Two-group BIBLIS library, compositions in source-deck order.

    The source deck gives the reflector a fission spectrum of 1.0 with no
    fission source, which ``finalize()`` warns about. It is kept as the deck
    has it: chi only ever multiplies a fission source that is zero there.
    """
    library = openndm.XSLibrary(2, len(COMPOSITIONS))
    for index, composition in enumerate(COMPOSITIONS):
        library.set_composition(index, **composition)
    library.finalize()
    return library


def build(subdivide: int = 1):
    """Extrude the node map into the deck's two axial planes."""
    radial = np.where(NODE_MAP == 0, openndm.INACTIVE, NODE_MAP - 1)
    core = np.repeat(radial[np.newaxis, :, :], len(DZ), axis=0)
    geometry = openndm.Geometry.from_lattice(
        core,
        pitch=(NODE_WIDTH, NODE_WIDTH, DZ[0]),
        dz=DZ,
        subdivide=(subdivide, subdivide, 1),
        boundaries=BOUNDARIES,
        outside="vacuum",
    )
    return geometry, biblis_library()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subdivide", type=int, default=1)
    args = parser.parse_args()

    geometry, library = build(args.subdivide)
    settings = benchmark_settings()
    print(
        f"BIBLIS-2D: {geometry.n_nodes} nodes, "
        f"{NODE_WIDTH / args.subdivide:.4f} cm mesh"
    )
    for kernel in ("fdm", "nem", "sanm"):
        settings.kernel = kernel
        result = openndm.Model(geometry, library, settings).solve()
        report(f"BIBLIS-2D sub={args.subdivide}", result, PUBLISHED_K_EFF)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
