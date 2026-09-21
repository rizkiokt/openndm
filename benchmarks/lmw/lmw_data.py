"""LMW transient benchmark data, parsed from the KOMODO sample deck.

Generated, not transcribed. Source:
  https://github.com/imronuke/KOMODO  smpl/transient/LMW
  commit b70d4ee262dcbd2547993302f6f86bef6d4dca00 (MIT)
  sha256 ce9b4dd1b20b48c02372e7f7630b3bd2e0675043d23fc4fc8db0dc62b6a4059d

KOMODO is a machine-readable transcription; the underlying data is the
published Langenbuch-Maurer-Werner operational transient. The deck states the
scenario in full and does not state its answer, so the power history it
produces here is reported rather than compared; see ``run.py``.
"""

from __future__ import annotations

import numpy as np

#: Energy groups.
N_GROUPS = 2

#: Uniform node widths after the deck's assembly divisions, in cm.
DX = 10.0
DY = 10.0
DZ = 5.0
#: Axial nodes.
N_Z = 40

#: Two-group constants per composition, KOMODO material 1..3 mapped to 0..2.
COMPOSITIONS = [
    {
        "D": [1.423912995498584, 0.35630598346419556],
        "absorption": [0.01040206, 0.08766217],
        "nu_fission": [0.006478, 0.112733],
        "kappa_fission": [0.006478, 0.112733],
        "chi": [1.0, 0.0],
        "scatter": [[0.0, 0.017556], [0.0, 0.0]],
        "inv_velocity": [8e-08, 4e-06],
    },
    {
        "D": [1.4256108421696259, 0.3505740018246676],
        "absorption": [0.01099263, 0.09925634],
        "nu_fission": [0.007503, 0.1378],
        "kappa_fission": [0.007503, 0.1378],
        "chi": [1.0, 0.0],
        "scatter": [[0.0, 0.017178], [0.0, 0.0]],
        "inv_velocity": [8e-08, 4e-06],
    },
    {
        "D": [1.6342272556421695, 0.264001999445543],
        "absorption": [0.00266057, 0.04936351],
        "nu_fission": [0.0, 0.0],
        "kappa_fission": [0.0, 0.0],
        "chi": [1.0, 0.0],
        "scatter": [[0.0, 0.027597], [0.0, 0.0]],
        "inv_velocity": [8e-08, 4e-06],
    },
]

#: Cross section increments a fully rodded node takes, per composition.
ROD_DELTA = [
    {
        "transport": [0.0, 0.0],
        "absorption": [0.00055, 0.0038],
        "nu_fission": [0.0, 0.0],
        "kappa_fission": [0.0, 0.0],
    },
    {
        "transport": [0.0, 0.0],
        "absorption": [0.0, 0.0],
        "nu_fission": [0.0, 0.0],
        "kappa_fission": [0.0, 0.0],
    },
    {
        "transport": [0.0, 0.0],
        "absorption": [0.0, 0.0],
        "nu_fission": [0.0, 0.0],
        "kappa_fission": [0.0, 0.0],
    },
]

#: Radial composition map of each planar type, on the node mesh,
#: south row first. 0 marks a lattice position outside the core.
PLANAR_TYPES = np.array([
    [
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0],
    ],
    [
        [1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3],
        [1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3],
        [1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3],
        [1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3],
        [1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3],
        [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3],
        [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3],
        [2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3],
        [2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0],
    ],
])

#: Planar type of each axial node, bottom to top.
PLANE_ASSIGNMENT = [
    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
]

#: Bank occupying each lattice position on the node mesh, 0 for none.
BANK_MAP = np.array([
    [1, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])

#: Boundary codes as the deck gives them: east, west, north, south,
#: bottom, top. 0 zero flux, 1 zero incoming current, 2 reflective.
BOUNDARY_CODES = [1, 2, 1, 2, 1, 1]

#: Rod travel: cm per step, tip height at step 0, and the step cap.
STEP_SIZE = 1.0
ZERO_POSITION = 0.0
MAX_STEPS = 180

#: Bank position in steps at t = 0.
INITIAL_POSITION = {
    "bank_1": 180.0,
    "bank_2": 100.0,
}

#: Bank motion: final position in steps, when it starts, steps per second.
MOTION = {
    "bank_1": {"final": 60.0, "start": 7.5, "speed": 3.0},
    "bank_2": {"final": 180.0, "start": 0.0, "speed": 3.0},
}

#: Transient duration and the deck's time step, in seconds.
TOTAL_TIME = 60.0
TIME_STEP = 0.25

#: Delayed neutron fraction and decay constant per precursor group.
BETA = [0.00025, 0.00138, 0.00122, 0.00265, 0.00083, 0.00017]
DECAY_CONSTANT = [0.0127, 0.0317, 0.115, 0.311, 1.4, 3.87]

#: Time integration weight the deck asks for.
THETA = 0.5
