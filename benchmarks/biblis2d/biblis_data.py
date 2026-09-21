"""BIBLIS 2D benchmark data, parsed from the KOMODO sample deck.

Generated, not transcribed. Source:
  https://github.com/imronuke/KOMODO  smpl/static/BIBLIS
  commit b70d4ee262dcbd2547993302f6f86bef6d4dca00 (MIT)
  sha256 0f1e275579fd13a22b944fd33207e3fe0bf442696715bc0260d3b8ac11442c6f

KOMODO is a machine-readable transcription; the underlying data is the
published BIBLIS PWR benchmark.
"""

from __future__ import annotations

import numpy as np

#: Published reference, stated in the source deck.
PUBLISHED_K_EFF = 1.02511

#: Uniform node width after the deck's assembly divisions, in cm.
NODE_WIDTH = 11.5613
#: Axial node widths, in cm.
DZ = [11.5613, 11.5613]

#: Two-group constants per composition, KOMODO order 1..8 mapped to 0..7.
COMPOSITIONS = [
    {  # composition 1 in the source deck
        "D": [1.4359998558256144, 0.3635000180841259],
        "absorption": [0.0095042, 0.075058],
        "nu_fission": [0.0058708, 0.096067],
        "kappa_fission": [0.0058708, 0.096067],
        "chi": [1.0, 0.0],
        "scatter": [[0.0, 0.017754], [0.0, 0.0]],
    },
    {  # composition 2 in the source deck
        "D": [1.4366001765006977, 0.3636000168564968],
        "absorption": [0.0096785, 0.078436],
        "nu_fission": [0.0061908, 0.10358],
        "kappa_fission": [0.0061908, 0.10358],
        "chi": [1.0, 0.0],
        "scatter": [[0.0, 0.017621], [0.0, 0.0]],
    },
    {  # composition 3 in the source deck
        "D": [1.3199997518400468, 0.277200000576576],
        "absorption": [0.0026562, 0.071596],
        "nu_fission": [0.0, 0.0],
        "kappa_fission": [0.0, 0.0],
        "chi": [1.0, 0.0],
        "scatter": [[0.0, 0.023106], [0.0, 0.0]],
    },
    {  # composition 4 in the source deck
        "D": [1.4389002657936572, 0.363799981024193],
        "absorption": [0.010363, 0.091408],
        "nu_fission": [0.0074527, 0.13236],
        "kappa_fission": [0.0074527, 0.13236],
        "chi": [1.0, 0.0],
        "scatter": [[0.0, 0.017101], [0.0, 0.0]],
    },
    {  # composition 5 in the source deck
        "D": [1.438100074220345, 0.3665000081179752],
        "absorption": [0.010003, 0.084828],
        "nu_fission": [0.0061908, 0.10358],
        "kappa_fission": [0.0061908, 0.10358],
        "chi": [1.0, 0.0],
        "scatter": [[0.0, 0.01729], [0.0, 0.0]],
    },
    {  # composition 6 in the source deck
        "D": [1.438499748334469, 0.3665000081179752],
        "absorption": [0.010132, 0.087314],
        "nu_fission": [0.0064285, 0.10911],
        "kappa_fission": [0.0064285, 0.10911],
        "chi": [1.0, 0.0],
        "scatter": [[0.0, 0.017192], [0.0, 0.0]],
    },
    {  # composition 7 in the source deck
        "D": [1.4389002657936572, 0.367900003601741],
        "absorption": [0.010165, 0.088024],
        "nu_fission": [0.0061908, 0.10358],
        "kappa_fission": [0.0061908, 0.10358],
        "chi": [1.0, 0.0],
        "scatter": [[0.0, 0.017125], [0.0, 0.0]],
    },
    {  # composition 8 in the source deck
        "D": [1.439299763393512, 0.3680000005888],
        "absorption": [0.010294, 0.09051],
        "nu_fission": [0.0064285, 0.10911],
        "kappa_fission": [0.0064285, 0.10911],
        "chi": [1.0, 0.0],
        "scatter": [[0.0, 0.017027], [0.0, 0.0]],
    },
]

#: Node composition map, 1-based as in the deck, 0 outside the core.
#: Row 0 is the south edge; the deck prints north first.
NODE_MAP = np.array(
    [
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0],
        [4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 8, 8, 4, 4, 4, 4, 3, 3, 3, 3, 0, 0],
        [1, 1, 1, 1, 1, 8, 8, 4, 4, 4, 4, 3, 3, 3, 3, 0, 0],
        [7, 1, 1, 7, 7, 1, 1, 5, 5, 4, 4, 4, 4, 3, 3, 0, 0],
        [7, 1, 1, 7, 7, 1, 1, 5, 5, 4, 4, 4, 4, 3, 3, 0, 0],
        [1, 8, 8, 2, 2, 8, 8, 2, 2, 5, 5, 4, 4, 3, 3, 3, 3],
        [1, 8, 8, 2, 2, 8, 8, 2, 2, 5, 5, 4, 4, 3, 3, 3, 3],
        [6, 2, 2, 8, 8, 2, 2, 8, 8, 1, 1, 8, 8, 4, 4, 3, 3],
        [6, 2, 2, 8, 8, 2, 2, 8, 8, 1, 1, 8, 8, 4, 4, 3, 3],
        [2, 8, 8, 1, 1, 8, 8, 2, 2, 7, 7, 1, 1, 4, 4, 3, 3],
        [2, 8, 8, 1, 1, 8, 8, 2, 2, 7, 7, 1, 1, 4, 4, 3, 3],
        [8, 1, 1, 8, 8, 2, 2, 8, 8, 1, 1, 1, 1, 4, 4, 3, 3],
        [8, 1, 1, 8, 8, 2, 2, 8, 8, 1, 1, 1, 1, 4, 4, 3, 3],
        [1, 8, 8, 2, 2, 6, 6, 1, 1, 7, 7, 1, 1, 4, 4, 3, 3],
    ],
    dtype=int,
)

#: Boundary conditions, KOMODO order east/west/north/south/bottom/top.
#: 0 zero flux, 1 zero incoming current, 2 reflective.
SOURCE_BC = [1, 2, 2, 1, 2, 2]
