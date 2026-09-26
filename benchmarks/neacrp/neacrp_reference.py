"""NEACRP PWR rod ejection reference solution, read from the primary source.

Source: "Results of LWR Core Transient Benchmarks", H. Finnemann and H. Bauer
(Siemens AG/KWU, Erlangen), A. Galati and R. Martinelli (ENEA Casaccia, Rome),
NEA/NSC/DOC(93)25, October 1993. Table 3.1, "PWR, Steady-State Reference
Solution", document page 18.
  https://www.oecd-nea.org/science/docs/1993/nsc-doc93-25.pdf
  sha256 bda2ab1bb9d252f36f68cc876480401b87158cabea8c1d56551df60aa3ba3490

The reference is a PANTHER calculation by P. K. Hutt at 4 radial nodes per
assembly and 16 axial nodes, finer than the standard submissions (page 6).

The document is a scan with no text layer; these values were read from a
render of page 18, not from an extraction. Every one of them is independently
confirmed by Pinem et al., Sci. Technol. Nucl. Install. 2014 art. 845832
Table 2 (A1, A2, B1, B2, as "PANTHER (1993)") and 2016 art. 7538681 Tables 1
to 3 (A1, C1, C2, as "Reference").
"""

from __future__ import annotations

#: Delayed neutron fraction the reference states, pcm.
BETA_PCM = 760.0

#: Radial and axial nodes per assembly in the reference calculation.
REFERENCE_MESH = (4, 16)

#: Table 3.1, initial steady state. Power is relative: 1e-6 is HZP, 1.0 is FP.
#: Temperatures in C, rod worth in pcm.
INITIAL_STEADY_STATE = {
    "A1": {"boron_ppm": 567.7, "power": 1.0e-6, "f_xy": 1.909, "f_q": 2.874,
           "t_doppler": 286.0, "t_centre": 286.0, "rod_worth_pcm": 821.8},
    "A2": {"boron_ppm": 1160.6, "power": 1.0, "f_xy": 1.198, "f_q": 2.221,
           "t_doppler": 546.1, "t_centre": 1671.9, "rod_worth_pcm": 89.5},
    "B1": {"boron_ppm": 1254.6, "power": 1.0e-6, "f_xy": 1.276, "f_q": 1.932,
           "t_doppler": 286.0, "t_centre": 286.0, "rod_worth_pcm": 831.0},
    "B2": {"boron_ppm": 1189.4, "power": 1.0, "f_xy": 1.170, "f_q": 2.109,
           "t_doppler": 543.7, "t_centre": 1576.6, "rod_worth_pcm": 99.1},
    "C1": {"boron_ppm": 1135.3, "power": 1.0e-6, "f_xy": 1.445, "f_q": 2.187,
           "t_doppler": 286.0, "t_centre": 286.0, "rod_worth_pcm": 958.0},
    "C2": {"boron_ppm": 1160.6, "power": 1.0, "f_xy": 1.198, "f_q": 2.221,
           "t_doppler": 546.1, "t_centre": 1671.9, "rod_worth_pcm": 78.1},
}

#: Table 3.1, final steady state at 5 s. Power relative, temperature in C.
FINAL_STEADY_STATE = {
    "A1": {"power": 0.263, "t_doppler": 344.1},
    "A2": {"power": 1.036, "t_doppler": 556.1},
    "B1": {"power": 0.416, "t_doppler": 378.9},
    "B2": {"power": 1.041, "t_doppler": 554.0},
    "C1": {"power": 0.206, "t_doppler": 331.9},
    "C2": {"power": 1.032, "t_doppler": 554.7},
}
