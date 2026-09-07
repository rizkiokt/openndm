"""Matplotlib plotting helpers (FR-OUT-5).

Every function takes an optional ``ax`` and returns it, matching
``openmc.plots`` conventions, so plots compose into a larger figure.
Matplotlib is an optional dependency; importing this module without it raises
a clear message rather than an ImportError from deep inside a call.
"""

from __future__ import annotations

import numpy as np

__all__ = ["plot_axial", "plot_convergence", "plot_radial"]


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without mpl
        raise ImportError(
            "plotting needs matplotlib; install it with "
            "`pip install openndm[plot]`"
        ) from exc
    return plt


def plot_radial(result, ax=None, *, annotate: bool = True, cmap: str = "viridis"):
    """Plot the radial power map as a core-map image.

    Parameters
    ----------
    result : Result
    annotate : bool
        Write the value into each powered position.
    """
    plt = _pyplot()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    radial = result.radial_power()
    masked = np.ma.masked_where(radial <= 0.0, radial)
    image = ax.imshow(masked, origin="lower", cmap=cmap)
    ax.figure.colorbar(image, ax=ax, label="relative assembly power")
    if annotate:
        for (j, i), value in np.ndenumerate(radial):
            if value > 0.0:
                ax.text(
                    i, j, f"{value:.2f}", ha="center", va="center", fontsize=7
                )
    ax.set_xlabel("assembly index, x")
    ax.set_ylabel("assembly index, y")
    ax.set_title(f"radial power, $F_{{\\Delta H}}$ = {result.f_dh:.3f}")
    return ax


def plot_axial(result, ax=None, *, heights=None):
    """Plot the axial power profile."""
    plt = _pyplot()
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 5))
    axial = result.axial_power()
    z = np.arange(axial.size) if heights is None else np.asarray(heights)
    ax.step(axial, z, where="mid")
    ax.set_xlabel("relative axial power")
    ax.set_ylabel("plane" if heights is None else "height [cm]")
    ax.grid(alpha=0.3)
    return ax


def plot_convergence(result, ax=None):
    """Plot the outer iteration history (FR-OUT-5)."""
    plt = _pyplot()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    history = result.history
    ax.semilogy(
        history["outer"], np.abs(history["k_change"]), label=r"$|\Delta k|$"
    )
    ax.semilogy(
        history["outer"], history["source_change"], label="fission source"
    )
    ax.set_xlabel("outer iteration")
    ax.set_ylabel("change")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    return ax
