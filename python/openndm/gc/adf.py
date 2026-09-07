"""Assembly discontinuity factor generation (FR-OMC-7).

An ADF is the ratio of the heterogeneous surface flux to the homogeneous
node-average flux, per face and per group. Both come from the same OpenMC
lattice calculation.

.. rubric:: Why a thin slab rather than a surface tally

A true surface flux tally converges slowly, and an ADF is a *ratio* of two
tallies, which amplifies the relative error of the noisier one. This module
therefore estimates the surface flux from a thin track-length-estimated slab
adjacent to the face, which converges far faster for the same particle count.
The slab has finite width, so it carries a small bias toward the node
average; :func:`compute_adf` reports the slab width used so the bias can be
studied by refining it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from ..exceptions import InputError

__all__ = ["DEFAULT_SLAB_FRACTION", "AdfResult", "add_adf_tallies", "compute_adf"]

#: Slab width as a fraction of the assembly pitch.
DEFAULT_SLAB_FRACTION = 0.05

#: ADF relative standard deviation above which a warning is issued.
ADF_SIGMA_WARNING = 0.02

_FACE_NAMES = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")


@dataclass
class AdfResult:
    """Discontinuity factors for one assembly.

    Attributes
    ----------
    values : numpy.ndarray, shape (n_faces, G)
        Face-ordered as ``-x, +x, -y, +y, -z, +z``, matching
        :meth:`openndm.XSLibrary.set_adf`.
    std_dev : numpy.ndarray, shape (n_faces, G)
        Propagated 1-sigma of the ratio.
    slab_fraction : float
        Slab width used, as a fraction of the pitch.
    """

    values: np.ndarray
    std_dev: np.ndarray
    slab_fraction: float

    def apply_to(self, library, composition: int) -> None:
        """Store these factors on a library composition."""
        library.set_adf(composition, self.values, n_axes=self.values.shape[0] // 2)

    def __repr__(self) -> str:
        return (
            f"<AdfResult {self.values.shape[0]} faces x {self.values.shape[1]} "
            f"groups, max sigma {np.nanmax(self.std_dev):.3g}>"
        )


def add_adf_tallies(
    model,
    lattice,
    energy_groups,
    *,
    slab_fraction: float = DEFAULT_SLAB_FRACTION,
    axes: str = "xy",
    prefix: str = "openndm_adf",
    z_bounds: tuple[float, float] = (-0.5, 0.5),
):
    """Instrument an ``openmc.Model`` with the tallies an ADF needs.

    Adds, per requested face, a thin mesh cell covering the slab adjacent to
    that face plus one covering the whole assembly, both scored with ``flux``
    on the supplied group structure.

    Parameters
    ----------
    model : openmc.Model
        Modified in place; its ``tallies`` gain the new entries.
    lattice : openmc.RectLattice
        The assembly lattice being homogenised. Its ``pitch`` and
        ``lower_left`` define the slab geometry.
    energy_groups : openmc.mgxs.EnergyGroups
        Group structure the ADFs are wanted on.
    slab_fraction : float
        Slab width as a fraction of the pitch, in ``(0, 0.5)``.
    axes : str
        Which axes to instrument, any subset of ``'xyz'``. Radial only by
        default, matching how ADFs are normally used.
    prefix : str
        Tally name prefix, used again by :func:`compute_adf`.
    z_bounds : (float, float)
        Axial extent of the tally meshes for a two-dimensional lattice, whose
        own extent is infinite in z. A discontinuity factor is a ratio of flux
        densities, so any consistent finite extent gives the same answer; the
        default is a unit height about the origin.

    Returns
    -------
    list of openmc.Tally
        The tallies that were added, in face order.

    Raises
    ------
    InputError
        On an out-of-range ``slab_fraction`` or an unknown axis letter.
    """
    from .mgxs import require_openmc

    openmc = require_openmc()

    if not 0.0 < slab_fraction < 0.5:
        raise InputError(
            f"slab_fraction must be in (0, 0.5), got {slab_fraction}"
        )
    unknown = set(axes) - set("xyz")
    if unknown:
        raise InputError(f"unknown axis letter(s) {sorted(unknown)}")

    pitch = np.atleast_1d(np.asarray(lattice.pitch, dtype=float))
    lower_left = np.atleast_1d(np.asarray(lattice.lower_left, dtype=float))
    shape = np.atleast_1d(np.asarray(lattice.shape, dtype=int))
    extent = pitch * shape
    upper_right = lower_left + extent
    if lower_left.size == 2:
        # A 2D lattice is infinite in z, but a mesh needs finite bounds.
        lower_left = np.append(lower_left, z_bounds[0])
        upper_right = np.append(upper_right, z_bounds[1])
        extent = np.append(extent, z_bounds[1] - z_bounds[0])

    energy_filter = openmc.EnergyFilter(energy_groups.group_edges)
    added = []

    whole = openmc.Tally(name=f"{prefix}_volume")
    whole_mesh = openmc.RegularMesh()
    whole_mesh.lower_left = list(lower_left)
    whole_mesh.upper_right = list(upper_right)
    whole_mesh.dimension = [1, 1, 1]
    whole.filters = [openmc.MeshFilter(whole_mesh), energy_filter]
    whole.scores = ["flux"]
    added.append(whole)

    for axis_index, letter in enumerate("xyz"):
        if letter not in axes:
            continue
        width = slab_fraction * extent[axis_index]
        for side, name in ((0, f"{letter}_min"), (1, f"{letter}_max")):
            ll = np.array(lower_left, dtype=float)
            ur = np.array(upper_right, dtype=float)
            if side == 0:
                ur[axis_index] = ll[axis_index] + width
            else:
                ll[axis_index] = ur[axis_index] - width
            mesh = openmc.RegularMesh()
            mesh.lower_left = list(ll)
            mesh.upper_right = list(ur)
            mesh.dimension = [1, 1, 1]
            tally = openmc.Tally(name=f"{prefix}_{name}")
            tally.filters = [openmc.MeshFilter(mesh), energy_filter]
            tally.scores = ["flux"]
            added.append(tally)

    if model.tallies is None:
        model.tallies = openmc.Tallies()
    model.tallies.extend(added)
    return added


def compute_adf(
    statepoint,
    *,
    prefix: str = "openndm_adf",
    slab_fraction: float = DEFAULT_SLAB_FRACTION,
    n_axes: int = 3,
    reverse_groups: bool = True,
    warn_sigma: float = ADF_SIGMA_WARNING,
) -> AdfResult:
    """Form the surface-to-volume flux ratios from a statepoint (FR-OMC-7).

    Parameters
    ----------
    statepoint : openmc.StatePoint
        Statepoint holding the tallies added by :func:`add_adf_tallies`.
    prefix : str
        The same prefix that was passed to :func:`add_adf_tallies`.
    n_axes : int
        Number of axes in the target geometry; faces not instrumented get 1.0.
    reverse_groups : bool
        An ``openmc.EnergyFilter`` orders its bins by *increasing* energy,
        while OpenMC's multi-group numbering and OpenNDM both put group 1 at
        the *highest* energy. The tally is therefore reversed by default, so
        that the result lines up with the library built by
        :func:`openndm.gc.from_mgxs_library`. Set this to False only for a
        filter already given in decreasing-energy order.

        Getting this backwards applies every discontinuity factor to the wrong
        group. It is silent: the values stay plausible, and only their sense
        inverts. A pin lattice is the giveaway, because its surface sits in
        water, where the thermal flux peaks and the fast flux does not, so the
        thermal factor must exceed one and the fast factor must fall below it.
    warn_sigma : float
        Warn when any ADF's relative standard deviation exceeds this. Noisy
        ADFs can leave the nodal solution worse than no ADFs at all, so this
        is worth heeding rather than suppressing.

    Returns
    -------
    AdfResult
    """
    from .mgxs import require_openmc

    require_openmc()

    def tally_flux(name):
        try:
            tally = statepoint.get_tally(name=name)
        except LookupError:
            return None, None
        mean = np.asarray(tally.mean).ravel()
        std = np.asarray(tally.std_dev).ravel()
        if reverse_groups:
            mean, std = mean[::-1], std[::-1]
        return mean, std

    volume_mean, volume_std = tally_flux(f"{prefix}_volume")
    if volume_mean is None:
        raise InputError(
            f"statepoint has no tally named {prefix!r}_volume; was "
            f"add_adf_tallies called with prefix={prefix!r}?"
        )
    # A mesh flux tally scores track length, so dividing by the cell volume is
    # unnecessary for a ratio of flux densities only if both cells are
    # normalised the same way; they are not, so normalise both by their width.
    n_groups = volume_mean.size
    values = np.ones((2 * n_axes, n_groups))
    std_dev = np.zeros((2 * n_axes, n_groups))

    with np.errstate(invalid="ignore", divide="ignore"):
        volume_rel = np.where(
            volume_mean > 0, volume_std / np.where(volume_mean > 0, volume_mean, 1), 0.0
        )

    for face, name in enumerate(_FACE_NAMES[: 2 * n_axes]):
        surface_mean, surface_std = tally_flux(f"{prefix}_{name}")
        if surface_mean is None:
            continue
        # Both tallies are volume-integrated track lengths, so the ratio of
        # flux densities divides out by the slab's fractional width.
        ratio = np.where(
            volume_mean > 0,
            (surface_mean / slab_fraction)
            / np.where(volume_mean > 0, volume_mean, 1),
            1.0,
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            surface_rel = np.where(
                surface_mean > 0,
                surface_std / np.where(surface_mean > 0, surface_mean, 1),
                0.0,
            )
        values[face] = ratio
        # Uncorrelated propagation is conservative here: the two tallies share
        # particle histories, so the true sigma of the ratio is smaller.
        std_dev[face] = np.abs(ratio) * np.hypot(surface_rel, volume_rel)

    worst = float(np.nanmax(std_dev)) if std_dev.size else 0.0
    if worst > warn_sigma:
        warnings.warn(
            f"ADF relative standard deviation reaches {100 * worst:.1f}%, above "
            f"the {100 * warn_sigma:.1f}% threshold. Noisy discontinuity factors "
            f"can degrade the nodal solution below the no-ADF baseline; run more "
            f"particles or widen slab_fraction.",
            stacklevel=2,
        )

    return AdfResult(
        values=values, std_dev=std_dev, slab_fraction=slab_fraction
    )
