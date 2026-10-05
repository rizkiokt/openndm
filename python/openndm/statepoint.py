"""HDF5 statepoint output and reader (FR-OUT-1, FR-OUT-2).

The layout follows OpenMC's conventions so that a user fluent in
``openmc.StatePoint`` can read an OpenNDM statepoint without new habits:
a ``filetype`` attribute identifying the file, an integer
``format_version``, scalars as root attributes and results as datasets.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np

from .exceptions import InputError

__all__ = ["STATEPOINT_FORMAT_VERSION", "StatePoint", "write_statepoint"]

STATEPOINT_FORMAT_VERSION = 1
"""Layout version of the statepoint file."""

_FILETYPE = "openndm_statepoint"


def write_statepoint(path, result, model, *, extra: dict | None = None) -> Path:
    """Write a solve result to ``statepoint.h5``.

    Parameters
    ----------
    path : path-like
        Destination file.
    result : Result
        The object returned by :meth:`openndm.Model.solve`.
    model : Model
        Provides the geometry and library metadata written alongside.
    extra : mapping, optional
        Additional scalar metadata stored as root attributes. Use it to record
        conventions that a downstream comparison needs, such as which leakage
        correction produced the group constants.

    Returns
    -------
    pathlib.Path
        The file that was written.
    """
    import h5py

    from . import __version__

    path = Path(path)
    with h5py.File(path, "w") as f:
        f.attrs["filetype"] = np.bytes_(_FILETYPE)
        f.attrs["format_version"] = STATEPOINT_FORMAT_VERSION
        f.attrs["version"] = np.bytes_(__version__)
        f.attrs["date_and_time"] = np.bytes_(
            _dt.datetime.now(_dt.timezone.utc).isoformat()
        )
        f.attrs["kernel"] = np.bytes_(result.kernel)
        f.attrs["mode"] = np.bytes_(model.settings.mode)
        f.attrs["k_eff"] = float(result.k_eff)
        f.attrs["converged"] = bool(result.converged)
        f.attrs["outer_iterations"] = int(result.outer_iterations)
        f.attrs["runtime_seconds"] = float(result.runtime)
        for key, value in (extra or {}).items():
            f.attrs[key] = (
                np.bytes_(value) if isinstance(value, str) else value
            )

        geom = f.create_group("geometry")
        geom.attrs["n_nodes"] = model.geometry.n_nodes
        geom.attrs["n_surfaces"] = model.geometry.n_surfaces
        geom.attrs["lattice_shape"] = np.asarray(model.geometry.shape)
        geom.create_dataset("volume", data=model.geometry.volumes)
        geom.create_dataset("composition", data=model.geometry.compositions)
        geom.create_dataset("lattice_to_node", data=model.geometry.lattice_to_node)

        results = f.create_group("results")
        results.attrs["n_groups"] = model.library.n_groups
        results.create_dataset("flux", data=np.asarray(result.flux))
        results.create_dataset("power", data=np.asarray(result.power))
        results.create_dataset("radial_power", data=result.radial_power())
        results.create_dataset("axial_power", data=result.axial_power())
        results.attrs["f_q"] = result.f_q
        results.attrs["f_dh"] = result.f_dh

        f.create_dataset("iteration_history", data=result.history)
    return path


class StatePoint:
    """Reader for an OpenNDM statepoint, mirroring ``openmc.StatePoint``.

    Use it as a context manager, or call :meth:`close` when done. Every array
    property reads its dataset afresh and returns a copy, never a view onto
    the file.

    Parameters
    ----------
    path : path-like
        Statepoint written by :func:`write_statepoint`.

    Raises
    ------
    InputError
        If the file is not an OpenNDM statepoint, or carries a
        ``format_version`` this build does not read.

    Examples
    --------
    >>> with StatePoint("statepoint.h5") as sp:      # doctest: +SKIP
    ...     print(sp.k_eff, sp.power.shape)
    """

    def __init__(self, path):
        import h5py

        self._path = Path(path)
        self._f = h5py.File(self._path, "r")
        filetype = self._f.attrs.get("filetype", b"")
        if isinstance(filetype, bytes):
            filetype = filetype.decode()
        if filetype != _FILETYPE:
            self._f.close()
            raise InputError(
                f"{path} is not an OpenNDM statepoint (filetype={filetype!r})"
            )
        version = int(self._f.attrs.get("format_version", 0))
        if version != STATEPOINT_FORMAT_VERSION:
            self._f.close()
            raise InputError(
                f"{path}: statepoint format version {version} is not supported "
                f"(this build reads version {STATEPOINT_FORMAT_VERSION})"
            )

    @property
    def k_eff(self) -> float:
        """Eigenvalue of the solve that wrote the file."""
        return float(self._f.attrs["k_eff"])

    @property
    def kernel(self) -> str:
        """Nodal kernel that ran, one of ``'fdm'``, ``'nem'``, ``'sanm'``."""
        return self._f.attrs["kernel"].decode()

    @property
    def mode(self) -> str:
        """Calculation mode, one of ``'forward'``, ``'adjoint'``, ``'fixed_source'``."""
        return self._f.attrs["mode"].decode()

    @property
    def converged(self) -> bool:
        """Whether the outer iteration met its convergence criteria."""
        return bool(self._f.attrs["converged"])

    @property
    def version(self) -> str:
        """OpenNDM version that wrote the file."""
        return self._f.attrs["version"].decode()

    @property
    def date_and_time(self) -> str:
        """UTC time the file was written, as an ISO 8601 string."""
        return self._f.attrs["date_and_time"].decode()

    @property
    def flux(self) -> np.ndarray:
        """Scalar flux, shape ``(n_nodes, n_groups)``, as a copy.

        Normalised so the volume-averaged total flux over all groups is one.
        """
        return self._f["results/flux"][()]

    @property
    def power(self) -> np.ndarray:
        """Relative node power, shape ``(n_nodes,)``, as a copy.

        Normalised to a mean of one over the nodes that carry power.
        """
        return self._f["results/power"][()]

    @property
    def radial_power(self) -> np.ndarray:
        """Volume-weighted radial power map, shape ``(ny, nx)``, as a copy."""
        return self._f["results/radial_power"][()]

    @property
    def axial_power(self) -> np.ndarray:
        """Volume-weighted axial power profile, shape ``(nz,)``, as a copy."""
        return self._f["results/axial_power"][()]

    @property
    def f_q(self) -> float:
        """Total peaking factor: peak node power over the core average."""
        return float(self._f["results"].attrs["f_q"])

    @property
    def f_dh(self) -> float:
        """Radial enthalpy-rise peaking factor: peak radial power."""
        return float(self._f["results"].attrs["f_dh"])

    @property
    def history(self) -> np.ndarray:
        """Outer iteration history as a structured array, as a copy.

        The fields are ``outer``, ``k_eff``, ``k_change``, ``source_change``
        and ``inner_iterations``.
        """
        return self._f["iteration_history"][()]

    @property
    def volumes(self) -> np.ndarray:
        """Node volumes in cm^3, shape ``(n_nodes,)``, as a copy."""
        return self._f["geometry/volume"][()]

    @property
    def lattice_shape(self) -> tuple[int, int, int]:
        """Lattice extent as ``(nz, ny, nx)``."""
        return tuple(int(v) for v in self._f["geometry"].attrs["lattice_shape"])

    def close(self) -> None:
        """Close the underlying HDF5 file."""
        self._f.close()

    def __enter__(self) -> StatePoint:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<StatePoint {self._path.name} k_eff={self.k_eff:.6f}>"
