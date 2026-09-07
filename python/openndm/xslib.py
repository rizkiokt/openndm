"""Cross section library construction and I/O (FR-XS, FR-IN-3)."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

from . import _core
from .exceptions import InputError

__all__ = ["BranchAxis", "XSLibrary"]

#: Layout version of the ``xslib.h5`` file this module reads and writes.
XSLIB_FORMAT_VERSION = 1

BranchAxis = _core.BranchAxis

_EXTRAPOLATION = {
    "clamp": _core.Extrapolation.clamp,
    "linear": _core.Extrapolation.linear,
    "error": _core.Extrapolation.error,
}


class XSLibrary:
    """Macroscopic group constants for every composition in a model.

    A library is either single-state or branch-parameterised. Adding branch
    axes with :meth:`set_axes` replaces the storage with one composition set
    per grid point; :meth:`interpolate` collapses it back to a single state at
    a requested set of state-variable values (FR-XS-5, FR-XS-6).

    Parameters
    ----------
    n_groups : int
        Number of energy groups, any value >= 1 (FR-XS-1).
    n_compositions : int
        Number of distinct homogenised compositions.

    Examples
    --------
    >>> lib = XSLibrary(2, 1)
    >>> lib.set_composition(
    ...     0, D=[1.5, 0.4], absorption=[0.01, 0.08],
    ...     nu_fission=[0.0, 0.135], chi=[1.0, 0.0],
    ...     scatter=[[0.0, 0.02], [0.0, 0.0]],
    ... )
    >>> lib.finalize()
    []
    >>> lib.n_groups
    2
    """

    def __init__(self, n_groups: int, n_compositions: int):
        self._lib = _core.XSLibrary(int(n_groups), int(n_compositions))

    # ------------------------------------------------------------ properties
    @property
    def n_groups(self) -> int:
        return self._lib.n_groups

    @property
    def n_compositions(self) -> int:
        return self._lib.n_compositions

    @property
    def n_states(self) -> int:
        """Number of branch grid points, 1 for a single-state library."""
        return self._lib.n_states

    @property
    def axes(self) -> list:
        return list(self._lib.axes)

    @property
    def finalized(self) -> bool:
        return self._lib.finalized

    @property
    def has_uncertainty(self) -> bool:
        """True when any composition carries 1-sigma values (FR-XS-9)."""
        return self._lib.has_uncertainty

    @property
    def extrapolation(self) -> str:
        return self._lib.extrapolation.name

    @extrapolation.setter
    def extrapolation(self, value: str) -> None:
        if isinstance(value, str):
            try:
                value = _EXTRAPOLATION[value.lower()]
            except KeyError:
                raise ValueError(
                    f"unknown extrapolation policy {value!r}; "
                    f"choose from {sorted(_EXTRAPOLATION)}"
                ) from None
        self._lib.extrapolation = value

    # --------------------------------------------------------------- writing
    def set_axes(self, axes: Sequence[tuple[str, Sequence[float]]]) -> None:
        """Declare the branch axes, outermost first (FR-XS-5).

        Parameters
        ----------
        axes : sequence of (name, points)
            Each axis is a name such as ``'fuel_temperature'`` and a strictly
            increasing sequence of grid points.

        Notes
        -----
        This discards any composition data already stored, because the number
        of state points changes. Declare the axes first, then fill states.
        """
        built = []
        for name, points in axes:
            axis = _core.BranchAxis()
            axis.name = str(name)
            axis.points = [float(p) for p in points]
            built.append(axis)
        self._lib.set_axes(built)

    def set_composition(
        self,
        index: int,
        *,
        state: int = 0,
        D=None,
        transport=None,
        absorption=None,
        nu_fission=None,
        kappa_fission=None,
        chi=None,
        scatter=None,
        inv_velocity=None,
        std=None,
    ) -> None:
        """Fill the group constants of one composition at one branch state.

        Parameters
        ----------
        index : int
            Composition index.
        state : int
            Branch grid point index; 0 for a single-state library.
        D : array_like, optional
            Diffusion coefficient per group, in cm. Mutually exclusive with
            ``transport``.
        transport : array_like, optional
            Transport cross section per group; ``D`` is set to ``1/(3*Sigma_tr)``.
        absorption, nu_fission, kappa_fission, chi, inv_velocity : array_like
            Per-group values. Anything omitted stays zero.
        scatter : array_like, shape (G, G)
            Scattering matrix indexed ``scatter[from_group][to_group]``.
        std : mapping, optional
            1-sigma uncertainties keyed by field name, e.g.
            ``{'absorption': [...], 'nu_fission': [...]}`` (FR-XS-9).
        """
        G = self.n_groups
        comp = self._lib.mutable_composition(int(index), int(state))

        if D is not None and transport is not None:
            raise InputError("give either D or transport, not both")
        if transport is not None:
            tr = self._vector(transport, G, "transport")
            if np.any(tr <= 0.0):
                raise InputError("transport cross section must be positive")
            comp.D = list(1.0 / (3.0 * tr))
        elif D is not None:
            comp.D = list(self._vector(D, G, "D"))

        for name, value in (
            ("absorption", absorption),
            ("nu_fission", nu_fission),
            ("kappa_fission", kappa_fission),
            ("chi", chi),
            ("inv_velocity", inv_velocity),
        ):
            if value is not None:
                setattr(comp, name, list(self._vector(value, G, name)))

        if scatter is not None:
            s = np.asarray(scatter, dtype=float)
            if s.shape != (G, G):
                raise InputError(
                    f"scatter must have shape ({G}, {G}), got {s.shape}"
                )
            comp.scatter = list(s.ravel(order="C"))

        for name, value in (std or {}).items():
            field = f"{name}_std"
            if not hasattr(comp, field):
                raise InputError(f"no uncertainty slot for {name!r}")
            arr = np.asarray(value, dtype=float).ravel()
            setattr(comp, field, list(arr))

    def set_adf(self, index: int, values, n_axes: int = 3) -> None:
        """Set discontinuity factors for one composition (FR-XS-4).

        Parameters
        ----------
        values : array_like, shape (2 * n_axes, G) or (G,)
            Per-face, per-group factors in the face order
            ``-x, +x, -y, +y, -z, +z``. A ``(G,)`` array is broadcast to every
            face.
        """
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1:
            arr = np.tile(arr, (2 * n_axes, 1))
        expected = (2 * n_axes, self.n_groups)
        if arr.shape != expected:
            raise InputError(
                f"ADF array must have shape {expected}, got {arr.shape}"
            )
        self._lib.set_adf(int(index), arr.ravel(order="C"), n_axes)

    def adf(self, index: int, n_axes: int = 3) -> np.ndarray:
        """Discontinuity factors as a ``(2 * n_axes, G)`` array."""
        return np.array(
            [
                [self._lib.adf_value(int(index), f, g) for g in range(self.n_groups)]
                for f in range(2 * n_axes)
            ]
        )

    def set_delayed(self, beta, decay_constant, chi_delayed=None) -> None:
        """Set delayed neutron data (FR-XS-3).

        Parameters
        ----------
        beta, decay_constant : array_like
            Delayed fraction and decay constant per precursor group, at most 8.
        chi_delayed : array_like, shape (n_precursors, G), optional
            Delayed spectrum; defaults to the prompt spectrum when omitted.
        """
        d = self._lib.delayed
        d.beta = [float(b) for b in beta]
        d.lambda_ = [float(v) for v in decay_constant]
        if chi_delayed is not None:
            arr = np.asarray(chi_delayed, dtype=float)
            if arr.shape != (len(d.beta), self.n_groups):
                raise InputError(
                    f"chi_delayed must have shape ({len(d.beta)}, "
                    f"{self.n_groups}), got {arr.shape}"
                )
            d.chi_delayed = list(arr.ravel(order="C"))

    def finalize(self, *, warn: bool = True) -> list[str]:
        """Validate and cache derived data (FR-XS-8).

        Returns the list of non-fatal findings, which are also issued as
        Python warnings unless ``warn=False``. Negative scattering transfers
        are reported this way rather than raised, because Monte Carlo noise
        produces them routinely.

        Raises
        ------
        LibraryError
            On a non-positive diffusion coefficient, a negative absorption
            cross section, a fission spectrum that does not sum to one, or a
            non-monotonic branch axis.
        """
        messages = self._lib.finalize()
        if warn:
            for message in messages:
                warnings.warn(message, stacklevel=2)
        return messages

    # --------------------------------------------------------------- reading
    def composition(self, index: int, state: int = 0):
        """Snapshot of one composition's group constants, for inspection.

        The returned record is a copy: mutating it does not change the
        library. Use :meth:`set_composition` to write.
        """
        return self._lib.composition(int(index), int(state))

    def interpolate(self, **state) -> XSLibrary:
        """Multilinear interpolation to a branch state point (FR-XS-6).

        Parameters
        ----------
        **state
            One keyword per branch axis, e.g.
            ``interpolate(fuel_temperature=900.0, boron=1200.0)``.

        Returns
        -------
        XSLibrary
            A finalized single-state library.
        """
        names = [a.name for a in self.axes]
        missing = set(names) - set(state)
        extra = set(state) - set(names)
        if missing or extra:
            raise InputError(
                f"branch state mismatch: missing {sorted(missing)}, "
                f"unexpected {sorted(extra)}; axes are {names}"
            )
        out = XSLibrary.__new__(XSLibrary)
        out._lib = self._lib.interpolate([float(state[n]) for n in names])
        return out

    def array(self, field: str, state: int = 0) -> np.ndarray:
        """Stack one field across compositions.

        Returns a ``(n_compositions, G)`` array, or ``(n_compositions, G, G)``
        for ``'scatter'``.
        """
        rows = [
            np.asarray(getattr(self.composition(c, state), field), dtype=float)
            for c in range(self.n_compositions)
        ]
        out = np.stack(rows)
        if field == "scatter":
            out = out.reshape(self.n_compositions, self.n_groups, self.n_groups)
        return out

    def __repr__(self) -> str:
        branch = (
            f", {self.n_states} branch states over "
            f"{[a.name for a in self.axes]}"
            if self.axes
            else ""
        )
        return (
            f"<XSLibrary {self.n_groups} groups, "
            f"{self.n_compositions} compositions{branch}>"
        )

    # ------------------------------------------------------------- hdf5 i/o
    def to_hdf5(self, path) -> None:
        """Write the library to ``xslib.h5`` (FR-IN-3).

        The layout is versioned through the root ``format_version`` attribute
        and follows OpenMC's conventions: scalars as attributes, arrays as
        datasets, one group per composition.
        """
        import h5py

        with h5py.File(path, "w") as f:
            f.attrs["filetype"] = np.bytes_("openndm_xslib")
            f.attrs["format_version"] = XSLIB_FORMAT_VERSION
            f.attrs["n_groups"] = self.n_groups
            f.attrs["n_compositions"] = self.n_compositions
            f.attrs["n_states"] = self.n_states

            if self.axes:
                axes = f.create_group("branch_axes")
                axes.attrs["names"] = [np.bytes_(a.name) for a in self.axes]
                for axis in self.axes:
                    axes.create_dataset(axis.name, data=np.asarray(axis.points))

            delayed = self._lib.delayed
            if delayed.beta:
                d = f.create_group("delayed")
                d.create_dataset("beta", data=np.asarray(delayed.beta))
                d.create_dataset("lambda", data=np.asarray(delayed.lambda_))
                if delayed.chi_delayed:
                    d.create_dataset(
                        "chi_delayed",
                        data=np.asarray(delayed.chi_delayed).reshape(
                            delayed.n_precursors, self.n_groups
                        ),
                    )

            comps = f.create_group("compositions")
            for c in range(self.n_compositions):
                g = comps.create_group(f"composition {c}")
                for field in (
                    "D",
                    "absorption",
                    "nu_fission",
                    "kappa_fission",
                    "chi",
                    "inv_velocity",
                ):
                    g.create_dataset(
                        field,
                        data=np.stack(
                            [
                                np.asarray(
                                    getattr(self.composition(c, s), field)
                                )
                                for s in range(self.n_states)
                            ]
                        ),
                    )
                g.create_dataset(
                    "scatter",
                    data=np.stack(
                        [
                            np.asarray(
                                self.composition(c, s).scatter
                            ).reshape(self.n_groups, self.n_groups)
                            for s in range(self.n_states)
                        ]
                    ),
                )
                g.create_dataset("adf", data=self.adf(c))
                std = {
                    field: np.asarray(
                        getattr(self.composition(c, 0), f"{field}_std")
                    )
                    for field in ("D", "absorption", "nu_fission")
                }
                if any(v.size for v in std.values()):
                    sg = g.create_group("std_dev")
                    for field, value in std.items():
                        if value.size:
                            sg.create_dataset(field, data=value)

    @classmethod
    def from_hdf5(cls, path) -> XSLibrary:
        """Read a library written by :meth:`to_hdf5`."""
        import h5py

        with h5py.File(path, "r") as f:
            version = int(f.attrs.get("format_version", 0))
            if version != XSLIB_FORMAT_VERSION:
                raise InputError(
                    f"{path}: xslib format version {version} is not supported "
                    f"(this build reads version {XSLIB_FORMAT_VERSION})"
                )
            n_groups = int(f.attrs["n_groups"])
            lib = cls(n_groups, int(f.attrs["n_compositions"]))

            if "branch_axes" in f:
                axes_group = f["branch_axes"]
                names = [
                    n.decode() if isinstance(n, bytes) else str(n)
                    for n in axes_group.attrs["names"]
                ]
                lib.set_axes([(n, axes_group[n][()]) for n in names])

            for c in range(lib.n_compositions):
                g = f["compositions"][f"composition {c}"]
                for s in range(lib.n_states):
                    lib.set_composition(
                        c,
                        state=s,
                        D=g["D"][s],
                        absorption=g["absorption"][s],
                        nu_fission=g["nu_fission"][s],
                        kappa_fission=g["kappa_fission"][s],
                        chi=g["chi"][s],
                        inv_velocity=g["inv_velocity"][s],
                        scatter=g["scatter"][s],
                    )
                if "adf" in g:
                    adf = g["adf"][()]
                    if not np.allclose(adf, 1.0):
                        lib.set_adf(c, adf, n_axes=adf.shape[0] // 2)
                if "std_dev" in g:
                    lib.set_composition(
                        c,
                        std={k: v[()] for k, v in g["std_dev"].items()},
                    )

            if "delayed" in f:
                d = f["delayed"]
                lib.set_delayed(
                    d["beta"][()],
                    d["lambda"][()],
                    d["chi_delayed"][()] if "chi_delayed" in d else None,
                )

        lib.finalize(warn=False)
        return lib

    # ----------------------------------------------------------------- utils
    @staticmethod
    def _vector(value, G: int, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=float).ravel()
        if arr.size != G:
            raise InputError(
                f"{name} must have {G} entries, got {arr.size}"
            )
        return arr
