"""Cross section library construction and I/O (FR-XS, FR-IN-3)."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

from . import _core
from .exceptions import InputError

__all__ = ["BranchAxis", "XSLibrary", "rotate_adf", "rotated_face"]

XSLIB_FORMAT_VERSION = 1
"""Layout version of the ``xslib.h5`` file this module reads and writes."""

BranchAxis = _core.BranchAxis

_EXTRAPOLATION = {
    "clamp": _core.Extrapolation.clamp,
    "linear": _core.Extrapolation.linear,
    "error": _core.Extrapolation.error,
}


def rotated_face(face: int, quarter_turns: int) -> int:
    """Face of an unrotated assembly that a turned one presents as ``face``.

    A quarter turn counter-clockwise about +z carries the assembly's +x face
    onto +y, +y onto -x, -x onto -y and -y onto +x, so the factor the lattice
    sees on a face is the one the unrotated assembly carried on the face
    returned here.

    Parameters
    ----------
    face : int
        Lattice face index in the order ``-x, +x, -y, +y, -z, +z``.
    quarter_turns : int
        Counter-clockwise turns about +z; reduced modulo 4.

    Returns
    -------
    int
        The axial faces are returned unchanged, since they do not move.

    Notes
    -----
    This is the same function the solver applies when it reads a rotated
    node's factors, rather than a second copy of the permutation that could
    drift from it.
    """
    return _core.rotated_face(int(face), int(quarter_turns))


def rotate_adf(values, quarter_turns: int) -> np.ndarray:
    """Permute a set of discontinuity factors onto a turned assembly.

    Parameters
    ----------
    values : array_like, shape (2 * n_axes, G)
        Per-face, per-group factors in the order ``-x, +x, -y, +y, -z, +z``.
    quarter_turns : int
        0 to 3, counter-clockwise about +z.

    Returns
    -------
    ndarray
        A new array; the input is not modified.

    Notes
    -----
    Rotation is a permutation and nothing else, so it is exactly invertible:
    four quarter turns are the identity, and two are the swap of each radial
    pair. A fully symmetric set is unchanged by any of them.

    Examples
    --------
    >>> import numpy as np
    >>> adf = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    >>> rotate_adf(adf, 1).ravel().tolist()
    [4.0, 3.0, 1.0, 2.0, 5.0, 6.0]
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[0] % 2 != 0:
        raise InputError(
            "discontinuity factors must be a (2 * n_axes, G) array, got "
            f"shape {arr.shape}"
        )
    if not 0 <= quarter_turns <= 3:
        raise InputError(
            f"quarter_turns must be 0, 1, 2 or 3, got {quarter_turns}"
        )
    order = [rotated_face(f, quarter_turns) for f in range(arr.shape[0])]
    return arr[order]


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
    ...     nu_fission=[0.0, 0.135], kappa_fission=[0.0, 0.135],
    ...     chi=[1.0, 0.0], scatter=[[0.0, 0.02], [0.0, 0.0]],
    ... )
    >>> lib.finalize()
    []
    >>> lib.n_groups
    2
    """

    def __init__(self, n_groups: int, n_compositions: int):
        self._lib = _core.XSLibrary(int(n_groups), int(n_compositions))

    @property
    def n_groups(self) -> int:
        """Number of energy groups."""
        return self._lib.n_groups

    @property
    def n_compositions(self) -> int:
        """Number of distinct homogenised compositions."""
        return self._lib.n_compositions

    @property
    def n_states(self) -> int:
        """Number of branch grid points, 1 for a single-state library."""
        return self._lib.n_states

    @property
    def axes(self) -> list:
        """Branch axes, outermost first, as a new list of ``BranchAxis``.

        Empty for a single-state library.
        """
        return list(self._lib.axes)

    @property
    def finalized(self) -> bool:
        """True once :meth:`finalize` has validated and cached the library.

        Reset by any mutable access to a composition or to the branch axes.
        """
        return self._lib.finalized

    @property
    def has_uncertainty(self) -> bool:
        """True when any composition carries 1-sigma values (FR-XS-9)."""
        return self._lib.has_uncertainty

    @property
    def extrapolation(self) -> str:
        """Off-grid interpolation policy: ``clamp``, ``linear`` or ``error``."""
        return self._lib.extrapolation.name

    @extrapolation.setter
    def extrapolation(self, value: str) -> None:
        """Set the off-grid policy from its name or an ``Extrapolation``.

        Parameters
        ----------
        value : {'clamp', 'linear', 'error'} or Extrapolation
            Policy applied to a state point outside the branch grid.
        """
        if isinstance(value, str):
            try:
                value = _EXTRAPOLATION[value.lower()]
            except KeyError:
                raise ValueError(
                    f"unknown extrapolation policy {value!r}; "
                    f"choose from {sorted(_EXTRAPOLATION)}"
                ) from None
        self._lib.extrapolation = value

    def set_axes(self, axes: Sequence[tuple[str, Sequence[float]]]) -> None:
        """Declare the branch axes, outermost first (FR-XS-5).

        Parameters
        ----------
        axes : sequence of (name, points)
            Each axis is a name such as ``'fuel_temperature'`` and a strictly
            increasing sequence of grid points, in the units of the state
            variable the name denotes: K for a temperature, ppm for a boron
            concentration.

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
        D : array_like, shape (G,), optional
            Diffusion coefficient per group, in cm. Mutually exclusive with
            ``transport``.
        transport : array_like, shape (G,), optional
            Transport cross section per group, in cm^-1; ``D`` is set to
            ``1/(3*Sigma_tr)``.
        absorption, nu_fission : array_like, shape (G,), optional
            Absorption and production cross section per group, in cm^-1.
        kappa_fission : array_like, shape (G,), optional
            Energy release per unit path, in J cm^-1. This is what makes
            power; ``nu_fission`` alone moves ``k_eff`` and nothing else.
        chi : array_like, shape (G,), optional
            Fission spectrum, dimensionless and summing to one over groups.
        inv_velocity : array_like, shape (G,), optional
            Reciprocal group velocity, in s cm^-1. Needed by a transient only.
        scatter : array_like, shape (G, G), optional
            Scattering matrix in cm^-1, indexed
            ``scatter[from_group][to_group]``.
        std : mapping of str to array_like, optional
            1-sigma uncertainties keyed by field name, each in the units of
            the field it belongs to, e.g.
            ``{'absorption': [...], 'nu_fission': [...]}`` (FR-XS-9).

        Notes
        -----
        Anything omitted stays zero.
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
        index : int
            Composition index.
        values : array_like, shape (2 * n_axes, G) or (G,)
            Per-face, per-group factors, dimensionless, in the face order
            ``-x, +x, -y, +y, -z, +z``. A ``(G,)`` array is broadcast to every
            face.
        n_axes : int, optional
            Axes the factors cover: 3 for a 3D mesh, 2 for a radial-only set.
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

    def rotated_adf(self, index: int, quarter_turns: int, n_axes: int = 3):
        """Discontinuity factors of one composition, turned counter-clockwise.

        Parameters
        ----------
        index : int
            Composition index.
        quarter_turns : int
            0 to 3, counter-clockwise about +z, as KOMODO's ``%ADF`` ``ROT``
            defines it.
        n_axes : int, optional
            Axes to read, in the face order ``-x, +x, -y, +y, -z, +z``.

        Returns
        -------
        ndarray, shape (2 * n_axes, G)
            The dimensionless set the assembly would present after turning. A
            new array. Four quarter turns return the original exactly.

        See Also
        --------
        openndm.rotate_adf : the same permutation on a plain array.
        openndm.Geometry.set_rotation : turn an assembly where it sits, which
            is what a core with one assembly type in several orientations
            wants.
        """
        return rotate_adf(self.adf(index, n_axes), quarter_turns)

    def adf(self, index: int, n_axes: int = 3) -> np.ndarray:
        """Discontinuity factors of one composition.

        Parameters
        ----------
        index : int
            Composition index.
        n_axes : int, optional
            Axes to read, in the face order ``-x, +x, -y, +y, -z, +z``.

        Returns
        -------
        ndarray, shape (2 * n_axes, G)
            Dimensionless per-face, per-group factors. A new array, not a
            view onto the C++ library.
        """
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
        beta : array_like, shape (n_precursors,)
            Delayed neutron fraction per precursor group, dimensionless. At
            most 8 groups.
        decay_constant : array_like, shape (n_precursors,)
            Precursor decay constant per group, in s^-1.
        chi_delayed : array_like, shape (n_precursors, G), optional
            Delayed spectrum, dimensionless; defaults to the prompt spectrum
            when omitted.
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

        Parameters
        ----------
        warn : bool, optional
            Issue each finding as a Python warning as well as returning it.

        Returns
        -------
        list of str
            Non-fatal findings. Negative scattering transfers are reported
            this way rather than raised, because Monte Carlo noise produces
            them routinely.

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

    def composition(self, index: int, state: int = 0):
        """Snapshot of one composition's group constants, for inspection.

        Parameters
        ----------
        index : int
            Composition index.
        state : int, optional
            Branch grid point index; 0 for a single-state library.

        Returns
        -------
        Composition
            A copy, with the fields and units of :meth:`set_composition`.
            Mutating it does not change the library; use
            :meth:`set_composition` to write.
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

    def interpolate_by_composition(self, state, *, out: XSLibrary | None = None):
        """Collapse the branch grid with a different state per composition.

        :meth:`interpolate` puts every composition at one state point, which a
        coupled solve cannot use: cross sections live per composition, so a
        temperature distribution is expressed by giving each region that can
        differ its own composition index and its own state.

        The grid is interpolated once per *distinct* state row, so a core with
        many compositions all at the same condition costs one interpolation,
        and one whose every composition differs costs one each.

        Parameters
        ----------
        state : mapping of str to array_like
            One entry per branch axis, each a scalar or one value per
            composition. Scalars broadcast.
        out : XSLibrary, optional
            Written in place and re-finalized, instead of a new library being
            built. This is what a coupled solve wants: the solver holds a
            reference to its library, so replacing the object would strand it.

        Returns
        -------
        XSLibrary
            Finalized and single-state. ``out`` itself when given.

        Raises
        ------
        InputError
            On a state that does not name exactly the branch axes, a value
            that is neither a scalar nor one per composition, or an ``out``
            library of a different shape.

        Examples
        --------
        >>> import numpy as np
        >>> lib = XSLibrary(1, 2)
        >>> lib.set_axes([("fuel_temperature", [500.0, 1000.0])])
        >>> for s, a in enumerate([0.08, 0.09]):
        ...     for c in range(2):
        ...         lib.set_composition(c, state=s, D=[1.0], absorption=[a],
        ...                             nu_fission=[0.1], kappa_fission=[0.1],
        ...                             chi=[1.0], scatter=[[0.0]])
        >>> _ = lib.finalize()
        >>> hot = lib.interpolate_by_composition(
        ...     {"fuel_temperature": [500.0, 1000.0]})
        >>> [round(hot.composition(c).absorption[0], 4) for c in range(2)]
        [0.08, 0.09]
        """
        names = [a.name for a in self.axes]
        if not names:
            raise InputError("the library has no branch axes to interpolate")
        missing = set(names) - set(state)
        extra = set(state) - set(names)
        if missing or extra:
            raise InputError(
                f"branch state mismatch: missing {sorted(missing)}, "
                f"unexpected {sorted(extra)}; axes are {names}"
            )

        n_comps = self.n_compositions
        columns = []
        for name in names:
            values = np.asarray(state[name], dtype=float)
            if values.ndim == 0:
                values = np.full(n_comps, float(values))
            if values.shape != (n_comps,):
                raise InputError(
                    f"branch axis {name!r} needs a scalar or one value per "
                    f"each of {n_comps} compositions, got shape {values.shape}"
                )
            columns.append(values)

        rows = np.stack(columns, axis=1)
        distinct, inverse = np.unique(rows, axis=0, return_inverse=True)
        collapsed = [
            self.interpolate(**dict(zip(names, row, strict=True)))
            for row in distinct
        ]

        target = self._collapse_target(out)
        for index in range(n_comps):
            _copy_composition(collapsed[int(inverse[index])], target, index)
        for index in range(n_comps):
            target.set_adf(index, self.adf(index))
        delayed = self._lib.delayed
        if delayed.beta:
            target.set_delayed(
                delayed.beta,
                delayed.lambda_,
                np.asarray(delayed.chi_delayed).reshape(
                    delayed.n_precursors, self.n_groups
                )
                if delayed.chi_delayed
                else None,
            )
        target.finalize(warn=False)
        return target

    def _collapse_target(self, out: XSLibrary | None) -> XSLibrary:
        """The library a per-composition collapse writes into."""
        if out is None:
            return XSLibrary(self.n_groups, self.n_compositions)
        if (out.n_groups, out.n_compositions) != (
            self.n_groups,
            self.n_compositions,
        ):
            raise InputError(
                f"out holds {out.n_groups} groups and {out.n_compositions} "
                f"compositions, not {self.n_groups} and {self.n_compositions}"
            )
        if out.n_states > 1:
            raise InputError("out must be single-state, not branch-parameterised")
        return out

    def array(self, field: str, state: int = 0) -> np.ndarray:
        """Stack one field across compositions.

        Parameters
        ----------
        field : str
            Field name as :meth:`set_composition` spells it, such as
            ``'absorption'`` or ``'scatter'``.
        state : int, optional
            Branch grid point index; 0 for a single-state library.

        Returns
        -------
        ndarray, shape (n_compositions, G)
            A new array in the units of the field, or
            ``(n_compositions, G, G)`` for ``'scatter'``.
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

    def to_hdf5(self, path) -> None:
        """Write the library to ``xslib.h5`` (FR-IN-3).

        Parameters
        ----------
        path : path-like
            Destination file. ``.h5`` is the conventional suffix and is not
            enforced.

        Notes
        -----
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
        """Read a library written by :meth:`to_hdf5`.

        Parameters
        ----------
        path : path-like
            File to read.

        Returns
        -------
        XSLibrary
            Finalized, with any findings suppressed: the library was already
            validated when it was written.

        Raises
        ------
        InputError
            On a file whose ``format_version`` this build does not read.
        """
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

    @staticmethod
    def _vector(value, G: int, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=float).ravel()
        if arr.size != G:
            raise InputError(
                f"{name} must have {G} entries, got {arr.size}"
            )
        return arr


def _copy_composition(source: XSLibrary, target: XSLibrary, index: int) -> None:
    """Write one composition's group constants from one library into another."""
    record = source.composition(index)
    groups = source.n_groups
    std = {}
    for name in ("D", "absorption", "nu_fission", "scatter"):
        values = getattr(record, f"{name}_std", None)
        if values:
            std[name] = np.asarray(values, dtype=float)
    target.set_composition(
        index,
        D=record.D,
        absorption=record.absorption,
        nu_fission=record.nu_fission,
        kappa_fission=record.kappa_fission,
        chi=record.chi,
        scatter=np.asarray(record.scatter, dtype=float).reshape(groups, groups),
        inv_velocity=record.inv_velocity,
        std=std or None,
    )
