"""Control rod banks (FR-MODE-5).

A bank is a set of radial lattice columns that share one axial position. The
position is given in *steps*, following the convention KOMODO's ``%CROD`` card
uses, so an existing deck translates without arithmetic:

``zero_position``
    Height of the rod tip above the bottom of the mesh at step 0, in cm.
``step_size``
    Centimetres of travel per step.

Step 0 is therefore the most inserted position and increasing steps withdraw
the bank upward. Rods enter from the top, so a node is rodded when it sits
above the tip.

Mutating the geometry does not by itself update a solver that is already
holding it, exactly as for :meth:`~openndm.Model.swap_assemblies`. Call
:meth:`~openndm.Model.refresh` after moving a bank.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .exceptions import InputError

__all__ = ["ControlRodBank", "ControlRods", "RodWorth"]


@dataclass(frozen=True)
class RodWorth:
    """A bank worth curve (FR-MODE-5).

    Worth is a reactivity difference, not a difference in ``k_eff``:

    .. math:: w(p) = \\rho_{ref} - \\rho(p) = 1/k(p) - 1/k_{ref}

    reported in pcm and signed so that **inserting a bank gives positive
    worth**, which is the usual convention. Using :math:`\\Delta k/k` instead
    is a common shortcut that drifts from this by of order its own square,
    which matters once a bank is worth several thousand pcm.

    Attributes
    ----------
    bank : str or None
        The bank swept, or None for a multi-bank sequence.
    steps : ndarray
        Position of each point, in steps. The sequence index for a
        multi-bank sweep.
    tip_height : ndarray
        Tip height above the bottom of the mesh, in cm.
    k_eff : ndarray
    integral : ndarray
        Worth relative to the reference position, in pcm.
    differential : ndarray
        The derivative of :attr:`integral` with respect to position, in pcm
        per step. Central differences inside the sweep, one-sided at its
        ends, over the actual spacing -- so a non-uniform sweep is handled,
        but a sweep of one point has no derivative and returns zero.

        Note the sign. Steps *withdraw* the bank and withdrawal reduces
        worth, so this is negative for a normal bank, largest in magnitude
        where the bank is most effective. A differential worth curve is
        conventionally plotted as its negative, the reactivity added per
        step of withdrawal. It is left as a plain derivative here so that
        integrating it returns :attr:`integral` exactly, rather than
        returning it with a sign flip.
    reference_k : float
        ``k_eff`` at the reference position.
    """

    bank: str | None
    steps: np.ndarray
    tip_height: np.ndarray
    k_eff: np.ndarray
    integral: np.ndarray
    differential: np.ndarray
    reference_k: float

    def __repr__(self) -> str:
        span = float(np.max(self.integral) - np.min(self.integral))
        return (
            f"<RodWorth bank={self.bank!r} points={self.steps.size} "
            f"span={span:.1f} pcm>"
        )


@dataclass(frozen=True)
class ControlRodBank:
    """One bank of control rods.

    Parameters
    ----------
    name : str
        Identifies the bank when setting positions.
    columns : array_like of bool, shape (ny, nx)
        Radial lattice positions this bank occupies. This is KOMODO's
        ``BMAP`` reduced to a single bank.
    rodded : mapping of int to int
        Composition substitution applied to a node the bank covers, keyed by
        the node's *unrodded* composition. Every composition the bank can
        reach must appear, including reflector compositions if the rods
        travel through an axial reflector; a missing key is an error rather
        than a silent no-op, because an unsubstituted node looks exactly like
        a correctly withdrawn one.
    step_size : float
        Centimetres of travel per step.
    zero_position : float
        Height of the tip above the bottom of the mesh at step 0, in cm.
        Note this is the bottom of the *mesh*, not of the active core: with a
        bottom axial reflector present, add its height.
    max_steps : int, optional
        Upper bound on the position, enforced when a position is set.
    cusp : mapping of int to int, optional
        Spare library composition index per unrodded composition, used to
        hold the homogenised mixture of the node the rod tip sits inside.
        Supplying it turns on the cusping correction for this bank; see
        :class:`ControlRods`. Without it the node is rounded to whichever
        state covers its centre.
    """

    name: str
    columns: np.ndarray
    rodded: Mapping[int, int]
    step_size: float = 1.0
    zero_position: float = 0.0
    max_steps: int | None = None
    cusp: Mapping[int, int] | None = None

    def __post_init__(self) -> None:
        columns = np.asarray(self.columns, dtype=bool)
        if columns.ndim != 2:
            raise InputError(
                f"bank {self.name!r}: columns must be a 2D (ny, nx) mask, "
                f"got {columns.ndim}D"
            )
        if not columns.any():
            raise InputError(f"bank {self.name!r}: no columns selected")
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "rodded", dict(self.rodded))
        if self.cusp is not None:
            cusp = dict(self.cusp)
            missing = sorted(set(cusp) - set(self.rodded))
            if missing:
                raise InputError(
                    f"bank {self.name!r}: cusp names composition(s) {missing} "
                    "that have no rodded counterpart"
                )
            slots = sorted(cusp.values())
            if len(set(slots)) != len(slots):
                raise InputError(
                    f"bank {self.name!r}: cusp slots {slots} are not distinct; "
                    "each mixture needs its own composition"
                )
            object.__setattr__(self, "cusp", cusp)
        if self.step_size <= 0.0:
            raise InputError(
                f"bank {self.name!r}: step_size must be positive, got {self.step_size}"
            )
        if self.max_steps is not None and self.max_steps < 0:
            raise InputError(f"bank {self.name!r}: max_steps must not be negative")

    def tip_height(self, steps: float) -> float:
        """Height of the rod tip above the bottom of the mesh, in cm."""
        return self.zero_position + float(steps) * self.step_size


class ControlRods:
    """Control rod banks bound to a geometry.

    The unrodded composition of every node the banks can reach is snapshotted
    at construction, so positions are absolute rather than incremental: moving
    a bank twice gives the same core as setting its final position once, and
    withdrawing it restores exactly what was there before.

    Parameters
    ----------
    geometry : Geometry
        Built with the banks **withdrawn** -- every node carrying its unrodded
        composition.
    banks : sequence of ControlRodBank
    library : XSLibrary, optional
        Required when any bank sets ``cusp``. The homogenised mixture is
        written into the spare composition each time a bank moves, so the
        library must be the one the model solves with. It must be collapsed:
        a branch-parameterised library cannot be solved either way.
    """

    def __init__(self, geometry, banks: Sequence[ControlRodBank], library=None):
        self._geometry = geometry
        self._library = library
        self._banks: dict[str, ControlRodBank] = {}
        for bank in banks:
            if bank.name in self._banks:
                raise InputError(f"duplicate bank name {bank.name!r}")
            self._banks[bank.name] = bank

        cusping = [b.name for b in self._banks.values() if b.cusp is not None]
        if cusping and library is None:
            raise InputError(
                f"bank(s) {cusping} ask for cusping, which needs the library "
                "the model solves with"
            )
        if library is not None and library.n_states > 1:
            raise InputError(
                "a branch-parameterised library must be collapsed before it "
                "can be used for cusping"
            )

        nz, ny, nx = geometry.shape
        edges = np.concatenate([[0.0], np.cumsum(geometry.dz)])
        centres = 0.5 * (edges[:-1] + edges[1:])
        mapping = geometry.lattice_to_node.reshape(nz, ny, nx)
        compositions = geometry.compositions

        self._nodes: dict[str, np.ndarray] = {}
        self._centres: dict[str, np.ndarray] = {}
        self._lo: dict[str, np.ndarray] = {}
        self._hi: dict[str, np.ndarray] = {}
        self._base: dict[str, np.ndarray] = {}
        self._column: dict[str, np.ndarray] = {}
        self._edges = edges
        for name, bank in self._banks.items():
            if bank.columns.shape != (ny, nx):
                raise InputError(
                    f"bank {name!r}: columns has shape {bank.columns.shape}, "
                    f"but the lattice is ({ny}, {nx})"
                )
            covered = np.broadcast_to(bank.columns, (nz, ny, nx))
            nodes = mapping[covered & (mapping >= 0)]
            self._nodes[name] = nodes
            active = covered & (mapping >= 0)

            def _per_plane(values, active=active, ny=ny, nx=nx, nz=nz):
                return np.repeat(values, ny * nx).reshape(nz, ny, nx)[active]

            self._centres[name] = _per_plane(centres)
            self._lo[name] = _per_plane(edges[:-1])
            self._hi[name] = _per_plane(edges[1:])
            self._base[name] = compositions[nodes].copy()
            where = np.argwhere(bank.columns)
            self._column[name] = np.stack([mapping[:, j, i] for j, i in where], axis=1)
            if bank.cusp is not None:
                reachable = {int(c) for c in np.unique(self._base[name])}
                uncovered = sorted(reachable - set(bank.cusp))
                if uncovered:
                    raise InputError(
                        f"bank {name!r}: cusp has no spare composition for "
                        f"{uncovered}, which the bank reaches"
                    )

        self._positions: dict[str, float | None] = dict.fromkeys(self._banks)

    # ------------------------------------------------------------ properties
    @property
    def banks(self) -> tuple[str, ...]:
        return tuple(self._banks)

    @property
    def positions(self) -> dict[str, float | None]:
        """Position of each bank in steps; ``None`` means fully withdrawn."""
        return dict(self._positions)

    def tip_height(self, name: str) -> float:
        """Tip height of one bank above the bottom of the mesh, in cm.

        A fully withdrawn bank returns the top of the mesh.
        """
        bank = self._bank(name)
        steps = self._positions[name]
        if steps is None:
            return float(np.sum(self._geometry.dz))
        return bank.tip_height(steps)

    # --------------------------------------------------------------- methods
    def insert(self, positions: Mapping[str, float | None] | None = None, **kwargs):
        """Set bank positions in steps and rewrite the node compositions.

        Accepts a mapping, keyword arguments, or both. A bank left unnamed
        keeps its current position; ``None`` withdraws one fully.

        Call :meth:`~openndm.Model.refresh` afterwards if a solver already
        holds this geometry.
        """
        requested: dict[str, float | None] = {}
        requested.update(positions or {})
        requested.update(kwargs)

        for name, steps in requested.items():
            bank = self._bank(name)
            if steps is not None:
                steps = float(steps)
                if steps < 0.0:
                    raise InputError(
                        f"bank {name!r}: position must not be negative, got {steps}"
                    )
                if bank.max_steps is not None and steps > bank.max_steps:
                    raise InputError(
                        f"bank {name!r}: position {steps} exceeds max_steps "
                        f"{bank.max_steps}"
                    )
            self._positions[name] = steps

        self._apply()
        return self

    def withdraw(self, *names: str):
        """Fully withdraw the named banks, or all of them if none are named."""
        targets = names or tuple(self._banks)
        return self.insert(dict.fromkeys(targets, None))

    def converge_cusping(
        self, model, max_sweeps: int = 8, tolerance: float = 1.0e-6, **overrides
    ):
        """Solve, re-weighting the tip node's mixture with the flux each time.

        :meth:`insert` weights the partial node by volume alone, which is the
        flat-flux limit and biased: the flux is depressed on the rodded side,
        so volume weighting over-counts the rodded absorption. Correcting it
        needs a flux, and a flux needs a solve, so this iterates.

        Returns the final :class:`~openndm.Result`. With no bank cusping, or
        with every tip on a plane boundary, it is a single solve.

        Parameters
        ----------
        model : Model
            Must hold this object's geometry and library.
        max_sweeps : int
            Cap on re-weightings after the first solve.
        tolerance : float
            Stop once the largest relative change in a mixed cross section
            falls below this.
        **overrides
            Settings overridden for every solve, e.g. ``warm_start=True``.
        """
        result = model.solve(**overrides)
        for _ in range(max_sweeps):
            change = self.reweight(result.flux)
            if change is None or change < tolerance:
                break
            model.refresh()
            result = model.solve(**overrides)
        return result

    def reweight(self, flux) -> float | None:
        """Re-weight every partially rodded node against a flux already in hand.

        :meth:`insert` weights the tip node by volume, which is the flat-flux
        limit and biased. :meth:`converge_cusping` corrects that by iterating
        static solves, which a transient cannot do: a static solve would
        discard the time-dependent flux and the precursors with it. Passing
        the previous step's flux here instead lags the correction by one step
        rather than dropping it.

        Parameters
        ----------
        flux : array_like, shape (n_nodes, n_groups)
            Flux to weight with, for instance
            :attr:`~openndm.Transient.flux`.

        Returns
        -------
        float or None
            Largest relative change in a mixed cross section, or ``None`` when
            no bank has a partially rodded node to correct.
        """
        return self._reweight_from_flux(np.asarray(flux))

    def worth_curve(
        self,
        model,
        bank: str | None = None,
        positions=None,
        *,
        reference=None,
        cusping: bool = True,
        warm_start: bool = True,
    ) -> RodWorth:
        """Integral and differential worth of a bank over a set of positions.

        Parameters
        ----------
        model : Model
            Must hold this object's geometry and library.
        bank : str, optional
            Bank to sweep. Every other bank stays where it is, which is how
            an overlapping sequence is modelled: set the others first. Pass
            None to sweep a prepared sequence instead, giving ``positions``
            as a sequence of ``{bank: steps}`` mappings.
        positions : sequence
            Positions in steps, or mappings when ``bank`` is None.
        reference : float or mapping, optional
            The position worth is measured from. Defaults to fully
            withdrawn, which is the usual convention and makes an inserted
            bank's worth positive.
        cusping : bool
            Converge the cusping correction at each point, where a bank has
            `cusp` slots. Turning it off measures the volume-weighted
            mixture instead, which is biased -- see
            :meth:`converge_cusping`.
        warm_start : bool
            Reuse the previous point's flux and coupling coefficients. A
            worth curve is the case warm starting exists for (FR-OPT-3):
            neighbouring positions differ in one node.

        Returns
        -------
        RodWorth

        Notes
        -----
        The bank positions are restored when the sweep finishes, and the
        model refreshed, so measuring worth does not move the rods.
        """
        if positions is None:
            raise InputError("worth_curve needs positions to sweep")
        if bank is None:
            states = [dict(p) for p in positions]
            steps = np.arange(len(states), dtype=float)
            for state in states:
                for name in state:
                    self._bank(name)
        else:
            self._bank(bank)
            steps = np.asarray(positions, dtype=float)
            states = [{bank: float(p)} for p in steps]
        if not states:
            raise InputError("worth_curve needs at least one position")

        if reference is None:
            reference_state = dict.fromkeys(self._banks)
        elif isinstance(reference, Mapping):
            reference_state = dict(reference)
        elif bank is None:
            raise InputError(
                "a multi-bank sweep needs its reference given as a mapping"
            )
        else:
            reference_state = {bank: float(reference)}

        restore = self.positions
        try:
            reference_k = self._solve_at(model, reference_state, cusping, warm_start)
            eigenvalues = [
                self._solve_at(model, state, cusping, warm_start) for state in states
            ]
        finally:
            self.insert(restore)
            model.refresh()

        k_eff = np.asarray(eigenvalues, dtype=float)
        # Worth is a reactivity difference: rho_ref - rho(p) with
        # rho = 1 - 1/k, which is 1/k - 1/k_ref. Positive when the bank is
        # further in than the reference, and unlike dk/k it stays additive
        # over a sequence.
        integral = 1.0e5 * (1.0 / k_eff - 1.0 / reference_k)
        if steps.size > 1:
            differential = np.gradient(integral, steps)
        else:
            differential = np.zeros_like(integral)

        tip = np.array(
            [self._tip_height_at(state, bank) for state in states], dtype=float
        )
        return RodWorth(
            bank=bank,
            steps=steps,
            tip_height=tip,
            k_eff=k_eff,
            integral=integral,
            differential=differential,
            reference_k=float(reference_k),
        )

    # ----------------------------------------------------------------- inner
    def _solve_at(self, model, state, cusping: bool, warm_start: bool) -> float:
        self.insert(state)
        model.refresh()
        wants_cusping = cusping and any(
            b.cusp is not None for b in self._banks.values()
        )
        if wants_cusping:
            return self.converge_cusping(model, warm_start=warm_start).k_eff
        return model.solve(warm_start=warm_start).k_eff

    def _tip_height_at(self, state, bank: str | None) -> float:
        if bank is not None:
            return self._banks[bank].tip_height(state[bank])
        heights = [
            self._banks[name].tip_height(steps)
            for name, steps in state.items()
            if steps is not None
        ]
        return min(heights) if heights else float(np.sum(self._geometry.dz))

    def _partial_plane(self, bank: ControlRodBank, steps):
        """Plane holding the rod tip and the share of it the rod occupies."""
        if steps is None:
            return None
        tip = bank.tip_height(steps)
        lo, hi = self._edges[:-1], self._edges[1:]
        inside = np.flatnonzero((lo < tip) & (tip < hi))
        if inside.size == 0:
            return None
        k = int(inside[0])
        return k, float((hi[k] - tip) / (hi[k] - lo[k]))

    def _reweight_from_flux(self, flux: np.ndarray) -> float | None:
        """Rewrite every cusp mixture using flux-volume weights.

        Returns the largest relative change, or None when no bank has a
        partially rodded node to correct.
        """
        flux = flux.reshape(self._geometry.n_nodes, -1)
        largest = None
        for name, bank in self._banks.items():
            if bank.cusp is None:
                continue
            found = self._partial_plane(bank, self._positions[name])
            if found is None:
                continue
            plane, fraction = found
            column = self._column[name]
            nz = column.shape[0]
            below = column[plane - 1] if plane > 0 else column[plane]
            here = column[plane]
            above = column[plane + 1] if plane + 1 < nz else column[plane]

            column_base = self._column_base(name, plane)
            for slot_base in np.unique(column_base[column_base >= 0]):
                slot_base = int(slot_base)
                take = column_base == slot_base
                active = take & (here >= 0) & (below >= 0) & (above >= 0)
                if not active.any():
                    continue
                # The lower half of the node is unrodded and the upper half is
                # rodded, so each half is represented by the average of the
                # node and the neighbour on its own side.
                phi_un = 0.5 * (
                    flux[below[active]].mean(axis=0) + flux[here[active]].mean(axis=0)
                )
                phi_rod = 0.5 * (
                    flux[here[active]].mean(axis=0) + flux[above[active]].mean(axis=0)
                )
                slot = int(bank.cusp[slot_base])
                before = np.asarray(
                    self._library.composition(slot).absorption, dtype=float
                )
                self._library.set_composition(
                    slot,
                    **_homogenise(
                        self._library,
                        slot_base,
                        self._rodded_of(bank, slot_base),
                        fraction,
                        flux=(phi_un, phi_rod),
                    ),
                )
                after = np.asarray(
                    self._library.composition(slot).absorption, dtype=float
                )
                scale = np.where(np.abs(before) > 0.0, np.abs(before), 1.0)
                moved = float(np.max(np.abs(after - before) / scale))
                largest = moved if largest is None else max(largest, moved)
        if largest is not None:
            self._library.finalize(warn=False)
        return largest

    def _column_base(self, name: str, plane: int) -> np.ndarray:
        """Unrodded composition of each column of a bank at one plane."""
        column = self._column[name]
        nodes = self._nodes[name]
        base = self._base[name]
        lookup = dict(zip(nodes.tolist(), base.tolist(), strict=True))
        return np.array([lookup.get(int(n), -1) for n in column[plane]], dtype=int)

    def _bank(self, name: str) -> ControlRodBank:
        try:
            return self._banks[name]
        except KeyError:
            raise InputError(
                f"unknown bank {name!r}; known banks are {sorted(self._banks)}"
            ) from None

    def _apply(self) -> None:
        geometry = self._geometry
        for name, bank in self._banks.items():
            nodes = self._nodes[name]
            base = self._base[name]
            steps = self._positions[name]

            if steps is None:
                rodded = np.zeros(nodes.shape, dtype=bool)
                partial = np.zeros(nodes.shape, dtype=bool)
                fraction = np.zeros(nodes.shape, dtype=float)
            else:
                tip = bank.tip_height(steps)
                lo, hi = self._lo[name], self._hi[name]
                # Rods enter from the top, so a node is rodded when it sits
                # above the tip. A node the tip falls inside is partial: the
                # fraction is how much of its height the rod occupies.
                rodded = lo >= tip
                partial = (lo < tip) & (tip < hi)
                fraction = np.zeros(nodes.shape, dtype=float)
                if bank.cusp is None:
                    # No mixture available, so round to whichever state covers
                    # the node centre. This is what makes k_eff a staircase in
                    # rod position.
                    rodded = self._centres[name] > tip
                    partial = np.zeros(nodes.shape, dtype=bool)
                else:
                    np.divide(hi - tip, hi - lo, out=fraction, where=partial)
                    self._write_cusp_compositions(bank, base, partial, fraction)

            for node, was, is_rodded, is_partial in zip(
                nodes, base, rodded, partial, strict=True
            ):
                if is_partial:
                    geometry.set_composition(int(node), int(bank.cusp[int(was)]))
                elif is_rodded:
                    geometry.set_composition(int(node), self._rodded_of(bank, int(was)))
                else:
                    geometry.set_composition(int(node), int(was))

    def _rodded_of(self, bank: ControlRodBank, unrodded: int) -> int:
        try:
            return int(bank.rodded[unrodded])
        except KeyError:
            raise InputError(
                f"bank {bank.name!r} covers a node of composition {unrodded}, "
                f"which has no rodded counterpart; rodded maps "
                f"{sorted(bank.rodded)}"
            ) from None

    def _write_cusp_compositions(self, bank, base, partial, fraction) -> None:
        """Homogenise the rodded and unrodded halves of each partial node.

        Every column of a bank shares one tip height and one axial mesh, so
        all partial nodes with the same unrodded composition share one
        mixture and one spare slot.
        """
        if not partial.any():
            return
        for unrodded in np.unique(base[partial]):
            unrodded = int(unrodded)
            shares = fraction[partial & (base == unrodded)]
            self._library.set_composition(
                int(bank.cusp[unrodded]),
                **_homogenise(
                    self._library,
                    unrodded,
                    self._rodded_of(bank, unrodded),
                    float(shares[0]),
                ),
            )
        # Writing a composition marks the library unfinalized, and its removal
        # cross sections stay cached until it is finalized again. Skipping this
        # would leave the mixture written but not solved with.
        self._library.finalize(warn=False)


def _homogenise(
    library,
    unrodded: int,
    rodded: int,
    rodded_fraction: float,
    flux: tuple[np.ndarray, np.ndarray] | None = None,
):
    """Weight two compositions into one, for the node holding the rod tip.

    ``rodded_fraction`` is the share of the node the rod occupies.

    With ``flux`` omitted the weights are the volumes alone. That is the
    flat-flux limit, and it is biased: the flux is depressed on the rodded
    side, so weighting by volume over-counts the rodded absorption and puts
    ``k_eff`` low. On a ten-plane test core it leaves a -103 pcm mean bias
    and up to 259 pcm of error.

    ``flux`` is ``(unrodded, rodded)`` group flux in the two halves of the
    node, which turns the weights into flux-volume weights and removes most
    of that: the same case falls to a -13 pcm bias and 55 pcm of error, which
    is the size of the coarse mesh's own discretisation error. Getting it
    needs a solved flux, so it is an iteration -- see
    :meth:`ControlRods.converge_cusping`.

    The fission spectrum follows the fission source rather than either weight,
    because that is what it is a spectrum of.
    """
    a = library.composition(unrodded)
    b = library.composition(rodded)
    groups = library.n_groups
    volume_rod = float(rodded_fraction)
    volume_un = 1.0 - volume_rod

    if flux is None:
        w_un = np.full(groups, volume_un)
        w_rod = np.full(groups, volume_rod)
    else:
        phi_un, phi_rod = (np.asarray(x, dtype=float) for x in flux)
        w_un = volume_un * phi_un
        w_rod = volume_rod * phi_rod
    total = w_un + w_rod
    with np.errstate(invalid="ignore", divide="ignore"):
        w_un = np.where(total > 0.0, w_un / total, volume_un)
        w_rod = np.where(total > 0.0, w_rod / total, volume_rod)

    def mix(field):
        x = np.asarray(getattr(a, field), dtype=float)
        y = np.asarray(getattr(b, field), dtype=float)
        if x.size == groups * groups:
            # A scattering matrix is weighted by the flux of the group the
            # transfer leaves, which is its row.
            shape = (groups, groups)
            return (
                w_un[:, None] * x.reshape(shape) + w_rod[:, None] * y.reshape(shape)
            ).ravel()
        return w_un * x + w_rod * y

    source_un = float(np.sum(w_un * np.asarray(a.nu_fission, dtype=float)))
    source_rod = float(np.sum(w_rod * np.asarray(b.nu_fission, dtype=float)))
    if source_un + source_rod > 0.0:
        chi = (
            source_un * np.asarray(a.chi, dtype=float)
            + source_rod * np.asarray(b.chi, dtype=float)
        ) / (source_un + source_rod)
    else:
        chi = mix("chi")

    fields = {
        "D": mix("D"),
        "absorption": mix("absorption"),
        "nu_fission": mix("nu_fission"),
        "kappa_fission": mix("kappa_fission"),
        "chi": chi,
        "scatter": mix("scatter").reshape(groups, groups),
    }
    inv_velocity = mix("inv_velocity")
    if np.any(inv_velocity != 0.0):
        fields["inv_velocity"] = inv_velocity
    return fields
