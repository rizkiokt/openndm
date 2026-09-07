"""Branch library generation (FR-OMC-11).

A branch table over five state variables and thirty compositions is thousands
of OpenMC runs. The driver is therefore built for that reality from the start:
it checkpoints after every completed branch, resumes from a partial run, and
can emit an HPC job array instead of executing anything itself.
"""

from __future__ import annotations

import itertools
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..exceptions import InputError
from ..xslib import XSLibrary

__all__ = ["BranchDriver", "BranchGrid", "BranchPoint"]


@dataclass(frozen=True)
class BranchPoint:
    """One state point of a branch grid."""

    index: int
    state: dict

    @property
    def key(self) -> str:
        """Stable identifier used for checkpoint filenames."""
        return "_".join(f"{k}{v:g}" for k, v in sorted(self.state.items()))


class BranchGrid:
    """The Cartesian product of the branch axes (FR-XS-5).

    Parameters
    ----------
    **axes
        One keyword per state variable, each a sequence of grid points, e.g.
        ``BranchGrid(fuel_temperature=[560, 900, 1200], boron=[0, 1500])``.
        Insertion order becomes axis order.

    Examples
    --------
    >>> grid = BranchGrid(fuel_temperature=[560.0, 900.0], boron=[0.0, 1500.0])
    >>> len(grid)
    4
    >>> grid[0].state
    {'fuel_temperature': 560.0, 'boron': 0.0}
    """

    def __init__(self, **axes: Sequence[float]):
        if not axes:
            raise InputError("a branch grid needs at least one axis")
        self.axes: dict[str, np.ndarray] = {}
        for name, points in axes.items():
            arr = np.asarray(points, dtype=float).ravel()
            if arr.size == 0:
                raise InputError(f"axis {name!r} has no points")
            if np.any(np.diff(arr) <= 0.0):
                raise InputError(
                    f"axis {name!r} must be strictly increasing, got {arr}"
                )
            self.axes[name] = arr

    @property
    def names(self) -> list[str]:
        return list(self.axes)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(a.size for a in self.axes.values())

    def __len__(self) -> int:
        return int(np.prod(self.shape))

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> BranchPoint:
        if not 0 <= index < len(self):
            raise IndexError(index)
        # Row-major over the axes in declaration order, matching the flat state
        # index the C++ library uses.
        state = {}
        remaining = index
        for name, points in reversed(list(self.axes.items())):
            remaining, position = divmod(remaining, points.size)
            state[name] = float(points[position])
        return BranchPoint(index=index, state=dict(reversed(list(state.items()))))

    def to_library_axes(self) -> list[tuple[str, np.ndarray]]:
        """Axis specification for :meth:`openndm.XSLibrary.set_axes`."""
        return [(name, points) for name, points in self.axes.items()]

    def __repr__(self) -> str:
        dims = ", ".join(f"{n}({a.size})" for n, a in self.axes.items())
        return f"<BranchGrid {dims} = {len(self)} points>"


@dataclass
class BranchDriver:
    """Execute a branch calculation and assemble the resulting library.

    Parameters
    ----------
    model_factory : callable
        ``model_factory(**state) -> openmc.Model``. Called once per branch
        point with that point's state variables as keyword arguments.
    grid : BranchGrid
    library_factory : callable
        ``library_factory(model, statepoint, **state) -> XSLibrary``, turning
        one completed OpenMC run into a single-state OpenNDM library.
        Typically a thin wrapper over
        :func:`openndm.gc.from_mgxs_library`.
    workdir : path-like
        Root directory for per-branch run directories and checkpoints.
    checkpoint : bool
        Write a JSON checkpoint after each branch so an interrupted run
        resumes where it stopped.

    Examples
    --------
    >>> grid = BranchGrid(boron=[0.0, 1500.0])
    >>> driver = BranchDriver(                       # doctest: +SKIP
    ...     model_factory=build_model, grid=grid,
    ...     library_factory=extract_library, workdir="branches")
    >>> library = driver.run(processes=4)            # doctest: +SKIP
    """

    model_factory: Callable
    grid: BranchGrid
    library_factory: Callable
    workdir: os.PathLike | str = "openndm_branches"
    checkpoint: bool = True
    _done: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self.workdir = Path(self.workdir)

    # ----------------------------------------------------------- checkpoints
    @property
    def checkpoint_path(self) -> Path:
        return self.workdir / "branch_checkpoint.json"

    def completed(self) -> set[int]:
        """Branch indices already finished, read from the checkpoint."""
        if not self.checkpoint_path.exists():
            return set()
        with open(self.checkpoint_path) as f:
            return {int(k) for k in json.load(f).get("completed", [])}

    def _record(self, index: int) -> None:
        if not self.checkpoint:
            return
        done = self.completed() | {index}
        self.workdir.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, "w") as f:
            json.dump(
                {"axes": self.grid.names, "completed": sorted(done)}, f, indent=2
            )

    def branch_dir(self, point: BranchPoint) -> Path:
        return self.workdir / f"branch_{point.index:05d}_{point.key}"

    # ------------------------------------------------------------ execution
    def run_one(self, point: BranchPoint) -> XSLibrary:
        """Execute one branch point and return its single-state library."""
        from .mgxs import require_openmc

        require_openmc()

        directory = self.branch_dir(point)
        directory.mkdir(parents=True, exist_ok=True)
        model = self.model_factory(**point.state)
        statepoint_path = model.run(cwd=str(directory))

        import openmc

        with openmc.StatePoint(statepoint_path) as sp:
            library = self.library_factory(model, sp, **point.state)
        self._record(point.index)
        return library

    def run(
        self,
        *,
        processes: int = 1,
        resume: bool = True,
        points: Sequence[BranchPoint] | None = None,
    ) -> XSLibrary:
        """Run every branch point and assemble the branch-parameterised library.

        Parameters
        ----------
        processes : int
            Run this many branches concurrently with :mod:`multiprocessing`.
            Each OpenMC run is itself parallel, so oversubscribing hurts.
        resume : bool
            Skip branch points recorded in the checkpoint (FR-OMC-11).
        points : sequence of BranchPoint, optional
            Restrict to these points; the rest are read from checkpoints.

        Returns
        -------
        XSLibrary
            Branch-parameterised and finalized.
        """
        todo = list(points if points is not None else self.grid)
        if resume:
            done = self.completed()
            todo = [p for p in todo if p.index not in done]

        results: dict[int, XSLibrary] = {}
        if processes > 1 and todo:
            from multiprocessing import Pool

            with Pool(processes) as pool:
                for point, library in zip(
                    todo, pool.map(self.run_one, todo), strict=True
                ):
                    results[point.index] = library
        else:
            for point in todo:
                results[point.index] = self.run_one(point)

        return self.assemble(results)

    def assemble(self, results: Mapping[int, XSLibrary]) -> XSLibrary:
        """Stitch single-state libraries into one branch-parameterised library.

        Parameters
        ----------
        results : mapping of int to XSLibrary
            Keyed by branch index. Must cover every grid point.

        Raises
        ------
        InputError
            If any grid point is missing, or the single-state libraries
            disagree on group or composition count.
        """
        missing = set(range(len(self.grid))) - set(results)
        if missing:
            raise InputError(
                f"{len(missing)} of {len(self.grid)} branch points are missing "
                f"(first few: {sorted(missing)[:5]})"
            )

        first = results[0]
        out = XSLibrary(first.n_groups, first.n_compositions)
        out.set_axes(self.grid.to_library_axes())

        for index in range(len(self.grid)):
            single = results[index]
            if (
                single.n_groups != first.n_groups
                or single.n_compositions != first.n_compositions
            ):
                raise InputError(
                    f"branch {index} has "
                    f"{single.n_groups}x{single.n_compositions}, expected "
                    f"{first.n_groups}x{first.n_compositions}"
                )
            for c in range(first.n_compositions):
                source = single.composition(c)
                out.set_composition(
                    c,
                    state=index,
                    D=source.D,
                    absorption=source.absorption,
                    nu_fission=source.nu_fission,
                    kappa_fission=source.kappa_fission,
                    chi=source.chi,
                    inv_velocity=source.inv_velocity,
                    scatter=np.asarray(source.scatter).reshape(
                        first.n_groups, first.n_groups
                    ),
                )
        out.finalize()
        return out

    # ------------------------------------------------------------ job arrays
    def write_job_array(
        self,
        path,
        *,
        command: str = "python -m openndm.gc.driver",
        scheduler: str = "slurm",
        job_name: str = "openndm-branch",
        time_limit: str = "01:00:00",
        cpus: int = 8,
        extra_directives: Sequence[str] = (),
    ) -> Path:
        """Emit a scheduler job array covering the whole grid (FR-OMC-11).

        One array task per branch point, so a five-axis table is one
        submission rather than thousands. Combined with ``resume=True``, a
        partially completed array can simply be resubmitted.

        Parameters
        ----------
        scheduler : {'slurm', 'pbs'}
        """
        path = Path(path)
        n = len(self.grid)
        if scheduler == "slurm":
            lines = [
                "#!/bin/bash",
                f"#SBATCH --job-name={job_name}",
                f"#SBATCH --array=0-{n - 1}",
                f"#SBATCH --cpus-per-task={cpus}",
                f"#SBATCH --time={time_limit}",
                *(f"#SBATCH {d}" for d in extra_directives),
                "",
                f"export OMP_NUM_THREADS={cpus}",
                f"{command} --index $SLURM_ARRAY_TASK_ID "
                f"--workdir {self.workdir}",
            ]
        elif scheduler == "pbs":
            lines = [
                "#!/bin/bash",
                f"#PBS -N {job_name}",
                f"#PBS -J 0-{n - 1}",
                f"#PBS -l select=1:ncpus={cpus}",
                f"#PBS -l walltime={time_limit}",
                *(f"#PBS {d}" for d in extra_directives),
                "",
                f"export OMP_NUM_THREADS={cpus}",
                f"{command} --index $PBS_ARRAY_INDEX --workdir {self.workdir}",
            ]
        else:
            raise InputError(
                f"unknown scheduler {scheduler!r}; expected 'slurm' or 'pbs'"
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
        path.chmod(0o755)
        return path


def _product(axes: Mapping[str, Sequence[float]]):
    """Cartesian product helper kept for readability in tests."""
    names = list(axes)
    for combination in itertools.product(*(axes[n] for n in names)):
        yield dict(zip(names, combination, strict=True))
