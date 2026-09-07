"""Pin form functions for power reconstruction (FR-OMC-10)."""

from __future__ import annotations

import numpy as np

from ..exceptions import InputError

__all__ = ["compute_form_functions"]


def compute_form_functions(
    statepoint,
    tally_name: str = "openndm_pin_fission",
    *,
    shape: tuple[int, int] | None = None,
    normalise: bool = True,
) -> np.ndarray:
    """Extract normalised pin-wise fission rate distributions.

    The form function is the pin fission rate divided by the assembly-average
    pin fission rate, so it multiplies a reconstructed nodal power directly.

    Parameters
    ----------
    statepoint : openmc.StatePoint
        Statepoint holding a mesh-filtered ``fission`` or ``kappa-fission``
        tally over the pin lattice.
    tally_name : str
        Name of that tally.
    shape : (ny, nx), optional
        Reshape the flat mesh result. Inferred from the mesh filter when
        omitted.
    normalise : bool
        Divide by the mean over pins with non-zero fission rate.

    Returns
    -------
    numpy.ndarray
        Form functions with the pin lattice's shape. Non-fuel positions are 0.

    Raises
    ------
    InputError
        If the named tally is absent or its shape cannot be resolved.
    """
    from .mgxs import require_openmc

    require_openmc()

    try:
        tally = statepoint.get_tally(name=tally_name)
    except LookupError as exc:
        raise InputError(
            f"statepoint has no tally named {tally_name!r}"
        ) from exc

    values = np.asarray(tally.mean).squeeze()
    if shape is None:
        for f in tally.filters:
            dimension = getattr(getattr(f, "mesh", None), "dimension", None)
            if dimension is not None:
                dims = [int(d) for d in dimension if int(d) > 1]
                if len(dims) == 2:
                    shape = (dims[1], dims[0])
                break
    if shape is None:
        raise InputError(
            f"cannot infer the pin lattice shape for {tally_name!r}; pass shape="
        )
    values = values.reshape(shape)

    if normalise:
        active = values > 0.0
        if not active.any():
            raise InputError(f"tally {tally_name!r} has no non-zero fission rate")
        values = values / values[active].mean()
    return values
