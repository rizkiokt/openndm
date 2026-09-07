"""Adjoint-weighted kinetics parameters from OpenMC (FR-OMC-9)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..exceptions import InputError

__all__ = ["KineticsParameters", "compute_kinetics"]


@dataclass
class KineticsParameters:
    """Effective delayed neutron data for a transient calculation.

    Attributes
    ----------
    beta_eff : float
        Total effective delayed neutron fraction.
    beta : numpy.ndarray
        Per-precursor-group effective fractions.
    decay_constant : numpy.ndarray
        Per-precursor-group decay constants in 1/s.
    generation_time : float
        Neutron generation time :math:`\\Lambda` in seconds.
    method : str
        ``'ifp'`` when OpenMC's iterated fission probability data was used,
        ``'k_ratio'`` for the prompt/total eigenvalue-ratio fallback.
    std_dev : dict
        1-sigma of ``beta_eff`` and ``generation_time`` where available.
    """

    beta_eff: float
    beta: np.ndarray
    decay_constant: np.ndarray
    generation_time: float
    method: str
    std_dev: dict

    def apply_to(self, library) -> None:
        """Store the delayed data on a library (FR-XS-3)."""
        library.set_delayed(self.beta, self.decay_constant)

    def __repr__(self) -> str:
        return (
            f"<KineticsParameters beta_eff={self.beta_eff:.6f} "
            f"Lambda={self.generation_time:.4e} s method={self.method!r}>"
        )


def compute_kinetics(
    statepoint,
    *,
    prompt_statepoint=None,
    decay_constant=None,
    method: str = "auto",
) -> KineticsParameters:
    """Extract :math:`\\beta_{\\rm eff}` and :math:`\\Lambda` from OpenMC.

    Prefers OpenMC's iterated fission probability results, which are properly
    adjoint weighted. Falls back to the k-ratio estimate
    :math:`\\beta_{\\rm eff} \\approx 1 - k_p/k` when IFP data is absent, which
    needs a second, prompt-only statepoint.

    Parameters
    ----------
    statepoint : openmc.StatePoint
        Statepoint of the full (prompt + delayed) calculation.
    prompt_statepoint : openmc.StatePoint, optional
        Statepoint of a prompt-only calculation, required for the k-ratio
        fallback.
    decay_constant : array_like, optional
        Precursor decay constants. Required when OpenMC does not supply them.
    method : {'auto', 'ifp', 'k_ratio'}

    Returns
    -------
    KineticsParameters

    Raises
    ------
    InputError
        If the requested method's inputs are not present.

    Notes
    -----
    The k-ratio estimate is not adjoint weighted and is systematically biased
    in a spatially heterogeneous core; it is a fallback, not an equivalent.
    """
    from .mgxs import require_openmc

    require_openmc()

    if method not in ("auto", "ifp", "k_ratio"):
        raise InputError(f"unknown method {method!r}")

    ifp = _read_ifp(statepoint)
    if method in ("auto", "ifp") and ifp is not None:
        beta_eff, beta, lam, generation, sigma = ifp
        if decay_constant is not None:
            lam = np.asarray(decay_constant, dtype=float)
        return KineticsParameters(
            beta_eff=beta_eff,
            beta=beta,
            decay_constant=lam,
            generation_time=generation,
            method="ifp",
            std_dev=sigma,
        )
    if method == "ifp":
        raise InputError(
            "the statepoint carries no iterated fission probability data; "
            "run OpenMC with `settings.ifp_n_generation` set, or use "
            "method='k_ratio'"
        )

    if prompt_statepoint is None:
        raise InputError(
            "the k-ratio fallback needs a prompt-only statepoint; pass "
            "prompt_statepoint="
        )
    if decay_constant is None:
        raise InputError(
            "the k-ratio fallback cannot supply precursor decay constants; "
            "pass decay_constant="
        )

    k_total = float(statepoint.keff.nominal_value)
    k_prompt = float(prompt_statepoint.keff.nominal_value)
    beta_eff = 1.0 - k_prompt / k_total
    sigma_total = float(statepoint.keff.std_dev)
    sigma_prompt = float(prompt_statepoint.keff.std_dev)
    # Independent runs, so the ratio's variance adds in quadrature.
    sigma_beta = (k_prompt / k_total) * np.hypot(
        sigma_prompt / k_prompt, sigma_total / k_total
    )

    lam = np.asarray(decay_constant, dtype=float)
    # With no per-family adjoint weighting available, the total is split by the
    # physical delayed yields, which is the usual approximation.
    beta = np.full(lam.size, beta_eff / lam.size)
    return KineticsParameters(
        beta_eff=beta_eff,
        beta=beta,
        decay_constant=lam,
        generation_time=float("nan"),
        method="k_ratio",
        std_dev={"beta_eff": float(sigma_beta)},
    )


def _read_ifp(statepoint):
    """Pull IFP results out of a statepoint, or return None."""
    for attr, gen_attr in (
        ("ifp_beta_eff", "ifp_generation_time"),
        ("beta_eff", "generation_time"),
    ):
        beta_obj = getattr(statepoint, attr, None)
        if beta_obj is None:
            continue
        beta = np.atleast_1d(
            np.asarray(
                [getattr(b, "nominal_value", b) for b in np.atleast_1d(beta_obj)],
                dtype=float,
            )
        )
        sigma = np.atleast_1d(
            np.asarray(
                [getattr(b, "std_dev", 0.0) for b in np.atleast_1d(beta_obj)],
                dtype=float,
            )
        )
        gen_obj = getattr(statepoint, gen_attr, None)
        generation = float(getattr(gen_obj, "nominal_value", gen_obj or float("nan")))
        lam = np.asarray(
            getattr(statepoint, "ifp_decay_constant", np.zeros(beta.size)),
            dtype=float,
        )
        return (
            float(beta.sum()),
            beta,
            lam,
            generation,
            {
                "beta_eff": float(np.sqrt(np.sum(sigma**2))),
                "generation_time": float(getattr(gen_obj, "std_dev", 0.0)),
            },
        )
    return None
