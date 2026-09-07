"""Critical spectrum and buckling search (FR-OMC-8).

An infinite-lattice OpenMC calculation produces group constants for a medium
with no net leakage. A core calculation needs constants generated at, or
corrected toward, the critical spectrum. This module performs the buckling
search and reports which convention it used, because B1 and P1 give
measurably different answers and cross-code comparisons go wrong silently
when the convention is not recorded (see the risk table in the specification).

.. rubric:: Equations implemented

For a homogeneous medium with buckling :math:`B^2`, the group balance is

.. math::

    \\left(\\Sigma_{t,g} + D_g(B^2)\\,B^2\\right)\\phi_g
        - \\sum_{g'}\\Sigma_{s0,g'\\to g}\\phi_{g'}
        = \\frac{\\chi_g}{k}\\sum_{g'}\\nu\\Sigma_{f,g'}\\phi_{g'} .

Because the fission source is rank one, :math:`k(B^2)` is available in closed
form as :math:`k = \\nu\\Sigma_f^{\\mathsf T} M(B^2)^{-1}\\chi`, and the search
reduces to a scalar root find on :math:`B^2`.

The two conventions differ only in :math:`D_g`:

``p1``
    :math:`D_g = 1 / (3\\Sigma_{tr,g})`, independent of buckling.
``b1``
    :math:`D_g = \\gamma_g / (3\\Sigma_{tr,g})` with the standard correction
    factor :math:`\\gamma_g = x_g / \\left(3(1/\\alpha_g - 1)\\right)`,
    :math:`x_g = B^2/\\Sigma_{t,g}^2`, and
    :math:`\\alpha_g = \\arctan(\\sqrt{x_g})/\\sqrt{x_g}` for
    :math:`B^2 > 0` (the ``artanh`` branch for :math:`B^2 < 0`).
    :math:`\\gamma \\to 1` as :math:`B \\to 0`, so B1 reduces to P1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..exceptions import ConvergenceError, InputError

__all__ = ["CriticalSpectrum", "b1_gamma", "critical_spectrum"]


def b1_gamma(buckling: float, total: np.ndarray) -> np.ndarray:
    """B1 leakage correction factor :math:`\\gamma_g`.

    Returns an array of ones when ``buckling`` is zero, which is exactly the
    P1 limit.
    """
    total = np.asarray(total, dtype=float)
    x = buckling / np.square(total)
    gamma = np.ones_like(x)
    # Series expansion near zero: alpha = 1 - x/3 + x^2/5, so gamma -> 1.
    small = np.abs(x) < 1.0e-8
    with np.errstate(invalid="ignore", divide="ignore"):
        positive = (~small) & (x > 0)
        if np.any(positive):
            r = np.sqrt(x[positive])
            alpha = np.arctan(r) / r
            gamma[positive] = x[positive] / (3.0 * (1.0 / alpha - 1.0))
        negative = (~small) & (x < 0)
        if np.any(negative):
            r = np.sqrt(-x[negative])
            # artanh diverges at r = 1; beyond that the B1 form has no real
            # solution and the P1 value is the defensible fallback.
            usable = r < 0.999999
            alpha = np.ones_like(r)
            alpha[usable] = np.arctanh(r[usable]) / r[usable]
            g = np.ones_like(r)
            g[usable] = x[negative][usable] / (3.0 * (1.0 / alpha[usable] - 1.0))
            gamma[negative] = g
    return gamma


@dataclass
class CriticalSpectrum:
    """Outcome of a buckling search.

    Attributes
    ----------
    buckling : float
        Critical geometric buckling :math:`B^2` in cm^-2. Positive for a
        leaking (supercritical-infinite-medium) system.
    k_infinity : float
        Eigenvalue at zero buckling.
    spectrum : numpy.ndarray
        Critical flux spectrum, normalised to sum to one.
    diffusion : numpy.ndarray
        Leakage-corrected diffusion coefficient per group.
    method : str
        ``'b1'`` or ``'p1'``. Recorded so a downstream comparison can state
        which convention produced the constants.
    iterations : int
    """

    buckling: float
    k_infinity: float
    spectrum: np.ndarray
    diffusion: np.ndarray
    method: str
    iterations: int = 0
    metadata: dict = field(default_factory=dict)

    def apply_to(self, library, composition: int) -> None:
        """Replace one composition's D with the leakage-corrected value.

        The group constants themselves are not re-collapsed: doing that
        correctly needs the fine-group data that lives inside the lattice
        calculation, not the already-condensed coarse-group library. What this
        does give you is the buckling and the corrected D, which is the part
        the nodal solve is most sensitive to.
        """
        library.set_composition(composition, D=self.diffusion)

    def __repr__(self) -> str:
        return (
            f"<CriticalSpectrum method={self.method!r} "
            f"B2={self.buckling:.6e} cm^-2 k_inf={self.k_infinity:.6f}>"
        )


def _k_of_buckling(buckling, total, scatter, nu_fission, chi, transport, method):
    """Eigenvalue of the homogeneous medium at a given buckling.

    Returns ``(k, phi, D)`` with ``k = None`` when the balance operator has
    stopped being physical. That happens at sufficiently negative buckling,
    where the leakage term cancels removal and the flux solution changes sign;
    the bracketing search uses it to stay inside the physical region.
    """
    if method == "b1":
        D = b1_gamma(buckling, total) / (3.0 * transport)
    else:
        D = 1.0 / (3.0 * transport)
    # M[g, g'] multiplies phi_{g'} in the balance of group g.
    M = np.diag(total + D * buckling) - scatter.T
    if np.any(np.diag(M) <= 0.0):
        return None, None, D
    try:
        phi = np.linalg.solve(M, chi)
    except np.linalg.LinAlgError:
        return None, None, D
    if np.any(phi < 0.0):
        return None, None, D
    return float(nu_fission @ phi), phi, D


def _bracket(residual, f_zero, width, tolerance):
    """Bracket the critical buckling, staying inside the physical region.

    ``residual`` decreases monotonically with buckling: more leakage means a
    smaller eigenvalue. So the root sits on the positive side when the medium
    is supercritical at zero buckling and on the negative side otherwise. On
    the negative side the physical region is bounded, so the search first
    walks *inward* from the trial point until the residual is defined, and
    only then expands outward.
    """
    if abs(f_zero) < tolerance:
        return 0.0, f_zero, 0.0, f_zero, 0

    steps = 0
    if f_zero > 0.0:
        lo, f_lo = 0.0, f_zero
        hi = width
        while steps < 60:
            f_hi = residual(hi)
            if f_hi is None:
                hi *= 0.5
            elif f_hi < 0.0:
                return lo, f_lo, hi, f_hi, steps
            else:
                lo, f_lo, hi = hi, f_hi, hi * 2.0
            steps += 1
        return lo, f_lo, hi, f_lo, steps

    hi, f_hi = 0.0, f_zero
    lo = -width
    # Walk inward until the residual exists at all.
    while steps < 60:
        f_lo = residual(lo)
        if f_lo is not None:
            break
        lo *= 0.5
        steps += 1
    else:
        return lo, f_hi, hi, f_hi, steps
    # Then outward until it changes sign, without leaving the physical region.
    while steps < 60:
        if f_lo > 0.0:
            return lo, f_lo, hi, f_hi, steps
        hi, f_hi = lo, f_lo
        candidate = lo * 2.0
        f_candidate = residual(candidate)
        if f_candidate is None:
            # The pole lies between lo and candidate; close in on it instead.
            candidate = 0.5 * (lo + candidate)
            f_candidate = residual(candidate)
            if f_candidate is None:
                return lo, f_lo, hi, f_hi, steps
        lo, f_lo = candidate, f_candidate
        steps += 1
    return lo, f_lo, hi, f_hi, steps


def critical_spectrum(
    total,
    scatter,
    nu_fission,
    chi,
    *,
    transport=None,
    method: str = "b1",
    target_k: float = 1.0,
    bracket: tuple[float, float] = (-0.05, 0.05),
    tolerance: float = 1.0e-10,
    max_iterations: int = 100,
) -> CriticalSpectrum:
    """Search for the buckling that makes a homogeneous medium critical.

    Parameters
    ----------
    total : array_like, shape (G,)
        Total cross section per group.
    scatter : array_like, shape (G, G)
        Scattering matrix indexed ``[from_group, to_group]``.
    nu_fission, chi : array_like, shape (G,)
    transport : array_like, shape (G,), optional
        Transport cross section. Defaults to ``total`` minus the P1
        within-group correction, which without P1 data is just ``total``.
    method : {'b1', 'p1'}
        Leakage convention. Recorded on the result.
    bracket : (float, float)
        Initial search bracket on :math:`B^2`, expanded automatically if the
        root lies outside it.

    Returns
    -------
    CriticalSpectrum

    Raises
    ------
    ConvergenceError
        If no sign change can be bracketed, or bisection fails to converge.

    Examples
    --------
    >>> import numpy as np
    >>> total = np.array([0.2, 0.9])
    >>> scatter = np.array([[0.17, 0.02], [0.0, 0.82]])
    >>> result = critical_spectrum(
    ...     total, scatter, np.array([0.0, 0.135]), np.array([1.0, 0.0]))
    >>> result.buckling > 0
    True
    """
    method = method.lower()
    if method not in ("b1", "p1"):
        raise InputError(f"method must be 'b1' or 'p1', got {method!r}")

    total = np.asarray(total, dtype=float)
    scatter = np.asarray(scatter, dtype=float)
    nu_fission = np.asarray(nu_fission, dtype=float)
    chi = np.asarray(chi, dtype=float)
    G = total.size
    if scatter.shape != (G, G):
        raise InputError(
            f"scatter must have shape ({G}, {G}), got {scatter.shape}"
        )
    transport = total if transport is None else np.asarray(transport, dtype=float)
    if np.any(transport <= 0.0):
        raise InputError("transport cross section must be positive")

    def residual(b2):
        k, _, _ = _k_of_buckling(
            b2, total, scatter, nu_fission, chi, transport, method
        )
        return None if k is None else k - target_k

    k_inf, _, _ = _k_of_buckling(
        0.0, total, scatter, nu_fission, chi, transport, method
    )
    if k_inf is None:
        raise InputError(
            "the group balance is already singular at zero buckling; check the "
            "total and scattering cross sections"
        )

    if abs(k_inf - target_k) < tolerance:
        # Already critical at infinite dilution: zero buckling is the answer.
        _, phi, D = _k_of_buckling(
            0.0, total, scatter, nu_fission, chi, transport, method
        )
        return CriticalSpectrum(
            buckling=0.0,
            k_infinity=k_inf,
            spectrum=phi / phi.sum() if phi.sum() else phi,
            diffusion=D,
            method=method,
            iterations=0,
            metadata={"target_k": target_k},
        )

    width = max(abs(bracket[0]), abs(bracket[1])) or 0.05
    lo, f_lo, hi, f_hi, expansions = _bracket(
        residual, k_inf - target_k, width, tolerance
    )
    if f_lo is None or f_hi is None or f_lo * f_hi > 0.0:
        raise ConvergenceError(
            f"no critical buckling bracketed in [{lo:.3e}, {hi:.3e}]; "
            f"k_inf = {k_inf:.6f} may be too far from {target_k}",
            iterations=expansions,
            residual=min(abs(f) for f in (f_lo, f_hi) if f is not None),
        )

    iterations = 0
    for step in range(1, max_iterations + 1):
        iterations = step
        mid = 0.5 * (lo + hi)
        f_mid = residual(mid)
        if f_mid is None:
            # Unphysical midpoint: the pole is below it, so keep the upper half.
            lo = mid
            continue
        if abs(f_mid) < tolerance or abs(hi - lo) < 1.0e-14:
            break
        if f_lo * f_mid <= 0.0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    else:
        raise ConvergenceError(
            "buckling search did not converge",
            iterations=max_iterations,
            residual=abs(f_mid),
        )

    b2 = 0.5 * (lo + hi)
    _, phi, D = _k_of_buckling(
        b2, total, scatter, nu_fission, chi, transport, method
    )
    total_flux = phi.sum()
    return CriticalSpectrum(
        buckling=b2,
        k_infinity=k_inf,
        spectrum=phi / total_flux if total_flux else phi,
        diffusion=D,
        method=method,
        iterations=iterations,
        metadata={"target_k": target_k},
    )
