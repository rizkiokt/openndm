"""Ingestion of OpenMC multi-group cross section data (FR-OMC-1..6).

The single rule this module exists to enforce: a user holding an
``openmc.mgxs.Library`` writes no translation code.

.. rubric:: Group ordering

``openmc.mgxs`` numbers groups from 1 at the highest energy, and
``MGXS.get_xs(groups='all')`` returns arrays in that order. OpenNDM uses the
same convention, so no reordering happens by default. Pass
``reverse_groups=True`` if a source ever hands over increasing-energy data.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

from ..exceptions import InputError
from ..xslib import XSLibrary

__all__ = [
    "DEFAULT_D_TOLERANCE",
    "from_mgxs_file",
    "from_mgxs_library",
    "from_statepoint",
    "require_openmc",
]

#: Relative disagreement between the two diffusion coefficient sources above
#: which a warning is issued (FR-OMC-5).
DEFAULT_D_TOLERANCE = 0.05

#: MGXS type names accepted for each OpenNDM field, in preference order.
_FIELD_TYPES = {
    "absorption": ("absorption",),
    "nu_fission": ("nu-fission",),
    "kappa_fission": ("kappa-fission",),
    "chi": ("chi",),
    "inv_velocity": ("inverse-velocity",),
}
_SCATTER_TYPES = (
    "consistent nu-scatter matrix",
    "nu-scatter matrix",
    "consistent scatter matrix",
    "scatter matrix",
)
#: Scattering matrix scores paired as (with multiplicity, without). The
#: difference of their row sums is the (n,xn) multiplicity excess.
_SCATTER_PAIRS = (
    ("consistent nu-scatter matrix", "consistent scatter matrix"),
    ("nu-scatter matrix", "scatter matrix"),
)
_DIFFUSION_TYPES = ("diffusion-coefficient",)
_TRANSPORT_TYPES = ("transport", "nu-transport")


def require_openmc():
    """Import and return :mod:`openmc`, with an actionable error if missing.

    OpenNDM's solver runs without OpenMC; only this subpackage needs it
    (FR-OMC-14).
    """
    try:
        import openmc
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "openndm.gc needs OpenMC. Install it with "
            "`pip install openndm[openmc]` or `conda install -c conda-forge openmc`."
        ) from exc
    return openmc


def _subdomain_id(domain):
    """Subdomain selector for ``MGXS.get_xs``.

    ``Library.get_mgxs`` takes the domain object, but ``MGXS.get_xs`` takes the
    integer domain *id*; passing the object raises a TypeError deep inside
    OpenMC's argument checking. Mesh domains are selected by element index
    instead, and are handled by the caller.
    """
    return getattr(domain, "id", domain)


def _mean_and_std(mgxs, domain, *, matrix: bool = False, subdomain=None):
    """Extract mean and 1-sigma from one MGXS object (FR-OMC-6)."""
    if subdomain is not None:
        selector = [subdomain]
    elif domain is not None:
        selector = [_subdomain_id(domain)]
    else:
        selector = "all"
    kwargs = {
        "subdomains": selector,
        "nuclides": "sum",
        "xs_type": "macro",
    }
    mean = np.atleast_1d(np.asarray(mgxs.get_xs(value="mean", **kwargs)).squeeze())
    try:
        std = np.atleast_1d(
            np.asarray(mgxs.get_xs(value="std_dev", **kwargs)).squeeze()
        )
    except Exception:
        std = None
    if matrix:
        n = int(np.sqrt(mean.size) + 0.5)
        mean = mean.reshape(n, n)
        if std is not None:
            std = std.reshape(n, n)
    return mean, std


def _first_available(library, domain, names):
    """Return the first MGXS in ``names`` that ``library`` actually holds."""
    available = set(getattr(library, "mgxs_types", ()))
    for name in names:
        if name not in available:
            continue
        try:
            return name, library.get_mgxs(domain, name)
        except (KeyError, ValueError):
            continue
    return None, None


def from_mgxs_library(
    library,
    *,
    domains: Sequence | None = None,
    reverse_groups: bool = False,
    d_tolerance: float = DEFAULT_D_TOLERANCE,
    prefer: str = "diffusion-coefficient",
    scattering_multiplicity: str = "correct",
) -> XSLibrary:
    r"""Build an OpenNDM library from an in-memory ``openmc.mgxs.Library``.

    Accepts every domain type OpenMC supports: ``material``, ``cell``,
    ``universe`` and ``mesh`` (FR-OMC-2). For a mesh domain the mesh elements
    become OpenNDM compositions in the mesh's own element order, so the
    resulting library maps one-to-one onto a node grid of the same shape.

    Parameters
    ----------
    library : openmc.mgxs.Library
        A library whose ``build_library`` and ``load_from_statepoint`` have
        already run.
    domains : sequence, optional
        Restrict and order the domains; defaults to ``library.domains``. The
        resulting composition index is the position in this sequence.
    reverse_groups : bool
        Reverse the group order on ingestion. Leave this alone unless you know
        the source uses increasing-energy ordering.
    d_tolerance : float
        Relative disagreement between ``diffusion-coefficient`` and
        ``transport`` above which a warning is raised (FR-OMC-5).
    prefer : {'diffusion-coefficient', 'transport'}
        Which source of D wins when both are present.
    scattering_multiplicity : {'correct', 'ignore'}
        How to handle neutron multiplication in scattering, that is (n,2n) and
        (n,3n). See the note below; leave this alone unless you are comparing
        against a code that makes the other choice.

    Returns
    -------
    XSLibrary
        Finalized, single branch state.

    Raises
    ------
    InputError
        If the library carries neither a diffusion coefficient nor a transport
        cross section, or lacks absorption or scattering data.

    Notes
    -----
    Standard deviations are carried through wherever OpenMC provides them
    (FR-XS-9, FR-OMC-6). Monte Carlo noise routinely produces small negative
    scattering transfers; those survive ingestion and are reported as warnings
    by :meth:`~openndm.XSLibrary.finalize`, not silently zeroed.

    .. rubric:: Scattering multiplicity

    OpenMC's ``absorption`` score does not include (n,2n) or (n,3n); those
    reactions destroy one neutron and create several, and the extra neutrons
    appear only in the ``nu-scatter matrix``, as row sums exceeding those of
    the plain ``scatter matrix``.

    A diffusion operator built with a single scattering matrix cannot see that
    production. Summed over groups, the in-scatter and out-scatter terms are
    the same double sum and cancel exactly, leaving
    :math:`k = \sum\nu\Sigma_f\phi / \sum\Sigma_a\phi` with the extra
    neutrons nowhere in it. On a homogeneous UO2 and water mixture that is
    worth 230 pcm, and the only symptom is the eigenvalue.

    The fix is exact rather than approximate. Writing the group balance with
    multiplicity and moving the self-scatter term across gives a removal cross
    section :math:`\Sigma_a + \Sigma_s^{\rm tot} - \nu\Sigma_{s,g\to g}`,
    while this implementation forms
    :math:`\Sigma_a + \nu\Sigma_s^{\rm tot} - \nu\Sigma_{s,g\to g}`. The
    two differ by exactly the multiplicity excess
    :math:`\nu\Sigma_s^{\rm tot} - \Sigma_s^{\rm tot}`, so subtracting that
    excess from the absorption cross section reproduces the correct operator
    group by group, not merely in the global balance.

    Doing so needs both the multiplied and the plain scattering matrix in the
    library. When only one is present the correction is impossible and a
    warning says so, because the resulting eigenvalue is biased low by roughly
    the (n,xn) production rate.
    """
    if prefer not in ("diffusion-coefficient", "transport"):
        raise InputError(
            f"prefer must be 'diffusion-coefficient' or 'transport', "
            f"got {prefer!r}"
        )

    domain_list = list(domains if domains is not None else library.domains)
    if not domain_list:
        raise InputError("the MGXS library has no domains")

    n_groups = int(library.energy_groups.num_groups)
    out = XSLibrary(n_groups, len(domain_list))

    def order(a):
        return a[::-1] if reverse_groups else a

    for index, domain in enumerate(domain_list):
        fields = {}
        std = {}
        for field, names in _FIELD_TYPES.items():
            _, mgxs = _first_available(library, domain, names)
            if mgxs is None:
                continue
            mean, sigma = _mean_and_std(mgxs, domain)
            fields[field] = order(mean)
            if sigma is not None and field in ("absorption", "nu_fission"):
                std[field] = order(sigma)

        scatter_name, scatter_mgxs = _first_available(
            library, domain, _SCATTER_TYPES
        )
        if scatter_mgxs is None:
            raise InputError(
                f"domain {domain!r} has no scattering matrix; add one of "
                f"{list(_SCATTER_TYPES)} to the MGXS library"
            )
        scatter, _ = _mean_and_std(scatter_mgxs, domain, matrix=True)
        if reverse_groups:
            scatter = scatter[::-1, ::-1]

        excess = _multiplicity_excess(
            library, domain, scatter_name, order, scattering_multiplicity, index
        )

        D = _diffusion_coefficient(
            library, domain, order, d_tolerance, prefer, index
        )

        if "absorption" not in fields:
            raise InputError(
                f"domain {domain!r} has no absorption cross section"
            )
        # OpenMC reports absorption including fission, which is what the
        # diffusion balance wants, so it is used unchanged.
        out.set_composition(
            index,
            D=D,
            absorption=fields["absorption"] - excess,
            nu_fission=fields.get("nu_fission"),
            kappa_fission=fields.get("kappa_fission"),
            chi=fields.get("chi"),
            inv_velocity=fields.get("inv_velocity"),
            scatter=scatter,
            std=std or None,
        )

    out.finalize()
    return out


def _multiplicity_excess(
    library, domain, scatter_name, order, policy, index
):
    """Per-group (n,xn) neutron production hidden in a nu-scatter matrix.

    Returns an array to subtract from the absorption cross section, or zeros
    when no correction is called for.
    """
    if policy not in ("correct", "ignore"):
        raise InputError(
            f"scattering_multiplicity must be 'correct' or 'ignore', "
            f"got {policy!r}"
        )
    n_groups = int(library.energy_groups.num_groups)
    zero = np.zeros(n_groups)
    if policy == "ignore":
        return zero

    pair = next(
        (p for p in _SCATTER_PAIRS if scatter_name in p), None
    )
    if pair is None:
        return zero
    multiplied, plain = pair

    available = set(getattr(library, "mgxs_types", ()))
    if multiplied not in available or plain not in available:
        # Only warn when the library actually carries multiplication. A plain
        # matrix on its own simply has no (n,xn) neutrons to account for, but
        # it also silently loses that production.
        warnings.warn(
            f"composition {index}: cannot correct for scattering multiplicity "
            f"because the library has only one of {multiplied!r} and "
            f"{plain!r}. Any (n,2n) or (n,3n) production is missing from the "
            f"neutron balance, which biases k_eff low by roughly the (n,xn) "
            f"reaction rate. Add both scores, or pass "
            f"scattering_multiplicity='ignore' to silence this.",
            stacklevel=3,
        )
        return zero

    try:
        nu_matrix, _ = _mean_and_std(
            library.get_mgxs(domain, multiplied), domain, matrix=True
        )
        plain_matrix, _ = _mean_and_std(
            library.get_mgxs(domain, plain), domain, matrix=True
        )
    except (KeyError, ValueError):
        return zero

    return order(np.asarray(nu_matrix).sum(axis=1)
                 - np.asarray(plain_matrix).sum(axis=1))


def _diffusion_coefficient(library, domain, order, tolerance, prefer, index):
    """Resolve D from the available MGXS scores (FR-OMC-5)."""
    name_d, mgxs_d = _first_available(library, domain, _DIFFUSION_TYPES)
    name_t, mgxs_t = _first_available(library, domain, _TRANSPORT_TYPES)

    D_direct = None
    D_transport = None
    if mgxs_d is not None:
        D_direct = order(_mean_and_std(mgxs_d, domain)[0])
    if mgxs_t is not None:
        transport = order(_mean_and_std(mgxs_t, domain)[0])
        if np.any(transport <= 0.0):
            warnings.warn(
                f"composition {index}: non-positive transport cross section "
                f"from {name_t!r}; that group's D is taken from the other source",
                stacklevel=3,
            )
            transport = np.where(transport > 0.0, transport, np.nan)
        D_transport = 1.0 / (3.0 * transport)

    if D_direct is None and D_transport is None:
        raise InputError(
            f"domain {domain!r} carries neither a 'diffusion-coefficient' nor "
            f"a 'transport' score, so D cannot be formed. Add one to the MGXS "
            f"library."
        )
    if D_direct is not None and D_transport is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            disagreement = np.abs(D_direct - D_transport) / np.abs(D_direct)
        worst = np.nanmax(disagreement) if disagreement.size else 0.0
        if worst > tolerance:
            warnings.warn(
                f"composition {index}: {name_d!r} and {name_t!r} disagree by "
                f"{100 * worst:.1f}% (tolerance {100 * tolerance:.1f}%). Using "
                f"{prefer!r}; the two estimators differ most in strongly "
                f"heterogeneous nodes.",
                stacklevel=3,
            )
    chosen = (
        D_direct
        if (prefer == "diffusion-coefficient" and D_direct is not None)
        else D_transport
    )
    if chosen is None:
        chosen = D_direct if D_direct is not None else D_transport
    other = D_transport if chosen is D_direct else D_direct
    if other is not None:
        chosen = np.where(np.isfinite(chosen), chosen, other)
    return chosen


def from_mgxs_file(path, *, reverse_groups: bool = False) -> XSLibrary:
    """Read an ``mgxs.h5`` written by ``Library.create_mg_mode`` (FR-OMC-3).

    This is the format OpenMC itself consumes in multi-group mode, so a
    library already prepared for an OpenMC MG run works unchanged.

    Parameters
    ----------
    path : path-like
        Path to the ``mgxs.h5`` file.

    Returns
    -------
    XSLibrary

    Notes
    -----
    The file stores a transport-corrected total cross section rather than a
    diffusion coefficient, so ``D = 1 / (3 * Sigma_tr)`` throughout. The file
    format carries no uncertainties.
    """
    openmc = require_openmc()

    mg = openmc.MGXSLibrary.from_hdf5(str(path))
    names = list(mg.names) if hasattr(mg, "names") else [
        x.name for x in mg.xsdatas
    ]
    n_groups = int(mg.energy_groups.num_groups)
    out = XSLibrary(n_groups, len(names))

    for index, xsdata in enumerate(mg.xsdatas):
        # xsdata is bound as a default so the closure cannot capture the loop
        # variable by reference.
        def pick(attr, default=None, xsdata=xsdata):
            value = getattr(xsdata, attr, None)
            if value is None:
                return default
            arr = np.asarray(value, dtype=float).squeeze()
            return arr[::-1] if reverse_groups else arr

        total = pick("total")
        absorption = pick("absorption")
        scatter = getattr(xsdata, "scatter_matrix", None)
        if scatter is None:
            raise InputError(f"{path}: xsdata {names[index]!r} has no scatter matrix")
        scatter = np.asarray(scatter, dtype=float).squeeze()
        # The MGXS file keeps a Legendre axis; only P0 feeds diffusion.
        while scatter.ndim > 2:
            scatter = scatter[..., 0] if scatter.shape[-1] <= 5 else scatter[0]
        if reverse_groups:
            scatter = scatter[::-1, ::-1]

        if total is None:
            raise InputError(f"{path}: xsdata {names[index]!r} has no total XS")
        if absorption is None:
            absorption = total - scatter.sum(axis=1)

        out.set_composition(
            index,
            D=1.0 / (3.0 * total),
            absorption=absorption,
            nu_fission=pick("nu_fission"),
            kappa_fission=pick("kappa_fission"),
            chi=pick("chi"),
            inv_velocity=pick("inverse_velocity"),
            scatter=scatter,
        )
    out.finalize()
    return out


def from_statepoint(
    statepoint,
    mgxs_library,
    *,
    domains: Sequence | None = None,
    **kwargs,
) -> XSLibrary:
    """Build a library straight from an ``openmc.StatePoint`` (FR-OMC-4).

    Parameters
    ----------
    statepoint : openmc.StatePoint
        The statepoint holding the tallies.
    mgxs_library : openmc.mgxs.Library
        The library object describing the tally specification that was used.
        It is loaded from ``statepoint`` in place.
    **kwargs
        Forwarded to :func:`from_mgxs_library`.

    Returns
    -------
    XSLibrary
    """
    require_openmc()
    mgxs_library.load_from_statepoint(statepoint)
    return from_mgxs_library(mgxs_library, domains=domains, **kwargs)
