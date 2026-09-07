#include "openndm/cmfd.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "openndm/error.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace openndm {

CmfdSystem::CmfdSystem(const Geometry& geom, const XSLibrary& xs)
    : geom_(geom),
      xs_(xs),
      n_groups_(xs.n_groups())
{
  if (!xs.finalized()) {
    throw InputError("cross section library must be finalized before use");
  }
  const int n = geom_.n_nodes();
  const int G = n_groups_;
  removal_.assign(static_cast<std::size_t>(n) * G, 0.0);
  nu_fission_.assign(static_cast<std::size_t>(n) * G, 0.0);
  chi_.assign(static_cast<std::size_t>(n) * G, 0.0);
  diffusion_.assign(static_cast<std::size_t>(n) * G, 0.0);
  scatter_.assign(static_cast<std::size_t>(n) * G * G, 0.0);
  dtilde_.assign(static_cast<std::size_t>(geom_.n_surfaces()) * G, 0.0);
  dhat_.assign(static_cast<std::size_t>(geom_.n_surfaces()) * G, 0.0);

  build_pattern();
  matrices_.reserve(static_cast<std::size_t>(G));
  precond_.resize(static_cast<std::size_t>(G));
  for (int g = 0; g < G; ++g) matrices_.emplace_back(&pattern_);

  refresh_cross_sections();
  build_coupling();
}

void CmfdSystem::build_pattern()
{
  const int n = geom_.n_nodes();
  std::vector<std::vector<int>> neighbours(static_cast<std::size_t>(n));
  for (const auto& s : geom_.surfaces()) {
    if (s.bc != BoundaryType::interior) continue;
    neighbours[static_cast<std::size_t>(s.lo)].push_back(s.hi);
    neighbours[static_cast<std::size_t>(s.hi)].push_back(s.lo);
  }
  pattern_ = SparsePattern(n, neighbours);
}

void CmfdSystem::refresh_cross_sections()
{
  const int G = n_groups_;
  const auto& nodes = geom_.nodes();
  has_upscatter_ = false;
  // Fraction of the removal term the shift is never allowed to consume.
  constexpr double SHIFT_MARGIN = 0.25;
  double limit = std::numeric_limits<double>::infinity();
  for (int i = 0; i < geom_.n_nodes(); ++i) {
    const Node& node = nodes[static_cast<std::size_t>(i)];
    const Composition& c = xs_.composition(node.composition);
    const double V = node.volume;
    for (int g = 0; g < G; ++g) {
      const std::size_t idx = static_cast<std::size_t>(i) * G + g;
      const std::size_t gg = static_cast<std::size_t>(g);
      diffusion_[idx] = c.D[gg];
      removal_[idx] = c.removal[gg] * V;
      // The adjoint operator is the transpose of the forward one, so the
      // emission spectrum and the production cross section exchange roles and
      // the scattering matrix is transposed. Everything downstream, including
      // the power iteration, is then identical to the forward case.
      if (adjoint_) {
        nu_fission_[idx] = c.chi[gg] * V;
        chi_[idx] = c.nu_fission[gg];
      } else {
        nu_fission_[idx] = c.nu_fission[gg] * V;
        chi_[idx] = c.chi[gg];
      }
      const double production = chi_[idx] * nu_fission_[idx];
      if (production > 0.0) {
        limit =
            std::min(limit, (1.0 - SHIFT_MARGIN) * removal_[idx] / production);
      }
      for (int gp = 0; gp < G; ++gp) {
        // scatter_[i][from][to]; transposed for the adjoint.
        const std::size_t src = adjoint_ ? static_cast<std::size_t>(gp) * G + g
                                         : static_cast<std::size_t>(g) * G + gp;
        const double value = c.scatter[src] * V;
        scatter_[(static_cast<std::size_t>(i) * G + g) * G + gp] = value;
        // scatter_[i][g][gp] transfers g -> gp, so gp < g is upscattering.
        if (gp < g && value != 0.0) has_upscatter_ = true;
      }
    }
  }
  max_inv_shift_ = std::max(0.0, limit);
}

void CmfdSystem::set_adjoint(bool adjoint)
{
  if (adjoint == adjoint_) return;
  adjoint_ = adjoint;
  // Dtilde depends only on the diffusion coefficients and the discontinuity
  // factors, and Dhat carries the nonlinear correction from a preceding
  // forward solve. Neither is rebuilt here: the adjoint of the corrected
  // operator is exactly the transpose of the corrected forward operator.
  refresh_cross_sections();
}

double CmfdSystem::boundary_coupling(
    const Surface& surf, int group, double D, double adf) const
{
  const double h = surf.boundary_is_lo_side() ? surf.h_hi : surf.h_lo;
  switch (surf.bc) {
    case BoundaryType::reflective:
      return 0.0;
    case BoundaryType::zero_flux:
      // phi_s = 0 makes the discontinuity factor irrelevant.
      return 2.0 * D / h;
    case BoundaryType::vacuum: {
      // Marshak: J = phi_s / 4 outgoing with no incoming partial current.
      const double gamma = 0.5 * adf;
      return 2.0 * D * gamma / (2.0 * D + gamma * h);
    }
    case BoundaryType::albedo: {
      const double beta = geom_.albedos()[static_cast<std::size_t>(
          surf.albedo_id)][static_cast<std::size_t>(group)];
      if (beta >= 1.0) return 0.0;
      if (beta <= -1.0) return 2.0 * D / h;
      const double gamma = adf * (1.0 - beta) / (2.0 * (1.0 + beta));
      return 2.0 * D * gamma / (2.0 * D + gamma * h);
    }
    default:
      return 0.0;
  }
}

void CmfdSystem::build_coupling()
{
  const int G = n_groups_;
  const auto& surfaces = geom_.surfaces();
  const auto& nodes = geom_.nodes();

  for (int s = 0; s < geom_.n_surfaces(); ++s) {
    const Surface& surf = surfaces[static_cast<std::size_t>(s)];
    const int axis = surf.axis;
    if (surf.bc == BoundaryType::interior) {
      const int L = surf.lo;
      const int R = surf.hi;
      const int comp_l = nodes[static_cast<std::size_t>(L)].composition;
      const int comp_r = nodes[static_cast<std::size_t>(R)].composition;
      for (int g = 0; g < G; ++g) {
        const double DL = diffusion_[static_cast<std::size_t>(L) * G + g];
        const double DR = diffusion_[static_cast<std::size_t>(R) * G + g];
        const double fL = xs_.adf_value(comp_l, 2 * axis + 1, g);
        const double fR = xs_.adf_value(comp_r, 2 * axis + 0, g);
        const double denom = fL * surf.h_lo * DR + fR * surf.h_hi * DL;
        const double base = 2.0 * DL * DR / denom;
        // J = base * (fL*phi_L - fR*phi_R), written in the canonical
        // J = -Dtilde (phi_R - phi_L) - Dhat (phi_R + phi_L) form so that the
        // discontinuity factors are already folded into the coupling before
        // any nonlinear correction is applied.
        dtilde_[static_cast<std::size_t>(s) * G + g] = base * 0.5 * (fL + fR);
        dhat_[static_cast<std::size_t>(s) * G + g] = base * 0.5 * (fR - fL);
      }
    } else {
      const int N = surf.boundary_node();
      const int comp = nodes[static_cast<std::size_t>(N)].composition;
      const int face = surf.boundary_is_lo_side() ? 2 * axis : 2 * axis + 1;
      for (int g = 0; g < G; ++g) {
        const double D = diffusion_[static_cast<std::size_t>(N) * G + g];
        const double f = xs_.adf_value(comp, face, g);
        dtilde_[static_cast<std::size_t>(s) * G + g] =
            boundary_coupling(surf, g, D, f);
        dhat_[static_cast<std::size_t>(s) * G + g] = 0.0;
      }
    }
  }
}

void CmfdSystem::assemble(double inv_k_shift)
{
  const int G = n_groups_;
  const auto& surfaces = geom_.surfaces();

  for (int g = 0; g < G; ++g) {
    GroupMatrix& A = matrices_[static_cast<std::size_t>(g)];
    A.zero();
    for (int i = 0; i < geom_.n_nodes(); ++i) {
      const std::size_t idx = static_cast<std::size_t>(i) * G + g;
      double diag = removal_[idx];
      // Within-group scattering never leaves the node, so it is excluded from
      // the removal term and never appears here.
      diag -= chi_[idx] * nu_fission_[idx] * inv_k_shift;
      A.add_diagonal(i, diag);
    }
    for (int s = 0; s < geom_.n_surfaces(); ++s) {
      const Surface& surf = surfaces[static_cast<std::size_t>(s)];
      const double area = surf.area;
      const double dt = dtilde_[static_cast<std::size_t>(s) * G + g];
      const double dh = dhat_[static_cast<std::size_t>(s) * G + g];
      if (surf.bc == BoundaryType::interior) {
        const int L = surf.lo;
        const int R = surf.hi;
        // Row L gains +J*A, row R gains -J*A. Transposing the two
        // off-diagonal entries turns the leakage operator into its adjoint.
        A.add_diagonal(L, area * (dt - dh));
        A.add_diagonal(R, area * (dt + dh));
        A.add(L, R, area * (adjoint_ ? (-dt + dh) : (-dt - dh)));
        A.add(R, L, area * (adjoint_ ? (-dt - dh) : (-dt + dh)));
      } else {
        A.add_diagonal(surf.boundary_node(), area * dt);
      }
    }
    precond_[static_cast<std::size_t>(g)].factor(A);
  }
}

void CmfdSystem::fission_source(
    const std::vector<double>& flux, std::vector<double>& source) const
{
  const int G = n_groups_;
  const int n = geom_.n_nodes();
  source.assign(static_cast<std::size_t>(n), 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (std::ptrdiff_t i = 0; i < n; ++i) {
    double s = 0.0;
    for (int g = 0; g < G; ++g) {
      const std::size_t idx = static_cast<std::size_t>(i) * G + g;
      s += nu_fission_[idx] * flux[idx];
    }
    source[static_cast<std::size_t>(i)] = s;
  }
}

int CmfdSystem::solve_groups(const std::vector<double>& fission_src,
    double k_eff, double inv_k_shift, std::vector<double>& flux,
    const Settings& s)
{
  const int G = n_groups_;
  const int n = geom_.n_nodes();
  const double lambda = 1.0 / k_eff - inv_k_shift;

  rhs_.resize(static_cast<std::size_t>(n));
  group_flux_.resize(static_cast<std::size_t>(n));
  int total_inner = 0;

  // A single lagged sweep is exact only when neither upscattering nor the
  // Wielandt shift couples the groups. Otherwise the sweep repeats until the
  // flux settles, because the shift enters entirely through the source of one
  // group formed from the fluxes of the others.
  const bool needs_repeats = (inv_k_shift > 0.0) || has_upscatter_;
  const int max_sweeps =
      needs_repeats ? std::max(2, s.group_sweeps) : std::max(1, s.group_sweeps);

  for (int sweep = 0; sweep < max_sweeps; ++sweep) {
    if (needs_repeats) prev_flux_.assign(flux.begin(), flux.end());
    for (int g = 0; g < G; ++g) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (std::ptrdiff_t i = 0; i < n; ++i) {
        const std::size_t base = static_cast<std::size_t>(i) * G;
        double b = 0.0;
        double fission_other = 0.0;
        for (int gp = 0; gp < G; ++gp) {
          if (gp == g) continue;
          b += scatter_[(base + gp) * G + g] * flux[base + gp];
          fission_other += nu_fission_[base + gp] * flux[base + gp];
        }
        const double chi = chi_[base + g];
        b += chi * (inv_k_shift * fission_other +
                       lambda * fission_src[static_cast<std::size_t>(i)]);
        rhs_[static_cast<std::size_t>(i)] = b;
        group_flux_[static_cast<std::size_t>(i)] = flux[base + g];
      }
      const LinearResult r = bicgstab(matrices_[static_cast<std::size_t>(g)],
          rhs_, group_flux_, precond_[static_cast<std::size_t>(g)],
          s.inner_tolerance, s.max_inner);
      total_inner += r.iterations;
      for (int i = 0; i < n; ++i) {
        flux[static_cast<std::size_t>(i) * G + g] =
            group_flux_[static_cast<std::size_t>(i)];
      }
    }

    if (!needs_repeats || sweep + 1 >= max_sweeps) break;
    double change = 0.0;
    double scale = 0.0;
    for (std::size_t i = 0; i < flux.size(); ++i) {
      change = std::max(change, std::abs(flux[i] - prev_flux_[i]));
      scale = std::max(scale, std::abs(flux[i]));
    }
    if (scale > 0.0 && change / scale < s.group_sweep_tolerance) break;
  }
  return total_inner;
}

int CmfdSystem::solve_fixed_source(const std::vector<double>& external,
    std::vector<double>& flux, const Settings& s)
{
  const int G = n_groups_;
  const int n = geom_.n_nodes();
  rhs_.resize(static_cast<std::size_t>(n));
  group_flux_.resize(static_cast<std::size_t>(n));
  int total_inner = 0;

  // The in-group fission term sits on the matrix diagonal here, so a single
  // sweep is exact unless the library upscatters.
  const int max_sweeps = has_upscatter_ ? std::max(2, s.group_sweeps)
                                        : std::max(1, s.group_sweeps);

  for (int sweep = 0; sweep < max_sweeps; ++sweep) {
    if (has_upscatter_) prev_flux_.assign(flux.begin(), flux.end());
    for (int g = 0; g < G; ++g) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (std::ptrdiff_t i = 0; i < n; ++i) {
        const std::size_t base = static_cast<std::size_t>(i) * G;
        double b = external[base + g] *
                   geom_.nodes()[static_cast<std::size_t>(i)].volume;
        double fission = 0.0;
        for (int gp = 0; gp < G; ++gp) {
          if (gp != g) b += scatter_[(base + gp) * G + g] * flux[base + gp];
          fission += nu_fission_[base + gp] * flux[base + gp];
        }
        // Subcritical multiplication: the fission source is treated as a
        // lagged source at k = 1 rather than as an eigenvalue problem.
        b += chi_[base + g] * fission;
        b -= chi_[base + g] * nu_fission_[base + g] * flux[base + g];
        rhs_[static_cast<std::size_t>(i)] = b;
        group_flux_[static_cast<std::size_t>(i)] = flux[base + g];
      }
      const LinearResult r = bicgstab(matrices_[static_cast<std::size_t>(g)],
          rhs_, group_flux_, precond_[static_cast<std::size_t>(g)],
          s.inner_tolerance, s.max_inner);
      total_inner += r.iterations;
      for (int i = 0; i < n; ++i) {
        flux[static_cast<std::size_t>(i) * G + g] =
            group_flux_[static_cast<std::size_t>(i)];
      }
    }
    if (!has_upscatter_ || sweep + 1 >= max_sweeps) break;
    double change = 0.0;
    double scale = 0.0;
    for (std::size_t i = 0; i < flux.size(); ++i) {
      change = std::max(change, std::abs(flux[i] - prev_flux_[i]));
      scale = std::max(scale, std::abs(flux[i]));
    }
    if (scale > 0.0 && change / scale < s.group_sweep_tolerance) break;
  }
  return total_inner;
}

void CmfdSystem::compute_currents(
    const std::vector<double>& flux, std::vector<double>& current) const
{
  const int G = n_groups_;
  const int ns = geom_.n_surfaces();
  const auto& surfaces = geom_.surfaces();
  current.assign(static_cast<std::size_t>(ns) * G, 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (std::ptrdiff_t s = 0; s < ns; ++s) {
    const Surface& surf = surfaces[static_cast<std::size_t>(s)];
    for (int g = 0; g < G; ++g) {
      const std::size_t si = static_cast<std::size_t>(s) * G + g;
      if (surf.bc == BoundaryType::interior) {
        const double pl = flux[static_cast<std::size_t>(surf.lo) * G + g];
        const double ph = flux[static_cast<std::size_t>(surf.hi) * G + g];
        current[si] = -dtilde_[si] * (ph - pl) - dhat_[si] * (ph + pl);
      } else {
        const double p =
            flux[static_cast<std::size_t>(surf.boundary_node()) * G + g];
        // The outward normal of a low-side boundary points along -axis, so the
        // current expressed along +axis changes sign.
        const double sign = surf.boundary_is_lo_side() ? -1.0 : 1.0;
        current[si] = sign * dtilde_[si] * p;
      }
    }
  }
}

double CmfdSystem::nodal_update(const Kernel& kernel,
    const std::vector<double>& flux, double k_eff, const Settings& s)
{
  if (kernel.is_finite_difference()) return 0.0;
  compute_currents(flux, current_);

  const int G = n_groups_;
  const int n_axes = geom_.n_axes();
  const auto& surfaces = geom_.surfaces();
  const auto& nodes = geom_.nodes();

  // Node-average transverse leakage per axis, from the coarse-mesh currents.
  leakage_.assign(static_cast<std::size_t>(geom_.n_nodes()) * n_axes * G, 0.0);
  const int nn = geom_.n_nodes();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (std::ptrdiff_t i = 0; i < nn; ++i) {
    const Node& node = nodes[static_cast<std::size_t>(i)];
    for (int a = 0; a < n_axes; ++a) {
      for (int g = 0; g < G; ++g) {
        double leak = 0.0;
        for (int b = 0; b < n_axes; ++b) {
          if (b == a) continue;
          const int s_lo = node.face[2 * b + 0];
          const int s_hi = node.face[2 * b + 1];
          const double j_lo = current_[static_cast<std::size_t>(s_lo) * G + g];
          const double j_hi = current_[static_cast<std::size_t>(s_hi) * G + g];
          leak += (j_hi - j_lo) / node.width[static_cast<std::size_t>(b)];
        }
        leakage_[(static_cast<std::size_t>(i) * n_axes + a) * G + g] = leak;
      }
    }
  }

  std::vector<int> composition(static_cast<std::size_t>(geom_.n_nodes()));
  for (int i = 0; i < geom_.n_nodes(); ++i) {
    composition[static_cast<std::size_t>(i)] =
        nodes[static_cast<std::size_t>(i)].composition;
  }

  const int ns = geom_.n_surfaces();
  std::vector<double> change(static_cast<std::size_t>(ns), 0.0);

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    std::vector<double> tl_lo(static_cast<std::size_t>(3) * G);
    std::vector<double> tl_hi(static_cast<std::size_t>(3) * G);
    std::vector<double> adf_lo(static_cast<std::size_t>(G));
    std::vector<double> adf_hi(static_cast<std::size_t>(G));
    std::vector<double> current(static_cast<std::size_t>(G));

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (std::ptrdiff_t si = 0; si < ns; ++si) {
      const Surface& surf = surfaces[static_cast<std::size_t>(si)];
      if (surf.bc != BoundaryType::interior) continue;
      const int L = surf.lo;
      const int R = surf.hi;
      const int a = surf.axis;
      const Node& nl = nodes[static_cast<std::size_t>(L)];
      const Node& nr = nodes[static_cast<std::size_t>(R)];

      // Neighbours used only for the quadratic transverse leakage fit. A
      // missing neighbour is replaced by a copy of the node itself, which is
      // the usual flat extrapolation at a core boundary.
      const Surface& s_prev =
          surfaces[static_cast<std::size_t>(nl.face[2 * a + 0])];
      const Surface& s_next =
          surfaces[static_cast<std::size_t>(nr.face[2 * a + 1])];
      const int prev = (s_prev.bc == BoundaryType::interior) ? s_prev.lo : L;
      const int next = (s_next.bc == BoundaryType::interior) ? s_next.hi : R;

      for (int g = 0; g < G; ++g) {
        const std::size_t la = static_cast<std::size_t>(a);
        tl_lo[static_cast<std::size_t>(0) * G + g] =
            leakage_[(static_cast<std::size_t>(prev) * n_axes + la) * G + g];
        tl_lo[static_cast<std::size_t>(1) * G + g] =
            leakage_[(static_cast<std::size_t>(L) * n_axes + la) * G + g];
        tl_lo[static_cast<std::size_t>(2) * G + g] =
            leakage_[(static_cast<std::size_t>(R) * n_axes + la) * G + g];
        tl_hi[static_cast<std::size_t>(0) * G + g] =
            leakage_[(static_cast<std::size_t>(L) * n_axes + la) * G + g];
        tl_hi[static_cast<std::size_t>(1) * G + g] =
            leakage_[(static_cast<std::size_t>(R) * n_axes + la) * G + g];
        tl_hi[static_cast<std::size_t>(2) * G + g] =
            leakage_[(static_cast<std::size_t>(next) * n_axes + la) * G + g];
        adf_lo[static_cast<std::size_t>(g)] =
            xs_.adf_value(nl.composition, 2 * a + 1, g);
        adf_hi[static_cast<std::size_t>(g)] =
            xs_.adf_value(nr.composition, 2 * a + 0, g);
      }

      TwoNodeProblem p;
      p.surface = static_cast<int>(si);
      p.node_lo = L;
      p.node_hi = R;
      p.axis = a;
      p.h_lo = surf.h_lo;
      p.h_hi = surf.h_hi;
      p.flux_lo = &flux[static_cast<std::size_t>(L) * G];
      p.flux_hi = &flux[static_cast<std::size_t>(R) * G];
      p.adf_lo = adf_lo.data();
      p.adf_hi = adf_hi.data();
      p.tl_lo = tl_lo.data();
      p.tl_hi = tl_hi.data();
      p.h_lo_prev = nodes[static_cast<std::size_t>(prev)]
                        .width[static_cast<std::size_t>(a)];
      p.h_hi_next = nodes[static_cast<std::size_t>(next)]
                        .width[static_cast<std::size_t>(a)];
      p.cmfd_current_lo =
          &current_[static_cast<std::size_t>(nl.face[2 * a + 0]) * G];
      p.cmfd_current_hi =
          &current_[static_cast<std::size_t>(nr.face[2 * a + 1]) * G];
      p.k_eff = k_eff;

      kernel.solve(p, xs_, composition, G, s.two_node_sweeps, current.data());

      double local_change = 0.0;
      for (int g = 0; g < G; ++g) {
        const std::size_t idx = static_cast<std::size_t>(si) * G + g;
        const double pl = flux[static_cast<std::size_t>(L) * G + g];
        const double ph = flux[static_cast<std::size_t>(R) * G + g];
        const double sum = pl + ph;
        if (std::abs(sum) < 1.0e-30) continue;
        double dh =
            -(current[static_cast<std::size_t>(g)] + dtilde_[idx] * (ph - pl)) /
            sum;
        // An unbounded correction makes the coarse-mesh matrix lose diagonal
        // dominance; both PARCS and KOMODO clamp for the same reason.
        const double limit = s.dhat_limit * dtilde_[idx];
        dh = std::max(-limit, std::min(limit, dh));
        local_change =
            std::max(local_change, std::abs(dh - dhat_[idx]) / (dtilde_[idx]));
        dhat_[idx] = dh;
      }
      change[static_cast<std::size_t>(si)] = local_change;
    }
  }

  double max_change = 0.0;
  for (double c : change) max_change = std::max(max_change, c);
  return max_change;
}

}  // namespace openndm
