#include "openndm/solver.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

#include "openndm/error.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace openndm {

namespace {

//! Reciprocal Wielandt shift for the current outer iteration.
//!
//! The shift is held back for the first few outers because a tight shift
//! applied to a poor flux guess makes the inner systems nearly singular
//! without buying any outer convergence.
double reciprocal_shift(double k, const Settings& s, int outer, double ceiling)
{
  if (s.wielandt_shift <= 0.0 || outer < s.wielandt_start) return 0.0;
  return std::min(1.0 / (k + s.wielandt_shift), ceiling);
}

double sum(const std::vector<double>& v)
{
  double t = 0.0;
  for (double x : v) t += x;
  return t;
}

}  // namespace

Solver::Solver(const Geometry& geom, const XSLibrary& xs)
    : geom_(geom),
      xs_(xs),
      cmfd_(geom, xs)
{
}

Solver::~Solver() = default;

void Solver::reset()
{
  has_solution_ = false;
  flux_.clear();
  k_eff_ = 1.0;
  // Cross sections may have been mutated in place since the last solve, for
  // instance by a boron search, so the cached per-node data is refreshed
  // before the coupling that depends on it is rebuilt.
  cmfd_.refresh_cross_sections();
  cmfd_.build_coupling();
}

Result Solver::solve(const Settings& settings)
{
  const auto t0 = std::chrono::steady_clock::now();
#ifdef _OPENMP
  if (settings.threads > 0) omp_set_num_threads(settings.threads);
#endif

  const int n = geom_.n_nodes();
  const int G = xs_.n_groups();
  const std::size_t size = static_cast<std::size_t>(n) * G;

  // An adjoint run reuses the nonlinear coupling from a forward solve, so a
  // cold adjoint request runs the forward problem first (see set_adjoint).
  if (settings.mode == SolveMode::adjoint && !cmfd_.adjoint()) {
    if (settings.kernel != KernelType::fdm && !has_solution_) {
      Settings forward = settings;
      forward.mode = SolveMode::forward;
      forward.verbosity = 0;
      solve(forward);
    }
    cmfd_.set_adjoint(true);
  } else if (settings.mode != SolveMode::adjoint && cmfd_.adjoint()) {
    cmfd_.set_adjoint(false);
    has_solution_ = false;
  }

  if (!settings.warm_start || !has_solution_ || flux_.size() != size) {
    flux_.assign(size, 1.0);
    k_eff_ = 1.0;
    // Discard the nonlinear correction as well, so that a cold solve of the
    // same model always retraces the same iteration path. An adjoint run is
    // the exception: it needs the Dhat converged by the forward solve.
    if (!cmfd_.adjoint()) cmfd_.build_coupling();
  }

  auto kernel = Kernel::create(settings.kernel);

  std::vector<double> source_old;
  cmfd_.fission_source(flux_, source_old);
  const double s0 = sum(source_old);
  if (!(s0 > 0.0)) {
    throw InputError(
        "the model has no fission source; use solve_fixed_source instead");
  }

  Result result;
  result.kernel = kernel->name();
  result.history.reserve(static_cast<std::size_t>(settings.max_outer));

  std::vector<double> source_new;
  double k = k_eff_;
  int total_inner = 0;

  // The two-node kernels solve the forward transverse-integrated problem, so
  // running them against an adjoint flux would produce meaningless coupling
  // corrections. The adjoint instead keeps the Dhat converged by the forward
  // solve, which is exactly the transpose of the corrected forward operator.
  const bool update_nodal = !kernel->is_finite_difference() && !cmfd_.adjoint();

  for (int outer = 1; outer <= settings.max_outer; ++outer) {
    if (update_nodal && outer >= settings.nodal_start &&
        ((outer - settings.nodal_start) % settings.nodal_update_interval ==
            0)) {
      cmfd_.nodal_update(*kernel, flux_, k, settings);
    }

    const double inv_shift =
        reciprocal_shift(k, settings, outer, cmfd_.max_reciprocal_shift());
    cmfd_.assemble(inv_shift);
    total_inner +=
        cmfd_.solve_groups(source_old, k, inv_shift, flux_, settings);

    cmfd_.fission_source(flux_, source_new);
    const double s_new = sum(source_new);
    const double s_old = sum(source_old);
    if (!(s_new > 0.0)) {
      throw ConvergenceError("fission source collapsed to zero", outer, 0.0);
    }

    const double k_prev = k;
    const double inv_k_new =
        inv_shift + (1.0 / k - inv_shift) * (s_old / s_new);
    k = 1.0 / inv_k_new;

    // Node-wise fission source change, on a source normalised to unit mean so
    // that the criterion is independent of the flux normalisation (FR-SOL-6).
    double src_change = 0.0;
    const double scale_new = static_cast<double>(n) / s_new;
    const double scale_old = static_cast<double>(n) / s_old;
    for (int i = 0; i < n; ++i) {
      const double a = source_new[static_cast<std::size_t>(i)] * scale_new;
      const double b = source_old[static_cast<std::size_t>(i)] * scale_old;
      if (a > 1.0e-12) {
        src_change = std::max(src_change, std::abs(a - b) / a);
      }
    }

    IterationRecord rec;
    rec.outer = outer;
    rec.k_eff = k;
    rec.k_change = k - k_prev;
    rec.source_change = src_change;
    rec.inner_iterations = total_inner;
    result.history.push_back(rec);

    if (settings.verbosity >= 2) {
      std::printf("  outer %4d   k = %.8f   dk = %+.2e   dS = %.2e\n", outer, k,
          rec.k_change, src_change);
      std::fflush(stdout);
    }

    source_old.swap(source_new);

    if (outer >= settings.min_outer &&
        std::abs(rec.k_change) < settings.k_tolerance &&
        src_change < settings.fission_source_tolerance) {
      result.converged = true;
      result.outer_iterations = outer;
      break;
    }
    result.outer_iterations = outer;
  }

  k_eff_ = k;
  fission_source_ = source_old;
  has_solution_ = true;

  if (!result.converged) {
    const double last =
        result.history.empty() ? 1.0 : result.history.back().source_change;
    throw ConvergenceError("outer iteration did not converge after " +
                               std::to_string(result.outer_iterations) +
                               " iterations",
        result.outer_iterations, last);
  }

  // Normalise the flux so that the volume-averaged total flux is one, which
  // makes results comparable between meshes and between runs.
  double integral = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int g = 0; g < G; ++g) {
      integral += flux_[static_cast<std::size_t>(i) * G + g] *
                  geom_.nodes()[static_cast<std::size_t>(i)].volume;
    }
  }
  const double norm = (integral > 0.0) ? geom_.total_volume() / integral : 1.0;
  for (auto& f : flux_) f *= norm;

  result.k_eff = k;
  result.flux = flux_;
  compute_power(result.power);
  result.runtime_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();

  if (settings.verbosity >= 1) {
    std::printf("openndm: %s  k_eff = %.6f  outers = %d  time = %.3f s\n",
        result.kernel.c_str(), result.k_eff, result.outer_iterations,
        result.runtime_seconds);
    std::fflush(stdout);
  }
  return result;
}

Result Solver::solve_fixed_source(
    const std::vector<double>& source, const Settings& settings)
{
  const auto t0 = std::chrono::steady_clock::now();
  const int n = geom_.n_nodes();
  const int G = xs_.n_groups();
  const std::size_t size = static_cast<std::size_t>(n) * G;
  if (source.size() != size) {
    throw InputError("external source must have n_nodes * n_groups entries");
  }
  if (!settings.warm_start || flux_.size() != size) flux_.assign(size, 0.0);

  auto kernel = Kernel::create(settings.kernel);
  Result result;
  result.kernel = kernel->name();

  // Assembling at a reciprocal shift of one puts the in-group fission term on
  // the diagonal, which is what makes this a subcritical multiplication
  // operator rather than a pure absorber.
  cmfd_.assemble(1.0);
  double previous = 0.0;
  for (int outer = 1; outer <= settings.max_outer; ++outer) {
    if (!kernel->is_finite_difference() && outer >= settings.nodal_start &&
        ((outer - settings.nodal_start) % settings.nodal_update_interval ==
            0)) {
      cmfd_.nodal_update(*kernel, flux_, 1.0, settings, &source);
      cmfd_.assemble(1.0);
    }
    cmfd_.solve_fixed_source(source, flux_, settings);

    double total = 0.0;
    for (double f : flux_) total += f;
    const double change =
        (previous > 0.0) ? std::abs(total - previous) / previous : 1.0;
    IterationRecord rec;
    rec.outer = outer;
    rec.k_eff = 0.0;
    rec.source_change = change;
    result.history.push_back(rec);
    previous = total;
    result.outer_iterations = outer;
    if (outer >= settings.min_outer &&
        change < settings.fission_source_tolerance) {
      result.converged = true;
      break;
    }
  }
  if (!result.converged) {
    throw ConvergenceError("fixed source iteration did not converge",
        result.outer_iterations, previous);
  }
  has_solution_ = true;
  result.k_eff = 0.0;
  result.flux = flux_;
  compute_power(result.power);
  result.runtime_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  return result;
}

std::vector<double> Solver::surface_currents() const
{
  if (!has_solution_) {
    throw InputError("no solution available; call solve() first");
  }
  std::vector<double> current;
  cmfd_.compute_currents(flux_, current);
  return current;
}

void Solver::compute_power(std::vector<double>& power) const
{
  const int n = geom_.n_nodes();
  const int G = xs_.n_groups();
  power.assign(static_cast<std::size_t>(n), 0.0);
  int n_fuel = 0;
  double total = 0.0;
  for (int i = 0; i < n; ++i) {
    const Node& node = geom_.nodes()[static_cast<std::size_t>(i)];
    const Composition& c = xs_.composition(node.composition);
    double p = 0.0;
    for (int g = 0; g < G; ++g) {
      p += c.kappa_fission[static_cast<std::size_t>(g)] *
           flux_[static_cast<std::size_t>(i) * G + g];
    }
    p *= node.volume;
    power[static_cast<std::size_t>(i)] = p;
    if (p > 0.0) {
      total += p;
      ++n_fuel;
    }
  }
  // Relative power, normalised to a mean of one over the powered nodes, which
  // is the convention peaking factors are quoted against (FR-OUT-3).
  if (total > 0.0 && n_fuel > 0) {
    const double norm = static_cast<double>(n_fuel) / total;
    for (auto& p : power) p *= norm;
  }
}

}  // namespace openndm
