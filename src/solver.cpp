#include "openndm/solver.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

#include "openndm/error.h"
#include "openndm/precursors.h"

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

//! Reciprocal shift that puts the in-group fission term on the diagonal.
//!
//! This is what makes a fixed-source solve a subcritical multiplication
//! operator rather than a pure absorber; see docs/theory.md section 6.
constexpr double kSubcriticalMultiplication = 1.0;

//! Largest node-wise change in a fission source between two outers.
//!
//! Each source is normalised to unit mean first, so the criterion does not
//! depend on how the flux happens to be scaled (FR-SOL-6).
double node_wise_source_change(const std::vector<double>& source_new,
    const std::vector<double>& source_old, double s_new, double s_old)
{
  constexpr double kSourceFloor = 1.0e-12;
  const std::size_t n = source_new.size();
  const double scale_new = static_cast<double>(n) / s_new;
  const double scale_old = static_cast<double>(n) / s_old;
  double largest = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double a = source_new[i] * scale_new;
    const double b = source_old[i] * scale_old;
    if (a > kSourceFloor) largest = std::max(largest, std::abs(a - b) / a);
  }
  return largest;
}

}  // namespace

Solver::Solver(const Geometry& geom, const XSLibrary& xs)
    : geom_(geom),
      xs_(xs),
      cmfd_(geom, xs)
{
}

Solver::~Solver() = default;

void Solver::normalise_to_unit_mean_flux()
{
  const int G = xs_.n_groups();
  const auto& nodes = geom_.nodes();
  double integral = 0.0;
  for (int i = 0; i < geom_.n_nodes(); ++i) {
    for (int g = 0; g < G; ++g) {
      integral += flux_[static_cast<std::size_t>(i) * G + g] *
                  nodes[static_cast<std::size_t>(i)].volume;
    }
  }
  const double norm = (integral > 0.0) ? geom_.total_volume() / integral : 1.0;
  for (auto& f : flux_) f *= norm;
}

void Solver::reset()
{
  if (in_transient_) {
    throw InputError(
        "a transient is in progress: resetting would discard the "
        "time-dependent flux, and the next step would run on an empty one. "
        "step() already re-reads the cross sections, the node compositions "
        "and the coupling, so nothing needs refreshing between steps");
  }
  has_solution_ = false;
  flux_.clear();
  k_eff_ = 1.0;
  cmfd_.refresh_cross_sections();
  cmfd_.build_coupling();
}

Result Solver::solve(const Settings& settings)
{
  in_transient_ = false;
  const auto t0 = std::chrono::steady_clock::now();
#ifdef _OPENMP
  if (settings.threads > 0) omp_set_num_threads(settings.threads);
#endif

  const int n = geom_.n_nodes();
  const int G = xs_.n_groups();
  const std::size_t size = static_cast<std::size_t>(n) * G;

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

    const double src_change =
        node_wise_source_change(source_new, source_old, s_new, s_old);

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

  normalise_to_unit_mean_flux();

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

  cmfd_.assemble(kSubcriticalMultiplication);
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
  if (total > 0.0 && n_fuel > 0) {
    const double norm = static_cast<double>(n_fuel) / total;
    for (auto& p : power) p *= norm;
  }
}

double Solver::chi_delayed(int node, int g, int d) const
{
  const DelayedData& delayed = xs_.delayed();
  const int G = xs_.n_groups();
  if (static_cast<int>(delayed.chi_delayed.size()) ==
      delayed.n_precursors() * G) {
    return delayed.chi_delayed[static_cast<std::size_t>(d) * G + g];
  }
  return cmfd_.chi(node, g);
}

double Solver::chi_prompt(int node, int g) const
{
  const DelayedData& delayed = xs_.delayed();
  const double beta = delayed.beta_total();
  if (!(beta < 1.0)) {
    throw InputError("total delayed fraction must be below one");
  }
  double weighted = 0.0;
  for (int d = 0; d < delayed.n_precursors(); ++d) {
    weighted +=
        delayed.beta[static_cast<std::size_t>(d)] * chi_delayed(node, g, d);
  }
  return (cmfd_.chi(node, g) - weighted) / (1.0 - beta);
}

void Solver::fission_power(
    const std::vector<double>& flux, std::vector<double>& power) const
{
  const int n = geom_.n_nodes();
  const int G = xs_.n_groups();
  power.assign(static_cast<std::size_t>(n), 0.0);
  const auto& nodes = geom_.nodes();
  for (int i = 0; i < n; ++i) {
    const Node& node = nodes[static_cast<std::size_t>(i)];
    const Composition& c = xs_.composition(node.composition);
    double total = 0.0;
    for (int g = 0; g < G; ++g) {
      total += c.kappa_fission[static_cast<std::size_t>(g)] *
               flux[static_cast<std::size_t>(i) * G + g];
    }
    power[static_cast<std::size_t>(i)] = total * node.volume;
  }
}

void Solver::time_derivative(const std::vector<double>& flux,
    const std::vector<double>& fission_norm,
    const std::vector<double>& delayed_source, std::vector<double>& out) const
{
  const int n = geom_.n_nodes();
  const int G = xs_.n_groups();
  const DelayedData& delayed = xs_.delayed();
  const double beta = delayed.beta_total();

  std::vector<double> applied;
  cmfd_.apply_operator(flux, applied);

  out.assign(static_cast<std::size_t>(n) * G, 0.0);
  for (int i = 0; i < n; ++i) {
    const std::size_t base = static_cast<std::size_t>(i) * G;
    for (int g = 0; g < G; ++g) {
      double value =
          -(applied[base + g] - cmfd_.time_removal(i, g) * flux[base + g]);
      for (int gp = 0; gp < G; ++gp) {
        if (gp == g) continue;
        value += cmfd_.scatter(i, gp, g) * flux[base + gp];
      }
      value += chi_prompt(i, g) * (1.0 - beta) *
               fission_norm[static_cast<std::size_t>(i)];
      value += delayed_source[base + g];
      out[base + g] = value;
    }
  }
}

void Solver::start_transient(const Settings& settings)
{
  const int n = geom_.n_nodes();
  const int G = xs_.n_groups();
  if (!has_solution_ || flux_.empty()) {
    throw InputError(
        "a transient starts from a converged static solution; call solve() "
        "first");
  }
  for (int c = 0; c < xs_.n_compositions(); ++c) {
    const Composition& comp = xs_.composition(c);
    bool any = false;
    for (double v : comp.inv_velocity) any = any || (v > 0.0);
    if (!any) {
      throw InputError("composition " + std::to_string(c) +
                       " has no inverse velocities; a time-dependent solve "
                       "cannot be scaled without them");
    }
  }

  precursors_ = std::make_unique<PrecursorState>(n, xs_.delayed());
  k_static_ = k_eff_;
  time_ = 0.0;

  std::vector<double> source;
  cmfd_.fission_source(flux_, source);
  fission_norm_.assign(static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    fission_norm_[static_cast<std::size_t>(i)] =
        source[static_cast<std::size_t>(i)] / k_static_;
  }
  precursors_->set_equilibrium(fission_norm_);

  cmfd_.set_time_removal(0.0);
  cmfd_.set_transient_chi({});
  cmfd_.set_transient_source({});
  cmfd_.assemble(0.0);
  std::vector<double> delayed_source;
  precursors_->delayed_source(xs_.delayed(), G, delayed_source);
  time_derivative(flux_, fission_norm_, delayed_source, derivative_);

  (void)settings;
  in_transient_ = true;
}

const std::vector<double>& Solver::precursors() const
{
  if (!precursors_) throw InputError("no transient in progress");
  return precursors_->concentrations();
}

int Solver::n_precursors() const
{
  return precursors_ ? precursors_->n_precursors() : 0;
}

TransientRecord Solver::step(double dt, const Settings& settings)
{
  if (!in_transient_ || !precursors_) {
    throw InputError("call start_transient() before step()");
  }
  if (!(dt > 0.0)) throw InputError("time step must be positive");
  const double theta = settings.theta;
  if (!(theta > 0.0) || theta > 1.0) {
    throw InputError("theta must be in (0, 1]");
  }

  const int n = geom_.n_nodes();
  const int G = xs_.n_groups();
  const DelayedData& delayed = xs_.delayed();
  const int D = delayed.n_precursors();
  const double beta = delayed.beta_total();

  cmfd_.refresh_cross_sections();
  cmfd_.refresh_coupling();

  if (theta < 1.0) {
    cmfd_.set_time_removal(0.0);
    cmfd_.set_transient_chi({});
    cmfd_.set_transient_source({});
    cmfd_.assemble(0.0);
    std::vector<double> previous_delayed;
    precursors_->delayed_source(delayed, G, previous_delayed);
    time_derivative(flux_, fission_norm_, previous_delayed, derivative_);
  }

  cmfd_.set_time_removal(theta * dt);

  std::vector<double> chi_eff(static_cast<std::size_t>(n) * G, 0.0);
  std::vector<double> known(static_cast<std::size_t>(n) * G, 0.0);
  std::vector<double> decay(static_cast<std::size_t>(D));
  std::vector<double> i0(static_cast<std::size_t>(D));
  std::vector<double> i1(static_cast<std::size_t>(D));
  for (int d = 0; d < D; ++d) {
    const double lambda = delayed.lambda[static_cast<std::size_t>(d)];
    decay[static_cast<std::size_t>(d)] = std::exp(-lambda * dt);
    i0[static_cast<std::size_t>(d)] = decay_integral_0(lambda, dt);
    i1[static_cast<std::size_t>(d)] = decay_integral_1(lambda, dt);
  }

  for (int i = 0; i < n; ++i) {
    const std::size_t base = static_cast<std::size_t>(i) * G;
    const double f_old = fission_norm_[static_cast<std::size_t>(i)];
    for (int g = 0; g < G; ++g) {
      double implicit = chi_prompt(i, g) * (1.0 - beta);
      double explicit_part = 0.0;
      for (int d = 0; d < D; ++d) {
        const std::size_t dd = static_cast<std::size_t>(d);
        const double lambda = delayed.lambda[dd];
        const double b = delayed.beta[dd];
        const double chid = chi_delayed(i, g, d);
        implicit += lambda * chid * b * i1[dd] / dt;
        explicit_part += lambda * chid *
                         (precursors_->concentration(i, d) * decay[dd] +
                             b * f_old * (i0[dd] - i1[dd] / dt));
      }
      chi_eff[base + g] = implicit;
      known[base + g] = explicit_part +
                        cmfd_.time_removal(i, g) * flux_[base + g] +
                        ((1.0 - theta) / theta) * derivative_[base + g];
    }
  }

  cmfd_.set_transient_chi(chi_eff);
  cmfd_.set_transient_source(known);
  cmfd_.assemble(0.0);

  std::vector<double> previous = flux_;
  std::vector<double> source(static_cast<std::size_t>(n));
  TransientRecord record;
  record.dt = dt;

  for (int iter = 1; iter <= settings.max_step_iterations; ++iter) {
    cmfd_.fission_source(flux_, source);
    record.inner_iterations +=
        cmfd_.solve_groups(source, k_static_, 0.0, flux_, settings);

    double change = 0.0;
    double scale = 0.0;
    for (std::size_t j = 0; j < flux_.size(); ++j) {
      change += std::abs(flux_[j] - previous[j]);
      scale += std::abs(flux_[j]);
    }
    previous = flux_;
    record.iterations = iter;
    if (scale > 0.0 && change / scale < settings.step_tolerance) {
      record.converged = true;
      break;
    }
  }

  std::vector<double> fission_new(static_cast<std::size_t>(n));
  cmfd_.fission_source(flux_, source);
  for (int i = 0; i < n; ++i) {
    fission_new[static_cast<std::size_t>(i)] =
        source[static_cast<std::size_t>(i)] / k_static_;
  }
  precursors_->advance(fission_norm_, fission_new, dt);
  fission_norm_ = fission_new;
  time_ += dt;

  std::vector<double> power;
  fission_power(flux_, power);
  double total = 0.0;
  double peak = 0.0;
  for (double value : power) {
    total += value;
    peak = std::max(peak, value);
  }
  record.time = time_;
  record.total_power = total;
  record.peak_power = peak;
  return record;
}

}  // namespace openndm
