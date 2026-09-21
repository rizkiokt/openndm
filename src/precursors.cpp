#include "openndm/precursors.h"

#include <cmath>
#include <string>

#include "openndm/error.h"

namespace openndm {

namespace {

//! Below this the closed forms for I0 and I1 lose their significant digits to
//! cancellation and the series is both faster and more accurate. At x = 1e-4
//! the truncated series is good to well under one ulp of the result.
constexpr double kSmall = 1.0e-4;

}  // namespace

double decay_integral_0(double lambda, double dt)
{
  const double x = lambda * dt;
  if (std::abs(x) < kSmall) {
    return dt * (1.0 - x / 2.0 + x * x / 6.0 - x * x * x / 24.0);
  }
  return -std::expm1(-x) / lambda;
}

double decay_integral_1(double lambda, double dt)
{
  const double x = lambda * dt;
  if (std::abs(x) < kSmall) {
    return dt * dt * (0.5 - x / 3.0 + x * x / 8.0 - x * x * x / 30.0);
  }
  return (dt - decay_integral_0(lambda, dt)) / lambda;
}

PrecursorState::PrecursorState(int n_nodes, const DelayedData& delayed)
    : n_nodes_(n_nodes),
      n_precursors_(delayed.n_precursors()),
      beta_(delayed.beta),
      lambda_(delayed.lambda)
{
  if (n_nodes <= 0) throw InputError("precursor state needs at least one node");
  if (n_precursors_ <= 0) {
    throw InputError(
        "the library carries no delayed neutron data; set beta and lambda "
        "before running a transient");
  }
  if (static_cast<int>(lambda_.size()) != n_precursors_) {
    throw InputError("beta and lambda disagree on the precursor count");
  }
  for (int d = 0; d < n_precursors_; ++d) {
    if (!(lambda_[static_cast<std::size_t>(d)] > 0.0)) {
      throw InputError("precursor group " + std::to_string(d) +
                       " has a non-positive decay constant");
    }
    if (beta_[static_cast<std::size_t>(d)] < 0.0) {
      throw InputError("precursor group " + std::to_string(d) +
                       " has a negative delayed fraction");
    }
  }
  c_.assign(static_cast<std::size_t>(n_nodes_) * n_precursors_, 0.0);
}

double PrecursorState::concentration(int node, int d) const
{
  return c_[static_cast<std::size_t>(node) * n_precursors_ + d];
}

void PrecursorState::set_equilibrium(const std::vector<double>& fission_source)
{
  if (static_cast<int>(fission_source.size()) != n_nodes_) {
    throw InputError("fission source must have one value per node");
  }
  for (int i = 0; i < n_nodes_; ++i) {
    const double f = fission_source[static_cast<std::size_t>(i)];
    for (int d = 0; d < n_precursors_; ++d) {
      c_[static_cast<std::size_t>(i) * n_precursors_ + d] =
          beta_[static_cast<std::size_t>(d)] * f /
          lambda_[static_cast<std::size_t>(d)];
    }
  }
}

void PrecursorState::advance(
    const std::vector<double>& f0, const std::vector<double>& f1, double dt)
{
  if (!(dt > 0.0)) throw InputError("time step must be positive");
  if (static_cast<int>(f0.size()) != n_nodes_ ||
      static_cast<int>(f1.size()) != n_nodes_) {
    throw InputError("fission source must have one value per node");
  }

  std::vector<double> decay(static_cast<std::size_t>(n_precursors_));
  std::vector<double> i0(static_cast<std::size_t>(n_precursors_));
  std::vector<double> i1(static_cast<std::size_t>(n_precursors_));
  for (int d = 0; d < n_precursors_; ++d) {
    const double lambda = lambda_[static_cast<std::size_t>(d)];
    decay[static_cast<std::size_t>(d)] = std::exp(-lambda * dt);
    i0[static_cast<std::size_t>(d)] = decay_integral_0(lambda, dt);
    i1[static_cast<std::size_t>(d)] = decay_integral_1(lambda, dt);
  }

#pragma omp parallel for schedule(static)
  for (int i = 0; i < n_nodes_; ++i) {
    const double a = f0[static_cast<std::size_t>(i)];
    const double slope = (f1[static_cast<std::size_t>(i)] - a) / dt;
    for (int d = 0; d < n_precursors_; ++d) {
      const std::size_t k = static_cast<std::size_t>(i) * n_precursors_ + d;
      const std::size_t dd = static_cast<std::size_t>(d);
      c_[k] = c_[k] * decay[dd] + beta_[dd] * (a * i0[dd] + slope * i1[dd]);
    }
  }
}

void PrecursorState::delayed_source(
    const DelayedData& delayed, int n_groups, std::vector<double>& out) const
{
  const bool has_spectrum =
      static_cast<int>(delayed.chi_delayed.size()) == n_precursors_ * n_groups;
  out.assign(static_cast<std::size_t>(n_nodes_) * n_groups, 0.0);
  for (int i = 0; i < n_nodes_; ++i) {
    for (int d = 0; d < n_precursors_; ++d) {
      const std::size_t dd = static_cast<std::size_t>(d);
      const double rate = lambda_[dd] * concentration(i, d);
      if (rate == 0.0) continue;
      for (int g = 0; g < n_groups; ++g) {
        const double chi =
            has_spectrum
                ? delayed
                      .chi_delayed[dd * n_groups + static_cast<std::size_t>(g)]
                : (g == 0 ? 1.0 : 0.0);
        out[static_cast<std::size_t>(i) * n_groups + g] += rate * chi;
      }
    }
  }
}

std::vector<double> PrecursorState::decay_rate() const
{
  std::vector<double> out(static_cast<std::size_t>(n_nodes_), 0.0);
  for (int i = 0; i < n_nodes_; ++i) {
    double total = 0.0;
    for (int d = 0; d < n_precursors_; ++d) {
      total += lambda_[static_cast<std::size_t>(d)] * concentration(i, d);
    }
    out[static_cast<std::size_t>(i)] = total;
  }
  return out;
}

}  // namespace openndm
