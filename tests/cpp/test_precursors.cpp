#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <vector>

#include "openndm/error.h"
#include "openndm/precursors.h"

using namespace openndm;
using Catch::Approx;

namespace {

//! A conventional six-group delayed set for U-235 thermal fission.
DelayedData six_group()
{
  DelayedData d;
  d.beta = {2.1e-4, 1.4e-3, 1.26e-3, 2.52e-3, 7.4e-4, 2.7e-4};
  d.lambda = {0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01};
  return d;
}

DelayedData one_group(double beta, double lambda)
{
  DelayedData d;
  d.beta = {beta};
  d.lambda = {lambda};
  return d;
}

}  // namespace

TEST_CASE("the decay integrals match their closed forms", "[precursors]")
{
  const double lambda = 0.0124;
  const double dt = 0.5;
  REQUIRE(decay_integral_0(lambda, dt) ==
          Approx((1.0 - std::exp(-lambda * dt)) / lambda).epsilon(1e-14));
  REQUIRE(decay_integral_1(lambda, dt) ==
          Approx((dt - (1.0 - std::exp(-lambda * dt)) / lambda) / lambda)
              .epsilon(1e-12));
}

TEST_CASE("the decay integrals stay accurate as lambda*dt goes to zero",
    "[precursors]")
{
  // The closed forms are (1-exp(-x))/lambda and (dt-I0)/lambda, both of which
  // are catastrophic cancellations at small x: I1 loses every significant
  // digit long before x underflows. The limits are dt and dt^2/2.
  const double dt = 1.0e-3;
  for (const double lambda : {1.0e-6, 1.0e-9, 1.0e-12, 0.0}) {
    const double i0 = decay_integral_0(lambda, dt);
    const double i1 = decay_integral_1(lambda, dt);
    REQUIRE(i0 == Approx(dt).epsilon(1.0e-8));
    REQUIRE(i1 == Approx(0.5 * dt * dt).epsilon(1.0e-8));
    REQUIRE(std::isfinite(i1));
  }
}

TEST_CASE("equilibrium is beta*F/lambda", "[precursors]")
{
  const DelayedData delayed = six_group();
  PrecursorState state(2, delayed);
  state.set_equilibrium({1.0, 3.0});
  for (int d = 0; d < state.n_precursors(); ++d) {
    const auto dd = static_cast<std::size_t>(d);
    REQUIRE(state.concentration(0, d) ==
            Approx(delayed.beta[dd] / delayed.lambda[dd]));
    REQUIRE(state.concentration(1, d) ==
            Approx(3.0 * delayed.beta[dd] / delayed.lambda[dd]));
  }
}

TEST_CASE("equilibrium is a fixed point of the integration", "[precursors]")
{
  // A reactor sitting at steady state must stay there for any step size. If
  // the integration and the equilibrium disagree, every transient starts with
  // a spurious jump that looks like physics.
  const DelayedData delayed = six_group();
  PrecursorState state(1, delayed);
  const std::vector<double> f = {2.5};
  state.set_equilibrium(f);
  const std::vector<double> before = state.concentrations();

  for (const double dt : {1.0e-4, 1.0e-2, 1.0, 100.0}) {
    state.advance(f, f, dt);
    for (std::size_t k = 0; k < before.size(); ++k) {
      REQUIRE(state.concentrations()[k] == Approx(before[k]).epsilon(1e-12));
    }
  }
}

TEST_CASE("a step up in fission source approaches the new equilibrium",
    "[precursors]")
{
  // With F held constant the exact solution is
  //   C(t) = C_eq + (C_0 - C_eq) exp(-lambda t)
  const double beta = 6.5e-3;
  const double lambda = 0.0785;
  const DelayedData delayed = one_group(beta, lambda);
  PrecursorState state(1, delayed);
  state.set_equilibrium({1.0});
  const double c0 = state.concentration(0, 0);
  const double c_eq = beta * 2.0 / lambda;

  const std::vector<double> f = {2.0};
  double t = 0.0;
  const double dt = 0.25;
  for (int step = 0; step < 200; ++step) {
    state.advance(f, f, dt);
    t += dt;
    const double exact = c_eq + (c0 - c_eq) * std::exp(-lambda * t);
    REQUIRE(state.concentration(0, 0) == Approx(exact).epsilon(1e-12));
  }
}

TEST_CASE("a ramp in fission source is integrated exactly", "[precursors]")
{
  // The integration assumes F is linear across the step, so a genuinely
  // linear F must be reproduced to round-off however coarse the step is.
  // For F(t) = a + b t starting from C(0) = 0 the exact solution is
  //   C = beta[ (a/l)(1-E) + (b/l)(t - (1-E)/l) ],  E = exp(-l t)
  const double beta = 5.0e-3;
  const double lambda = 0.3;
  const double a = 1.0;
  const double b = 0.4;
  const DelayedData delayed = one_group(beta, lambda);

  for (const double dt : {0.1, 1.0, 5.0}) {
    PrecursorState state(1, delayed);
    double t = 0.0;
    for (int step = 0; step < 4; ++step) {
      state.advance({a + b * t}, {a + b * (t + dt)}, dt);
      t += dt;
    }
    const double e = std::exp(-lambda * t);
    const double exact = beta * ((a / lambda) * (1.0 - e) +
                                    (b / lambda) * (t - (1.0 - e) / lambda));
    REQUIRE(state.concentration(0, 0) == Approx(exact).epsilon(1e-12));
  }
}

TEST_CASE("decaying with no source is a pure exponential", "[precursors]")
{
  const double lambda = 0.111;
  PrecursorState state(1, one_group(1.0e-3, lambda));
  state.set_equilibrium({1.0});
  const double c0 = state.concentration(0, 0);
  const std::vector<double> none = {0.0};
  for (int step = 0; step < 50; ++step) state.advance(none, none, 0.2);
  REQUIRE(state.concentration(0, 0) ==
          Approx(c0 * std::exp(-lambda * 10.0)).epsilon(1e-12));
}

TEST_CASE("the delayed source splits over the delayed spectrum", "[precursors]")
{
  DelayedData delayed = one_group(1.0e-3, 0.5);
  delayed.chi_delayed = {0.3, 0.7};
  PrecursorState state(2, delayed);
  state.set_equilibrium({1.0, 2.0});

  std::vector<double> source;
  state.delayed_source(delayed, 2, source);
  const double rate0 = 0.5 * state.concentration(0, 0);
  REQUIRE(source[0] == Approx(0.3 * rate0));
  REQUIRE(source[1] == Approx(0.7 * rate0));
  REQUIRE(source[2] == Approx(2.0 * source[0]));

  const auto rates = state.decay_rate();
  REQUIRE(rates[0] == Approx(rate0));
  REQUIRE(source[0] + source[1] == Approx(rates[0]));
}

TEST_CASE("bad precursor data is rejected", "[precursors]")
{
  REQUIRE_THROWS_AS(PrecursorState(1, DelayedData{}), InputError);
  REQUIRE_THROWS_AS(PrecursorState(0, six_group()), InputError);
  REQUIRE_THROWS_AS(PrecursorState(1, one_group(1.0e-3, 0.0)), InputError);
  REQUIRE_THROWS_AS(PrecursorState(1, one_group(-1.0e-3, 0.1)), InputError);

  PrecursorState state(1, six_group());
  REQUIRE_THROWS_AS(state.advance({1.0}, {1.0}, 0.0), InputError);
  REQUIRE_THROWS_AS(state.advance({1.0, 2.0}, {1.0}, 1.0), InputError);
  REQUIRE_THROWS_AS(state.set_equilibrium({1.0, 2.0}), InputError);
}
