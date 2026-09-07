#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "kernel_common.h"

using namespace openndm::detail;
using Catch::Approx;

namespace {

//! Average of the quadratic leakage fit over the neighbour node, which is what
//! the fit is constructed to reproduce.
double neighbour_average(
    double l_self, double l1, double l2, double lo, double hi)
{
  // integral of (l_self + l1*P1 + l2*P2) over [lo, hi], divided by the width
  const auto antiderivative = [&](double x) {
    return l_self * x + l1 * x * x + l2 * (2.0 * x * x * x - 0.5 * x);
  };
  return (antiderivative(hi) - antiderivative(lo)) / (hi - lo);
}

}  // namespace

TEST_CASE(
    "the transverse leakage fit reproduces all three averages", "[kernel]")
{
  SECTION("uniform mesh")
  {
    const double prev = 3.0, self = 5.0, next = 4.0;
    const auto fit = leakage_fit(prev, self, next, 20.0, 20.0, 20.0);
    REQUIRE(
        neighbour_average(self, fit[0], fit[1], -1.5, -0.5) == Approx(prev));
    REQUIRE(neighbour_average(self, fit[0], fit[1], 0.5, 1.5) == Approx(next));
    REQUIRE(neighbour_average(self, fit[0], fit[1], -0.5, 0.5) == Approx(self));
  }
  SECTION("non-uniform mesh")
  {
    const double prev = -2.0, self = 1.0, next = 7.0;
    const double h_prev = 10.0, h_self = 20.0, h_next = 40.0;
    const auto fit = leakage_fit(prev, self, next, h_prev, h_self, h_next);
    const double a = h_prev / h_self;
    const double b = h_next / h_self;
    REQUIRE(neighbour_average(self, fit[0], fit[1], -0.5 - a, -0.5) ==
            Approx(prev));
    REQUIRE(
        neighbour_average(self, fit[0], fit[1], 0.5, 0.5 + b) == Approx(next));
  }
}

TEST_CASE("a flat leakage produces a flat fit", "[kernel]")
{
  const auto fit = leakage_fit(2.0, 2.0, 2.0, 20.0, 20.0, 20.0);
  REQUIRE(fit[0] == Approx(0.0).margin(1e-14));
  REQUIRE(fit[1] == Approx(0.0).margin(1e-14));
}

TEST_CASE("the analytic basis agrees with direct quadrature", "[kernel]")
{
  const auto quadrature = [](auto f) {
    // Composite Simpson over [-1/2, 1/2] with plenty of points.
    const int n = 20000;
    const double h = 1.0 / n;
    double total = f(-0.5) + f(0.5);
    for (int i = 1; i < n; ++i) {
      const double x = -0.5 + i * h;
      total += (i % 2 == 0 ? 2.0 : 4.0) * f(x);
    }
    return total * h / 3.0;
  };

  SECTION("hyperbolic branch")
  {
    const double k2 = 6.0;
    const auto basis = AnalyticBasis::make(k2);
    const double k = std::sqrt(k2);
    REQUIRE(basis.hyperbolic);
    REQUIRE(basis.even_avg ==
            Approx(quadrature([k](double x) { return std::cosh(k * x); })));
    REQUIRE(basis.odd_m1 == Approx(quadrature([k](double x) {
      return std::sinh(k * x) * 2.0 * x;
    })));
    REQUIRE(basis.even_m2 == Approx(quadrature([k](double x) {
      return std::cosh(k * x) * (6.0 * x * x - 0.5);
    })));
    REQUIRE(basis.odd_face == Approx(std::sinh(0.5 * k)));
    REQUIRE(basis.odd_dface == Approx(k * std::cosh(0.5 * k)));
  }

  SECTION("trigonometric branch, reached when in-group fission exceeds removal")
  {
    const double k2 = -3.0;
    const auto basis = AnalyticBasis::make(k2);
    const double w = std::sqrt(-k2);
    REQUIRE_FALSE(basis.hyperbolic);
    REQUIRE(basis.even_avg ==
            Approx(quadrature([w](double x) { return std::cos(w * x); })));
    REQUIRE(basis.odd_m1 == Approx(quadrature([w](double x) {
      return std::sin(w * x) * 2.0 * x;
    })));
    REQUIRE(basis.even_m2 == Approx(quadrature([w](double x) {
      return std::cos(w * x) * (6.0 * x * x - 0.5);
    })));
    REQUIRE(basis.odd_face == Approx(std::sin(0.5 * w)));
    REQUIRE(basis.even_dface == Approx(-w * std::sin(0.5 * w)));
  }
}

TEST_CASE("the analytic basis stays finite as kappa vanishes", "[kernel]")
{
  const auto basis = AnalyticBasis::make(0.0);
  REQUIRE(std::isfinite(basis.even_avg));
  REQUIRE(std::isfinite(basis.odd_m1));
  REQUIRE(std::isfinite(basis.even_m2));
  REQUIRE(basis.even_avg == Approx(1.0).epsilon(1e-6));
}

TEST_CASE(
    "the dense solver handles pivoting and reports singularity", "[kernel]")
{
  SECTION("a system needing a row swap")
  {
    double a[4] = {0.0, 2.0, 3.0, 1.0};
    double b[2] = {4.0, 5.0};
    REQUIRE(solve_dense(a, b, 2));
    // 0*x + 2*y = 4 and 3*x + 1*y = 5 give y = 2, x = 1.
    REQUIRE(b[0] == Approx(1.0));
    REQUIRE(b[1] == Approx(2.0));
  }
  SECTION("a singular system")
  {
    double a[4] = {1.0, 2.0, 2.0, 4.0};
    double b[2] = {1.0, 2.0};
    REQUIRE_FALSE(solve_dense(a, b, 2));
  }
}
