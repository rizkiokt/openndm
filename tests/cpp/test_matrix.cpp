#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <numeric>
#include <random>

#include "openndm/error.h"
#include "openndm/matrix.h"

using namespace openndm;
using Catch::Approx;

namespace {

//! Tridiagonal 1D Laplacian plus a shift, a stand-in for a within-group
//! CMFD row structure that has an exactly known solution.
SparsePattern line_pattern(int n)
{
  std::vector<std::vector<int>> neighbours(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    if (i > 0) neighbours[static_cast<std::size_t>(i)].push_back(i - 1);
    if (i + 1 < n) neighbours[static_cast<std::size_t>(i)].push_back(i + 1);
  }
  return SparsePattern(n, neighbours);
}

GroupMatrix laplacian(const SparsePattern& pattern, double shift)
{
  GroupMatrix A(&pattern);
  const int n = pattern.n_rows();
  for (int i = 0; i < n; ++i) {
    A.add_diagonal(i, 2.0 + shift);
    if (i > 0) A.add(i, i - 1, -1.0);
    if (i + 1 < n) A.add(i, i + 1, -1.0);
  }
  return A;
}

}  // namespace

TEST_CASE("deterministic reductions do not depend on ordering", "[matrix]")
{
  // Values spanning many magnitudes make plain summation order dependent.
  std::vector<double> x(10000);
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> dist(-1.0e6, 1.0e6);
  for (auto& v : x) v = dist(rng);
  std::vector<double> y = x;

  const double first = deterministic_dot(x, y);
  for (int repeat = 0; repeat < 5; ++repeat) {
    REQUIRE(deterministic_dot(x, y) == first);
  }
  REQUIRE(deterministic_norm(x) == Approx(std::sqrt(first)).epsilon(1e-12));
}

TEST_CASE("the sparsity pattern stores columns in ascending order", "[matrix]")
{
  const auto pattern = line_pattern(5);
  const auto& rowptr = pattern.rowptr();
  const auto& colind = pattern.colind();
  for (int r = 0; r < pattern.n_rows(); ++r) {
    for (int p = rowptr[r] + 1; p < rowptr[r + 1]; ++p) {
      REQUIRE(colind[p] > colind[p - 1]);
    }
    REQUIRE(colind[pattern.diag_pos()[r]] == r);
  }
}

TEST_CASE("writing outside the pattern is an error, not silent", "[matrix]")
{
  const auto pattern = line_pattern(5);
  GroupMatrix A(&pattern);
  REQUIRE_THROWS_AS(A.add(0, 4, 1.0), InputError);
}

TEST_CASE("BiCGSTAB with ILU0 solves a shifted Laplacian", "[matrix]")
{
  const int n = 200;
  const auto pattern = line_pattern(n);
  const auto A = laplacian(pattern, 0.1);

  std::vector<double> exact(n);
  std::iota(exact.begin(), exact.end(), 1.0);
  std::vector<double> b;
  A.multiply(exact, b);

  Ilu0 precond;
  precond.factor(A);
  REQUIRE(precond.valid());

  std::vector<double> x(n, 0.0);
  const auto result = bicgstab(A, b, x, precond, 1.0e-12, 500);
  REQUIRE(result.converged);
  for (int i = 0; i < n; ++i) {
    REQUIRE(x[i] == Approx(exact[i]).epsilon(1e-9));
  }
}

TEST_CASE("BiCGSTAB accepts a good initial guess", "[matrix]")
{
  const int n = 50;
  const auto pattern = line_pattern(n);
  const auto A = laplacian(pattern, 1.0);
  std::vector<double> exact(n, 3.0);
  std::vector<double> b;
  A.multiply(exact, b);

  Ilu0 precond;
  precond.factor(A);
  std::vector<double> x = exact;
  const auto result = bicgstab(A, b, x, precond, 1.0e-12, 100);
  REQUIRE(result.converged);
  REQUIRE(result.iterations == 0);  // already converged, no work needed
}
