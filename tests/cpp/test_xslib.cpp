#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "openndm/error.h"
#include "openndm/xslib.h"

using namespace openndm;
using Catch::Approx;

namespace {

void fill_two_group(Composition& c, double absorption_thermal)
{
  c.D = {1.5, 0.4};
  c.absorption = {0.01, absorption_thermal};
  c.nu_fission = {0.0, 0.135};
  c.kappa_fission = {0.0, 0.135};
  c.chi = {1.0, 0.0};
  c.scatter = {0.0, 0.02, 0.0, 0.0};
}

} // namespace

TEST_CASE("removal is absorption plus out-scatter", "[xslib]")
{
  XSLibrary lib(2, 1);
  fill_two_group(lib.composition(0), 0.08);
  lib.finalize();
  // Read through a const reference: the mutable accessor marks the library
  // unfinalized by design, because the caller can invalidate the cache
  // through it.
  const XSLibrary& read = lib;
  REQUIRE(read.composition(0).removal[0] == Approx(0.03));
  REQUIRE(read.composition(0).removal[1] == Approx(0.08));
  REQUIRE(lib.finalized());
}

TEST_CASE("taking a mutable composition reference invalidates the library",
  "[xslib]")
{
  XSLibrary lib(2, 1);
  fill_two_group(lib.composition(0), 0.08);
  lib.finalize();
  REQUIRE(lib.finalized());
  (void)lib.composition(0);
  REQUIRE_FALSE(lib.finalized());
}

TEST_CASE("validation rejects unphysical data", "[xslib]")
{
  SECTION("non-positive diffusion coefficient")
  {
    XSLibrary lib(2, 1);
    fill_two_group(lib.composition(0), 0.08);
    lib.composition(0).D[0] = 0.0;
    REQUIRE_THROWS_AS(lib.finalize(), LibraryError);
  }
  SECTION("fission spectrum that does not sum to one")
  {
    XSLibrary lib(2, 1);
    fill_two_group(lib.composition(0), 0.08);
    lib.composition(0).chi = {0.5, 0.0};
    REQUIRE_THROWS_AS(lib.finalize(), LibraryError);
  }
  SECTION("negative absorption")
  {
    XSLibrary lib(2, 1);
    fill_two_group(lib.composition(0), -0.01);
    REQUIRE_THROWS_AS(lib.finalize(), LibraryError);
  }
}

TEST_CASE("negative scattering warns but does not fail", "[xslib]")
{
  XSLibrary lib(2, 1);
  fill_two_group(lib.composition(0), 0.08);
  lib.composition(0).scatter[2] = -1.0e-6; // thermal up to fast
  std::vector<std::string> warnings;
  REQUIRE_NOTHROW(lib.finalize(&warnings));
  REQUIRE_FALSE(warnings.empty());
  REQUIRE(lib.finalized());
}

TEST_CASE("discontinuity factors default to one", "[xslib]")
{
  XSLibrary lib(2, 1);
  fill_two_group(lib.composition(0), 0.08);
  lib.finalize();
  for (int face = 0; face < 6; ++face) {
    for (int g = 0; g < 2; ++g) {
      REQUIRE(lib.adf_value(0, face, g) == Approx(1.0));
    }
  }
  lib.adf(0).value[0] = 1.07;
  REQUIRE(lib.adf_value(0, 0, 0) == Approx(1.07));
  REQUIRE(lib.adf_value(0, 0, 1) == Approx(1.0));
}

TEST_CASE("multilinear interpolation is exact on linear data", "[xslib]")
{
  XSLibrary lib(1, 1);
  lib.set_axes({{"temperature", {500.0, 1500.0}}, {"boron", {0.0, 2000.0}}});
  const double values[4][2] = {
    {500.0, 0.0}, {500.0, 2000.0}, {1500.0, 0.0}, {1500.0, 2000.0}};
  for (int state = 0; state < 4; ++state) {
    auto& c = lib.composition(0, state);
    c.D = {1.0};
    c.absorption = {0.01 + 1.0e-6 * values[state][0] + 1.0e-6 * values[state][1]};
    c.nu_fission = {0.1};
    c.chi = {1.0};
    c.scatter = {0.0};
  }
  lib.finalize();
  REQUIRE(lib.n_states() == 4);

  const auto mid = lib.interpolate({1000.0, 1000.0});
  REQUIRE(mid.n_states() == 1);
  REQUIRE(mid.composition(0).absorption[0] == Approx(0.012));
  REQUIRE(mid.finalized());

  // Reproducing the corners exactly is the check that catches an index
  // ordering mismatch between the axes and the flat state array.
  for (int state = 0; state < 4; ++state) {
    const auto corner =
      lib.interpolate({values[state][0], values[state][1]});
    const double expected =
      0.01 + 1.0e-6 * values[state][0] + 1.0e-6 * values[state][1];
    REQUIRE(corner.composition(0).absorption[0] == Approx(expected));
  }
}

TEST_CASE("extrapolation policy is honoured", "[xslib]")
{
  XSLibrary lib(1, 1);
  lib.set_axes({{"boron", {0.0, 1000.0}}});
  for (int state = 0; state < 2; ++state) {
    auto& c = lib.composition(0, state);
    c.D = {1.0};
    c.absorption = {0.01 + 1.0e-5 * (state == 0 ? 0.0 : 1000.0)};
    c.chi = {0.0};
    c.scatter = {0.0};
  }
  lib.finalize();

  lib.set_extrapolation(Extrapolation::clamp);
  REQUIRE(lib.interpolate({5000.0}).composition(0).absorption[0] ==
    Approx(0.02));

  lib.set_extrapolation(Extrapolation::linear);
  REQUIRE(lib.interpolate({2000.0}).composition(0).absorption[0] ==
    Approx(0.03));

  lib.set_extrapolation(Extrapolation::error);
  REQUIRE_THROWS_AS(lib.interpolate({5000.0}), InputError);
}

TEST_CASE("a non-monotonic branch axis is rejected", "[xslib]")
{
  XSLibrary lib(1, 1);
  REQUIRE_THROWS_AS(
    lib.set_axes({{"boron", {1000.0, 0.0}}}), LibraryError);
}
