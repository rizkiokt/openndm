#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "openndm/error.h"
#include "openndm/geometry.h"

using namespace openndm;
using Catch::Approx;

namespace {

CartesianSpec uniform(int nx, int ny, int nz, double h)
{
  CartesianSpec spec;
  spec.dx.assign(nx, h);
  spec.dy.assign(ny, h);
  spec.dz.assign(nz, h);
  spec.composition.assign(
    static_cast<std::size_t>(nx) * ny * nz, 0);
  return spec;
}

} // namespace

TEST_CASE("every node face is wired to exactly one surface", "[geometry]")
{
  const auto g = Geometry::from_cartesian(uniform(3, 4, 5, 10.0));
  REQUIRE(g.n_nodes() == 60);
  for (const auto& node : g.nodes()) {
    for (int face = 0; face < 6; ++face) {
      REQUIRE(node.face[face] >= 0);
      REQUIRE(node.face[face] < g.n_surfaces());
    }
  }
}

TEST_CASE("surface adjacency is consistent in both directions", "[geometry]")
{
  const auto g = Geometry::from_cartesian(uniform(3, 3, 3, 10.0));
  for (int s = 0; s < g.n_surfaces(); ++s) {
    const auto& surf = g.surfaces()[static_cast<std::size_t>(s)];
    if (surf.bc == BoundaryType::interior) {
      const auto& lo = g.nodes()[static_cast<std::size_t>(surf.lo)];
      const auto& hi = g.nodes()[static_cast<std::size_t>(surf.hi)];
      REQUIRE(lo.face[2 * surf.axis + 1] == s);
      REQUIRE(hi.face[2 * surf.axis + 0] == s);
      REQUIRE(surf.h_lo == Approx(lo.width[surf.axis]));
      REQUIRE(surf.h_hi == Approx(hi.width[surf.axis]));
    } else {
      REQUIRE((surf.lo == NO_NODE) != (surf.hi == NO_NODE));
    }
  }
}

TEST_CASE("surface areas sum to the transverse cross section", "[geometry]")
{
  const auto g = Geometry::from_cartesian(uniform(2, 3, 4, 10.0));
  double area_x = 0.0;
  for (const auto& surf : g.surfaces()) {
    // Count only the low boundary faces normal to x.
    if (surf.axis == 0 && surf.bc != BoundaryType::interior &&
      surf.boundary_is_lo_side()) {
      area_x += surf.area;
    }
  }
  REQUIRE(area_x == Approx(3 * 10.0 * 4 * 10.0));
  REQUIRE(g.total_volume() == Approx(2 * 3 * 4 * 1000.0));
}

TEST_CASE("non-uniform widths give the right volumes", "[geometry]")
{
  CartesianSpec spec;
  spec.dx = {5.0, 15.0};
  spec.dy = {10.0};
  spec.dz = {2.0};
  spec.composition = {0, 0};
  const auto g = Geometry::from_cartesian(spec);
  REQUIRE(g.nodes()[0].volume == Approx(100.0));
  REQUIRE(g.nodes()[1].volume == Approx(300.0));
}

TEST_CASE("an out-of-core neighbour uses the inactive condition", "[geometry]")
{
  CartesianSpec spec = uniform(2, 2, 1, 10.0);
  spec.composition[3] = COMP_INACTIVE; // the (1,1) position
  spec.bc.fill(BoundaryType::reflective);
  spec.inactive_bc = BoundaryType::vacuum;
  const auto g = Geometry::from_cartesian(spec);

  int vacuum = 0;
  for (const auto& surf : g.surfaces()) {
    if (surf.bc == BoundaryType::vacuum) ++vacuum;
  }
  REQUIRE(g.n_nodes() == 3);
  REQUIRE(vacuum == 2); // the two faces looking at the dropped position
}

TEST_CASE("bad geometry input is rejected", "[geometry]")
{
  CartesianSpec spec = uniform(2, 2, 1, 10.0);
  spec.composition.pop_back();
  REQUIRE_THROWS_AS(Geometry::from_cartesian(spec), InputError);

  CartesianSpec negative = uniform(1, 1, 1, 10.0);
  negative.dx[0] = -1.0;
  REQUIRE_THROWS_AS(Geometry::from_cartesian(negative), InputError);

  CartesianSpec empty = uniform(2, 2, 1, 10.0);
  empty.composition.assign(4, COMP_INACTIVE);
  REQUIRE_THROWS_AS(Geometry::from_cartesian(empty), InputError);
}
