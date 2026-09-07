//! \file geometry.h
//! Node/surface connectivity graph and its Cartesian builder.
//!
//! Solver kernels never index nodes by \f$(i,j,k)\f$; they walk the surface
//! list and the per-node face table declared here (FR-GEO-6, NFR-EXT-1). A
//! hexagonal geometry is therefore a new builder in this file plus a kernel
//! that understands six lateral axes, and nothing else.

#ifndef OPENNDM_GEOMETRY_H
#define OPENNDM_GEOMETRY_H

#include <array>
#include <string>
#include <vector>

#include "openndm/constants.h"

namespace openndm {

//! A single interface between two nodes, or between a node and the outside.
struct Surface {
  int lo = NO_NODE;   //!< node on the negative side of the normal
  int hi = NO_NODE;   //!< node on the positive side of the normal
  double area = 0.0;  //!< surface area [cm^2]
  double h_lo = 0.0;  //!< width of \c lo normal to this surface [cm]
  double h_hi = 0.0;  //!< width of \c hi normal to this surface [cm]
  int axis = 0;       //!< transverse-integration axis this surface belongs to
  BoundaryType bc = BoundaryType::interior;
  int albedo_id = -1; //!< row of Geometry::albedos() when \c bc is \c albedo

  bool is_boundary() const { return bc != BoundaryType::interior; }

  //! Index of the single adjacent node on a boundary surface.
  int boundary_node() const { return lo == NO_NODE ? hi : lo; }

  //! True when the domain lies on the negative side, i.e. the outward normal
  //! of the adjacent node points along \c -axis.
  bool boundary_is_lo_side() const { return lo == NO_NODE; }
};

//! One computational node (homogenised region) of the core.
struct Node {
  double volume = 0.0;      //!< [cm^3]
  int composition = 0;      //!< row of the cross section library
  std::array<double, 3> width {0.0, 0.0, 0.0}; //!< extent along each axis [cm]
  //! Surface index per (axis, side); side 0 is the low face, 1 the high face.
  //! Sized \c 2*n_axes by the builder.
  std::array<int, 6> face {{-1, -1, -1, -1, -1, -1}};
  //! Original lattice position, retained for reporting and for the loading
  //! pattern API (FR-OPT-7). Meaningless for non-Cartesian builders.
  std::array<int, 3> ijk {{0, 0, 0}};
};

//! Structured description used by the Cartesian builder (FR-GEO-1..5).
struct CartesianSpec {
  std::vector<double> dx; //!< node widths along x [cm]
  std::vector<double> dy; //!< node widths along y [cm]
  std::vector<double> dz; //!< node widths along z [cm]
  //! Composition index per (k, j, i), C order, with COMP_INACTIVE marking a
  //! position outside the core (FR-GEO-2).
  std::vector<int> composition;
  //! Boundary condition per face in the order -x, +x, -y, +y, -z, +z.
  std::array<BoundaryType, 6> bc {{BoundaryType::vacuum, BoundaryType::vacuum,
    BoundaryType::vacuum, BoundaryType::vacuum, BoundaryType::vacuum,
    BoundaryType::vacuum}};
  //! Per-face albedo \f$\beta_g = J^-/J^+\f$; only read where \c bc is
  //! \c albedo. Each entry is either empty or has one value per group.
  std::array<std::vector<double>, 6> albedo;
  //! Condition applied to a face whose neighbour is an inactive position in
  //! the interior of the mesh, as opposed to a face on the mesh edge.
  //!
  //! The distinction matters on a symmetric core map: the mesh edges carry the
  //! symmetry conditions, while a face looking at an out-of-core position is a
  //! real outer boundary and must not inherit a reflective symmetry condition.
  BoundaryType inactive_bc = BoundaryType::vacuum;
  std::vector<double> inactive_albedo;

  int nx() const { return static_cast<int>(dx.size()); }
  int ny() const { return static_cast<int>(dy.size()); }
  int nz() const { return static_cast<int>(dz.size()); }
};

//! Immutable node/surface graph handed to the solver.
class Geometry {
public:
  //! Build a Cartesian grid, dropping COMP_INACTIVE positions and applying
  //! the requested face boundary conditions to every exposed surface.
  //!
  //! \throws InputError on an empty mesh, a size mismatch, a non-positive
  //!         node width, or a fully inactive core map.
  static Geometry from_cartesian(const CartesianSpec& spec);

  int n_nodes() const { return static_cast<int>(nodes_.size()); }
  int n_surfaces() const { return static_cast<int>(surfaces_.size()); }
  int n_axes() const { return n_axes_; }

  const std::vector<Node>& nodes() const { return nodes_; }
  const std::vector<Surface>& surfaces() const { return surfaces_; }

  //! Albedo table, one row per distinct boundary spec, each of length \c n_groups.
  const std::vector<std::vector<double>>& albedos() const { return albedos_; }

  double total_volume() const;

  //! Shape of the originating structured grid, for reshaping results back to
  //! (nz, ny, nx). Empty when the geometry was not built from a lattice.
  const std::array<int, 3>& lattice_shape() const { return shape_; }

  //! Map from flat lattice index (k*ny*nx + j*nx + i) to node index, or
  //! NO_NODE for an inactive position.
  const std::vector<int>& lattice_to_node() const { return lattice_to_node_; }

  //! Mutate the composition of a node in place (FR-OPT-7). Cheap enough to sit
  //! inside a loading pattern optimisation loop; no surfaces are rebuilt.
  void set_composition(int node, int composition);

  //! Number of compositions referenced by the map, i.e. max index + 1.
  int n_compositions() const;

private:
  Geometry() = default;

  std::vector<Node> nodes_;
  std::vector<Surface> surfaces_;
  std::vector<std::vector<double>> albedos_;
  std::vector<int> lattice_to_node_;
  std::array<int, 3> shape_ {{0, 0, 0}};
  int n_axes_ = 3;
};

} // namespace openndm

#endif // OPENNDM_GEOMETRY_H
