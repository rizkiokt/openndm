#include "openndm/geometry.h"

#include <algorithm>
#include <numeric>

#include "openndm/error.h"

namespace openndm {

namespace {

//! Register an albedo row, reusing an identical existing one so that the
//! common case of a single albedo spec per face does not grow the table.
int intern_albedo(
    std::vector<std::vector<double>>& table, const std::vector<double>& values)
{
  for (std::size_t i = 0; i < table.size(); ++i) {
    if (table[i] == values) return static_cast<int>(i);
  }
  table.push_back(values);
  return static_cast<int>(table.size()) - 1;
}

}  // namespace

Geometry Geometry::from_cartesian(const CartesianSpec& spec)
{
  const int nx = spec.nx();
  const int ny = spec.ny();
  const int nz = spec.nz();

  if (nx == 0 || ny == 0 || nz == 0) {
    throw InputError("Cartesian mesh must have at least one node per axis");
  }
  const std::size_t expected = static_cast<std::size_t>(nx) * ny * nz;
  if (spec.composition.size() != expected) {
    throw InputError("composition map has " +
                     std::to_string(spec.composition.size()) +
                     " entries but the mesh has " + std::to_string(expected));
  }
  for (const auto* d : {&spec.dx, &spec.dy, &spec.dz}) {
    for (double w : *d) {
      if (!(w > 0.0)) throw InputError("node widths must be positive");
    }
  }

  Geometry g;
  g.n_axes_ = 3;
  g.shape_ = {nz, ny, nx};
  g.lattice_to_node_.assign(expected, NO_NODE);

  // Pass 1: create a node for every active lattice position.
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const std::size_t flat =
            (static_cast<std::size_t>(k) * ny + j) * nx + i;
        const int comp = spec.composition[flat];
        if (comp == COMP_INACTIVE) continue;
        if (comp < 0) {
          throw InputError("negative composition index " +
                           std::to_string(comp) +
                           " is not the inactive marker");
        }
        Node n;
        n.composition = comp;
        n.width = {spec.dx[i], spec.dy[j], spec.dz[k]};
        n.volume = n.width[0] * n.width[1] * n.width[2];
        n.ijk = {i, j, k};
        g.lattice_to_node_[flat] = static_cast<int>(g.nodes_.size());
        g.nodes_.push_back(n);
      }
    }
  }
  if (g.nodes_.empty()) {
    throw InputError("core map contains no active positions");
  }

  // Pass 2: walk every axis and create one surface per node face. Sweeping
  // from the low side means each interior interface is visited exactly once,
  // and a face whose neighbour is missing or inactive becomes a boundary.
  const std::array<int, 3> extent{nx, ny, nz};
  for (int axis = 0; axis < 3; ++axis) {
    const int stride_i = (axis == 0) ? 1 : 0;
    const int stride_j = (axis == 1) ? 1 : 0;
    const int stride_k = (axis == 2) ? 1 : 0;

    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
          const std::size_t flat =
              (static_cast<std::size_t>(k) * ny + j) * nx + i;
          const int node = g.lattice_to_node_[flat];
          if (node == NO_NODE) continue;

          const std::array<int, 3> ijk{i, j, k};

          // Low face: a surface is created only when the low neighbour is
          // absent, otherwise it was already created by that neighbour.
          int lo_neighbour = NO_NODE;
          if (ijk[axis] > 0) {
            const std::size_t nb =
                (static_cast<std::size_t>(k - stride_k) * ny + (j - stride_j)) *
                    nx +
                (i - stride_i);
            lo_neighbour = g.lattice_to_node_[nb];
          }
          if (lo_neighbour == NO_NODE) {
            Surface s;
            s.lo = NO_NODE;
            s.hi = node;
            s.axis = axis;
            s.h_hi = g.nodes_[node].width[axis];
            s.area = g.nodes_[node].volume / s.h_hi;
            const int face = 2 * axis;  // -x, -y, -z
            const bool mesh_edge = (ijk[axis] == 0);
            s.bc = mesh_edge ? spec.bc[face] : spec.inactive_bc;
            if (s.bc == BoundaryType::albedo) {
              const std::vector<double>& alb =
                  mesh_edge ? spec.albedo[face] : spec.inactive_albedo;
              if (alb.empty()) {
                throw InputError("albedo boundary requested on face " +
                                 std::to_string(face) +
                                 " but no albedo values were given");
              }
              s.albedo_id = intern_albedo(g.albedos_, alb);
            }
            g.nodes_[node].face[2 * axis + 0] =
                static_cast<int>(g.surfaces_.size());
            g.surfaces_.push_back(s);
          }

          // High face: interior when the high neighbour exists.
          int hi_neighbour = NO_NODE;
          if (ijk[axis] + 1 < extent[axis]) {
            const std::size_t nb =
                (static_cast<std::size_t>(k + stride_k) * ny + (j + stride_j)) *
                    nx +
                (i + stride_i);
            hi_neighbour = g.lattice_to_node_[nb];
          }
          Surface s;
          s.lo = node;
          s.hi = hi_neighbour;
          s.axis = axis;
          s.h_lo = g.nodes_[node].width[axis];
          s.area = g.nodes_[node].volume / s.h_lo;
          if (hi_neighbour != NO_NODE) {
            s.bc = BoundaryType::interior;
            s.h_hi = g.nodes_[hi_neighbour].width[axis];
          } else {
            const int face = 2 * axis + 1;  // +x, +y, +z
            const bool mesh_edge = (ijk[axis] + 1 == extent[axis]);
            s.bc = mesh_edge ? spec.bc[face] : spec.inactive_bc;
            if (s.bc == BoundaryType::albedo) {
              const std::vector<double>& alb =
                  mesh_edge ? spec.albedo[face] : spec.inactive_albedo;
              if (alb.empty()) {
                throw InputError("albedo boundary requested on face " +
                                 std::to_string(face) +
                                 " but no albedo values were given");
              }
              s.albedo_id = intern_albedo(g.albedos_, alb);
            }
          }
          const int sid = static_cast<int>(g.surfaces_.size());
          g.nodes_[node].face[2 * axis + 1] = sid;
          if (hi_neighbour != NO_NODE) {
            g.nodes_[hi_neighbour].face[2 * axis + 0] = sid;
          }
          g.surfaces_.push_back(s);
        }
      }
    }
  }

  return g;
}

double Geometry::total_volume() const
{
  double v = 0.0;
  for (const auto& n : nodes_) v += n.volume;
  return v;
}

void Geometry::set_composition(int node, int composition)
{
  if (node < 0 || node >= n_nodes()) {
    throw InputError("node index out of range");
  }
  if (composition < 0) {
    throw InputError("composition index must be non-negative");
  }
  nodes_[node].composition = composition;
}

int Geometry::n_compositions() const
{
  int m = -1;
  for (const auto& n : nodes_) m = std::max(m, n.composition);
  return m + 1;
}

}  // namespace openndm
