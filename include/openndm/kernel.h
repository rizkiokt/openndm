//! \file kernel.h
//! Abstract nodal kernel interface for the nonlinear two-node iteration.
//!
//! A kernel's whole job is to answer one question: given the CMFD node-average
//! fluxes, the eigenvalue and the transverse leakage, what is the net current
//! across this interface? The nonlinear iteration then chooses the corrected
//! coupling coefficient \f$\hat{D}\f$ that makes the coarse-mesh system
//! reproduce that current (FR-SOL-4).

#ifndef OPENNDM_KERNEL_H
#define OPENNDM_KERNEL_H

#include <memory>
#include <vector>

#include "openndm/constants.h"

namespace openndm {

class Geometry;
class XSLibrary;
struct Settings;

//! Everything a two-node problem needs, gathered by CmfdSystem.
struct TwoNodeProblem {
  int surface = 0;
  int node_lo = 0;
  int node_hi = 0;
  int axis = 0;
  double h_lo = 0.0;
  double h_hi = 0.0;
  //! Node-average scalar flux from the coarse-mesh solution, [node][group].
  const double* flux_lo = nullptr;
  const double* flux_hi = nullptr;
  //! Discontinuity factors on the shared interface, per group.
  const double* adf_lo = nullptr;
  const double* adf_hi = nullptr;
  //! External source, per group, as three node-average values per node for
  //! the same quadratic fit the transverse leakage uses. Null for an
  //! eigenvalue solve, where there is no external source.
  //!
  //! Omitting this term leaves a fixed-source nodal solve reconstructing the
  //! within-node shape from the scattering and fission sources alone, which
  //! is wrong wherever the external source carries any of the shape.
  const double* src_lo = nullptr;  //!< 3*G: neighbour-, self, neighbour+
  const double* src_hi = nullptr;

  //! Transverse leakage expansion, per group: three node-average values used
  //! for the quadratic fit, for each of the two nodes.
  const double* tl_lo = nullptr;  //!< 3*G: neighbour-, self, neighbour+
  const double* tl_hi = nullptr;
  //! Widths of the neighbouring nodes along the axis, for the leakage fit.
  double h_lo_prev = 0.0;
  double h_hi_next = 0.0;
  //! Net current across the interface from the coarse-mesh solution, per
  //! group; used by kernels that need outer-face partial currents.
  const double* cmfd_current_lo = nullptr;  //!< at the outer face of node_lo
  const double* cmfd_current_hi = nullptr;  //!< at the outer face of node_hi
  const double* cmfd_surface_flux_lo = nullptr;
  const double* cmfd_surface_flux_hi = nullptr;
  double k_eff = 1.0;
};

//! Interface implemented by FDM, NEM and SANM.
class Kernel {
public:
  virtual ~Kernel() = default;

  //! Human readable name, written into the statepoint metadata.
  virtual const char* name() const = 0;

  //! Solve one two-node problem.
  //!
  //! \param[out] current net current across the interface, per group, positive
  //!             along the surface normal (from \c node_lo to \c node_hi).
  virtual void solve(const TwoNodeProblem& p, const XSLibrary& xs,
      const std::vector<int>& composition, int n_groups, int sweeps,
      double* current) const = 0;

  //! True when the kernel leaves Dhat at its finite difference value, in which
  //! case the nonlinear iteration is skipped entirely.
  virtual bool is_finite_difference() const { return false; }

  static std::unique_ptr<Kernel> create(KernelType type);
};

}  // namespace openndm

#endif  // OPENNDM_KERNEL_H
