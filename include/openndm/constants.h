//! \file constants.h
//! Compile-time constants and small enumerations shared across the core.

#ifndef OPENNDM_CONSTANTS_H
#define OPENNDM_CONSTANTS_H

#include <cstddef>

namespace openndm {

//! Maximum number of delayed neutron precursor groups (FR-XS-3).
constexpr int MAX_DELAYED_GROUPS = 8;

//! Sentinel composition index marking an out-of-core lattice position
//! (FR-GEO-2).
constexpr int COMP_INACTIVE = -1;

//! Sentinel node index used on a surface that bounds the problem domain.
constexpr int NO_NODE = -1;

//! Boundary condition kinds available on an exterior surface (FR-GEO-5).
enum class BoundaryType {
  interior,    //!< not a boundary; both sides carry a node
  zero_flux,   //!< \f$\phi_s = 0\f$
  vacuum,      //!< zero incoming partial current (Marshak)
  reflective,  //!< zero net current
  albedo       //!< user supplied \f$\beta = J^-/J^+\f$, per group
};

//! Nodal kernel selected at run time, never at compile time (FR-SOL-8).
enum class KernelType {
  fdm,  //!< finite difference (FR-SOL-1)
  nem,  //!< polynomial nodal, quartic expansion (FR-SOL-2)
  sanm  //!< semi-analytic nodal, the default (FR-SOL-3)
};

//! What the solver is being asked to compute (FR-MODE).
enum class SolveMode {
  forward,      //!< forward static eigenvalue (FR-MODE-1)
  adjoint,      //!< adjoint static eigenvalue (FR-MODE-2)
  fixed_source  //!< fixed external source (FR-MODE-3)
};

}  // namespace openndm

#endif  // OPENNDM_CONSTANTS_H
