#include <memory>

#include "openndm/kernel.h"

namespace openndm {

//! Finite difference kernel (FR-SOL-1).
//!
//! The coupling coefficients built by CmfdSystem::build_coupling() already are
//! the finite difference ones, so this kernel exists to name that choice and to
//! short-circuit the nonlinear iteration. It is the reference against which
//! the nodal kernels are verified on a refined mesh (V-3).
class FdmKernel : public Kernel {
public:
  const char* name() const override { return "fdm"; }
  bool is_finite_difference() const override { return true; }

  //! Write a zero current for every group.
  //!
  //! CmfdSystem skips the nodal update for a finite difference kernel, so
  //! this is never reached from the solver; filling the currents keeps the
  //! contract honest for a caller that invokes the kernel directly.
  void solve(const TwoNodeProblem& p, const XSLibrary&, const std::vector<int>&,
      int n_groups, int, double* current) const override
  {
    for (int g = 0; g < n_groups; ++g) current[g] = 0.0;
    (void)p;
  }
};

std::unique_ptr<Kernel> make_fdm_kernel()
{
  return std::make_unique<FdmKernel>();
}

}  // namespace openndm
