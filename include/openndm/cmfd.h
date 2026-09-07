//! \file cmfd.h
//! Coarse mesh finite difference system and the nonlinear two-node iteration.

#ifndef OPENNDM_CMFD_H
#define OPENNDM_CMFD_H

#include <memory>
#include <vector>

#include "openndm/geometry.h"
#include "openndm/kernel.h"
#include "openndm/matrix.h"
#include "openndm/settings.h"
#include "openndm/xslib.h"

namespace openndm {

//! The coarse-mesh system solved by the outer eigenvalue iteration.
//!
//! Holds the coupling coefficients, the assembled within-group matrices and
//! the scratch space for the nonlinear update. One instance is reused across
//! solves so that a perturbed re-solve can warm start (FR-OPT-3).
class CmfdSystem {
public:
  CmfdSystem(const Geometry& geom, const XSLibrary& xs);

  int n_nodes() const { return geom_.n_nodes(); }
  int n_groups() const { return n_groups_; }

  //! Recompute the finite difference coupling coefficients and reset Dhat to
  //! the value implied by the discontinuity factors alone.
  void build_coupling();

  //! Refresh cached per-node cross sections after a composition change.
  void refresh_cross_sections();

  //! Assemble the within-group matrices for the current coupling.
  //!
  //! \param k_shift reciprocal Wielandt shift subtracted from the removal
  //!        term; pass 0 for the unshifted operator.
  void assemble(double k_shift);

  //! Run the nonlinear two-node update, refreshing Dhat (FR-SOL-4).
  //! \return the largest relative change in Dhat, as a convergence indicator.
  double nodal_update(const Kernel& kernel, const std::vector<double>& flux,
    double k_eff, const Settings& s);

  //! Fission source per node, \f$\sum_g \nu\Sigma_{f,g}\phi_g V\f$.
  void fission_source(const std::vector<double>& flux,
    std::vector<double>& source) const;

  //! One Gauss-Seidel pass over energy groups, solving each within-group
  //! system with preconditioned BiCGSTAB.
  int solve_groups(const std::vector<double>& fission_src, double k_eff,
    double inv_k_shift, std::vector<double>& flux, const Settings& s);

  //! Solve with a user supplied external source instead of fission
  //! (FR-MODE-3).
  int solve_fixed_source(const std::vector<double>& external,
    std::vector<double>& flux, const Settings& s);

  //! Largest reciprocal Wielandt shift that leaves every shifted diagonal
  //! entry safely positive.
  //!
  //! The shift subtracts \f$\chi_g\nu\Sigma_{f,g}/k_s\f$ from the removal
  //! term. Where the emission spectrum and the production cross section share
  //! a group, as they do in a one-group model, a tight shift cancels removal
  //! outright and leaves a singular operator. Capping the shift here keeps the
  //! acceleration without ever handing the inner solver a matrix it cannot
  //! factor.
  double max_reciprocal_shift() const { return max_inv_shift_; }

  //! Transpose the operator in place for the adjoint solve (FR-MODE-2).
  void set_adjoint(bool adjoint);
  bool adjoint() const { return adjoint_; }

  //! Net current per surface per group from the current flux, evaluated with
  //! the present coupling coefficients.
  void compute_currents(const std::vector<double>& flux,
    std::vector<double>& current) const;

  const std::vector<double>& dtilde() const { return dtilde_; }
  const std::vector<double>& dhat() const { return dhat_; }
  std::vector<double>& dhat() { return dhat_; }

  const Geometry& geometry() const { return geom_; }
  const XSLibrary& library() const { return xs_; }

private:
  void build_pattern();
  //! Boundary coupling coefficient for one exterior surface and group.
  double boundary_coupling(const Surface& surf, int group, double D,
    double adf) const;

  const Geometry& geom_;
  const XSLibrary& xs_;
  int n_groups_;
  bool adjoint_ = false;
  //! True when any composition transfers neutrons to a lower group index.
  bool has_upscatter_ = false;
  double max_inv_shift_ = 0.0;

  //! Cached per node/group data, flattened as node*G + g.
  std::vector<double> removal_;    //!< \f$\Sigma_r V\f$
  std::vector<double> nu_fission_; //!< \f$\nu\Sigma_f V\f$
  std::vector<double> chi_;
  std::vector<double> diffusion_;
  //! Scattering, node*G*G + from*G + to, already multiplied by volume.
  std::vector<double> scatter_;

  //! Coupling coefficients, surface*G + g.
  std::vector<double> dtilde_;
  std::vector<double> dhat_;

  SparsePattern pattern_;
  std::vector<GroupMatrix> matrices_;
  std::vector<Ilu0> precond_;

  //! Scratch buffers reused across iterations to keep the solve allocation
  //! free once warmed up (DP-4).
  mutable std::vector<double> rhs_;
  mutable std::vector<double> group_flux_;
  mutable std::vector<double> current_;
  mutable std::vector<double> leakage_;
  mutable std::vector<double> prev_flux_;
};

} // namespace openndm

#endif // OPENNDM_CMFD_H
