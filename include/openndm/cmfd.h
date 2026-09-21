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

  //! Recompute Dtilde from the current diffusion coefficients, keeping the
  //! nonlinear Dhat a kernel has already converged.
  //!
  //! Dtilde is derived from D, so refreshing the cross sections alone leaves
  //! the leakage operator reading the D it was built with while the removal
  //! operator reads the new one. Inside a time step there is no nonlinear
  //! update to rebuild Dhat, so \c build_coupling() cannot be used there: it
  //! would discard the correction the static solve converged.
  void refresh_coupling();

  //! Refresh cached per-node cross sections after a composition change.
  void refresh_cross_sections();

  //! Assemble the within-group matrices for the current coupling.
  //!
  //! \param k_shift reciprocal Wielandt shift subtracted from the removal
  //!        term; pass 0 for the unshifted operator.
  void assemble(double k_shift);

  //! Run the nonlinear two-node update, refreshing Dhat (FR-SOL-4).
  //! \return the largest relative change in Dhat, as a convergence indicator.
  //! \param external optional node-average external source density,
  //!        node*G + g; null for an eigenvalue solve.
  double nodal_update(const Kernel& kernel, const std::vector<double>& flux,
      double k_eff, const Settings& s,
      const std::vector<double>* external = nullptr);

  //! Fission source per node, \f$\sum_g \nu\Sigma_{f,g}\phi_g V\f$.
  void fission_source(
      const std::vector<double>& flux, std::vector<double>& source) const;

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

  //! Add \f$V/(v_g\,\theta\,\Delta t)\f$ to every diagonal, for a
  //! time-dependent solve (FR-KIN-2).
  //!
  //! Pass a non-positive \c theta_dt to clear it and return to the static
  //! operator. The term is applied by the next assemble().
  void set_time_removal(double theta_dt);

  //! Emission spectrum used in place of chi while a step is being solved.
  //!
  //! The analytic precursor solution makes the delayed source linear in the
  //! new fission source, so its implicit part folds into the fission
  //! spectrum rather than needing an inner iteration over the delayed
  //! source. Empty restores the library spectrum.
  void set_transient_chi(const std::vector<double>& chi);

  //! Source added to every group's right hand side, node*G + g. Empty clears
  //! it. Used for the terms of a time step that are known from the previous
  //! one.
  void set_transient_source(const std::vector<double>& source);

  //! Apply the assembled within-group operators to \c flux, writing
  //! node*G + g. Excludes scattering and fission, which the group sweep
  //! carries on the right hand side.
  void apply_operator(
      const std::vector<double>& flux, std::vector<double>& out) const;

  //! Time removal term for one node and group, or zero when not set.
  double time_removal(int node, int group) const;

  //! Scattering out of \c from into \c to for \c node, volume weighted.
  double scatter(int node, int from, int to) const;

  //! \f$\nu\Sigma_f V\f$ for one node and group.
  double nu_fission(int node, int group) const;

  //! Library emission spectrum for one node and group.
  double chi(int node, int group) const;

  //! Transpose the operator in place for the adjoint solve (FR-MODE-2).
  void set_adjoint(bool adjoint);
  bool adjoint() const { return adjoint_; }

  //! Net current per surface per group from the current flux, evaluated with
  //! the present coupling coefficients.
  void compute_currents(
      const std::vector<double>& flux, std::vector<double>& current) const;

  const std::vector<double>& dtilde() const { return dtilde_; }
  const std::vector<double>& dhat() const { return dhat_; }
  std::vector<double>& dhat() { return dhat_; }

  const Geometry& geometry() const { return geom_; }
  const XSLibrary& library() const { return xs_; }

private:
  void build_pattern();
  //! Boundary coupling coefficient for one exterior surface and group.
  double boundary_coupling(
      const Surface& surf, int group, double D, double adf) const;

  const Geometry& geom_;
  const XSLibrary& xs_;
  int n_groups_;
  bool adjoint_ = false;
  //! True when any composition transfers neutrons to a lower group index.
  bool has_upscatter_ = false;
  double max_inv_shift_ = 0.0;

  //! Cached per node/group data, flattened as node*G + g.
  std::vector<double> removal_;     //!< \f$\Sigma_r V\f$
  std::vector<double> nu_fission_;  //!< \f$\nu\Sigma_f V\f$
  std::vector<double> chi_;
  std::vector<double> diffusion_;
  //! Scattering, node*G*G + from*G + to, already multiplied by volume.
  std::vector<double> scatter_;

  //! Time-dependent diagonal, node*G + g. Empty for a static solve.
  std::vector<double> time_removal_;
  //! Emission spectrum overriding chi_ for a transient step. Empty when off.
  std::vector<double> transient_chi_;
  //! Extra right hand side for a transient step, node*G + g. Empty when off.
  std::vector<double> transient_source_;

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

}  // namespace openndm

#endif  // OPENNDM_CMFD_H
