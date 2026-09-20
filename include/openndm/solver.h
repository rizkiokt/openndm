//! \file solver.h
//! Outer eigenvalue iteration and the top level solve entry points.

#ifndef OPENNDM_SOLVER_H
#define OPENNDM_SOLVER_H

#include <memory>
#include <string>
#include <vector>

#include "openndm/cmfd.h"
#include "openndm/geometry.h"
#include "openndm/precursors.h"
#include "openndm/settings.h"
#include "openndm/xslib.h"

namespace openndm {

//! Everything a completed solve produces (FR-OUT-3).
struct Result {
  double k_eff = 0.0;
  //! Scalar flux, node*G + g.
  std::vector<double> flux;
  //! Node power [W], normalised so that the total equals Settings power when
  //! set, otherwise to a core average of 1.
  std::vector<double> power;
  std::vector<IterationRecord> history;
  bool converged = false;
  int outer_iterations = 0;
  double runtime_seconds = 0.0;
  std::string kernel;
};

//! One completed time step (FR-KIN-6).
struct TransientRecord {
  double time = 0.0;
  double dt = 0.0;
  //! Total fission power, in the units of the library's kappa-fission. Not
  //! renormalised, because the whole point of a transient is that it moves.
  double total_power = 0.0;
  double peak_power = 0.0;
  //! Iterations over the implicit fission source within this step.
  int iterations = 0;
  int inner_iterations = 0;
  bool converged = false;
};

//! Drives the nonlinear nodal iteration to a converged eigenvalue.
//!
//! The object owns the CMFD system, so a caller can perturb the model and call
//! solve() again to warm start from the previous solution (FR-OPT-3).
class Solver {
public:
  Solver(const Geometry& geom, const XSLibrary& xs);
  ~Solver();

  //! Forward or adjoint static eigenvalue (FR-MODE-1, FR-MODE-2).
  Result solve(const Settings& settings);

  //! Fixed external source, node*G + g (FR-MODE-3).
  Result solve_fixed_source(
      const std::vector<double>& source, const Settings& settings);

  //! Begin a transient from the retained static solution (FR-KIN-1).
  //!
  //! The static eigenvalue is generally not one, so the fission source is
  //! divided by it for the whole transient. That is the standard criticality
  //! normalisation and it is what makes a null transient actually null:
  //! without it a reactor at k = 1.03 would ramp from the first step and the
  //! ramp would look like physics.
  //!
  //! Precursors start at the equilibrium of the initial fission source.
  //! Starting them at zero instead produces a prompt drop that is entirely
  //! an artefact.
  //!
  //! \throws InputError without a converged static solution, or if the
  //!         library carries no delayed data or no inverse velocities.
  void start_transient(const Settings& settings);

  //! Advance one step of \c dt seconds (FR-KIN-2).
  //!
  //! Cross sections and geometry may be changed between steps; the operator
  //! is reassembled here. The explicit half of the theta scheme is evaluated
  //! against the state as it stood at the end of the previous step, so a
  //! change made just before this call belongs to the new step.
  TransientRecord step(double dt, const Settings& settings);

  //! Precursor concentrations, node*n_precursors + d.
  const std::vector<double>& precursors() const;
  int n_precursors() const;
  double transient_time() const { return time_; }

  //! Discard the retained flux and coupling coefficients so the next solve
  //! starts cold.
  void reset();

  //! Access the retained solution, for warm starting or for inspection.
  const std::vector<double>& flux() const { return flux_; }
  double k_eff() const { return k_eff_; }

  //! Net current on every surface for the retained flux, surface*G + g.
  //!
  //! Positive along the surface normal, which runs from the low-side node to
  //! the high-side node. A caller checking the node balance adds \c +J*A to
  //! the low-side node and \c -J*A to the high-side one, which works
  //! unchanged for a boundary surface where only one side exists.
  std::vector<double> surface_currents() const;

  //! Number of surfaces, so a binding can shape the current array.
  int geometry_surfaces() const { return geom_.n_surfaces(); }

  CmfdSystem& system() { return cmfd_; }

private:
  //! Normalised node power from the retained flux.
  void compute_power(std::vector<double>& power) const;

  //! Un-normalised fission power per node, sum_g kappa_f phi V.
  void fission_power(
      const std::vector<double>& flux, std::vector<double>& power) const;

  //! Delayed spectrum of precursor group \c d in \c node, falling back to
  //! the library emission spectrum when none is supplied.
  double chi_delayed(int node, int g, int d) const;
  //! Prompt spectrum, (chi - sum_d beta_d chi_d)/(1 - beta).
  double chi_prompt(int node, int g) const;

  //! Right hand side of the time derivative, (V/v) dphi/dt, for the given
  //! state. Zero at a converged critical steady state, by construction.
  void time_derivative(const std::vector<double>& flux,
      const std::vector<double>& fission_norm,
      const std::vector<double>& delayed_source,
      std::vector<double>& out) const;

  const Geometry& geom_;
  const XSLibrary& xs_;
  CmfdSystem cmfd_;
  std::vector<double> flux_;
  std::vector<double> fission_source_;
  double k_eff_ = 1.0;
  bool has_solution_ = false;

  // ------------------------------------------------------------- transient
  std::unique_ptr<PrecursorState> precursors_;
  //! Fission source divided by the static eigenvalue: the criticality
  //! normalisation, held for the whole transient.
  double k_static_ = 1.0;
  double time_ = 0.0;
  bool in_transient_ = false;
  //! Fission source / k_static at the start of the current step.
  std::vector<double> fission_norm_;
  //! (V/v) dphi/dt at the end of the previous step, for the explicit half of
  //! the theta scheme.
  std::vector<double> derivative_;
};

}  // namespace openndm

#endif  // OPENNDM_SOLVER_H
