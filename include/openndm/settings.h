//! \file settings.h
//! Run-time solver configuration (FR-SOL-6, FR-SOL-8).

#ifndef OPENNDM_SETTINGS_H
#define OPENNDM_SETTINGS_H

#include "openndm/constants.h"

namespace openndm {

//! Convergence and algorithm selection for a static solve.
//!
//! Every field is run-time settable; no kernel choice is a compile-time
//! decision (FR-SOL-8).
struct Settings {
  KernelType kernel = KernelType::sanm;
  SolveMode mode = SolveMode::forward;

  //! Outer (eigenvalue) iteration control (FR-SOL-6).
  double k_tolerance = 1.0e-9;
  double fission_source_tolerance = 1.0e-8;
  int max_outer = 500;
  int min_outer = 2;

  //! Inner (within-group) linear solve control.
  double inner_tolerance = 1.0e-5;
  int max_inner = 50;
  //! Maximum Gauss-Seidel sweeps over energy groups per outer iteration.
  //!
  //! One sweep is exact for a purely down-scattering problem with no Wielandt
  //! shift. Both upscattering and the shift couple groups in a way a single
  //! lagged sweep cannot represent, so the sweep repeats until the flux stops
  //! changing or this bound is reached. In particular, when chi and nu-fission
  //! occupy different groups, as they do in any two-group LWR library, the
  //! entire shift lives off the group diagonal and a single sweep cancels it
  //! exactly.
  int group_sweeps = 50;
  //! Relative flux change between group sweeps below which the sweep stops.
  double group_sweep_tolerance = 1.0e-8;

  //! Additive Wielandt shift: the shifted operator subtracts the fission
  //! operator evaluated at k_shift = k_eff + wielandt_shift. Smaller
  //! accelerates the outer iteration at the cost of a harder inner solve; the
  //! system caps it internally so the shifted operator never turns singular.
  //! Set to 0 to disable.
  double wielandt_shift = 0.05;
  //! Outer iteration at which the shift starts being applied.
  int wielandt_start = 3;

  //! Nonlinear two-node update cadence: run the nodal kernel every
  //! \c nodal_update_interval outer iterations once past \c nodal_start.
  int nodal_update_interval = 1;
  int nodal_start = 2;
  //! Gauss-Seidel sweeps over groups inside one two-node problem.
  int two_node_sweeps = 2;
  //! Clip on the magnitude of Dhat relative to Dtilde. Large corrected
  //! coefficients destabilise the CMFD system; PARCS and KOMODO both clamp.
  //!
  //! Interior faces clamp to this. Boundary faces *discard* an update that
  //! exceeds it instead, keeping the value they had. A one-node correction
  //! that lands outside the band has been asked for a current the coarse mesh
  //! cannot express as Dhat times a node flux, and clamping it would replace
  //! an untrustworthy number with a large wrong one.
  double dhat_limit = 10.0;
  //! Under-relaxation on the boundary faces' Dhat, in (0, 1].
  //!
  //! The one-node boundary problem writes the face current as \c Dhat times
  //! the node-average flux. In a group whose flux at the boundary is small
  //! next to the within-node source driving it -- an intermediate group of a
  //! long down-scatter chain, at a zero flux face -- that ratio is badly
  //! conditioned, and the undamped update overshoots, saturates against
  //! \c dhat_limit and settles into a two-cycle instead of converging.
  //! Damping the step removes the cycle without moving the fixed point,
  //! since at convergence the update is a no-op whatever the factor.
  //!
  //! Interior faces are not damped: their Dhat divides by the sum of two node
  //! fluxes and is tied to a neighbour by continuity, so it does not suffer
  //! the same conditioning.
  double boundary_relaxation = 0.5;

  //! Time integration weighting (FR-KIN-2): 1 is fully implicit, 0.5 is
  //! Crank-Nicolson. KOMODO allows 0.01 to 1 and defaults to 1; matching
  //! that makes its decks portable.
  double theta = 1.0;
  //! Iterations within one time step, over the implicit fission source.
  int max_step_iterations = 50;
  //! Relative flux change between step iterations below which a step stops.
  double step_tolerance = 1.0e-9;

  //! Reuse the incoming flux and Dhat as the initial guess (FR-OPT-3).
  //!
  //! A cold solve discards the nonlinear correction along with the flux, so
  //! that solving the same model twice always retraces the same iteration
  //! path. An adjoint run is the exception: it needs the Dhat the forward
  //! solve converged.
  bool warm_start = false;

  //! Verbosity: 0 silent, 1 summary, 2 per-outer iteration history (FR-OUT-7).
  int verbosity = 1;

  //! Threads for the OpenMP loops; 0 leaves the OpenMP default alone.
  int threads = 0;
};

//! Iteration history entry, retained for the statepoint and for plotting.
struct IterationRecord {
  int outer = 0;
  double k_eff = 0.0;
  double k_change = 0.0;
  double source_change = 0.0;
  int inner_iterations = 0;
};

}  // namespace openndm

#endif  // OPENNDM_SETTINGS_H
