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
  double dhat_limit = 10.0;

  //! Reuse the incoming flux and Dhat as the initial guess (FR-OPT-3).
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

} // namespace openndm

#endif // OPENNDM_SETTINGS_H
