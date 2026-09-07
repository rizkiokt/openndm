//! \file solver.h
//! Outer eigenvalue iteration and the top level solve entry points.

#ifndef OPENNDM_SOLVER_H
#define OPENNDM_SOLVER_H

#include <memory>
#include <string>
#include <vector>

#include "openndm/cmfd.h"
#include "openndm/geometry.h"
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
  Result solve_fixed_source(const std::vector<double>& source,
    const Settings& settings);

  //! Discard the retained flux and coupling coefficients so the next solve
  //! starts cold.
  void reset();

  //! Access the retained solution, for warm starting or for inspection.
  const std::vector<double>& flux() const { return flux_; }
  double k_eff() const { return k_eff_; }

  CmfdSystem& system() { return cmfd_; }

private:
  //! Normalised node power from the retained flux.
  void compute_power(std::vector<double>& power) const;

  const Geometry& geom_;
  const XSLibrary& xs_;
  CmfdSystem cmfd_;
  std::vector<double> flux_;
  std::vector<double> fission_source_;
  double k_eff_ = 1.0;
  bool has_solution_ = false;
};

} // namespace openndm

#endif // OPENNDM_SOLVER_H
