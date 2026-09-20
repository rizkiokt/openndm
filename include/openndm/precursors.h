//! \file precursors.h
//! Delayed neutron precursor concentrations and the delayed source (FR-KIN-1).
//!
//! The precursor equation is linear in the concentration once the fission
//! source is known, so it is integrated in closed form over a step rather
//! than by the same scheme used for the flux. For
//!
//! \f[ \frac{dC_d}{dt} = \beta_d F(t) - \lambda_d C_d \f]
//!
//! with the fission source \f$F\f$ taken linear across the step, the exact
//! solution is
//!
//! \f[ C_d(\Delta t) = C_d(0) e^{-\lambda_d \Delta t}
//!     + \beta_d \left[ F_0 I_0 + \frac{F_1 - F_0}{\Delta t} I_1 \right] \f]
//!
//! where \f$I_0 = (1 - e^{-\lambda \Delta t})/\lambda\f$ and
//! \f$I_1 = (\Delta t - I_0)/\lambda\f$. Both are evaluated by series near
//! \f$\lambda \Delta t = 0\f$, where the closed forms lose all their
//! significant digits to cancellation.

#ifndef OPENNDM_PRECURSORS_H
#define OPENNDM_PRECURSORS_H

#include <vector>

#include "openndm/xslib.h"

namespace openndm {

//! Precursor concentrations for every node, indexed \c value[node*D + d].
//!
//! The concentrations carry the units of the fission source divided by a
//! decay constant, so they are consistent with whatever normalisation the
//! caller's flux uses. Nothing here imposes one.
class PrecursorState {
public:
  //! \throws InputError if the library carries no delayed data, or if any
  //!         decay constant is non-positive.
  PrecursorState(int n_nodes, const DelayedData& delayed);

  int n_nodes() const { return n_nodes_; }
  int n_precursors() const { return n_precursors_; }

  const std::vector<double>& concentrations() const { return c_; }
  std::vector<double>& concentrations() { return c_; }

  //! Concentration of precursor group \c d in \c node.
  double concentration(int node, int d) const;

  //! Set every node to the equilibrium of its own fission source,
  //! \f$C_d = \beta_d F / \lambda_d\f$.
  //!
  //! This is the steady state a critical reactor sits in, and it is what a
  //! transient must start from: initialising to zero instead makes the first
  //! step see no delayed neutrons at all and produces a prompt drop that is
  //! entirely an artefact.
  //!
  //! \param fission_source one value per node.
  void set_equilibrium(const std::vector<double>& fission_source);

  //! Advance every node across \c dt, with the fission source going linearly
  //! from \c f0 to \c f1.
  //!
  //! \throws InputError on a non-positive step or a size mismatch.
  void advance(
      const std::vector<double>& f0, const std::vector<double>& f1, double dt);

  //! Delayed neutron source per node and group,
  //! \f$\sum_d \lambda_d C_d \chi^d_g\f$, written into \c out sized
  //! \c n_nodes*n_groups.
  void delayed_source(
      const DelayedData& delayed, int n_groups, std::vector<double>& out) const;

  //! Total delayed production per node, \f$\sum_d \lambda_d C_d\f$.
  std::vector<double> decay_rate() const;

private:
  int n_nodes_;
  int n_precursors_;
  std::vector<double> beta_;
  std::vector<double> lambda_;
  std::vector<double> c_;
};

//! \f$I_0 = (1 - e^{-\lambda t})/\lambda\f$, accurate as \f$\lambda t \to 0\f$.
double decay_integral_0(double lambda, double dt);

//! \f$I_1 = (\Delta t - I_0)/\lambda\f$, accurate as \f$\lambda t \to 0\f$.
double decay_integral_1(double lambda, double dt);

}  // namespace openndm

#endif  // OPENNDM_PRECURSORS_H
