//! \file xslib.h
//! Macroscopic cross section storage, validation and branch interpolation.

#ifndef OPENNDM_XSLIB_H
#define OPENNDM_XSLIB_H

#include <string>
#include <vector>

#include "openndm/constants.h"

namespace openndm {

//! Group constants for one composition at one branch state point.
//!
//! Layout is group-major and flat so that a whole composition is contiguous;
//! the scattering matrix is stored row-major with \c scatter[from*G + to].
struct Composition {
  std::vector<double> D;              //!< diffusion coefficient [cm]
  std::vector<double> absorption;     //!< \f$\Sigma_a\f$ [1/cm]
  std::vector<double> nu_fission;     //!< \f$\nu\Sigma_f\f$ [1/cm]
  std::vector<double> kappa_fission;  //!< \f$\kappa\Sigma_f\f$ [J/cm]
  std::vector<double> chi;            //!< prompt + delayed fission spectrum
  std::vector<double> scatter;        //!< \f$\Sigma_{s,g\to g'}\f$, size G*G
  std::vector<double> inv_velocity;   //!< \f$1/v\f$ [s/cm]

  //! Removal cross section \f$\Sigma_a + \sum_{g'\neq g}\Sigma_{s,g\to g'}\f$.
  //! Cached by XSLibrary::finalize() because every kernel needs it.
  std::vector<double> removal;

  //! Optional 1-sigma uncertainties, parallel to the arrays above and empty
  //! when the source data carried none (FR-XS-9, FR-OMC-6).
  std::vector<double> D_std;
  std::vector<double> absorption_std;
  std::vector<double> nu_fission_std;
  std::vector<double> scatter_std;

  bool has_uncertainty() const { return !absorption_std.empty(); }
};

//! One axis of a branch-parameterised library (FR-XS-5).
struct BranchAxis {
  std::string name;            //!< e.g. "fuel_temperature"
  std::vector<double> points;  //!< strictly increasing grid points
};

//! Extrapolation behaviour outside the branch grid (FR-XS-6).
enum class Extrapolation { clamp, linear, error };

//! Delayed neutron data, shared by every composition (FR-XS-3).
struct DelayedData {
  std::vector<double> beta;    //!< delayed fraction per precursor group
  std::vector<double> lambda;  //!< decay constant per precursor group [1/s]
  //! Delayed spectrum, row-major \c chi_delayed[i*G + g].
  std::vector<double> chi_delayed;

  int n_precursors() const { return static_cast<int>(beta.size()); }
  double beta_total() const;
};

//! Assembly discontinuity factors for one composition (FR-XS-4).
//!
//! Indexed \c value[face*G + g] with \c face running over the 2*n_axes faces
//! in the same order as Node::face. Absent entries default to 1.0.
struct AdfSet {
  std::vector<double> value;
};

//! A complete, possibly branch-parameterised, macroscopic library.
class XSLibrary {
public:
  XSLibrary(int n_groups, int n_compositions);

  int n_groups() const { return n_groups_; }
  int n_compositions() const { return n_comps_; }

  //! Number of branch state points stored per composition.
  int n_states() const;

  //! Branch axes, outermost first. Empty for a single-state library.
  const std::vector<BranchAxis>& axes() const { return axes_; }
  void set_axes(std::vector<BranchAxis> axes);

  void set_extrapolation(Extrapolation e) { extrapolation_ = e; }
  Extrapolation extrapolation() const { return extrapolation_; }

  //! Mutable access to the composition at \c (comp, state).
  //!
  //! Taking this reference marks the library unfinalized, because the caller
  //! can invalidate the cached removal cross sections through it. Call
  //! finalize() again before solving.
  Composition& composition(int comp, int state = 0);

  //! Read-only access, which leaves the finalized state alone.
  const Composition& composition(int comp, int state = 0) const;

  DelayedData& delayed() { return delayed_; }
  const DelayedData& delayed() const { return delayed_; }

  //! ADFs for \c comp, sized on first use to 2*n_axes*G ones.
  AdfSet& adf(int comp, int n_axes = 3);
  //! ADF value, or 1.0 when the composition declared none.
  double adf_value(int comp, int face, int group) const;

  //! Cache derived data and run the FR-XS-8 checks.
  //!
  //! \param warnings collects non-fatal findings, e.g. negative scattering
  //!        elements produced by Monte Carlo noise, which are reported but
  //!        never treated as errors.
  //! \throws LibraryError on a negative total cross section, a fission
  //!         spectrum that does not sum to one, or a non-monotonic branch axis.
  void finalize(std::vector<std::string>* warnings = nullptr);

  bool finalized() const { return finalized_; }

  //! Multilinear interpolation over every branch axis (FR-XS-6).
  //!
  //! \param state one coordinate per axis, in axis order.
  //! \return a single-state library holding the interpolated compositions.
  XSLibrary interpolate(const std::vector<double>& state) const;

  //! True when any composition carries uncertainties.
  bool has_uncertainty() const;

private:
  //! Flat index of a branch grid point from its per-axis indices.
  int state_index(const std::vector<int>& idx) const;

  int n_groups_;
  int n_comps_;
  std::vector<BranchAxis> axes_;
  //! comps_[state * n_comps_ + comp]
  std::vector<Composition> comps_;
  std::vector<AdfSet> adfs_;
  DelayedData delayed_;
  Extrapolation extrapolation_ = Extrapolation::clamp;
  bool finalized_ = false;
};

}  // namespace openndm

#endif  // OPENNDM_XSLIB_H
