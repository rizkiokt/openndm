//! \file xslib.h
//! Macroscopic cross section storage, validation and branch interpolation.

#pragma once

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

  //! True when the arrays above carry 1-sigma uncertainties.
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

  //! Precursor groups stored, at most MAX_DELAYED_GROUPS.
  int n_precursors() const { return static_cast<int>(beta.size()); }
  //! \f$\beta = \sum_d \beta_d\f$.
  double beta_total() const;
};

//! Assembly discontinuity factors for one composition (FR-XS-4).
//!
//! Indexed \c value[face*G + g] with \c face running over the 2*n_axes faces
//! in the same order as Node::face. Absent entries default to 1.0.
struct AdfSet {
  std::vector<double> value;  //!< empty until the composition declares one
};

//! A complete, possibly branch-parameterised, macroscopic library.
class XSLibrary {
public:
  //! Zeroed library of one state, to be filled through composition().
  //!
  //! \throws InputError on fewer than one group or one composition.
  XSLibrary(int n_groups, int n_compositions);

  //! Energy groups, shared by every composition.
  int n_groups() const { return n_groups_; }
  //! Compositions, i.e. rows the core map may refer to.
  int n_compositions() const { return n_comps_; }

  //! Number of branch state points stored per composition.
  int n_states() const;

  //! Branch axes, outermost first. Empty for a single-state library.
  const std::vector<BranchAxis>& axes() const { return axes_; }
  //! Replace the branch axes, resizing the library to their grid and
  //! discarding whatever it held.
  //!
  //! \throws InputError on an axis with no points.
  //! \throws LibraryError on an axis that is not strictly increasing.
  void set_axes(std::vector<BranchAxis> axes);

  //! Choose what happens outside the branch grid (FR-XS-6).
  void set_extrapolation(Extrapolation e) { extrapolation_ = e; }
  //! Current behaviour outside the branch grid.
  Extrapolation extrapolation() const { return extrapolation_; }

  //! Mutable access to the composition at \c (comp, state).
  //!
  //! Taking this reference marks the library unfinalized, because the caller
  //! can invalidate the cached removal cross sections through it. Call
  //! finalize() again before solving.
  Composition& composition(int comp, int state = 0);

  //! Read-only access, which leaves the finalized state alone.
  const Composition& composition(int comp, int state = 0) const;

  //! Delayed data shared by every composition, empty for a static library.
  DelayedData& delayed() { return delayed_; }
  //! Delayed data shared by every composition, empty for a static library.
  const DelayedData& delayed() const { return delayed_; }

  //! ADFs for \c comp, sized on first use to 2*n_axes*G ones.
  AdfSet& adf(int comp, int n_axes = 3);
  //! ADF value, or 1.0 when the composition declared none.
  double adf_value(int comp, int face, int group) const;

  //! Cache derived data and run the FR-XS-8 checks.
  //!
  //! \param warnings collects non-fatal findings, never errors: negative
  //!        scattering elements produced by Monte Carlo noise, a fission
  //!        spectrum with no fission source, and a composition that produces
  //!        neutrons but carries no kappa-fission and so will carry no power,
  //!        whose only symptom is a silently zero power distribution.
  //! \throws LibraryError on a negative total cross section, a fission
  //!         spectrum that does not sum to one, or a non-monotonic branch axis.
  void finalize(std::vector<std::string>* warnings = nullptr);

  //! True while the cached removal cross sections match the compositions.
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

}
