#include "openndm/xslib.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "openndm/error.h"

namespace openndm {

double DelayedData::beta_total() const
{
  return std::accumulate(beta.begin(), beta.end(), 0.0);
}

XSLibrary::XSLibrary(int n_groups, int n_compositions)
    : n_groups_(n_groups),
      n_comps_(n_compositions)
{
  if (n_groups < 1) throw InputError("a library needs at least one group");
  if (n_compositions < 1) {
    throw InputError("a library needs at least one composition");
  }
  comps_.resize(static_cast<std::size_t>(n_comps_));
  for (auto& c : comps_) {
    c.D.assign(n_groups_, 0.0);
    c.absorption.assign(n_groups_, 0.0);
    c.nu_fission.assign(n_groups_, 0.0);
    c.kappa_fission.assign(n_groups_, 0.0);
    c.chi.assign(n_groups_, 0.0);
    c.inv_velocity.assign(n_groups_, 0.0);
    c.scatter.assign(static_cast<std::size_t>(n_groups_) * n_groups_, 0.0);
  }
  adfs_.resize(static_cast<std::size_t>(n_comps_));
}

int XSLibrary::n_states() const
{
  return static_cast<int>(comps_.size()) / n_comps_;
}

void XSLibrary::set_axes(std::vector<BranchAxis> axes)
{
  std::size_t total = 1;
  for (const auto& a : axes) {
    if (a.points.size() < 1) {
      throw InputError("branch axis '" + a.name + "' has no points");
    }
    for (std::size_t i = 1; i < a.points.size(); ++i) {
      if (!(a.points[i] > a.points[i - 1])) {
        throw LibraryError(
            "branch axis '" + a.name + "' is not strictly increasing");
      }
    }
    total *= a.points.size();
  }
  axes_ = std::move(axes);
  comps_.assign(total * n_comps_, Composition{});
  for (auto& c : comps_) {
    c.D.assign(n_groups_, 0.0);
    c.absorption.assign(n_groups_, 0.0);
    c.nu_fission.assign(n_groups_, 0.0);
    c.kappa_fission.assign(n_groups_, 0.0);
    c.chi.assign(n_groups_, 0.0);
    c.inv_velocity.assign(n_groups_, 0.0);
    c.scatter.assign(static_cast<std::size_t>(n_groups_) * n_groups_, 0.0);
  }
  finalized_ = false;
}

Composition& XSLibrary::composition(int comp, int state)
{
  if (comp < 0 || comp >= n_comps_) {
    throw InputError("composition index " + std::to_string(comp) +
                     " out of range [0, " + std::to_string(n_comps_) + ")");
  }
  if (state < 0 || state >= n_states()) {
    throw InputError("branch state index out of range");
  }
  finalized_ = false;
  return comps_[static_cast<std::size_t>(state) * n_comps_ + comp];
}

const Composition& XSLibrary::composition(int comp, int state) const
{
  if (comp < 0 || comp >= n_comps_) {
    throw InputError("composition index " + std::to_string(comp) +
                     " out of range [0, " + std::to_string(n_comps_) + ")");
  }
  if (state < 0 || state >= n_states()) {
    throw InputError("branch state index out of range");
  }
  return comps_[static_cast<std::size_t>(state) * n_comps_ + comp];
}

AdfSet& XSLibrary::adf(int comp, int n_axes)
{
  if (comp < 0 || comp >= n_comps_) {
    throw InputError("composition index out of range");
  }
  auto& a = adfs_[static_cast<std::size_t>(comp)];
  if (a.value.empty()) {
    a.value.assign(static_cast<std::size_t>(2 * n_axes) * n_groups_, 1.0);
  }
  return a;
}

double XSLibrary::adf_value(int comp, int face, int group) const
{
  const auto& a = adfs_[static_cast<std::size_t>(comp)];
  if (a.value.empty()) return 1.0;
  const std::size_t idx = static_cast<std::size_t>(face) * n_groups_ + group;
  return idx < a.value.size() ? a.value[idx] : 1.0;
}

void XSLibrary::finalize(std::vector<std::string>* warnings)
{
  const int G = n_groups_;
  for (int state = 0; state < n_states(); ++state) {
    for (int c = 0; c < n_comps_; ++c) {
      auto& comp = comps_[static_cast<std::size_t>(state) * n_comps_ + c];
      const std::string where = "composition " + std::to_string(c) + " state " +
                                std::to_string(state);

      comp.removal.assign(G, 0.0);
      for (int g = 0; g < G; ++g) {
        if (!(comp.D[g] > 0.0)) {
          throw LibraryError(where + " group " + std::to_string(g) +
                             " has a non-positive diffusion coefficient");
        }
        if (comp.absorption[g] < 0.0) {
          throw LibraryError(where + " group " + std::to_string(g) +
                             " has a negative absorption cross section");
        }
        double out_scatter = 0.0;
        for (int gp = 0; gp < G; ++gp) {
          const double s = comp.scatter[static_cast<std::size_t>(g) * G + gp];
          if (s < 0.0) {
            // Monte Carlo noise routinely produces small negative transfers;
            // FR-XS-8 requires a warning rather than a failure.
            if (warnings) {
              warnings->push_back(where + ": negative scattering transfer " +
                                  std::to_string(g) + "->" +
                                  std::to_string(gp) + " = " +
                                  std::to_string(s));
            }
          }
          if (gp != g) out_scatter += s;
        }
        comp.removal[g] = comp.absorption[g] + out_scatter;
        if (comp.removal[g] <= 0.0 && warnings) {
          warnings->push_back(where + " group " + std::to_string(g) +
                              " has a non-positive removal cross section");
        }
      }

      const double chi_sum =
          std::accumulate(comp.chi.begin(), comp.chi.end(), 0.0);
      const bool fissile = std::any_of(comp.nu_fission.begin(),
          comp.nu_fission.end(), [](double v) { return v > 0.0; });
      if (fissile) {
        if (std::abs(chi_sum - 1.0) > 1.0e-6) {
          throw LibraryError(where + ": fission spectrum sums to " +
                             std::to_string(chi_sum) + ", expected 1");
        }
      } else if (chi_sum > 0.0) {
        // Harmless but almost always a mistake worth surfacing.
        if (warnings) {
          warnings->push_back(
              where + " has a fission spectrum but no fission source");
        }
      }

      if (fissile && warnings) {
        const bool heating = std::any_of(comp.kappa_fission.begin(),
            comp.kappa_fission.end(), [](double v) { return v > 0.0; });
        if (!heating) {
          // Power is formed from kappa-fission, so a fissile composition
          // without it contributes to the eigenvalue and to nothing else. The
          // symptom is a silently zero power distribution.
          warnings->push_back(
              where +
              " produces neutrons but has no kappa-fission, so it "
              "will carry no power; set kappa_fission to get a power "
              "distribution");
        }
      }
    }
  }

  if (!delayed_.beta.empty()) {
    if (delayed_.beta.size() != delayed_.lambda.size()) {
      throw LibraryError("beta and lambda have different precursor counts");
    }
    if (delayed_.n_precursors() > MAX_DELAYED_GROUPS) {
      throw LibraryError("at most " + std::to_string(MAX_DELAYED_GROUPS) +
                         " delayed precursor groups are supported");
    }
  }
  finalized_ = true;
}

bool XSLibrary::has_uncertainty() const
{
  return std::any_of(comps_.begin(), comps_.end(),
      [](const Composition& c) { return c.has_uncertainty(); });
}

int XSLibrary::state_index(const std::vector<int>& idx) const
{
  int flat = 0;
  for (std::size_t a = 0; a < axes_.size(); ++a) {
    flat = flat * static_cast<int>(axes_[a].points.size()) + idx[a];
  }
  return flat;
}

XSLibrary XSLibrary::interpolate(const std::vector<double>& state) const
{
  if (axes_.empty()) {
    XSLibrary out = *this;
    return out;
  }
  if (state.size() != axes_.size()) {
    throw InputError("expected " + std::to_string(axes_.size()) +
                     " branch coordinates, got " +
                     std::to_string(state.size()));
  }

  const int n_axes = static_cast<int>(axes_.size());
  std::vector<int> lo(n_axes, 0);
  std::vector<double> frac(n_axes, 0.0);

  for (int a = 0; a < n_axes; ++a) {
    const auto& pts = axes_[a].points;
    const double x = state[a];
    const int last = static_cast<int>(pts.size()) - 1;
    if (last == 0) {
      lo[a] = 0;
      frac[a] = 0.0;
      continue;
    }
    if (x < pts.front() || x > pts.back()) {
      switch (extrapolation_) {
        case Extrapolation::error:
          throw InputError("branch coordinate " + std::to_string(x) +
                           " is outside axis '" + axes_[a].name + "'");
        case Extrapolation::clamp:
          lo[a] = (x < pts.front()) ? 0 : last - 1;
          frac[a] = (x < pts.front()) ? 0.0 : 1.0;
          continue;
        case Extrapolation::linear:
          lo[a] = (x < pts.front()) ? 0 : last - 1;
          frac[a] = (x - pts[lo[a]]) / (pts[lo[a] + 1] - pts[lo[a]]);
          continue;
      }
    }
    int i = 0;
    while (i < last - 1 && pts[i + 1] < x) ++i;
    lo[a] = i;
    frac[a] = (x - pts[i]) / (pts[i + 1] - pts[i]);
  }

  XSLibrary out(n_groups_, n_comps_);
  out.delayed_ = delayed_;
  out.adfs_ = adfs_;
  out.extrapolation_ = extrapolation_;

  const int n_corners = 1 << n_axes;
  for (int corner = 0; corner < n_corners; ++corner) {
    double weight = 1.0;
    std::vector<int> idx(n_axes);
    for (int a = 0; a < n_axes; ++a) {
      const bool high = (corner >> a) & 1;
      const int last = static_cast<int>(axes_[a].points.size()) - 1;
      idx[a] = std::min(lo[a] + (high ? 1 : 0), last);
      weight *= high ? frac[a] : (1.0 - frac[a]);
    }
    if (weight == 0.0) continue;
    const int s = state_index(idx);
    for (int c = 0; c < n_comps_; ++c) {
      const Composition& src =
          comps_[static_cast<std::size_t>(s) * n_comps_ + c];
      Composition& dst = out.comps_[c];
      for (int g = 0; g < n_groups_; ++g) {
        dst.D[g] += weight * src.D[g];
        dst.absorption[g] += weight * src.absorption[g];
        dst.nu_fission[g] += weight * src.nu_fission[g];
        dst.kappa_fission[g] += weight * src.kappa_fission[g];
        dst.chi[g] += weight * src.chi[g];
        dst.inv_velocity[g] += weight * src.inv_velocity[g];
      }
      for (std::size_t i = 0; i < src.scatter.size(); ++i) {
        dst.scatter[i] += weight * src.scatter[i];
      }
    }
  }
  out.finalize();
  return out;
}

}  // namespace openndm
