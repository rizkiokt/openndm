#include <cmath>
#include <memory>
#include <vector>

#include "kernel_common.h"
#include "openndm/kernel.h"
#include "openndm/xslib.h"

namespace openndm {

namespace {

//! Quartic polynomial nodal expansion, NEM (FR-SOL-2).
//!
//! \f[
//!   \phi(\xi) = \bar\phi + a_1 f_1 + a_2 f_2 + a_3 f_3 + a_4 f_4
//! \f]
//! with
//! \f$f_1 = 2\xi\f$, \f$f_2 = 6\xi^2 - 1/2\f$,
//! \f$f_3 = \xi^3 - \xi/4\f$, \f$f_4 = \xi^4 - 0.3\xi^2 + 0.0125\f$.
//! All four integrate to zero over the node, and \f$f_3, f_4\f$ additionally
//! vanish at both faces, so the surface fluxes involve only \f$a_1, a_2\f$.
//!
//! The first two coefficients follow from the \f$f_1\f$ and \f$f_2\f$ weighted
//! residual (moment) equations, leaving \f$a_3, a_4\f$ per node. Those four
//! unknowns are closed by flux and current continuity at the shared interface
//! and by the coarse-mesh net currents at the two outer faces.
struct NodeExpansion {
  double phibar = 0.0;
  double alpha1 = 0.0, beta1 = 0.0;
  double alpha2 = 0.0, beta2 = 0.0;
  double a1 = 0.0, a2 = 0.0, a3 = 0.0, a4 = 0.0;
  double D_over_h = 0.0;

  void update_moment_coefficients()
  {
    a1 = alpha1 + beta1 * a3;
    a2 = alpha2 + beta2 * a4;
  }
  double moment1() const { return a1 - a3 / 20.0; }
  double moment2() const { return a2 - a4 / 70.0; }
  //! d(phi)/d(xi) at xi = sign/2.
  double face_derivative(double sign) const
  {
    return 2.0 * a1 + sign * 6.0 * a2 + 0.5 * a3 + sign * 0.2 * a4;
  }
};

class NemKernel : public Kernel {
public:
  const char* name() const override { return "nem"; }

  void solve(const TwoNodeProblem& p, const XSLibrary& xs,
      const std::vector<int>& composition, int G, int sweeps,
      double* current) const override
  {
    const Composition& cl =
        xs.composition(composition[static_cast<std::size_t>(p.node_lo)]);
    const Composition& cr =
        xs.composition(composition[static_cast<std::size_t>(p.node_hi)]);
    const double inv_k = 1.0 / p.k_eff;

    std::vector<NodeExpansion> ex(static_cast<std::size_t>(2 * G));
    std::vector<double> l0(static_cast<std::size_t>(2 * G));
    std::vector<double> l1(static_cast<std::size_t>(2 * G));
    std::vector<double> l2(static_cast<std::size_t>(2 * G));
    // External source on the same quadratic basis. Only its first and second
    // moments reach the NEM unknowns: a flat source cannot change a shape
    // whose node average is already fixed by the coarse-mesh solution.
    std::vector<double> q_ext1(static_cast<std::size_t>(2 * G), 0.0);
    std::vector<double> q_ext2(static_cast<std::size_t>(2 * G), 0.0);
    std::vector<double> sr_eff(static_cast<std::size_t>(2 * G));

    for (int side = 0; side < 2; ++side) {
      const Composition& c = side == 0 ? cl : cr;
      const double h = side == 0 ? p.h_lo : p.h_hi;
      const double* tl = side == 0 ? p.tl_lo : p.tl_hi;
      const double h_prev = side == 0 ? p.h_lo_prev : p.h_lo;
      const double h_next = side == 0 ? p.h_hi : p.h_hi_next;
      for (int g = 0; g < G; ++g) {
        const std::size_t e = static_cast<std::size_t>(side) * G + g;
        const std::size_t gg = static_cast<std::size_t>(g);
        ex[e].phibar = (side == 0 ? p.flux_lo : p.flux_hi)[gg];
        ex[e].D_over_h = c.D[gg] / h;
        double sr = c.removal[gg] - c.chi[gg] * c.nu_fission[gg] * inv_k;
        // The moment equations divide by the removal term, so a vanishing
        // effective removal is floored rather than allowed to blow up.
        if (std::abs(sr) < 1.0e-10) sr = (sr < 0.0 ? -1.0e-10 : 1.0e-10);
        sr_eff[e] = sr;
        const double dh2 = c.D[gg] / (h * h);
        ex[e].beta1 = 3.0 * dh2 / sr + 1.0 / 20.0;
        ex[e].beta2 = 2.0 * dh2 / sr + 1.0 / 70.0;
        const auto fit =
            detail::leakage_fit(tl[gg], tl[static_cast<std::size_t>(G) + gg],
                tl[static_cast<std::size_t>(2 * G) + gg], h_prev, h, h_next);
        l0[e] = tl[static_cast<std::size_t>(G) + gg];
        l1[e] = fit[0];
        l2[e] = fit[1];
        (void)l0[e];
        const double* sq = side == 0 ? p.src_lo : p.src_hi;
        if (sq) {
          const auto sfit =
              detail::leakage_fit(sq[gg], sq[static_cast<std::size_t>(G) + gg],
                  sq[static_cast<std::size_t>(2 * G) + gg], h_prev, h, h_next);
          q_ext1[e] = sfit[0];
          q_ext2[e] = sfit[1];
        }
      }
    }

    const int n_sweeps = sweeps > 0 ? sweeps : 1;
    for (int sweep = 0; sweep < n_sweeps; ++sweep) {
      for (int g = 0; g < G; ++g) {
        for (int side = 0; side < 2; ++side) {
          const Composition& c = side == 0 ? cl : cr;
          const std::size_t e = static_cast<std::size_t>(side) * G + g;
          const std::size_t gg = static_cast<std::size_t>(g);
          double q1 = q_ext1[e] - l1[e];
          double q2 = q_ext2[e] - l2[e];
          for (int gp = 0; gp < G; ++gp) {
            if (gp == g) continue;
            const std::size_t ep = static_cast<std::size_t>(side) * G + gp;
            const double coeff =
                c.scatter[static_cast<std::size_t>(gp) * G + g] +
                c.chi[gg] * c.nu_fission[static_cast<std::size_t>(gp)] * inv_k;
            q1 += coeff * ex[ep].moment1();
            q2 += coeff * ex[ep].moment2();
          }
          ex[e].alpha1 = q1 / sr_eff[e];
          ex[e].alpha2 = q2 / sr_eff[e];
        }

        NodeExpansion& L = ex[static_cast<std::size_t>(g)];
        NodeExpansion& R = ex[static_cast<std::size_t>(G) + g];
        const double fL = p.adf_lo[static_cast<std::size_t>(g)];
        const double fR = p.adf_hi[static_cast<std::size_t>(g)];
        const double cL3 = 2.0 * L.beta1 + 0.5;
        const double cL4 = 6.0 * L.beta2 + 0.2;
        const double cR3 = 2.0 * R.beta1 + 0.5;
        const double cR4 = 6.0 * R.beta2 + 0.2;

        double a[16] = {0.0};
        double rhs[4] = {0.0};
        // Row 0: ADF weighted flux continuity at the shared interface.
        a[0] = fL * L.beta1;
        a[1] = fL * L.beta2;
        a[2] = fR * R.beta1;
        a[3] = -fR * R.beta2;
        rhs[0] = -fL * (L.phibar + L.alpha1 + L.alpha2) +
                 fR * (R.phibar - R.alpha1 + R.alpha2);
        // Row 1: current continuity at the shared interface.
        a[4] = -L.D_over_h * cL3;
        a[5] = -L.D_over_h * cL4;
        a[6] = R.D_over_h * cR3;
        a[7] = -R.D_over_h * cR4;
        rhs[1] = L.D_over_h * (2.0 * L.alpha1 + 6.0 * L.alpha2) -
                 R.D_over_h * (2.0 * R.alpha1 - 6.0 * R.alpha2);
        // Row 2: the coarse-mesh net current at the outer face of node lo.
        a[8] = -L.D_over_h * cL3;
        a[9] = L.D_over_h * cL4;
        rhs[2] = p.cmfd_current_lo[static_cast<std::size_t>(g)] +
                 L.D_over_h * (2.0 * L.alpha1 - 6.0 * L.alpha2);
        // Row 3: the coarse-mesh net current at the outer face of node hi.
        a[14] = -R.D_over_h * cR3;
        a[15] = -R.D_over_h * cR4;
        rhs[3] = p.cmfd_current_hi[static_cast<std::size_t>(g)] +
                 R.D_over_h * (2.0 * R.alpha1 + 6.0 * R.alpha2);

        if (detail::solve_dense(a, rhs, 4)) {
          L.a3 = rhs[0];
          L.a4 = rhs[1];
          R.a3 = rhs[2];
          R.a4 = rhs[3];
        } else {
          L.a3 = L.a4 = R.a3 = R.a4 = 0.0;
        }
        L.update_moment_coefficients();
        R.update_moment_coefficients();
      }
    }

    for (int g = 0; g < G; ++g) {
      const NodeExpansion& L = ex[static_cast<std::size_t>(g)];
      current[g] = -L.D_over_h * L.face_derivative(1.0);
    }
  }
};

}  // namespace

std::unique_ptr<Kernel> make_nem_kernel()
{
  return std::make_unique<NemKernel>();
}

}  // namespace openndm
