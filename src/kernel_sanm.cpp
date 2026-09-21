#include <cmath>
#include <memory>
#include <vector>

#include "kernel_common.h"
#include "openndm/kernel.h"
#include "openndm/xslib.h"

namespace openndm {

namespace {

//! Row of the two-node system carrying factor-weighted flux continuity,
//! \f$f_{lo}\phi_{lo}(+1/2) = f_{hi}\phi_{hi}(-1/2)\f$.
constexpr int kFluxContinuity = 0;

//! Row carrying current continuity, with \f$J = -(D/h)\,d\phi/d\xi\f$.
constexpr int kCurrentContinuity = 1;

//! Unknowns of the two-node system: the odd coefficient of each node, which
//! the columns of the system and its solution share.
constexpr int kALo = 0;
constexpr int kAHi = 1;

//! Expansion of one node/group flux on the SANM basis.
//!
//! \f[
//!   \phi(\xi) = A\,\mathrm{odd}(\xi) + C\,\mathrm{even}(\xi)
//!             + b_0 + b_1 P_1(\xi) + b_2 P_2(\xi)
//! \f]
//! with \f$C\f$ fixed by the node-average constraint, leaving the odd
//! coefficient \f$A\f$ as the only free parameter per node.
struct NodeExpansion {
  detail::AnalyticBasis basis;
  double phibar = 0.0;
  double A = 0.0;
  double C = 0.0;
  double b0 = 0.0, b1 = 0.0, b2 = 0.0;
  double D_over_h = 0.0;

  //! Projection of the flux onto P1, used as the source moment for the other
  //! groups in the next Gauss-Seidel pass.
  double moment1() const { return 3.0 * (A * basis.odd_m1) + b1; }
  double moment2() const { return 5.0 * (C * basis.even_m2) + b2; }

  //! Set the even coefficient from the node-average constraint, which fixes
  //! it outright and leaves \c A as the only free coefficient.
  void apply_average_constraint() { C = (phibar - b0) / basis.even_avg; }

  double face_flux(double sign) const
  {
    return sign * A * basis.odd_face + C * basis.even_face + b0 + sign * b1 +
           b2;
  }
  //! d(phi)/d(xi) at xi = sign/2.
  double face_derivative(double sign) const
  {
    return A * basis.odd_dface + sign * C * basis.even_dface + 2.0 * b1 +
           sign * 6.0 * b2;
  }
};

//! Semi-analytic nodal method (FR-SOL-3), the default kernel.
//!
//! The transverse leakage and any external source are expanded on the same
//! quadratic basis, and the one free coefficient left per node is closed by
//! discontinuity-factor weighted flux continuity and by current continuity at
//! the shared interface, a 2x2 solve per surface and group. See
//! \c docs/theory.md 3.4.
class SanmKernel : public Kernel {
public:
  const char* name() const override { return "sanm"; }

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
    std::vector<double> l1(static_cast<std::size_t>(2 * G));
    std::vector<double> l2(static_cast<std::size_t>(2 * G));
    std::vector<double> l0(static_cast<std::size_t>(2 * G));
    std::vector<double> q_ext0(static_cast<std::size_t>(2 * G), 0.0);
    std::vector<double> q_ext1(static_cast<std::size_t>(2 * G), 0.0);
    std::vector<double> q_ext2(static_cast<std::size_t>(2 * G), 0.0);

    for (int side = 0; side < 2; ++side) {
      const Composition& c = side == 0 ? cl : cr;
      const double h = side == 0 ? p.h_lo : p.h_hi;
      const double* tl = side == 0 ? p.tl_lo : p.tl_hi;
      const double h_prev = side == 0 ? p.h_lo_prev : p.h_lo;
      const double h_next = side == 0 ? p.h_hi : p.h_hi_next;
      for (int g = 0; g < G; ++g) {
        const std::size_t e = static_cast<std::size_t>(side) * G + g;
        const std::size_t gg = static_cast<std::size_t>(g);
        const double D = c.D[gg];
        const double sr = detail::effective_removal(
            c.removal[gg], c.chi[gg], c.nu_fission[gg], inv_k);
        ex[e].basis = detail::AnalyticBasis::make(sr * h * h / D);
        ex[e].D_over_h = D / h;
        ex[e].phibar = (side == 0 ? p.flux_lo : p.flux_hi)[gg];
        const auto fit =
            detail::leakage_fit(tl[gg], tl[static_cast<std::size_t>(G) + gg],
                tl[static_cast<std::size_t>(2 * G) + gg], h_prev, h, h_next);
        l0[e] = tl[static_cast<std::size_t>(G) + gg];
        l1[e] = fit[0];
        l2[e] = fit[1];
        const double* sq = side == 0 ? p.src_lo : p.src_hi;
        if (sq) {
          const auto sfit =
              detail::leakage_fit(sq[gg], sq[static_cast<std::size_t>(G) + gg],
                  sq[static_cast<std::size_t>(2 * G) + gg], h_prev, h, h_next);
          q_ext0[e] = sq[static_cast<std::size_t>(G) + gg];
          q_ext1[e] = sfit[0];
          q_ext2[e] = sfit[1];
        }
        ex[e].apply_average_constraint();
      }
    }

    const int n_sweeps = sweeps > 0 ? sweeps : 1;
    for (int sweep = 0; sweep < n_sweeps; ++sweep) {
      for (int g = 0; g < G; ++g) {
        for (int side = 0; side < 2; ++side) {
          const Composition& c = side == 0 ? cl : cr;
          const double h = side == 0 ? p.h_lo : p.h_hi;
          const std::size_t e = static_cast<std::size_t>(side) * G + g;
          const std::size_t gg = static_cast<std::size_t>(g);
          double q0 = q_ext0[e] - l0[e];
          double q1 = q_ext1[e] - l1[e];
          double q2 = q_ext2[e] - l2[e];
          for (int gp = 0; gp < G; ++gp) {
            if (gp == g) continue;
            const std::size_t ep = static_cast<std::size_t>(side) * G + gp;
            const double coeff =
                c.scatter[static_cast<std::size_t>(gp) * G + g] +
                c.chi[gg] * c.nu_fission[static_cast<std::size_t>(gp)] * inv_k;
            q0 += coeff * ex[ep].phibar;
            q1 += coeff * ex[ep].moment1();
            q2 += coeff * ex[ep].moment2();
          }
          const double scale = h * h / c.D[gg];
          const double k2 = ex[e].basis.k2;
          ex[e].b2 = scale * q2 / k2;
          ex[e].b1 = scale * q1 / k2;
          ex[e].b0 = (scale * q0 + 12.0 * ex[e].b2) / k2;
          ex[e].apply_average_constraint();
        }

        NodeExpansion& L = ex[static_cast<std::size_t>(g)];
        NodeExpansion& R = ex[static_cast<std::size_t>(G) + g];
        const double fL = p.adf_lo[static_cast<std::size_t>(g)];
        const double fR = p.adf_hi[static_cast<std::size_t>(g)];

        double a[4];
        double rhs[2];
        a[2 * kFluxContinuity + kALo] = fL * L.basis.odd_face;
        a[2 * kFluxContinuity + kAHi] = fR * R.basis.odd_face;
        rhs[kFluxContinuity] =
            fR * (R.C * R.basis.even_face + R.b0 - R.b1 + R.b2) -
            fL * (L.C * L.basis.even_face + L.b0 + L.b1 + L.b2);
        a[2 * kCurrentContinuity + kALo] = L.D_over_h * L.basis.odd_dface;
        a[2 * kCurrentContinuity + kAHi] = -R.D_over_h * R.basis.odd_dface;
        rhs[kCurrentContinuity] =
            R.D_over_h * (-R.C * R.basis.even_dface + 2.0 * R.b1 - 6.0 * R.b2) -
            L.D_over_h * (L.C * L.basis.even_dface + 2.0 * L.b1 + 6.0 * L.b2);

        if (detail::solve_dense(a, rhs, 2)) {
          L.A = rhs[kALo];
          R.A = rhs[kAHi];
        } else {
          L.A = 0.0;
          R.A = 0.0;
        }
      }
    }

    for (int g = 0; g < G; ++g) {
      const NodeExpansion& L = ex[static_cast<std::size_t>(g)];
      current[g] = -L.D_over_h * L.face_derivative(1.0);
    }
  }
};

}  // namespace

std::unique_ptr<Kernel> make_sanm_kernel()
{
  return std::make_unique<SanmKernel>();
}

}  // namespace openndm
