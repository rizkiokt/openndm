//! \file kernel_common.h
//! Shared machinery for the transverse-integrated nodal kernels.
//!
//! Both NEM and SANM expand the transverse leakage quadratically over three
//! consecutive nodes and expand the group-coupling source on the same
//! quadratic basis. Only the homogeneous basis functions differ: quartic
//! polynomials for NEM, hyperbolic (or trigonometric) functions for SANM.
//!
//! Every node is mapped to the local coordinate \f$\xi \in [-1/2, 1/2]\f$ and
//! the basis is
//! \f[ P_0 = 1, \quad P_1 = 2\xi, \quad P_2 = 6\xi^2 - \tfrac{1}{2}, \f]
//! all of which except \f$P_0\f$ integrate to zero over the node.

#ifndef OPENNDM_KERNEL_COMMON_H
#define OPENNDM_KERNEL_COMMON_H

#include <array>
#include <cmath>

namespace openndm {
namespace detail {

//! Quadratic transverse leakage fit (FR-SOL-2).
//!
//! Returns the \f$P_1\f$ and \f$P_2\f$ coefficients of the leakage shape in
//! the centre node, chosen so that the polynomial reproduces the node-average
//! leakage of all three nodes. \c h_prev or \c h_next equal to \c h_self with
//! the corresponding leakage repeated is the flat extrapolation used at a core
//! boundary.
inline std::array<double, 2> leakage_fit(double l_prev, double l_self,
    double l_next, double h_prev, double h_self, double h_next)
{
  const double a = h_prev / h_self;
  const double b = h_next / h_self;
  const double t = 0.5 + a;
  const double s = 0.5 + b;

  const double A1 = -(1.0 + a);
  const double B1 = (2.0 * t * t * t - 0.5 * t) / a;
  const double A2 = (1.0 + b);
  const double B2 = (2.0 * s * s * s - 0.5 * s) / b;
  const double r1 = l_prev - l_self;
  const double r2 = l_next - l_self;

  const double det = A1 * B2 - A2 * B1;
  if (std::abs(det) < 1.0e-300) return {0.0, 0.0};
  return {(r1 * B2 - r2 * B1) / det, (A1 * r2 - A2 * r1) / det};
}

//! Analytic basis pair for one node and group in the SANM kernel.
//!
//! Holds \f$\kappa^2 = \Sigma_r h^2 / D\f$ and the handful of face values,
//! derivatives and moments the two-node solve needs. When \f$\kappa^2\f$ is
//! negative, which happens in a strongly multiplying group once the in-group
//! fission source is moved onto the left-hand side, the hyperbolic functions
//! are replaced by their trigonometric counterparts.
struct AnalyticBasis {
  double k2 = 1.0;  //!< \f$\kappa^2\f$, signed
  double x = 1.0;   //!< \f$|\kappa|\f$
  bool hyperbolic = true;
  double odd_face = 0.0;    //!< odd(1/2)
  double even_face = 0.0;   //!< even(1/2)
  double odd_dface = 0.0;   //!< odd'(1/2)
  double even_dface = 0.0;  //!< even'(1/2)
  double even_avg = 0.0;    //!< \f$\int\f$ even
  double odd_m1 = 0.0;      //!< \f$\int\f$ odd * P1
  double even_m2 = 0.0;     //!< \f$\int\f$ even * P2

  //! \param k2_in signed \f$\kappa^2\f$; magnitudes below \c K2_FLOOR are
  //!        raised to it, which keeps the basis well conditioned in the
  //!        physically irrelevant limit of a vanishing removal cross section.
  static AnalyticBasis make(double k2_in)
  {
    constexpr double K2_FLOOR = 1.0e-8;
    AnalyticBasis b;
    b.hyperbolic = (k2_in >= 0.0);
    b.k2 = (std::abs(k2_in) < K2_FLOOR) ? (b.hyperbolic ? K2_FLOOR : -K2_FLOOR)
                                        : k2_in;
    b.x = std::sqrt(std::abs(b.k2));
    const double x = b.x;
    const double x2 = x * x;
    const double x3 = x2 * x;
    if (b.hyperbolic) {
      const double sh = std::sinh(0.5 * x);
      const double ch = std::cosh(0.5 * x);
      b.odd_face = sh;
      b.even_face = ch;
      b.odd_dface = x * ch;
      b.even_dface = x * sh;
      b.even_avg = 2.0 * sh / x;
      b.odd_m1 = (2.0 / x) * ch - (4.0 / x2) * sh;
      b.even_m2 = (2.0 / x) * sh - (12.0 / x2) * ch + (24.0 / x3) * sh;
    } else {
      const double sn = std::sin(0.5 * x);
      const double cs = std::cos(0.5 * x);
      b.odd_face = sn;
      b.even_face = cs;
      b.odd_dface = x * cs;
      b.even_dface = -x * sn;
      b.even_avg = 2.0 * sn / x;
      b.odd_m1 = (4.0 / x2) * sn - (2.0 / x) * cs;
      b.even_m2 = (2.0 / x) * sn + (12.0 / x2) * cs - (24.0 / x3) * sn;
    }
    return b;
  }
};

//! In-place Gaussian elimination with partial pivoting for a small dense
//! system. \c a is row-major n*n, \c b is the right-hand side and holds the
//! solution on return. Returns false on a singular matrix.
inline bool solve_dense(double* a, double* b, int n)
{
  for (int col = 0; col < n; ++col) {
    int pivot = col;
    double best = std::abs(a[col * n + col]);
    for (int r = col + 1; r < n; ++r) {
      const double v = std::abs(a[r * n + col]);
      if (v > best) {
        best = v;
        pivot = r;
      }
    }
    if (best < 1.0e-300) return false;
    if (pivot != col) {
      for (int c = 0; c < n; ++c) std::swap(a[col * n + c], a[pivot * n + c]);
      std::swap(b[col], b[pivot]);
    }
    const double inv = 1.0 / a[col * n + col];
    for (int r = col + 1; r < n; ++r) {
      const double factor = a[r * n + col] * inv;
      if (factor == 0.0) continue;
      for (int c = col; c < n; ++c) a[r * n + c] -= factor * a[col * n + c];
      b[r] -= factor * b[col];
    }
  }
  for (int r = n - 1; r >= 0; --r) {
    double s = b[r];
    for (int c = r + 1; c < n; ++c) s -= a[r * n + c] * b[c];
    b[r] = s / a[r * n + r];
  }
  return true;
}

}  // namespace detail
}  // namespace openndm

#endif  // OPENNDM_KERNEL_COMMON_H
