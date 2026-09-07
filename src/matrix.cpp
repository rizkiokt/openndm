#include "openndm/matrix.h"

#include <algorithm>
#include <cmath>

#include "openndm/error.h"

namespace openndm {

namespace {

//! Sum a vector in fixed-size chunks so the reduction tree is identical for
//! any thread count (FR-OPT-4). The chunk partials are combined in index
//! order, which makes the result independent of completion order too.
double chunked_sum(const double* a, const double* b, std::size_t n)
{
  const std::size_t n_chunks = (n + REDUCTION_CHUNK - 1) / REDUCTION_CHUNK;
  std::vector<double> partial(n_chunks, 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (std::ptrdiff_t c = 0; c < static_cast<std::ptrdiff_t>(n_chunks); ++c) {
    const std::size_t begin = static_cast<std::size_t>(c) * REDUCTION_CHUNK;
    const std::size_t end = std::min(begin + REDUCTION_CHUNK, n);
    double s = 0.0;
    for (std::size_t i = begin; i < end; ++i) {
      s += b ? a[i] * b[i] : a[i] * a[i];
    }
    partial[static_cast<std::size_t>(c)] = s;
  }
  double total = 0.0;
  for (double p : partial) total += p;
  return total;
}

}  // namespace

double deterministic_dot(
    const std::vector<double>& x, const std::vector<double>& y)
{
  if (x.size() != y.size()) throw InputError("dot product size mismatch");
  if (x.empty()) return 0.0;
  return chunked_sum(x.data(), y.data(), x.size());
}

double deterministic_norm(const std::vector<double>& x)
{
  if (x.empty()) return 0.0;
  return std::sqrt(chunked_sum(x.data(), nullptr, x.size()));
}

SparsePattern::SparsePattern(
    int n_rows, const std::vector<std::vector<int>>& neighbours)
    : n_rows_(n_rows)
{
  rowptr_.assign(static_cast<std::size_t>(n_rows) + 1, 0);
  diag_pos_.assign(static_cast<std::size_t>(n_rows), -1);
  std::size_t nnz = 0;
  for (int r = 0; r < n_rows; ++r) nnz += neighbours[r].size() + 1;
  colind_.reserve(nnz);

  std::vector<int> row;
  for (int r = 0; r < n_rows; ++r) {
    rowptr_[static_cast<std::size_t>(r)] = static_cast<int>(colind_.size());
    row = neighbours[r];
    row.push_back(r);
    // Ascending column order keeps the ILU0 sweep and the matrix-vector
    // product cache friendly and makes the factorisation order well defined.
    std::sort(row.begin(), row.end());
    row.erase(std::unique(row.begin(), row.end()), row.end());
    for (int c : row) {
      if (c == r)
        diag_pos_[static_cast<std::size_t>(r)] =
            static_cast<int>(colind_.size());
      colind_.push_back(c);
    }
  }
  rowptr_[static_cast<std::size_t>(n_rows)] = static_cast<int>(colind_.size());
}

GroupMatrix::GroupMatrix(const SparsePattern* pattern)
    : pattern_(pattern)
{
  values_.assign(static_cast<std::size_t>(pattern->nnz()), 0.0);
}

void GroupMatrix::zero()
{
  std::fill(values_.begin(), values_.end(), 0.0);
}

int GroupMatrix::find(int row, int col) const
{
  const auto& rp = pattern_->rowptr();
  const auto& ci = pattern_->colind();
  const int begin = rp[static_cast<std::size_t>(row)];
  const int end = rp[static_cast<std::size_t>(row) + 1];
  const auto it = std::lower_bound(ci.begin() + begin, ci.begin() + end, col);
  if (it == ci.begin() + end || *it != col) {
    throw InputError("matrix entry (" + std::to_string(row) + ", " +
                     std::to_string(col) + ") is not in the sparsity pattern");
  }
  return static_cast<int>(it - ci.begin());
}

void GroupMatrix::add(int row, int col, double v)
{
  values_[static_cast<std::size_t>(find(row, col))] += v;
}

void GroupMatrix::add_diagonal(int row, double v)
{
  values_[static_cast<std::size_t>(
      pattern_->diag_pos()[static_cast<std::size_t>(row)])] += v;
}

double GroupMatrix::diagonal(int row) const
{
  return values_[static_cast<std::size_t>(
      pattern_->diag_pos()[static_cast<std::size_t>(row)])];
}

void GroupMatrix::multiply(
    const std::vector<double>& x, std::vector<double>& y) const
{
  const auto& rp = pattern_->rowptr();
  const auto& ci = pattern_->colind();
  const int n = pattern_->n_rows();
  y.resize(static_cast<std::size_t>(n));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (std::ptrdiff_t r = 0; r < n; ++r) {
    double s = 0.0;
    for (int p = rp[static_cast<std::size_t>(r)];
        p < rp[static_cast<std::size_t>(r) + 1]; ++p) {
      s += values_[static_cast<std::size_t>(p)] *
           x[static_cast<std::size_t>(ci[static_cast<std::size_t>(p)])];
    }
    y[static_cast<std::size_t>(r)] = s;
  }
}

void Ilu0::factor(const GroupMatrix& A)
{
  pattern_ = &A.pattern();
  lu_ = A.values();
  const int n = pattern_->n_rows();
  const auto& rp = pattern_->rowptr();
  const auto& ci = pattern_->colind();
  const auto& dp = pattern_->diag_pos();
  diag_inv_.assign(static_cast<std::size_t>(n), 0.0);
  valid_ = true;

  // Standard IKJ incomplete LU with no fill-in. Row positions are looked up
  // through a scatter array, which is O(nnz) overall for this pattern.
  std::vector<int> position(static_cast<std::size_t>(n), -1);
  for (int i = 0; i < n; ++i) {
    for (int p = rp[static_cast<std::size_t>(i)];
        p < rp[static_cast<std::size_t>(i) + 1]; ++p) {
      position[static_cast<std::size_t>(ci[static_cast<std::size_t>(p)])] = p;
    }
    for (int p = rp[static_cast<std::size_t>(i)];
        p < dp[static_cast<std::size_t>(i)]; ++p) {
      const int k = ci[static_cast<std::size_t>(p)];
      const double pivot = lu_[static_cast<std::size_t>(p)] *
                           diag_inv_[static_cast<std::size_t>(k)];
      lu_[static_cast<std::size_t>(p)] = pivot;
      if (pivot == 0.0) continue;
      for (int q = dp[static_cast<std::size_t>(k)] + 1;
          q < rp[static_cast<std::size_t>(k) + 1]; ++q) {
        const int j = ci[static_cast<std::size_t>(q)];
        const int target = position[static_cast<std::size_t>(j)];
        if (target >= 0) {
          lu_[static_cast<std::size_t>(target)] -=
              pivot * lu_[static_cast<std::size_t>(q)];
        }
      }
    }
    const double d =
        lu_[static_cast<std::size_t>(dp[static_cast<std::size_t>(i)])];
    if (std::abs(d) < 1.0e-300) {
      // Rather than fail, degrade to Jacobi on this row. A zero pivot here
      // means a decoupled node, which is a modelling problem the solver
      // reports through non-convergence, not a crash.
      diag_inv_[static_cast<std::size_t>(i)] = 0.0;
      valid_ = false;
    } else {
      diag_inv_[static_cast<std::size_t>(i)] = 1.0 / d;
    }
    for (int p = rp[static_cast<std::size_t>(i)];
        p < rp[static_cast<std::size_t>(i) + 1]; ++p) {
      position[static_cast<std::size_t>(ci[static_cast<std::size_t>(p)])] = -1;
    }
  }
}

void Ilu0::apply(const std::vector<double>& b, std::vector<double>& x) const
{
  const int n = pattern_->n_rows();
  const auto& rp = pattern_->rowptr();
  const auto& ci = pattern_->colind();
  const auto& dp = pattern_->diag_pos();
  x.resize(static_cast<std::size_t>(n));

  // Forward substitution with a unit lower triangle.
  for (int i = 0; i < n; ++i) {
    double s = b[static_cast<std::size_t>(i)];
    for (int p = rp[static_cast<std::size_t>(i)];
        p < dp[static_cast<std::size_t>(i)]; ++p) {
      s -= lu_[static_cast<std::size_t>(p)] *
           x[static_cast<std::size_t>(ci[static_cast<std::size_t>(p)])];
    }
    x[static_cast<std::size_t>(i)] = s;
  }
  // Backward substitution.
  for (int i = n - 1; i >= 0; --i) {
    double s = x[static_cast<std::size_t>(i)];
    for (int p = dp[static_cast<std::size_t>(i)] + 1;
        p < rp[static_cast<std::size_t>(i) + 1]; ++p) {
      s -= lu_[static_cast<std::size_t>(p)] *
           x[static_cast<std::size_t>(ci[static_cast<std::size_t>(p)])];
    }
    x[static_cast<std::size_t>(i)] = s * diag_inv_[static_cast<std::size_t>(i)];
  }
}

LinearResult bicgstab(const GroupMatrix& A, const std::vector<double>& b,
    std::vector<double>& x, const Ilu0& precond, double tol, int max_iter)
{
  const std::size_t n = b.size();
  LinearResult result;
  std::vector<double> r(n), r0(n), p(n), v(n), s(n), t(n), y(n), z(n), Ax(n);

  A.multiply(x, Ax);
  for (std::size_t i = 0; i < n; ++i) r[i] = b[i] - Ax[i];
  r0 = r;

  const double b_norm = deterministic_norm(b);
  const double scale = (b_norm > 0.0) ? b_norm : 1.0;
  double resid = deterministic_norm(r) / scale;
  if (resid <= tol) {
    result.converged = true;
    result.residual = resid;
    return result;
  }

  double rho = 1.0, alpha = 1.0, omega = 1.0;
  std::fill(p.begin(), p.end(), 0.0);
  std::fill(v.begin(), v.end(), 0.0);

  for (int it = 1; it <= max_iter; ++it) {
    const double rho_new = deterministic_dot(r0, r);
    if (std::abs(rho_new) < 1.0e-300) break;  // breakdown; take what we have
    const double beta = (rho_new / rho) * (alpha / omega);
    for (std::size_t i = 0; i < n; ++i) {
      p[i] = r[i] + beta * (p[i] - omega * v[i]);
    }
    precond.apply(p, y);
    A.multiply(y, v);
    const double r0v = deterministic_dot(r0, v);
    if (std::abs(r0v) < 1.0e-300) break;
    alpha = rho_new / r0v;
    for (std::size_t i = 0; i < n; ++i) s[i] = r[i] - alpha * v[i];

    const double s_norm = deterministic_norm(s);
    if (s_norm / scale <= tol) {
      for (std::size_t i = 0; i < n; ++i) x[i] += alpha * y[i];
      result.iterations = it;
      result.residual = s_norm / scale;
      result.converged = true;
      return result;
    }

    precond.apply(s, z);
    A.multiply(z, t);
    const double tt = deterministic_dot(t, t);
    omega = (tt > 0.0) ? deterministic_dot(t, s) / tt : 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      x[i] += alpha * y[i] + omega * z[i];
      r[i] = s[i] - omega * t[i];
    }
    rho = rho_new;
    resid = deterministic_norm(r) / scale;
    result.iterations = it;
    result.residual = resid;
    if (resid <= tol) {
      result.converged = true;
      return result;
    }
    if (std::abs(omega) < 1.0e-300) break;
  }
  return result;
}

}  // namespace openndm
