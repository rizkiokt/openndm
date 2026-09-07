//! \file matrix.h
//! Sparse linear algebra for the within-group CMFD systems.
//!
//! The systems solved here are small, extremely sparse (one diagonal plus one
//! entry per node face) and are re-assembled every nonlinear update, so the
//! pattern is built once and shared by every energy group while only the
//! values are refreshed. Reductions use a thread-count-independent chunked
//! summation so that results are bit-identical regardless of how many threads
//! run (FR-OPT-4).

#ifndef OPENNDM_MATRIX_H
#define OPENNDM_MATRIX_H

#include <cstddef>
#include <vector>

namespace openndm {

//! Number of vector entries summed serially inside one reduction chunk.
//! Fixed at compile time so the reduction tree does not depend on the number
//! of threads.
constexpr std::size_t REDUCTION_CHUNK = 512;

//! Thread-count-independent inner product (FR-OPT-4).
double deterministic_dot(
    const std::vector<double>& x, const std::vector<double>& y);

//! Thread-count-independent 2-norm.
double deterministic_norm(const std::vector<double>& x);

//! CSR sparsity pattern of the node adjacency graph, shared across groups.
class SparsePattern {
public:
  SparsePattern() = default;

  //! \param n_rows number of nodes
  //! \param neighbours neighbour node index list per node (diagonal excluded)
  SparsePattern(int n_rows, const std::vector<std::vector<int>>& neighbours);

  int n_rows() const { return n_rows_; }
  int nnz() const { return static_cast<int>(colind_.size()); }
  const std::vector<int>& rowptr() const { return rowptr_; }
  const std::vector<int>& colind() const { return colind_; }
  //! Position in \c colind_ of the diagonal entry of each row.
  const std::vector<int>& diag_pos() const { return diag_pos_; }

private:
  int n_rows_ = 0;
  std::vector<int> rowptr_;
  std::vector<int> colind_;
  std::vector<int> diag_pos_;
};

//! Values of one within-group operator laid out on a shared SparsePattern.
class GroupMatrix {
public:
  GroupMatrix() = default;
  explicit GroupMatrix(const SparsePattern* pattern);

  void zero();
  const SparsePattern& pattern() const { return *pattern_; }

  std::vector<double>& values() { return values_; }
  const std::vector<double>& values() const { return values_; }

  //! y = A * x
  void multiply(const std::vector<double>& x, std::vector<double>& y) const;

  //! Accumulate \c v into entry (row, col). The entry must exist.
  void add(int row, int col, double v);
  void add_diagonal(int row, double v);
  double diagonal(int row) const;

private:
  int find(int row, int col) const;

  const SparsePattern* pattern_ = nullptr;
  std::vector<double> values_;
};

//! Zero-fill incomplete LU factorisation used as the BiCGSTAB preconditioner.
//!
//! Reuses the matrix pattern, so the factors cost no extra memory beyond one
//! value array. Falls back to plain Jacobi on a zero pivot rather than
//! failing, which keeps a badly scaled problem solvable if slower.
class Ilu0 {
public:
  Ilu0() = default;
  void factor(const GroupMatrix& A);

  //! Solve (LU) x = b in place-safe fashion.
  void apply(const std::vector<double>& b, std::vector<double>& x) const;

  bool valid() const { return valid_; }

private:
  const SparsePattern* pattern_ = nullptr;
  std::vector<double> lu_;
  std::vector<double> diag_inv_;
  bool valid_ = false;
};

//! Convergence report from an inner linear solve.
struct LinearResult {
  int iterations = 0;
  double residual = 0.0;
  bool converged = false;
};

//! Preconditioned BiCGSTAB (FR-SOL-5).
//!
//! \param x is used as the initial guess and holds the solution on return.
LinearResult bicgstab(const GroupMatrix& A, const std::vector<double>& b,
    std::vector<double>& x, const Ilu0& precond, double tol, int max_iter);

}  // namespace openndm

#endif  // OPENNDM_MATRIX_H
