// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <pup.h>

#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/DenseVector.hpp"
#include "Options/Options.hpp"
#include "Parallel/PupStlCpp17.hpp"
#include "Utilities/Gsl.hpp"

namespace TestHelpers::LinearSolver {

struct ApplyMatrix {
  DenseMatrix<double> matrix;
  void operator()(const gsl::not_null<DenseVector<double>*> result,
                  const DenseVector<double>& operand) const noexcept {
    *result = matrix * operand;
  }
};

// Use the exact inverse of the matrix as preconditioner. This should solve
// problems in 1 iteration.
struct ExactInversePreconditioner {
  void solve(const gsl::not_null<DenseVector<double>*> solution,
             const ApplyMatrix& linear_operator,
             const DenseVector<double>& source) const noexcept {
    if (not inv_matrix_.has_value()) {
      inv_matrix_ = blaze::inv(linear_operator.matrix);
    }
    *solution = *inv_matrix_ * source;
  }

  void reset() noexcept { inv_matrix_.reset(); }

  void pup(PUP::er& p) noexcept { p | inv_matrix_; }  // NOLINT

  // Make option-creatable for factory tests
  using options = tmpl::list<>;
  static constexpr Options::String help{"halp"};

 private:
  mutable std::optional<DenseMatrix<double>> inv_matrix_{};
};

// Use the inverse of the diagonal as preconditioner.
struct JacobiPreconditioner {
  void solve(const gsl::not_null<DenseVector<double>*> solution,
             const ApplyMatrix& linear_operator,
             const DenseVector<double>& source) const noexcept {
    if (not inv_diagonal_.has_value()) {
      inv_diagonal_ = DenseVector<double>(source.size(), 1.);
      for (size_t i = 0; i < source.size(); ++i) {
        (*inv_diagonal_)[i] /= linear_operator.matrix(i, i);
      }
    }
    *solution = source;
    for (size_t i = 0; i < solution->size(); ++i) {
      (*solution)[i] *= (*inv_diagonal_)[i];
    }
  }

  void reset() noexcept { inv_diagonal_.reset(); }

  void pup(PUP::er& p) noexcept { p | inv_diagonal_; }  // NOLINT

 private:
  mutable std::optional<DenseVector<double>> inv_diagonal_{};
};

// Run a few Richardson iterations as preconditioner.
struct RichardsonPreconditioner {
  RichardsonPreconditioner() = default;
  RichardsonPreconditioner(const double relaxation_parameter,
                           const size_t num_iterations)
      : relaxation_parameter_(relaxation_parameter),
        num_iterations_(num_iterations) {}

  void solve(
      const gsl::not_null<DenseVector<double>*> initial_guess_in_solution_out,
      const ApplyMatrix& linear_operator,
      const DenseVector<double>& source) const noexcept {
    for (size_t i = 0; i < num_iterations_; ++i) {
      linear_operator(make_not_null(&correction_buffer_),
                      *initial_guess_in_solution_out);
      correction_buffer_ *= -1.;
      correction_buffer_ += source;
      *initial_guess_in_solution_out +=
          relaxation_parameter_ * correction_buffer_;
    }
  }

  static void reset() noexcept {}

  void pup(PUP::er& p) noexcept {  // NOLINT
    p | relaxation_parameter_;
    p | num_iterations_;
  }

 private:
  double relaxation_parameter_{};
  size_t num_iterations_{};
  mutable DenseVector<double> correction_buffer_{};
};

}  // namespace TestHelpers::LinearSolver
