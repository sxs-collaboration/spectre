// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <pup.h>

#include "DataStructures/DynamicMatrix.hpp"
#include "DataStructures/DynamicVector.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace TestHelpers::LinearSolver {

struct ApplyMatrix {
  blaze::DynamicMatrix<double> matrix;
  // NOLINTNEXTLINE(spectre-mutable)
  mutable size_t invocations = 0;
  template <typename ResultVectorType, typename OperandVectorType>
  void operator()(const gsl::not_null<ResultVectorType*> result,
                  const OperandVectorType& operand) const {
    *result = matrix * operand;
    ++invocations;
  }
};

// Use the exact inverse of the matrix as preconditioner. This should solve
// problems in 1 iteration.
struct ExactInversePreconditioner {
  template <typename SolutionVectorType, typename SourceVectorType>
  void solve(const gsl::not_null<SolutionVectorType*> solution,
             const ApplyMatrix& linear_operator, const SourceVectorType& source,
             const std::tuple<>& /*operator_args*/) const {
    if (not inv_matrix_.has_value()) {
      inv_matrix_ = blaze::inv(linear_operator.matrix);
    }
    *solution = *inv_matrix_ * source;
  }

  void reset() { inv_matrix_.reset(); }

  void pup(PUP::er& p) { p | inv_matrix_; }  // NOLINT

  // Make option-creatable for factory tests
  using options = tmpl::list<>;
  static constexpr Options::String help{"halp"};

 private:
  // NOLINTNEXTLINE(spectre-mutable)
  mutable std::optional<blaze::DynamicMatrix<double>> inv_matrix_{};
};

// Use the inverse of the diagonal as preconditioner.
struct JacobiPreconditioner {
  template <typename SolutionVectorType, typename SourceVectorType>
  void solve(const gsl::not_null<SolutionVectorType*> solution,
             const ApplyMatrix& linear_operator, const SourceVectorType& source,
             const std::tuple<>& /*operator_args*/) const {
    if (not inv_diagonal_.has_value()) {
      inv_diagonal_ = blaze::DynamicVector<double>(source.size(), 1.);
      for (size_t i = 0; i < source.size(); ++i) {
        (*inv_diagonal_)[i] /= linear_operator.matrix(i, i);
      }
    }
    *solution = source;
    for (size_t i = 0; i < solution->size(); ++i) {
      (*solution)[i] *= (*inv_diagonal_)[i];
    }
  }

  void reset() { inv_diagonal_.reset(); }

  void pup(PUP::er& p) { p | inv_diagonal_; }  // NOLINT

 private:
  // NOLINTNEXTLINE(spectre-mutable)
  mutable std::optional<blaze::DynamicVector<double>> inv_diagonal_{};
};

// Run a few Richardson iterations as preconditioner.
struct RichardsonPreconditioner {
  RichardsonPreconditioner() = default;
  RichardsonPreconditioner(const double relaxation_parameter,
                           const size_t num_iterations)
      : relaxation_parameter_(relaxation_parameter),
        num_iterations_(num_iterations) {}

  template <typename SolutionVectorType, typename SourceVectorType>
  void solve(
      const gsl::not_null<SolutionVectorType*> initial_guess_in_solution_out,
      const ApplyMatrix& linear_operator, const SourceVectorType& source,
      const std::tuple<>& /*operator_args*/) const {
    for (size_t i = 0; i < num_iterations_; ++i) {
      linear_operator(make_not_null(&correction_buffer_),
                      *initial_guess_in_solution_out);
      correction_buffer_ *= -1.;
      correction_buffer_ += source;
      *initial_guess_in_solution_out +=
          relaxation_parameter_ * correction_buffer_;
    }
  }

  static void reset() {}

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) {
    p | relaxation_parameter_;
    p | num_iterations_;
  }

 private:
  double relaxation_parameter_{};
  size_t num_iterations_{};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable blaze::DynamicVector<double> correction_buffer_{};
};

}  // namespace TestHelpers::LinearSolver
