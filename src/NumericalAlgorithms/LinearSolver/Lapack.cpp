// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearSolver/Lapack.hpp"

#include <cstddef>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

extern "C" {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
extern void dgesv_(int*, int*, double*, int*, int*, double*, int*,  // NOLINT
                   int*);
#pragma GCC diagnostic pop
}

namespace lapack {
int general_matrix_linear_solve(
    const gsl::not_null<DataVector*> rhs_in_solution_out,
    const gsl::not_null<Matrix*> matrix_operator,
    const int number_of_rhs) noexcept {
  const size_t rhs_vector_size = matrix_operator->rows();
  std::vector<int> ipiv(rhs_vector_size);
  return general_matrix_linear_solve(rhs_in_solution_out, make_not_null(&ipiv),
                                     matrix_operator, number_of_rhs);
}

int general_matrix_linear_solve(
    const gsl::not_null<DataVector*> rhs_in_solution_out,
    const gsl::not_null<std::vector<int>*> pivots,
    const gsl::not_null<Matrix*> matrix_operator, int number_of_rhs) noexcept {
  int output_vector_size = matrix_operator->columns();
  int rhs_vector_size = matrix_operator->rows();
  int matrix_spacing = matrix_operator->spacing();
  ASSERT(output_vector_size == rhs_vector_size,
         "The LAPACK-based general linear solve requires a square matrix "
         "input, not "
             << rhs_vector_size << " by " << output_vector_size);

  if (number_of_rhs == 0) {
    ASSERT(
        (rhs_in_solution_out->size() % matrix_operator->rows()) ==
            0_st,
        "The provided DataVector does not have size equal to (number of "
        "equations) * (larger matrix dimension), so the number of right-hand "
        "sides cannot be inferred and must be provided explicitly");
    number_of_rhs = static_cast<int>(
        rhs_in_solution_out->size() /
        std::max(matrix_operator->rows(), matrix_operator->columns()));
  }
  int info = 0;
  ASSERT(static_cast<size_t>(rhs_vector_size * number_of_rhs) <=
                 static_cast<size_t>(rhs_in_solution_out->size()) and
             static_cast<size_t>(output_vector_size * number_of_rhs) <=
                 static_cast<size_t>(rhs_in_solution_out->size()),
         "The single DataVector passed to the LAPACK call must be sufficiently "
         "large to contain x and b in A x = b");
  dgesv_(&rhs_vector_size, &number_of_rhs, matrix_operator->data(),
         &matrix_spacing, pivots->data(), rhs_in_solution_out->data(),
         &output_vector_size, &info);
  return info;
}

int general_matrix_linear_solve(const gsl::not_null<DataVector*> solution,
                                const gsl::not_null<Matrix*> matrix_operator,
                                const DataVector& rhs,
                                const int number_of_rhs) noexcept {
  const size_t rhs_vector_size = matrix_operator->rows();
  std::vector<int> ipiv(rhs_vector_size);
  return general_matrix_linear_solve(solution, make_not_null(&ipiv),
                                     matrix_operator, rhs, number_of_rhs);
}

int general_matrix_linear_solve(const gsl::not_null<DataVector*> solution,
                                const gsl::not_null<std::vector<int>*> pivots,
                                const gsl::not_null<Matrix*> matrix_operator,
                                const DataVector& rhs,
                                int number_of_rhs) noexcept {
  // NOLINTNEXTLINE(clang-analyzer-deadcode)
  const int output_vector_size = matrix_operator->columns();
  const int rhs_vector_size = matrix_operator->rows();
  if (number_of_rhs == 0) {
    ASSERT(rhs.size() % matrix_operator->rows() == 0,
           "The provided DataVector does not have size equal to (number of "
           "equations) * (number_of_matrix_rows), so the number of right-hand "
           "sides cannot be inferred and must be provided explicitly");
    number_of_rhs = static_cast<int>(rhs.size() / matrix_operator->rows());
  }
  ASSERT(solution->size() >=
             static_cast<size_t>(number_of_rhs) * matrix_operator->columns(),
         "The provided pointer for the output of the LAPACK linear solve is "
         "too small for the operation. Solution size is: "
             << solution->size() << " and should be: "
             << number_of_rhs * output_vector_size << ".");
  ASSERT(rhs.size() >=
             static_cast<size_t>(number_of_rhs) * matrix_operator->rows(),
         "The provided pointer for the input right-hand side vector of the "
         "LAPACK linear solve is too small for the operation. Vector size is: "
             << rhs.size()
             << " and should be: " << number_of_rhs * rhs_vector_size << ".");
  std::copy(rhs.begin(), rhs.begin() + number_of_rhs * rhs_vector_size,
            solution->begin());
  return general_matrix_linear_solve(solution, pivots, matrix_operator,
                                     number_of_rhs);
}

int general_matrix_linear_solve(
    const gsl::not_null<DataVector*> rhs_in_solution_out,
    const Matrix& matrix_operator, const int number_of_rhs) noexcept {
  // LAPACK is permitted to modify the matrix in-place, so we copy before
  // providing the operator if the original must be preserved.
  Matrix copied_matrix_operator = matrix_operator;
  return general_matrix_linear_solve(rhs_in_solution_out,
                                     make_not_null(&copied_matrix_operator),
                                     number_of_rhs);
}

int general_matrix_linear_solve(const gsl::not_null<DataVector*> solution,
                                const Matrix& matrix_operator,
                                const DataVector& rhs,
                                const int number_of_rhs) noexcept {
  // LAPACK is permitted to modify the matrix in-place, so we copy before
  // providing the operator if the original must be preserved.
  Matrix copied_matrix_operator = matrix_operator;
  return general_matrix_linear_solve(
      solution, make_not_null(&copied_matrix_operator), rhs, number_of_rhs);
}

}  // namespace lapack
