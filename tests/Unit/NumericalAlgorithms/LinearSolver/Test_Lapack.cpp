// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Matrix.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/LinearSolver/Lapack.hpp"
#include "Utilities/Gsl.hpp"

template <typename Generator>
void test_square_general_matrix_linear_solve(
    const gsl::not_null<Generator*> generator) noexcept {
  UniformCustomDistribution<size_t> size_dist(2, 6);
  // the size of the matrix in the linear solve
  const size_t rows = size_dist(*generator);
  const size_t columns = rows;

  // number of distinct rhs that will be solved, as LAPACK can efficiently do
  // several linear solves at once
  const size_t number_of_rhs = size_dist(*generator);
  // refer to the canonical linear system A x = b

  UniformCustomDistribution<double> value_dist(0.1, 0.5);
  // the vector x
  auto expected_solution_vector = make_with_random_values<DataVector>(
      generator, make_not_null(&value_dist), number_of_rhs * rows);
  // the matrix A
  Matrix operator_matrix{rows, columns};
  for (size_t row = 0; row < rows; ++row) {
    for (size_t column = 0; column < columns; ++column) {
      operator_matrix(row, column) = value_dist(*generator);
    }
  }
  for (size_t i = 0; i < rows; ++i) {
    operator_matrix(i, i) += 1.0;
  }
  CAPTURE_PRECISE(operator_matrix);
  // the vector b
  auto input_vector = apply_matrices<DataVector, Matrix>(
      {{operator_matrix, Matrix{}}}, expected_solution_vector,
      Index<2>{rows, number_of_rhs});
  CAPTURE_PRECISE(input_vector);
  // version which copies the matrix, and preserves the input
  DataVector solution_vector{number_of_rhs * rows, 0.0};
  lapack::general_matrix_linear_solve(make_not_null(&solution_vector),
                                      operator_matrix, input_vector);
  CHECK_ITERABLE_APPROX(solution_vector, expected_solution_vector);

  // solve only a subset of the rhs's given
  DataVector single_column_solution_vector{rows};
  lapack::general_matrix_linear_solve(
      make_not_null(&single_column_solution_vector), operator_matrix,
      input_vector, 1);
  for (size_t i = 0; i < rows; ++i) {
    CHECK(single_column_solution_vector[i] ==
          approx(expected_solution_vector[i]));
  }

  // version which copies the matrix, overwrites the input
  solution_vector = input_vector;
  lapack::general_matrix_linear_solve(make_not_null(&solution_vector),
                                      operator_matrix);
  CHECK_ITERABLE_APPROX(solution_vector, expected_solution_vector);

  // solve only a subset of the rhs's given
  solution_vector = input_vector;
  lapack::general_matrix_linear_solve(make_not_null(&solution_vector),
                                      operator_matrix, 1);
  for (size_t i = 0; i < rows; ++i) {
    CHECK(solution_vector[i] == approx(expected_solution_vector[i]));
  }
  // the rest of the entries should not have been changed by the LAPACK call.
  for (size_t i = rows; i < rows * number_of_rhs; ++i) {
    CHECK(solution_vector[i] == input_vector[i]);
  }

  // version which overwrites the matrix but preserves the input
  Matrix operator_matrix_copy = operator_matrix;
  lapack::general_matrix_linear_solve(make_not_null(&solution_vector),
                                      make_not_null(&operator_matrix_copy),
                                      input_vector);
  CHECK_ITERABLE_APPROX(solution_vector, expected_solution_vector);
  CHECK(operator_matrix_copy != operator_matrix);

  // version which overwrites the matrix and the input
  operator_matrix_copy = operator_matrix;
  solution_vector = input_vector;
  lapack::general_matrix_linear_solve(make_not_null(&solution_vector),
                                      make_not_null(&operator_matrix));
  CHECK_ITERABLE_APPROX(solution_vector, expected_solution_vector);
  CHECK(operator_matrix_copy != operator_matrix);
}

SPECTRE_TEST_CASE("Unit.Numerical.LinearSolver.Lapack",
                  "[Unit][NumericalAlgorithms][LinearSolver]") {
  MAKE_GENERATOR(gen);
  {
    INFO("Test general linear solve on invertible square matrix")
    test_square_general_matrix_linear_solve(make_not_null(&gen));
  }
}
