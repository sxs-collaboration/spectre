// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Expressions/AddSubtract.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearAlgebra/FindRealEigensystem.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"

namespace {
// Check the eigensystem of a matrix (at a single grid point) with the given
// eigenvalues.
template <size_t MatrixSize>
void check_real_eigensystem(
    const tnsr::iJ<DataVector, MatrixSize>& matrix,
    std::array<double, MatrixSize> expected_eigenvalues) {
  // Note: the errors for these matrices are at round-off level (<~ 2e-16).
  const Approx custom_approx = Approx::custom().epsilon(1.0e-15).scale(1.0);

  // Find the eigensystem.
  // Note: the outputs are left unsized to check that find_real_eigensystem
  //       sizes them.
  tnsr::i<DataVector, MatrixSize> eigenvalues{};
  tnsr::ij<DataVector, MatrixSize> right_eigenvectors{};
  tnsr::IJ<DataVector, MatrixSize> left_eigenvectors{};
  find_real_eigensystem(make_not_null(&eigenvalues),
                        make_not_null(&right_eigenvectors),
                        make_not_null(&left_eigenvectors), matrix);

  // Check that the eigenvalues are the expected ones, in increasing order.
  alg::sort(expected_eigenvalues);
  for (size_t i = 0; i < MatrixSize; ++i) {
    CHECK(eigenvalues.get(i)[0] ==
          custom_approx(gsl::at(expected_eigenvalues, i)));
  }

  // Check the left and right eigensystems for each eigenvalue.
  const size_t num_points = get<0, 0>(matrix).size();
  for (size_t i = 0; i < MatrixSize; ++i) {
    // Use non-owning DataVector views to avoid copying data
    const Scalar<DataVector> eigenvalue{};
    const tnsr::i<DataVector, MatrixSize> right_eigenvector{};
    const tnsr::I<DataVector, MatrixSize> left_eigenvector{};
    make_const_view(make_not_null(&get(eigenvalue)), eigenvalues.get(i), 0,
                    num_points);
    for (size_t k = 0; k < MatrixSize; ++k) {
      make_const_view(make_not_null(&right_eigenvector.get(k)),
                      right_eigenvectors.get(i, k), 0, num_points);
      make_const_view(make_not_null(&left_eigenvector.get(k)),
                      left_eigenvectors.get(i, k), 0, num_points);
    }

    const Scalar<DataVector> right_eigensystem_error = magnitude(
        tenex::evaluate<ti::k>(matrix(ti::k, ti::J) * right_eigenvector(ti::j) -
                               eigenvalue() * right_eigenvector(ti::k)));
    const Scalar<DataVector> left_eigensystem_error = magnitude(
        tenex::evaluate<ti::K>(left_eigenvector(ti::J) * matrix(ti::j, ti::K) -
                               eigenvalue() * left_eigenvector(ti::K)));
    CHECK(get(right_eigensystem_error)[0] == custom_approx(0.0));
    CHECK(get(left_eigensystem_error)[0] == custom_approx(0.0));
  }

  // Check that the left eigenvectors are the inverse of the right ones.
  const auto left_times_right_eigenvectors = tenex::evaluate<ti::I, ti::j>(
      left_eigenvectors(ti::I, ti::M) * right_eigenvectors(ti::j, ti::m));
  for (size_t i = 0; i < MatrixSize; ++i) {
    for (size_t j = 0; j < MatrixSize; ++j) {
      const double kronecker_delta = i == j ? 1.0 : 0.0;
      CHECK(left_times_right_eigenvectors.get(i, j)[0] ==
            custom_approx(kronecker_delta));
    }
  }
}

void test_one_by_one_matrix() {
  // Build a 1x1 matrix, whose only entry is its eigenvalue.
  const double eigenvalue = -2.0;
  const tnsr::iJ<DataVector, 1> matrix{size_t{1}, eigenvalue};

  // Check its eigensystem.
  check_real_eigensystem<1>(matrix, {{eigenvalue}});
}

void test_diagonal_matrix() {
  // Build a diagonal matrix with distinct eigenvalues, not in increasing
  // order.
  constexpr size_t matrix_size = 3;
  const std::array<double, matrix_size> eigenvalues{{3.0, -1.0, 0.5}};
  tnsr::iJ<DataVector, matrix_size> matrix{size_t{1}, 0.0};
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix.get(i, i) = gsl::at(eigenvalues, i);
  }

  // Check its eigensystem.
  check_real_eigensystem<matrix_size>(matrix, eigenvalues);
}

void test_repeated_eigenvalue() {
  // Build a non-diagonal matrix with a repeated eigenvalue, A = S Lambda S^-1.
  // Note: S is upper triangular with ones on and above the diagonal, so its
  //       inverse has 1 on the diagonal, -1 on the first superdiagonal, and 0
  //       elsewhere.
  constexpr size_t matrix_size = 3;
  const double distinct_eigenvalue = -2.0;
  const double repeated_eigenvalue = 3.0;
  const std::array<double, matrix_size> eigenvalues{
      {repeated_eigenvalue, distinct_eigenvalue, repeated_eigenvalue}};
  const auto similarity_matrix = [](const size_t row, const size_t col) {
    return col >= row ? 1.0 : 0.0;
  };
  const auto inverse_similarity_matrix = [](const size_t row,
                                            const size_t col) {
    if (col == row) {
      return 1.0;
    }
    return col == row + 1 ? -1.0 : 0.0;
  };
  tnsr::iJ<DataVector, matrix_size> matrix{size_t{1}, 0.0};
  for (size_t row = 0; row < matrix_size; ++row) {
    for (size_t col = 0; col < matrix_size; ++col) {
      for (size_t k = 0; k < matrix_size; ++k) {
        matrix.get(row, col) += similarity_matrix(row, k) *
                                gsl::at(eigenvalues, k) *
                                inverse_similarity_matrix(k, col);
      }
    }
  }

  // Check its eigensystem.
  check_real_eigensystem<matrix_size>(matrix, eigenvalues);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearAlgebra.FindRealEigensystem",
                  "[NumericalAlgorithms][LinearAlgebra][Unit]") {
  test_one_by_one_matrix();
  test_diagonal_matrix();
  test_repeated_eigenvalue();
}
