// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iostream>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/LinearAlgebra/FindGeneralizedEigenvaluesArpack.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace {
void test_find_generalized_eigenvalues_arpack(
    Matrix& matrix_a, Matrix& matrix_b, const size_t eigenvector_size,
    const size_t eigenvalue_size, const size_t number_of_eigenvalues_to_find,
    const std::string which_eigenvalues_to_find) {
  // Set up vectors and a matrix to store the eigenvalues and eigenvectors
  DataVector eigenvalues_real_part(eigenvalue_size, 0.0);
  DataVector eigenvalues_im_part(eigenvalue_size, 0.0);
  Matrix eigenvectors_real(eigenvector_size, eigenvalue_size, 0.0);

  const double sigma = -1.0;

  find_generalized_eigenvalues_arpack(
      eigenvalues_real_part, eigenvalues_im_part, eigenvectors_real, matrix_a,
      matrix_b, number_of_eigenvalues_to_find, sigma,
      which_eigenvalues_to_find);

  // Expected eigenvalues and eigenvectors
  constexpr double expected_closest_to_zero_eigenvalue = 1.0;
  constexpr double expected_second_closest_to_zero_eigenvalue = 2.0;

  CHECK(min(eigenvalues_real_part) ==
        approx(expected_closest_to_zero_eigenvalue));
  CHECK(eigenvalues_im_part[0] == approx(0.0));

  CHECK(max(eigenvalues_real_part) ==
        approx(expected_second_closest_to_zero_eigenvalue));
  CHECK(eigenvalues_im_part[1] == approx(0.0));

  // We don't know which order the eigenvalues are returned by Arpack, so check
  // which eigenvalue is larger and then verify that the corresponding
  // eigenvector's nontrivial component has the expected value.
  if (eigenvalues_real_part[1] > eigenvalues_real_part[0]) {
    CHECK(abs(eigenvectors_real(0, 0)) == approx(1.0));
    CHECK(abs(eigenvectors_real(1, 1)) == approx(1.0));
  } else {
    CHECK(abs(eigenvectors_real(0, 1)) == approx(1.0));
    CHECK(abs(eigenvectors_real(1, 0)) == approx(1.0));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearAlgebra.GeneralizedEigenvalueArpack",
                  "[NumericalAlgorithms][LinearAlgebra][Unit]") {
  // We will get an error if we ask Arpack to find more than n-2 eigenvalues for
  // a matrix of size n*n. Thus we are using a 4x4 matrix and asking to find 2
  // eigenvalues.
  constexpr std::size_t num_rows = 4;
  Matrix matrix_a(num_rows, num_rows, 0.0);
  Matrix matrix_b(num_rows, num_rows, 0.0);
  const std::string which_eigenvalues_to_find = "LM";

  for (std::size_t i = 0; i < num_rows; i++) {
    for (std::size_t j = 0; j < num_rows; j++) {
      if (i == j) {
        matrix_a(i, j) = j + 1;
        matrix_b(i, j) = 1;
      }
    }
  }
  test_find_generalized_eigenvalues_arpack(matrix_a, matrix_b, num_rows, 2, 2,
                                           which_eigenvalues_to_find);
}

// [[OutputRegex, Matrix A should be square]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Numerical.LinearAlgebra.GenEvalArpackAssertSquare",
    "[NumericalAlgorithms][LinearAlgebra][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Matrix matrix_a{{1.0}, {-3.0}};
  Matrix matrix_b{{4.0, -3.0}, {-2.0, 1.0}};
  const std::string which_eigenvalues_to_find = "LM";

  test_find_generalized_eigenvalues_arpack(matrix_a, matrix_b, 1, 1, 1,
                                           which_eigenvalues_to_find);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Matrix A and matrix B should be the same size]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Numerical.LinearAlgebra.GenEvalArpackAssertABSameSize",
    "[NumericalAlgorithms][LinearAlgebra][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Matrix matrix_a{{1.0, 2.0}, {-3.0, -4.0}};
  Matrix matrix_b{{4.0}};
  const std::string which_eigenvalues_to_find = "LM";

  test_find_generalized_eigenvalues_arpack(matrix_a, matrix_b, 1, 1, 1,
                                           which_eigenvalues_to_find);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Eigenvalue DataVector size and number of columns in the]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Numerical.LinearAlgebra.GenEvalArpackAssertSizeEigenvalues",
    "[NumericalAlgorithms][LinearAlgebra][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Matrix matrix_a{{1.0, 2.0}, {-3.0, -4.0}};
  Matrix matrix_b{{4.0, -3.0}, {-2.0, 1.0}};
  const std::string which_eigenvalues_to_find = "LM";

  test_find_generalized_eigenvalues_arpack(matrix_a, matrix_b, 2, 1, 2,
                                           which_eigenvalues_to_find);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Matrix A and matrix eigenvectors should have]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Numerical.LinearAlgebra.GenEvalArpackAssertSizeEigenvectors",
    "[NumericalAlgorithms][LinearAlgebra][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Matrix matrix_a{{1.0, 2.0}, {-3.0, -4.0}};
  Matrix matrix_b{{4.0, -3.0}, {-2.0, 1.0}};
  const std::string which_eigenvalues_to_find = "LM";

  test_find_generalized_eigenvalues_arpack(matrix_a, matrix_b, 1, 2, 2,
                                           which_eigenvalues_to_find);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, which_eigenvalues_to_find must be one of]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Numerical.LinearAlgebra.GenEvalArpackCheckWhichEigenvalueParameter",
    "[NumericalAlgorithms][LinearAlgebra][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Matrix matrix_a{{1.0, 2.0}, {-3.0, -4.0}};
  Matrix matrix_b{{4.0, -3.0}, {-2.0, 1.0}};
  const std::string which_eigenvalues_to_find = "Wrong_Input";

  test_find_generalized_eigenvalues_arpack(matrix_a, matrix_b, 2, 2, 2,
                                           which_eigenvalues_to_find);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
