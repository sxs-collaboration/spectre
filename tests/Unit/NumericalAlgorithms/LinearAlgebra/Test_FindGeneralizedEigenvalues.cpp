// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "ErrorHandling/Error.hpp"
#include "NumericalAlgorithms/LinearAlgebra/FindGeneralizedEigenvalues.hpp"
#include "Utilities/Gsl.hpp"

namespace {
void test_find_generalized_eigenvalues(const Matrix& matrix_a,
                                       const Matrix& matrix_b,
                                       const size_t eigenvector_size,
                                       const size_t eigenvalue_size) {
  // Set up vectors and a matrix to store the eigenvalues and eigenvectors
  DataVector eigenvalues_real_part(eigenvalue_size, 0.0);
  DataVector eigenvalues_im_part(eigenvalue_size, 0.0);
  Matrix eigenvectors(eigenvector_size, eigenvector_size, 0.0);

  find_generalized_eigenvalues(&eigenvalues_real_part, &eigenvalues_im_part,
                               &eigenvectors, matrix_a, matrix_b);

  // Expected eigenvalues and eigenvectors
  constexpr double expected_larger_eigenvalue = 10.099019513592784;
  constexpr double expected_smaller_eigenvalue = -0.09901951359278449;
  // We are testing a 2x2 matrix. Eigenvectors have two components.
  // Normalize the eigenvectors so the second component is 1.
  // Store the remaining components of the expected eigenvectors.
  constexpr double expected_larger_eigenvector_value = 0.819803902718557;
  constexpr double expected_smaller_eigenvector_value = -1.2198039027185574;

  CHECK(eigenvalues_im_part[0] == approx(0.0));
  CHECK(eigenvalues_im_part[1] == approx(0.0));
  CHECK(max(eigenvalues_real_part) == approx(expected_larger_eigenvalue));
  CHECK(min(eigenvalues_real_part) == approx(expected_smaller_eigenvalue));

  // We don't know which order the eigenvalues are returned by lapack, so
  // check which eigenvalue is larger and then verify that the corresponding
  // eigenvector's nontrivial component has the expected value.
  if (eigenvalues_real_part[0] > eigenvalues_real_part[1]) {
    CHECK(eigenvectors(0, 0) / eigenvectors(1, 0) ==
          approx(expected_larger_eigenvector_value));
    CHECK(eigenvectors(0, 1) / eigenvectors(1, 1) ==
          approx(expected_smaller_eigenvector_value));
  } else {
    CHECK(eigenvectors(0, 0) / eigenvectors(1, 0) ==
          approx(expected_smaller_eigenvector_value));
    CHECK(eigenvectors(0, 1) / eigenvectors(1, 1) ==
          approx(expected_larger_eigenvector_value));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearAlgebra.GeneralizedEigenvalue",
                  "[NumericalAlgorithms][LinearAlgebra][Unit]") {
  Matrix matrix_a{{1.0, 2.0}, {-3.0, -4.0}};
  Matrix matrix_b{{4.0, -3.0}, {-2.0, 1.0}};
  test_find_generalized_eigenvalues(matrix_a, matrix_b, 2, 2);
}

// [[OutputRegex, Matrix A should be square]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Numerical.LinearAlgebra.GeneralizedEigenvalueAssertSquare",
    "[NumericalAlgorithms][LinearAlgebra][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Matrix matrix_a{{1.0}, {-3.0}};
  Matrix matrix_b{{4.0, -3.0}, {-2.0, 1.0}};
  test_find_generalized_eigenvalues(matrix_a, matrix_b, 2, 2);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Matrix A and matrix B should be the same size]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Numerical.LinearAlgebra.GeneralizedEigenvalueAssertABSameSize",
    "[NumericalAlgorithms][LinearAlgebra][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Matrix matrix_a{{1.0, 2.0}, {-3.0, -4.0}};
  Matrix matrix_b{{4.0}};
  test_find_generalized_eigenvalues(matrix_a, matrix_b, 2, 2);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Matrix A and matrix eigenvectors should have the same size]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Numerical.LinearAlgebra.GeneralizedEigenvalueAssertSizeEigenvectors",
    "[NumericalAlgorithms][LinearAlgebra][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Matrix matrix_a{{1.0, 2.0}, {-3.0, -4.0}};
  Matrix matrix_b{{4.0, -3.0}, {-2.0, 1.0}};
  test_find_generalized_eigenvalues(matrix_a, matrix_b, 1, 2);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, eigenvalues DataVector sizes should equal number of columns]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Numerical.LinearAlgebra.GeneralizedEigenvalueAssertSizeEigenvalues",
    "[NumericalAlgorithms][LinearAlgebra][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Matrix matrix_a{{1.0, 2.0}, {-3.0, -4.0}};
  Matrix matrix_b{{4.0, -3.0}, {-2.0, 1.0}};
  test_find_generalized_eigenvalues(matrix_a, matrix_b, 2, 1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
