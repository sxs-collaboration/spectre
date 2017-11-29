// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <sstream>

#include "DataStructures/Matrix.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Matrix", "[DataStructures][Unit]") {
  Matrix matrix(3, 5, 1.0);
  CHECK(matrix.rows() == 3);
  CHECK(matrix.columns() == 5);
  const double* a_ptr = matrix.data();
  std::stringstream ss;
  for (size_t i = 0; i < matrix.rows(); ++i) {
    ss << "{ ";
    for (size_t j = 0; j < matrix.columns(); ++j) {
      CHECK(1.0 == matrix(i, j));
      // clang-tidy: do not use pointer arithmetic
      CHECK(&a_ptr[i + j * matrix.rows()] == &matrix(i, j));  // NOLINT
      ss << matrix(i, j) << " ";
    }
    ss << "}\n";
  }
  std::stringstream os;
  os << matrix;
  CHECK(ss.str() == os.str());

  test_copy_semantics(matrix);
  auto matrix_copy = matrix;
  test_move_semantics(std::move(matrix), matrix_copy);
}

SPECTRE_TEST_CASE("Unit.Serialization.Matrix",
                  "[DataStructures][Unit][Serialization]") {
  Matrix matrix(3, 7, 1.0);
  test_serialization(matrix);
}
