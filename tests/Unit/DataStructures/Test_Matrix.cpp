// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

#include "DataStructures/Matrix.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Matrix", "[DataStructures][Unit]") {
  Matrix matrix(3, 5, 1.0);
  CHECK(matrix.rows() == 3);
  CHECK(matrix.columns() == 5);
  Matrix matrix_vectinit({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
  CHECK(matrix_vectinit.rows() == 3);
  CHECK(matrix_vectinit.columns() == 2);
  CHECK(matrix_vectinit(0, 0) == 1.0);
  CHECK(matrix_vectinit(0, 1) == 2.0);
  CHECK(matrix_vectinit(1, 0) == 3.0);
  CHECK(matrix_vectinit(1, 1) == 4.0);
  CHECK(matrix_vectinit(2, 0) == 5.0);
  CHECK(matrix_vectinit(2, 1) == 6.0);
  Matrix matrix_empty(std::vector<std::vector<double>>{});
  CHECK(matrix_empty.rows() == 0);
  CHECK(matrix_empty.columns() == 0);
  CHECK(matrix_empty == Matrix(0, 0, 1.0));
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
