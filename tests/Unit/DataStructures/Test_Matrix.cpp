// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <string>

#include "DataStructures/Matrix.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Matrix", "[DataStructures][Unit]") {
  Matrix matrix(3, 5, 1.0);
  CHECK(matrix.rows() == 3);
  CHECK(matrix.columns() == 5);
  // const double* a_ptr = matrix.data();
  // std::stringstream ss;
  for (size_t i = 0; i < matrix.rows(); ++i) {
    // ss << "{ ";
    for (size_t j = 0; j < matrix.columns(); ++j) {
      CHECK(1.0 == matrix(i, j));
      // TODO: Do we really need this test? It checks the data layout, which
      // should be irrelevant as long as all functionality works correctly.
      // clang-tidy: do not use pointer arithmetic
      // CHECK(&a_ptr[i * matrix.columns() + j] == &matrix(i, j));  // NOLINT
      // ss << matrix(i, j) << " ";
    }
    // ss << "}\n";
  }
  // TODO: Do we really need this test, too? Blaze provides a nicely
  // formatted `<<` that's not straight-forward to reproduce here due to
  // number padding. If this exact formatting is required somewhere it
  // should be their responsibility to enforce it and test it.
  // std::stringstream os;
  // os << matrix;
  // CHECK(ss.str() == os.str());

  test_copy_semantics(matrix);
  auto matrix_copy = matrix;
  test_move_semantics(std::move(matrix), matrix_copy);
}

SPECTRE_TEST_CASE("Unit.Serialization.Matrix",
                  "[DataStructures][Unit][Serialization]") {
  Matrix matrix(3, 7, 1.0);
  test_serialization(matrix);
}
