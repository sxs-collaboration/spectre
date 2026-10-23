// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <vector>

#include "DataStructures/SimpleSparseMatrix.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"

namespace {
void test_simple_sparse_matrix() {
  SimpleSparseMatrix A;

  // Important that these are emplaced_back sorted by row.
  // Otherwise we should use SparseMatrixFiller to fill the matrices.
  std::vector<SparseMatrixElement> elements{
      {0, 0, 4.0}, {0, 3, 3.0}, {1, 3, 6.0}, {1, 5, 1.0},
      {2, 2, 7.0}, {4, 3, 3.0}, {4, 5, 5.0}};

  A.fill(elements);

  // Check that elements are filled correctly and that
  // SimpleSparseMatrix indexing is working.
  for (size_t row = 0; row < 5; ++row) {
    for (size_t column = 0; column < 5; ++column) {
      const auto it =
          alg::find_if(elements, [row, column](const SparseMatrixElement& t) {
            return t.row_dest == row and t.column_src == column;
          });
      if (it != elements.end()) {
        CHECK(A(row, column) == it->value);
      } else {
        CHECK(A(row, column) == 0.0);
      }
    }
  }

  // Create vector x
  const std::vector<double> x{2.0, 3.0, 4.0, 5.0, 6.0, 7.0};

  // Create vector y = Ax
  const std::vector<double> y{23.0, 37.0, 28.0, 0.0, 50.0};

  std::vector<double> ytest(5, 0);
  A.increment_multiply_on_right(make_not_null(&ytest), 0, 1, x, 0, 1);

  for (size_t i = 0; i < ytest.size(); ++i) {
    CHECK(y[i] == approx(ytest[i]));
  }

  // Now test offsets and strides
  // xx is the same as x, except offset is 1 and stride is 2.
  const std::vector<double> xx{1.0,  2.0, 4.0,  3.0, 9.0,  4.0,
                               25.0, 5.0, 49.0, 6.0, 64.0, 7.0};
  // yytest will be the same as ytest, except offset is 2 and stride is 3.
  std::vector<double> yytest(17, 0);
  A.increment_multiply_on_right(make_not_null(&yytest), 2, 3, xx, 1, 2);

  for (size_t i = 0; i < y.size(); ++i) {
    CHECK(y[i] == approx(yytest[2 + 3 * i]));
  }

  // Compare all offsets with separate scalar multiplies, including zero
  // stride and the large-stride fallback.
  for (size_t stride = 0; stride <= 33; ++stride) {
    CAPTURE(stride);
    // Offset the spans and leave sentinels on both sides of the used storage.
    std::vector<double> source(6 * stride + 2);
    std::vector<double> expected(5 * stride + 2);
    for (size_t i = 0; i < source.size(); ++i) {
      source[i] = 0.13 * static_cast<double>(i) - 1.7;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
      expected[i] = -0.017 * static_cast<double>(i) + 0.31;
    }
    auto actual = expected;
    for (size_t offset = 0; offset < stride; ++offset) {
      A.increment_multiply_on_right(make_not_null(&expected), offset + 1,
                                    stride, source, offset + 1, stride);
    }
    const auto source_span =
        gsl::make_span(source).subspan(1, source.size() - 2);
    auto actual_span = gsl::make_span(actual).subspan(1, actual.size() - 2);
    A.increment_multiply_on_right_all_offsets(make_not_null(&actual_span),
                                              source_span, stride);
    CHECK_ITERABLE_APPROX(actual, expected);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.SimpleSparseMatrix",
                  "[DataStructures][Unit]") {
  test_simple_sparse_matrix();
}
