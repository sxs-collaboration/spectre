// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <blaze/math/CompressedMatrix.h>
#include <cstddef>

#include "DataStructures/SparseMatrixFiller.hpp"
#include "Utilities/Gsl.hpp"

namespace {

template <typename SparseMatrixType>
void test_sparse_matrix_filler(const bool use_map_method) {
  const size_t num_cols = 6;

  SparseMatrixFiller filler(num_cols, use_map_method, 1.0);
  // Add in random order.
  // Note that empty rows must be explicitly added in the blaze
  // routines, so here we have rows with zero elements, rows with 1
  // element, and rows with >1 element to test all the edge cases.
  // Also note that the last row is empty, which tests another edge case.
  filler.add(4.0, 0, 0);
  filler.add(6.0, 1, 3);
  filler.add(2.0, 1, 3);  // This element is repeated! Important for test.
  filler.add(3.0, 0, 3);
  filler.add(7.0, 2, 2);
  filler.add(5.0, 4, 5);
  filler.add(3.0, 4, 3);
  filler.add(7.0, 0, 3);  // This element is repeated! Important for test.

  SparseMatrixType matrix;
  filler.fill(make_not_null(&matrix));

  if constexpr (not std::is_same_v<SparseMatrixType, SimpleSparseMatrix>) {
    // Matrix is square so number of rows and columns is the same.
    // SimpleSparseMatrix doesn't have rows, columns, or size member fns.
    CHECK(matrix.rows() == num_cols);
    CHECK(matrix.columns() == num_cols);
    CHECK(size(matrix) == num_cols * num_cols);
  }

  CHECK(matrix(0, 0) == 4.0);
  CHECK(matrix(1, 3) == 8.0);
  CHECK(matrix(0, 3) == 10.0);
  CHECK(matrix(2, 2) == 7.0);
  CHECK(matrix(4, 5) == 5.0);
  CHECK(matrix(4, 3) == 3.0);

  // Values that we did not fill
  CHECK(matrix(1, 0) == 0.0);
  CHECK(matrix(2, 0) == 0.0);
  CHECK(matrix(3, 0) == 0.0);
  CHECK(matrix(4, 0) == 0.0);
  CHECK(matrix(5, 0) == 0.0);
  CHECK(matrix(0, 1) == 0.0);
  CHECK(matrix(1, 1) == 0.0);
  CHECK(matrix(2, 1) == 0.0);
  CHECK(matrix(3, 1) == 0.0);
  CHECK(matrix(4, 1) == 0.0);
  CHECK(matrix(5, 1) == 0.0);
  CHECK(matrix(0, 2) == 0.0);
  CHECK(matrix(1, 2) == 0.0);
  CHECK(matrix(3, 2) == 0.0);
  CHECK(matrix(4, 2) == 0.0);
  CHECK(matrix(5, 2) == 0.0);
  CHECK(matrix(2, 3) == 0.0);
  CHECK(matrix(3, 3) == 0.0);
  CHECK(matrix(5, 3) == 0.0);
  CHECK(matrix(0, 4) == 0.0);
  CHECK(matrix(1, 4) == 0.0);
  CHECK(matrix(2, 4) == 0.0);
  CHECK(matrix(3, 4) == 0.0);
  CHECK(matrix(4, 4) == 0.0);
  CHECK(matrix(5, 4) == 0.0);
  CHECK(matrix(0, 5) == 0.0);
  CHECK(matrix(1, 5) == 0.0);
  CHECK(matrix(2, 5) == 0.0);
  CHECK(matrix(3, 5) == 0.0);
  CHECK(matrix(5, 5) == 0.0);
}

// This test touches some edge cases for Blaze matrices
// that the other tests do not.
void test_sparse_matrix_filler_last_row_filled(const bool use_map_method) {
  const size_t num_cols = 3;

  SparseMatrixFiller filler(num_cols, use_map_method, 1.0);
  // Add in random order.
  // First row has only one element (unlike the other tests)
  // Last row is nonempty with one nonzero column (unlike the other tests)
  filler.add(6.0, 1, 2);
  filler.add(7.0, 2, 2);
  filler.add(4.0, 0, 0);
  filler.add(2.0, 1, 2);  // This element is repeated! Important for test.

  blaze::CompressedMatrix<double, blaze::rowMajor> matrix;
  filler.fill(make_not_null(&matrix));

  // Matrix is square so number of rows and columns is the same.
  CHECK(matrix.rows() == num_cols);
  CHECK(matrix.columns() == num_cols);
  CHECK(size(matrix) == num_cols * num_cols);

  CHECK(matrix(0, 0) == 4.0);
  CHECK(matrix(1, 2) == 8.0);
  CHECK(matrix(2, 2) == 7.0);

  // Values that we did not fill
  CHECK(matrix(1, 0) == 0.0);
  CHECK(matrix(2, 0) == 0.0);
  CHECK(matrix(0, 1) == 0.0);
  CHECK(matrix(1, 1) == 0.0);
  CHECK(matrix(2, 1) == 0.0);
  CHECK(matrix(0, 2) == 0.0);
}

// This test touches some edge cases for Blaze matrices
// that the other tests do not.
void test_sparse_matrix_filler_first_row_empty(const bool use_map_method) {
  const size_t num_cols = 3;

  SparseMatrixFiller filler(num_cols, use_map_method, 1.0);
  // Add in random order.
  // First row is empty (unlike the other tests)
  // Last row is nonempty with multiple nonzero columns (unlike the other tests)
  filler.add(7.0, 2, 2);
  filler.add(6.0, 1, 2);
  filler.add(9.0, 2, 0);
  filler.add(2.0, 1, 2);  // This element is repeated! Important for test.

  blaze::CompressedMatrix<double, blaze::rowMajor> matrix;
  filler.fill(make_not_null(&matrix));

  // Matrix is square so number of rows and columns is the same.
  CHECK(matrix.rows() == num_cols);
  CHECK(matrix.columns() == num_cols);
  CHECK(size(matrix) == num_cols * num_cols);

  CHECK(matrix(2, 0) == 9.0);
  CHECK(matrix(1, 2) == 8.0);
  CHECK(matrix(2, 2) == 7.0);

  // Values that we did not fill
  CHECK(matrix(0, 0) == 0.0);
  CHECK(matrix(1, 0) == 0.0);
  CHECK(matrix(0, 1) == 0.0);
  CHECK(matrix(1, 1) == 0.0);
  CHECK(matrix(2, 1) == 0.0);
  CHECK(matrix(0, 2) == 0.0);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.SparseMatrixFiller",
                  "[DataStructures][Unit]") {
  test_sparse_matrix_filler<SimpleSparseMatrix>(true);
  test_sparse_matrix_filler<SimpleSparseMatrix>(false);
  test_sparse_matrix_filler<blaze::CompressedMatrix<double, blaze::rowMajor>>(
      true);
  test_sparse_matrix_filler<blaze::CompressedMatrix<double, blaze::rowMajor>>(
      false);
  test_sparse_matrix_filler_last_row_filled(true);
  test_sparse_matrix_filler_last_row_filled(false);
  test_sparse_matrix_filler_first_row_empty(true);
  test_sparse_matrix_filler_first_row_empty(false);
}
