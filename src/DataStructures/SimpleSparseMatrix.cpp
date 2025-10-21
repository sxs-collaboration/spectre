// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/SimpleSparseMatrix.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <vector>

#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

double SimpleSparseMatrix::operator()(const size_t row_dest_index,
                                      const size_t column_src_index) const {
  bool found_dest = false;
  const auto dest_it =
      alg::find_if(row_dest_indices_,
                   [&found_dest, row_dest_index](const size_t dest_index) {
                     found_dest = row_dest_index == dest_index;
                     return found_dest or row_dest_index < dest_index;
                   });
  if (not found_dest) {
    return 0.0;
  }
  const auto dest_index =
      static_cast<size_t>(std::distance(row_dest_indices_.begin(), dest_it));

  const size_t src_index = std::accumulate(
      num_columns_per_row_.begin(),
      num_columns_per_row_.begin() + (dest_it - row_dest_indices_.begin()),
      0_st);

  bool found_src = false;
  const auto src_it = std::find_if(
      column_src_indices_.begin() +
          static_cast<std::vector<size_t>::difference_type>(src_index),
      column_src_indices_.begin() +
          static_cast<std::vector<size_t>::difference_type>(
              src_index + num_columns_per_row_[dest_index]),
      [&found_src, column_src_index](const size_t val) {
        found_src = column_src_index == val;
        return found_src or column_src_index < val;
      });
  if (not found_src) {
    return 0.0;
  }
  return matrix_elements_[static_cast<size_t>(src_it -
                                              column_src_indices_.begin())];
}

void SimpleSparseMatrix::fill(const std::vector<SparseMatrixElement>& data) {
  if (data.empty()) {
    // Nothing to do.
    return;
  }

  // Figure out number of rows with nonzero matrix elements.  Note that data has
  // already been sorted by row.
  size_t num_used_rows = 1;
  for (size_t i = 1; i < data.size(); ++i) {
    if (data[i].row_dest != data[i - 1].row_dest) {
      ++num_used_rows;
    }
  }

  num_columns_per_row_.resize(num_used_rows, 0);
  row_dest_indices_.resize(num_used_rows);
  matrix_elements_.resize(data.size());
  column_src_indices_.resize(data.size());

  // Current_row_index is the index of the current row in row_dest_indices
  // and num_columns_per_row.
  // Current_row_index goes from zero to num_used_rows - 1
  size_t current_row_index = 0;
  // current_row is the actual row number in the full matrix.
  size_t current_row = data[0].row_dest;
  row_dest_indices_[0] = current_row;
  for (const auto& d : data) {
    if (d.row_dest != current_row) {
      ++current_row_index;
      current_row = d.row_dest;
      row_dest_indices_[current_row_index] = current_row;
    }
    ++num_columns_per_row_[current_row_index];
  }
  for (size_t i = 0; i < data.size(); ++i) {
    matrix_elements_[i] = data[i].value;
    column_src_indices_[i] = data[i].column_src;
  }
}

template <typename T>
void SimpleSparseMatrix::increment_multiply_on_right(
    const gsl::not_null<T*> a, const size_t a_offset, const T& b,
    const size_t b_offset) const {
  if (matrix_elements_.empty()) {
    // Nothing to do
    return;
  }

  ASSERT(matrix_elements_.size() == column_src_indices_.size(),
         "Size mismatch between the size of matrix_elements_: "
             << matrix_elements_.size()
             << " and column_src_indices_: " << column_src_indices_.size());
  ASSERT(num_columns_per_row_.size() == row_dest_indices_.size(),
         "Size mismatch between the size of num_columns_per_row_: "
             << num_columns_per_row_.size()
             << " and row_dest_indices_: " << row_dest_indices_.size());

  // This particular implementation (with raw pointers and pointer
  // arithmetic) was found in SpEC to be fastest, back when we
  // tested several variations of this implementation with SpEC.
  const double* mat = matrix_elements_.data();
  const size_t* src_ind = column_src_indices_.data();
  const size_t* num_inner = num_columns_per_row_.data();
  const size_t* dest_ind = row_dest_indices_.data();

  // num_comps_a is the number of components of 'a' that we will touch.
  // (the matrix is sparse, so we don't touch all components of 'a')
  size_t num_comps_a = num_columns_per_row_.size();
  while (num_comps_a-- > 0) {
    // num_comps_b is the number of components of 'b' that we will touch,
    // for this component of 'a'.
    // (the matrix is sparse, so we don't touch all components of 'b')
    size_t num_comps_b = *num_inner;
    std::advance(num_inner, 1);
    double sum = 0.0;
    while (num_comps_b-- > 0) {
      sum += *mat * b[b_offset + *src_ind];
      std::advance(mat, 1);
      std::advance(src_ind, 1);
    }
    (*a)[a_offset + *dest_ind] += sum;
    std::advance(dest_ind, 1);
  }
}

// Explicit instantiations
template void SimpleSparseMatrix::increment_multiply_on_right(
    const gsl::not_null<std::vector<double>*> a, const size_t a_offset,
    const std::vector<double>& b, const size_t b_offset) const;
template void SimpleSparseMatrix::increment_multiply_on_right(
    const gsl::not_null<gsl::span<double>*> a, const size_t a_offset,
    const gsl::span<double>& b, const size_t b_offset) const;
