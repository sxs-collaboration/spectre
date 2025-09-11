// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/SparseMatrixFiller.hpp"

#include "DataStructures/SimpleSparseMatrix.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

SparseMatrixFiller::SparseMatrixFiller(const size_t num_cols,
                                       const bool use_map_method)
    : num_rows_(num_cols),
      num_cols_(num_cols),
      use_map_method_(use_map_method) {
  if (not use_map_method) {
    matrix_elements_.assign(square(num_cols), 0.0);
  }
}

void SparseMatrixFiller::add(const double element, const size_t dest_index,
                             const size_t src_index) {
  if (not use_map_method_) {
    matrix_elements_[src_index + num_cols_ * dest_index] += element;
  } else {
    const std::pair index{dest_index, src_index};
    if (const auto iter = element_index_.find(index);
        iter != element_index_.end()) {
      matrix_elements_[iter->second] += element;
    } else {
      element_index_.insert({index, matrix_elements_.size()});
      matrix_elements_.push_back(element);
      dest_indices_.push_back(dest_index);
      src_indices_.push_back(src_index);
    }
  }
}

void SparseMatrixFiller::fill_sparse_matrix_elements(
    const gsl::not_null<std::vector<SparseMatrixElement>*> data) const {
  // First fill data for sorting.
  if (not use_map_method_) {
    const auto num_rows =
        static_cast<size_t>(sqrt(double(matrix_elements_.size())));
    ASSERT(num_rows * num_rows == matrix_elements_.size(),
           "Size should be a perfect square, not " << matrix_elements_.size());
    // For this method, many elements are zero so filter them out here.
    const auto num_zeros =
        std::count(matrix_elements_.begin(), matrix_elements_.end(), 0.0);
    data->reserve(matrix_elements_.size() - static_cast<size_t>(num_zeros));
    size_t indx = 0;
    for (size_t i_dest = 0; i_dest < num_rows; ++i_dest) {
      for (size_t j_src = 0; j_src < num_rows; ++j_src, ++indx) {
        const double element = matrix_elements_[indx];
        if (element != 0.0) {
          data->emplace_back(i_dest, j_src, element);
        }
      }
    }
  } else {
    ASSERT(matrix_elements_.size() == dest_indices_.size(),
           "Size mismatch value " << matrix_elements_.size() << " vs dest "
                                  << dest_indices_.size());
    ASSERT(matrix_elements_.size() == src_indices_.size(),
           "Size mismatch value " << matrix_elements_.size() << " vs src "
                                  << src_indices_.size());
    data->reserve(matrix_elements_.size());
    for (size_t i = 0; i < dest_indices_.size(); ++i) {
      data->emplace_back(dest_indices_[i], src_indices_[i],
                         matrix_elements_[i]);
    }
  }

  // Now sort the data by row and column, so we can fill in required order.
  std::sort(data->begin(), data->end(),
            [](const SparseMatrixElement& a, const SparseMatrixElement& b) {
              return a.row_dest == b.row_dest ? a.column_src < b.column_src
                                              : a.row_dest < b.row_dest;
            });
}

void SparseMatrixFiller::fill(
    const gsl::not_null<blaze::CompressedMatrix<double, blaze::rowMajor>*>
        matrix) const {
  // fill the elements and sort them.
  std::vector<SparseMatrixElement> data;
  fill_sparse_matrix_elements(make_not_null(&data));

  // Fill the matrix.
  // Do this by reserving a size, and appending elements row by row
  // (which must be done in order or else blaze dox say undefined
  // behavior), which is the reason for the above sort.
  // Note that this filling order assumes a rowMajor matrix; for columnMajor
  // matrices (not treated here) we must fill columns in order.
  // We need to explicitly call finalize on every row, *even empty ones*.
  matrix->resize(num_rows_, num_cols_, false);
  matrix->reserve(data.size());
  size_t current_row = 0;
  for (const auto& d : data) {
    while (d.row_dest != current_row) {
      matrix->finalize(current_row++);
    }
    matrix->append(d.row_dest, d.column_src, d.value);
  }
  // After the last data point, we need to finalize the last row(s).
  while (current_row < num_rows_) {
    matrix->finalize(current_row++);
  }
}

void SparseMatrixFiller::fill(
    const gsl::not_null<SimpleSparseMatrix*> matrix) const {
  // fill the elements and sort them.
  std::vector<SparseMatrixElement> data;
  fill_sparse_matrix_elements(make_not_null(&data));
  // Now fill.
  matrix->fill(data);
}
