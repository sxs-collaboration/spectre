// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "Utilities/Gsl.hpp"

/// Struct used to fill SimpleSparsematrix
struct SparseMatrixElement {
  SparseMatrixElement(const size_t row, const size_t column, const double val)
      : row_dest(row), column_src(column), value(val) {}
  size_t row_dest, column_src;
  double value;
};

/*!
 * \brief A simple fast sparse matrix.
 *
 * Supports only a limited number of operations to keep it simple:
 * filling the matrix, accessing elements, and multiplication by
 * a vector on the right.
 *
 * We use SimpleSparseMatrix because filling a Blaze sparse matrix is
 * enormously slow. For example, here are timings for creating a
 * rank-one TensorYlmFilter sparse matrix using SimpleSparseMatrix
 * (column labeled "this") and using blaze::CompressedMatrix (column
 * labeled "Blaze").  The column "Blaze-wo-blaze" is the same as the
 * "Blaze" column (which includes timing of all the TensorYlmFilter
 * code that doesn't explicitly depend on blaze) minus the timings of
 * the blaze calls that 1. resize the matrix, 2. reserve elements, and
 * 3. fill the blaze matrix.  The timings for those individual blaze
 * calls are in subsequent columns.  Note that "Blaze-wo-blaze" is
 * roughly equivalent to "this" in speed.  Also note that the "this"
 * timings were done in SpEC, which uses SimpleSparseMatrix [which was
 * ported from SpEC to SpECTRE and simplified here].
 *
 * l_max this Blaze Blaze-wo-blaze blaze.resize blaze.reserve blaze.fill
 * ----------------------------------------------------------------------------
 *  8    0ms     5ms     1ms              1ms           2ms          0ms
 * 18    1ms   101ms     2ms             39ms          51ms          8ms
 * 28    3ms   479ms     3ms            181ms         246ms         48ms
 * 38    4ms  1555ms     4ms            571ms         822ms        157ms
 * 48    5ms  3668ms     6ms           1395ms        1873ms        393ms
 * 58    6ms  7604ms     6ms           2908ms        3855ms        832ms
 * 68    7ms 14176ms     8ms           5416ms        7214ms       1536ms
 * 78    8ms 24234ms     8ms           9288ms       12304ms       2632ms
 *
 */
class SimpleSparseMatrix {
 public:
  SimpleSparseMatrix() = default;

  /// Fills with data.
  /// Normally called only by SparseMatrixFiller.
  /// ASSUMES that data has already been sorted by row, then column.
  void fill(const std::vector<SparseMatrixElement>& data);

  /// Does a += m*b where m is the sparse matrix.
  /// T is a container of doubles, like a std::vector<double>.
  /// Assumes b points into contiguous storage that allows one
  /// to access b[i] for b_offset <= i <= b_offset+max(column_indices).
  /// Assumes a points into contiguous storage that allows one
  /// to access a[i] for a_offset <= a <= a_offset+max(row_indices).
  ///
  /// Both source and destination are allowed to have a stride.
  template <typename T>
  void increment_multiply_on_right(gsl::not_null<T*> a, size_t a_offset,
                                   size_t a_stride, const T& b, size_t b_offset,
                                   size_t b_stride) const;

  /// \brief Does `a += m * b` for `stride` right-hand sides at once.
  ///
  /// The result is the same as calling
  /// `increment_multiply_on_right(a, k, stride, b, k, stride)` once for every
  /// `k` in `[0, stride)`.
  ///
  /// This is faster than calling `increment_multiply_on_right` once per
  /// offset for two reasons:
  ///
  /// - The sparse matrix (its values and its index arrays) is read from
  ///   memory once instead of `stride` times.
  /// - For every matrix entry, the `stride` source values are read as one
  ///   contiguous block and added to `stride` independent partial sums, so
  ///   the compiler can use vector instructions. The per-offset loop instead
  ///   picks one value out of each block and adds it to a single running
  ///   sum, so every addition has to wait for the previous one.
  ///
  /// For each offset, the entries of a row are summed in the same order as
  /// in `increment_multiply_on_right`. Source and destination must not
  /// overlap. Does not allocate.
  ///
  /// Strides up to `max_fused_stride` use this fused kernel. Larger strides
  /// fall back to one call of `increment_multiply_on_right` per offset.
  void increment_multiply_on_right_all_offsets(
      gsl::not_null<gsl::span<double>*> a, const gsl::span<double>& b,
      size_t stride) const;

  /// Largest `stride` for which `increment_multiply_on_right_all_offsets`
  /// uses the fused kernel. One kernel is compiled per stride up to this
  /// value, which bounds the amount of generated code.
  static constexpr size_t max_fused_stride = 24;

  /// Number of nonzero elements.
  size_t size() const { return matrix_elements_.size(); }

  /// Obtains element at (destindex,srcindex).
  /// Slow, should not use very often in critical situations.
  double operator()(size_t row_dest_index, size_t column_src_index) const;

 private:
  // Holds all nonzero matrix elements.
  std::vector<double> matrix_elements_;
  // Holds indices of all rows of the matrix that have nonzero matrix elements.
  std::vector<size_t> row_dest_indices_;
  // Holds the number of columns in every row that has nonzero matrix elements.
  // row_dest_indices_ and num_columns_per_row_ have the same size (which is the
  // number of rows with nonzero matrix elements).
  // The sum of all the num_columns_per_row_ is the size of matrix_elements_.
  std::vector<size_t> num_columns_per_row_;
  // Holds the indices of all columns that have nonzero matrix elements.
  // column_indices_ has the same size as matrix_elements_, so many
  // of the column_src_indices_ might be repeated.
  std::vector<size_t> column_src_indices_;
};
