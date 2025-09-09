// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <blaze/math/CompressedMatrix.h>
#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/SimpleSparseMatrix.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace sparse_matrix_filler_detail {
/// Hash on a pair of size_ts
struct PairHasher {
  size_t operator()(const std::pair<size_t, size_t>& key) const {
    ASSERT(key.first < 4294967295, "Key.first too large: " << key.first);
    ASSERT(key.second < 4294967295, "Key.second too large: " << key.second);
    return key.first bitor (key.second << 32);
  }
};
}  // namespace sparse_matrix_filler_detail

/*!
 * \ingroup DataStructuresGroup
 * \brief A data structure used to fill sparse matrices.
 *
 * We normally use blaze::CompressedMatrix to hold sparse matrices and to
 * operate on them.
 *
 * Unfortunately to fill a blaze::CompressedMatrix efficiently you
 * need to fill nonzero matrix elements in a well-defined order
 * (i.e. fill all nonzero elements in the top row in the order of
 * nonzero columns, and repeat for each other row, in order) using
 * CompressedMatrix's reserve(), append(), and finalize() functions.
 *
 * However, we are interested in a use case where:
 *
 *   1. We are filling the matrix elements using a complicated
 *      algorithm that computes (as opposed to systematically loops
 *      over) the matrix element indices to be filled, so we are
 *      effectively filling matrix elements in a random order.
 *
 *   2. The algorithm can traverse the same matrix element hundreds
 *      of times, meaning that the first time an element is traversed
 *      it should be filled, but subsequent times it is traversed it
 *      should be incremented.
 *
 * SparseMatrixFiller provides a function to fill matrix elements, keeping
 * in mind the above two issues.  And it provides a function to copy its
 * sparse matrix into blaze::CompressedMatrix efficiently.
 *
 * We consider two ways to keep track of unique elements:
 *
 * 1. Store matrix elements in a full vector that is indexed by
 *    matrix_elements[src_index+max_src_index*dest_index].  Set this
 *    vector to zero initially, and increment it each time 'add' is called.
 *    This is simple, but has a major issue: memory.
 *    In SpEC, I (Mark Scheel) have measured that computing TensorYlm
 *    filtering matrices using this method, after l_max=40 , the
 *    slope of CPU-h vs l_max for this method has a kink, and CPU-h
 *    increases like l_max^2 instead of l_max.  Presumably this is
 *    because of cache misses when accessing the very large
 *    matrix_elements vector at random locations.
 *
 * 2. Have a std::unordered_map<std::pair<size_t,size_t>, size_t>>
 *    called `element_index` that keeps track of the unique elements.
 *    Also have three vectors `dest_indices`, `src_indices`,
 *    and `matrix_elements` that keep the nonzero elements.
 *    element_index is indexed by
 *    element_index[{dest_index,src_index}] = index-into-matrix-elements
 *    So each time we add a new element, we check element_index to see
 *    if that element has not already been added, and if not, we push_back
 *    the three vectors.  If it has already been added, we simply do
 *    matrix_elements[index-into-matrix-elements] += new_element.
 *
 * I (Mark Scheel) have found in SpEC for computing TensorYlm filtering
 * matrices that the CPU cost of algorithm 2. above scales linearly
 * with l_max at least up to l_max=140, but for l_max < 40 or so,
 * algorithm 2. is about a factor of 3 slower than algorithm 1.
 * So SpEC uses both algorithms, depending on the value of l_max.
 *
 * For SpECTRE, we plan to do the same: for small l_max we will use
 * algorithm 1 and for large l_max we will use algorithm 2.  The
 * cutoff for l_max will depend on other factors (like the rank of the
 * Tensor being filtered) and will be determined by whoever calls
 * SparseMatrixFiller.
 */
class SparseMatrixFiller {
 public:
  /// Assume num_rows and num_cols are equal.
  /// If use_map_method is true, uses algorithm 2.
  SparseMatrixFiller(size_t num_cols, bool use_map_method);

  /// Add (or increment) an element such that for v = M x, where M is
  /// the matrix and v and x are vectors, we compute the contribution
  /// to v[dest_index] from x[src_index].
  void add(double element, size_t dest_index, size_t src_index);

  /// Fills the matrix.
  /// Note that the order in which blaze matrices must be filled depends
  /// on whether the matrix is rowMajor or columnMajor, so changing
  /// the template parameter here will fail without changing the code.
  void fill(gsl::not_null<blaze::CompressedMatrix<double, blaze::rowMajor>*>
                matrix) const;

  /// Fills a SimpleSparseMatrix.
  void fill(gsl::not_null<SimpleSparseMatrix*> matrix) const;

 private:
  void fill_sparse_matrix_elements(
      gsl::not_null<std::vector<SparseMatrixElement>*> data) const;
  size_t num_rows_{0}, num_cols_{0};
  bool use_map_method_{true};
  std::vector<double> matrix_elements_{};
  std::unordered_map<std::pair<size_t, size_t>, size_t,
                     sparse_matrix_filler_detail::PairHasher>
      element_index_{};
  std::vector<size_t> dest_indices_{};
  std::vector<size_t> src_indices_{};
};
