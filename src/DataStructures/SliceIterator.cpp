// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/SliceIterator.hpp"

#include <functional>
#include <numeric>

#include "DataStructures/Index.hpp"  // IWYU pragma: keep
#include "Utilities/Literals.hpp"

template <size_t Dim>
SliceIterator::SliceIterator(const Index<Dim>& extents, const size_t fixed_dim,
                             const size_t fixed_index)
    : size_(extents.product()),
      stride_(std::accumulate(extents.begin(), extents.begin() + fixed_dim,
                              1_st, std::multiplies<size_t>())),
      stride_count_(0),
      jump_((extents[fixed_dim] - 1) * stride_),
      initial_offset_(fixed_index * stride_),
      volume_offset_(initial_offset_),
      slice_offset_(0) {}

SliceIterator& SliceIterator::operator++() {
  ++volume_offset_;
  ++slice_offset_;
  ++stride_count_;
  if (stride_count_ == stride_) {
    volume_offset_ += jump_;
    stride_count_ = 0;
  }
  return *this;
}

void SliceIterator::reset() {
  volume_offset_ = initial_offset_;
  slice_offset_ = 0;
}

/// \cond HIDDEN_SYMBOLS
template SliceIterator::SliceIterator(const Index<1>&, const size_t,
                                      const size_t);
template SliceIterator::SliceIterator(const Index<2>&, const size_t,
                                      const size_t);
template SliceIterator::SliceIterator(const Index<3>&, const size_t,
                                      const size_t);
/// \endcond
