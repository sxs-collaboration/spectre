// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/SliceIterator.hpp"

#include <numeric>

#include "DataStructures/Index.hpp"
#include "Utilities/Literals.hpp"

template <size_t Dim>
SliceIterator::SliceIterator(const Index<Dim>& extents, const size_t fixed_dim,
                             const size_t fixed_index)
    : size_(extents.product()),
      stride_(std::accumulate(extents.begin(), extents.begin() + fixed_dim,
                              1_st, std::multiplies<size_t>())),
      jump_((extents[fixed_dim] - 1) * stride_),
      initial_offset_(fixed_index * stride_),
      volume_offset_(initial_offset_),
      slice_offset_(0) {}

SliceIterator& SliceIterator::operator++() {
  ++volume_offset_;
  ++slice_offset_;
  if (0 == (volume_offset_ % stride_)) {
    volume_offset_ += jump_;
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
