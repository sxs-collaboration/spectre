// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "Utilities/Gsl.hpp"

template <size_t>
class Index;

/*!
 * \ingroup DataStructuresGroup
 * \brief Iterates over the 1-dimensional stripes with info on how to
 * iterate over the current stripe
 */
class StripeIterator {
 public:
  /// Construct from the grid points in each direction and which dimension the
  /// stripes are in.
  template <size_t Dim>
  StripeIterator(const Index<Dim>& extents, size_t stripe_dim);

  /// Returns `true` if the iterator is valid
  explicit operator bool() const { return offset_ < size_; }

  /// Increment to the next stripe.
  StripeIterator& operator++() {
    ++offset_;
    ++stride_count_;
    if (UNLIKELY(stride_count_ == stride_)) {
      offset_ += jump_;
      stride_count_ = 0;
    }
    return *this;
  }

  /// Offset into DataVector for first element of stripe.
  size_t offset() const { return offset_; }

  /// Stride of elements in DataVector for the stripe.
  size_t stride() const { return stride_; }

 private:
  size_t offset_ = std::numeric_limits<size_t>::max();
  size_t size_ = std::numeric_limits<size_t>::max();
  size_t stride_ = std::numeric_limits<size_t>::max();
  size_t stride_count_ = std::numeric_limits<size_t>::max();
  size_t jump_ = std::numeric_limits<size_t>::max();
};
