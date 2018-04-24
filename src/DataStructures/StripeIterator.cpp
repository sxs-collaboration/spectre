// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/StripeIterator.hpp"

#include <functional>
#include <numeric>

#include "DataStructures/Index.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"

template <size_t Dim>
StripeIterator::StripeIterator(const Index<Dim>& extents,
                               const size_t stripe_dim)
    : offset_(0),
      size_(extents.product()),
      stride_(std::accumulate(extents.begin(), extents.begin() + stripe_dim,
                              1_st, std::multiplies<size_t>())),
      stride_count_(0),
      jump_((extents[stripe_dim] - 1) * stride_) {}

StripeIterator& StripeIterator::operator++() {
  ++offset_;
  ++stride_count_;
  if (UNLIKELY(stride_count_ == stride_)) {
    offset_ += jump_;
    stride_count_ = 0;
  }
  return *this;
}

/// \cond HIDDEN_SYMBOLS
template StripeIterator::StripeIterator(const Index<1>&, const size_t);
template StripeIterator::StripeIterator(const Index<2>&, const size_t);
template StripeIterator::StripeIterator(const Index<3>&, const size_t);
/// \endcond
