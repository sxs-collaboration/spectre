// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/StripeIterator.hpp"

#include <numeric>

#include "DataStructures/Index.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"

template <size_t Dim>
StripeIterator::StripeIterator(const Index<Dim>& extents,
                               const size_t stripe_dim)
    : offset_(0),
      size_(extents.product()),
      stride_(std::accumulate(extents.begin(), extents.begin() + stripe_dim,
                              1_st, std::multiplies<size_t>())),
      jump_((extents[stripe_dim] - 1) * stride_) {}

StripeIterator& StripeIterator::operator++() {
  ++offset_;
  if (UNLIKELY(0 == (offset_ % stride_))) {
    offset_ += jump_;
  }
  return *this;
}

/// \cond HIDDEN_SYMBOLS
template StripeIterator::StripeIterator(const Index<1>&, const size_t);
template StripeIterator::StripeIterator(const Index<2>&, const size_t);
template StripeIterator::StripeIterator(const Index<3>&, const size_t);
/// \endcond
