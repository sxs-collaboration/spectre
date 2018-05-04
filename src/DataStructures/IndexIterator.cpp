// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/IndexIterator.hpp"

#include "Utilities/Gsl.hpp"

template <size_t Dim>
IndexIterator<Dim>::IndexIterator(Index<Dim> extents)
    : extents_(std::move(extents)),
      index_(0),
      collapsed_index_(0),
      valid_(true) {}

template <size_t Dim>
IndexIterator<Dim>& IndexIterator<Dim>::operator++() {
  for (size_t d = 0;; ++d) {
    ++index_[d];
    if (extents_[d] > index_[d]) {
      break;
    }
    index_[d] = 0;
    if (UNLIKELY(Dim == d + 1)) {
      valid_ = false;
      break;
    }
  }
  ++collapsed_index_;
  return *this;
}

/// \cond
template <>
IndexIterator<0>& IndexIterator<0>::operator++() {
  valid_ = false;
  return *this;
}

template class IndexIterator<0>;
template class IndexIterator<1>;
template class IndexIterator<2>;
template class IndexIterator<3>;
template class IndexIterator<4>;
/// \endcond
