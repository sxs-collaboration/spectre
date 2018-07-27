// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/ModalVector.hpp"

#include <algorithm>
#include <pup.h>

#include "Utilities/StdHelpers.hpp"

ModalVector::ModalVector(const size_t size, const double value)
    : owned_data_(size, value) {
  reset_pointer_vector();
}

ModalVector::ModalVector(double* start, size_t size) noexcept
    : BaseType(start, size), owned_data_(0), owning_(false) {}

template <class T, Requires<cpp17::is_same_v<T, double>>>
ModalVector::ModalVector(std::initializer_list<T> list)
    : owned_data_(std::move(list)) {
  reset_pointer_vector();
}

/// \cond HIDDEN_SYMBOLS
// clang-tidy: calling a base constructor other than the copy constructor.
//             We reset the base class in reset_pointer_vector after calling its
//             default constructor
ModalVector::ModalVector(const ModalVector& rhs) : BaseType{} {  // NOLINT
  if (rhs.is_owning()) {
    owned_data_ = rhs.owned_data_;
  } else {
    owned_data_.assign(rhs.begin(), rhs.end());
  }
  reset_pointer_vector();
}

ModalVector& ModalVector::operator=(const ModalVector& rhs) {
  if (this != &rhs) {
    if (owning_) {
      if (rhs.is_owning()) {
        owned_data_ = rhs.owned_data_;
      } else {
        owned_data_.assign(rhs.begin(), rhs.end());
      }
      reset_pointer_vector();
    } else {
      ASSERT(rhs.size() == size(), "Must copy into same size, not "
                                       << rhs.size() << " into " << size());
      std::copy(rhs.begin(), rhs.end(), begin());
    }
  }
  return *this;
}

ModalVector::ModalVector(ModalVector&& rhs) noexcept {
  owned_data_ = std::move(rhs.owned_data_);
  ~*this = ~rhs;  // PointerVector is trivially copyable
  owning_ = rhs.owning_;

  rhs.owning_ = true;
  rhs.reset();
}

ModalVector& ModalVector::operator=(ModalVector&& rhs) noexcept {
  if (this != &rhs) {
    if (owning_) {
      owned_data_ = std::move(rhs.owned_data_);
      ~*this = ~rhs;  // PointerVector is trivially copyable
      owning_ = rhs.owning_;
    } else {
      ASSERT(rhs.size() == size(), "Must copy into same size, not "
                                       << rhs.size() << " into " << size());
      std::copy(rhs.begin(), rhs.end(), begin());
    }
    rhs.owning_ = true;
    rhs.reset();
  }
  return *this;
}
/// \endcond

void ModalVector::pup(PUP::er& p) noexcept {  // NOLINT
  auto my_size = size();
  p | my_size;
  if (my_size > 0) {
    if (p.isUnpacking()) {
      owning_ = true;
      owned_data_.resize(my_size);
      reset_pointer_vector();
    }
    PUParray(p, data(), size());
  }
}

std::ostream& operator<<(std::ostream& os, const ModalVector& d) {
  // This function is inside the detail namespace StdHelpers.hpp
  StdHelpers_detail::print_helper(os, d.begin(), d.end());
  return os;
}

/// Equivalence operator for ModalVector
bool operator==(const ModalVector& lhs, const ModalVector& rhs) noexcept {
  return lhs.size() == rhs.size() and
         std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

/// Inequivalence operator for ModalVector
bool operator!=(const ModalVector& lhs, const ModalVector& rhs) noexcept {
  return not(lhs == rhs);
}

/// \cond
template ModalVector::ModalVector(std::initializer_list<double> list);
/// \endcond
