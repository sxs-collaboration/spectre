// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataVector.hpp"

#include <algorithm>
#include <pup.h>

#include "Utilities/StdHelpers.hpp"

template <>
DataVector::DataVectorImpl(const size_t size, const double value) noexcept
  : owned_data_(size, value) {
  reset_pointer_vector();
}

template <>
DataVector::DataVectorImpl(double* start, size_t size) noexcept
    : BaseType(start, size), owned_data_(0), owning_(false) {}

template <>
template <>
DataVector::DataVectorImpl(std::initializer_list<double> list) noexcept
  // clang-tidy: move trivially copyable type
  : owned_data_(std::move(list)) { //NOLINT
  reset_pointer_vector();
}

/// \cond HIDDEN_SYMBOLS
// clang-tidy: calling a base constructor other than the copy constructor.
//             We reset the base class in reset_pointer_vector after calling its
//             default constructor
template <>
DataVector::DataVectorImpl(const DataVector& rhs) : BaseType{} {  // NOLINT
  if (rhs.is_owning()) {
    owned_data_ = rhs.owned_data_;
  } else {
    owned_data_.assign(rhs.begin(), rhs.end());
  }
  reset_pointer_vector();
}

template <>
DataVector& DataVector::operator=(const DataVector& rhs) {
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

template <>
DataVector::DataVectorImpl(DataVector&& rhs) noexcept {
  owned_data_ = std::move(rhs.owned_data_);
  ~*this = ~rhs;  // PointerVector is trivially copyable
  owning_ = rhs.owning_;

  rhs.owning_ = true;
  rhs.reset();
}

template <>
DataVector& DataVector::operator=(DataVector&& rhs) noexcept {
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

template <>
void DataVector::pup(PUP::er& p) noexcept {  // NOLINT
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

std::ostream& operator<<(std::ostream& os, const DataVector& d) {
  // This function is inside the detail namespace StdHelpers.hpp
  StdHelpers_detail::print_helper(os, d.begin(), d.end());
  return os;
}

/// Equivalence operator for DataVector
bool operator==(const DataVector& lhs, const DataVector& rhs) noexcept {
  return lhs.size() == rhs.size() and
         std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

/// Inequivalence operator for DataVector
bool operator!=(const DataVector& lhs, const DataVector& rhs) noexcept {
  return not(lhs == rhs);
}
