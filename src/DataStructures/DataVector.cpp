// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataVector.hpp"

#include <pup_stl.h>

#include "Utilities/StdHelpers.hpp"

DataVector::DataVector(const size_t size, const double value)
    : size_(size),
      owned_data_(size_, value),
      data_(owned_data_.data(), size_) {}

template <class T, Requires<cpp17::is_same_v<T, double>>>
DataVector::DataVector(std::initializer_list<T> list)
    : size_(list.size()),
      owned_data_(std::move(list)),
      data_(owned_data_.data(), size_) {}

DataVector::DataVector(double* start, size_t size)
    : size_(size), owned_data_(0), data_(start, size_), owning_(false) {}

/// \cond HIDDEN_SYMBOLS
DataVector::DataVector(const DataVector& rhs) {
  size_ = rhs.size();
  if (rhs.is_owning()) {
    owned_data_ = rhs.owned_data_;
  } else {
    owned_data_ = InternalStorage_t(rhs.begin(), rhs.end());
  }
  data_ = decltype(data_){owned_data_.data(), size_};
}

DataVector& DataVector::operator=(const DataVector& rhs) {
  if (this == &rhs) {
    return *this;
  }
  if (owning_) {
    size_ = rhs.size();
    if (rhs.is_owning()) {
      owned_data_ = rhs.owned_data_;
    } else {
      owned_data_ = InternalStorage_t(rhs.begin(), rhs.end());
    }
    data_ = decltype(data_){owned_data_.data(), size_};
  } else {
    ASSERT(rhs.size() == size(), "Must copy into same size");
    std::copy(rhs.begin(), rhs.end(), begin());
  }
  return *this;
}

DataVector::DataVector(DataVector&& rhs) noexcept {
  size_ = rhs.size_;
  owned_data_ = std::move(rhs.owned_data_);
  // clang-tidy: move trivially copyable type, future proof in case impl
  // changes
  data_ = std::move(rhs.data_);  // NOLINT
  owning_ = rhs.owning_;

  rhs.owning_ = true;
  rhs.size_ = 0;
  rhs.data_ = decltype(rhs.data_){};
}

DataVector& DataVector::operator=(DataVector&& rhs) noexcept {
  if (this == &rhs) {
    return *this;
  }
  if (owning_) {
    size_ = rhs.size_;
    owned_data_ = std::move(rhs.owned_data_);
    // clang-tidy: move trivially copyable type, future proof in case impl
    // changes
    data_ = std::move(rhs.data_);  // NOLINT
    owning_ = rhs.owning_;
  } else {
    ASSERT(rhs.size() == size(), "Must copy into same size");
    std::copy(rhs.begin(), rhs.end(), begin());
  }
  rhs.owning_ = true;
  rhs.size_ = 0;
  rhs.data_ = decltype(rhs.data_){};
  return *this;
}
/// \endcond

void DataVector::pup(PUP::er& p) noexcept {  // NOLINT
  p | size_;
  if (p.isUnpacking()) {
    owning_ = true;
    p | owned_data_;
    data_ = decltype(data_){owned_data_.data(), size_};
  } else {
    if (not owning_) {
      owned_data_ =
          InternalStorage_t(data_.data(), data_.data() + size_);  // NOLINT
      p | owned_data_;
      owned_data_.clear();
    } else {
      p | owned_data_;
    }
  }
}

std::ostream& operator<<(std::ostream& os, const DataVector& d) {
  // This function is inside the detail namespace StdHelpers.hpp
  StdHelpers_detail::print_helper(os, d.begin(), d.end());
  return os;
}

/// Equivalence operator for DataVector
bool operator==(const DataVector& lhs, const DataVector& rhs) {
  return lhs.size() == rhs.size() and
         std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

/// Inequivalence operator for DataVector
bool operator!=(const DataVector& lhs, const DataVector& rhs) {
  return not(lhs == rhs);
}

/// \cond
template DataVector::DataVector(std::initializer_list<double> list);
/// \endcond
