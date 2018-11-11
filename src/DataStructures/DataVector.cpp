// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataVector.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>  // for memcpy
#include <limits>
#include <pup.h>
#include <utility>  // IWYU pragma: keep

#include "Utilities/PrintHelpers.hpp"

DataVector::DataVector(const size_t size) noexcept
    : owned_data_(size > 0 ? static_cast<double*>(malloc(size * sizeof(double)))
                           : nullptr,
                  &free) {
#if defined(SPECTRE_DEBUG) || defined(SPECTRE_NAN_INIT)
  std::fill(owned_data_.get(), owned_data_.get() + size,
            std::numeric_limits<double>::signaling_NaN());
#endif  // SPECTRE_DEBUG
  reset_pointer_vector(size);
}

DataVector::DataVector(const size_t size, const double value) noexcept
    : owned_data_(size > 0 ? static_cast<double*>(malloc(size * sizeof(double)))
                           : nullptr,
                  &free) {
  std::fill(owned_data_.get(), owned_data_.get() + size, value);
  reset_pointer_vector(size);
}

DataVector::DataVector(double* start, size_t size) noexcept
    : BaseType(start, size), owning_(false) {}

template <class T, Requires<cpp17::is_same_v<T, double>>>
DataVector::DataVector(std::initializer_list<T> list) noexcept
    : owned_data_(list.size() > 0 ? static_cast<double*>(
                                        malloc(list.size() * sizeof(double)))
                                  : nullptr,
                  &free) {
  // Note: can't use memcpy with an initializer list.
  std::copy(list.begin(), list.end(), owned_data_.get());
  reset_pointer_vector(list.size());
}

/// \cond HIDDEN_SYMBOLS
#pragma GCC diagnostic push // Incorrect GCC warning.
#pragma GCC diagnostic ignored "-Wextra"
DataVector::DataVector(const DataVector& rhs)
    : owned_data_(rhs.size() > 0 ? static_cast<double*>(
                                       malloc(rhs.size() * sizeof(double)))
                                 : nullptr,
                  &free) {
#pragma GCC diagnostic pop
  reset_pointer_vector(rhs.size());
  std::memcpy(data(), rhs.data(), size() * sizeof(double));
}

DataVector& DataVector::operator=(const DataVector& rhs) {
  if (this != &rhs) {
    if (owning_) {
      if (size() != rhs.size()) {
        owned_data_.reset(rhs.size() > 0 ? static_cast<double*>(malloc(
                                               rhs.size() * sizeof(double)))
                                         : nullptr);
      }
      reset_pointer_vector(rhs.size());
    } else {
      ASSERT(rhs.size() == size(), "Must copy into same size, not "
                                       << rhs.size() << " into " << size());
    }
    std::memcpy(data(), rhs.data(), size() * sizeof(double));
  }
  return *this;
}

DataVector::DataVector(DataVector&& rhs) noexcept {
  owned_data_ = std::move(rhs.owned_data_);
  ~*this = ~rhs;  // PointerVector is trivially copyable
  owning_ = rhs.owning_;

  rhs.owning_ = true;
  rhs.reset();
}

DataVector& DataVector::operator=(DataVector&& rhs) noexcept {
  if (this != &rhs) {
    if (owning_) {
      owned_data_ = std::move(rhs.owned_data_);
      ~*this = ~rhs;  // PointerVector is trivially copyable
      owning_ = rhs.owning_;
    } else {
      ASSERT(rhs.size() == size(), "Must copy into same size, not "
                                       << rhs.size() << " into " << size());
      std::memcpy(data(), rhs.data(), size() * sizeof(double));
    }
    rhs.owning_ = true;
    rhs.reset();
  }
  return *this;
}
/// \endcond

void DataVector::pup(PUP::er& p) noexcept {  // NOLINT
  auto my_size = size();
  p | my_size;
  if (my_size > 0) {
    if (p.isUnpacking()) {
      owning_ = true;
      owned_data_.reset(
          my_size > 0 ? static_cast<double*>(malloc(my_size * sizeof(double)))
                      : nullptr);
      reset_pointer_vector(my_size);
    }
    PUParray(p, data(), size());
  }
}

std::ostream& operator<<(std::ostream& os, const DataVector& d) {
  sequence_print_helper(os, d.begin(), d.end());
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

/// \cond
template DataVector::DataVector(std::initializer_list<double> list) noexcept;
/// \endcond
