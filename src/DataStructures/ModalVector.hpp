// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class ModalVector.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <ostream>
#include <type_traits>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PointerVector.hpp" // IWYU pragma: keep
#include "Utilities/Requires.hpp"

/// \cond HIDDEN_SYMBOLS
namespace PUP {
class er;
}  // namespace PUP

// clang-tidy: no using declarations in header files
//             We want the std::abs to be used
using std::abs;  // NOLINT
/// \endcond

/*!
 * \ingroup DataStructuresGroup
 * \brief A class for storing spectral coefficients on a mesh.
 *
 * A ModalVector holds an array of spectral coefficients, and can be
 * either owning (the array is deleted when the ModalVector goes out of scope)
 * or non-owning, meaning it just has a pointer to an array.
 *
 * Only basic mathematical operations are supported with ModalVectors. In
 * addition to addition, subtraction, multiplication, division, there
 * are the following element-wise operations:
 *
 * - abs
 * - max
 * - min
 *
 * In order to allow filtering, multiplication (*, *=) and division (/, /=)
 * operations with a DataVectors (holding filters) is supported.
 *
 * Note : Because of no tagging feature yet, expressions containing purely
 * DataVectors will evaluate and get copy-constructed to a ModalVector.
 * Disallowing this would also mean disallowing expressions such as
 * ModalVector coeffs_filtered(coeffs * filter) - where:
 *  - coeffs is a ModalVector
 *  - filter is a DataVector
 */
class ModalVector {
  /// \cond HIDDEN_SYMBOLS
  static constexpr int private_asserts() {
    static_assert(std::is_nothrow_move_constructible<ModalVector>::value,
                  "Missing move semantics");
    return 0;
  }
  /// \endcond
 public:
  using value_type = double;
  using allocator_type = std::allocator<value_type>;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;

  // The type alias ElementType is needed because blaze::IsInvertible<T> is not
  // SFINAE friendly to an ElementType type alias
  using ElementType = double;

 private:
  /// The type of the "pointer" used internally
  using InternalModalVector_t = PointerVector<double>;
  /// The type used to store the data in
  using InternalStorage_t = std::vector<double, allocator_type>;

 public:
  /// Create with the given size and value.
  ///
  /// \param size number of values
  /// \param value the value to initialize each element.
  explicit
  ModalVector(size_t size,
              double value = std::numeric_limits<double>::signaling_NaN());

  /// Create a non-owning ModalVector that points to `start`
  ModalVector(double* start, size_t size);

  /// Create from an initializer list of doubles. All elements in the
  /// `std::initializer_list` must have decimal points
  // cppcheck-suppress syntaxError
  template <class T, Requires<cpp17::is_same_v<T, double>> = nullptr>
  ModalVector(std::initializer_list<T> list);

  /// Empty ModalVector
  ModalVector() = default;
  /// \cond HIDDEN_SYMBOLS
  ~ModalVector() = default;

  ModalVector(const ModalVector& rhs);
  ModalVector(ModalVector&& rhs) noexcept;
  ModalVector& operator=(const ModalVector& rhs);
  ModalVector& operator=(ModalVector&& rhs) noexcept;

  // This is a converting constructor. clang-tidy complains that it's not
  // explicit, but we want it to allow conversion.
  // clang-tidy: mark as explicit (we want conversion to ModalVector)
  template <typename VT, bool VF>
  ModalVector(const blaze::Vector<VT, VF>& expression);  // NOLINT

  template <typename VT, bool VF>
  ModalVector& operator=(const blaze::Vector<VT, VF>& expression);
  /// \endcond

  /// Number of values stored
  size_t size() const noexcept { return size_; }

  // @{
  /// Set the ModalVector to be a reference to another ModalVector
  /// object
  void set_data_ref(gsl::not_null<ModalVector*> rhs) noexcept {
    set_data_ref(rhs->data(), rhs->size_);
  }
  void set_data_ref(double* start, size_t size) noexcept {
    size_ = size;
    owned_data_ = decltype(owned_data_){};
    data_ = decltype(data_){start, size_};
    owning_ = false;
  }
  // @}

  /// Returns true if the class owns the data
  bool is_owning() const noexcept { return owning_; }

  // @{
  /// Access ith element
  double& operator[](const size_type i) noexcept {
    ASSERT(i < size_, "i = " << i << ", size = " << size_);
    // clang-tidy: do not use pointer arithmetic
    return data_[i];  // NOLINT
  }
  const double& operator[](const size_type i) const noexcept {
    ASSERT(i < size_, "i = " << i << ", size = " << size_);
    // clang-tidy: do not use pointer arithmetic
    return data_[i];  // NOLINT
  }
  // @}

  // @{
  /// Access to the pointer
  double* data() noexcept { return data_.data(); }
  const double* data() const noexcept { return data_.data(); }
  // @}

  // @{
  /// Returns iterator to beginning of data
  decltype(auto) begin() noexcept { return data_.begin(); }
  decltype(auto) begin() const noexcept { return data_.begin(); }
  // @}
  // @{
  /// Returns iterator to end of data
  decltype(auto) end() noexcept { return data_.end(); }
  decltype(auto) end() const noexcept { return data_.end(); }
  // @}

  /// Serialization for Charm++
  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT

  // @{
  /// See the Blaze library documentation for details on these functions since
  /// they merely forward to Blaze.
  /// https://bitbucket.org/blaze-lib/blaze/overview
  ModalVector& operator=(const double& rhs) noexcept {
    data_ = rhs;
    return *this;
  }

  ModalVector& operator+=(const ModalVector& rhs) noexcept {
    data_ += rhs.data_;
    return *this;
  }
  template <typename VT, bool VF>
  ModalVector& operator+=(const blaze::Vector<VT, VF>& rhs) noexcept {
    data_ += rhs;
    return *this;
  }
  ModalVector& operator+=(const double& rhs) noexcept {
    data_ += rhs;
    return *this;
  }

  ModalVector& operator-=(const ModalVector& rhs) noexcept {
    data_ -= rhs.data_;
    return *this;
  }
  template <typename VT, bool VF>
  ModalVector& operator-=(const blaze::Vector<VT, VF>& rhs) noexcept {
    data_ -= rhs;
    return *this;
  }
  ModalVector& operator-=(const double& rhs) noexcept {
    data_ -= rhs;
    return *this;
  }

  template <typename VT, bool VF>
  ModalVector& operator*=(const blaze::Vector<VT, VF>& rhs) noexcept {
    data_ *= rhs;
    return *this;
  }
  ModalVector& operator*=(const double& rhs) noexcept {
    data_ *= rhs;
    return *this;
  }
  ModalVector& operator*=(const ModalVector& rhs) noexcept {
    data_ *= rhs.data_;
    return *this;
  }
  // FIXME PK: Can we avoid the instantiation of new DV object in this function
  ModalVector& operator*=(const DataVector& rhs) noexcept {
    const blaze::DynamicVector<double> _rhs_local(rhs.size(), rhs.data());
    data_ *= _rhs_local;
    return *this;
  }

  template <typename VT, bool VF>
  ModalVector& operator/=(const blaze::Vector<VT, VF>& rhs) noexcept {
    data_ /= rhs;
    return *this;
  }
  ModalVector& operator/=(const double& rhs) noexcept {
    data_ /= rhs;
    return *this;
  }
  // FIXME PK: Can we avoid the instantiation of new DV object in this function
  ModalVector& operator/=(const DataVector& rhs) noexcept {
    const blaze::DynamicVector<double> _rhs_local(rhs.size(), rhs.data());
    data_ /= _rhs_local;
    return *this;
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator+(
      const ModalVector& lhs, const ModalVector& rhs) noexcept {
    return lhs.data_ + rhs.data_;
  }
  template <typename VT, bool VF>
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator+(
      const blaze::Vector<VT, VF>& lhs, const ModalVector& rhs) noexcept {
    return ~lhs + rhs.data_;
  }
  template <typename VT, bool VF>
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator+(
      const ModalVector& lhs, const blaze::Vector<VT, VF>& rhs) noexcept {
    return lhs.data_ + ~rhs;
  }
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator+(
      const ModalVector& lhs, const double& rhs) noexcept {
    return lhs.data_ + rhs;
  }
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator+(
      const double& lhs, const ModalVector& rhs) noexcept {
    return lhs + rhs.data_;
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator-(
      const ModalVector& lhs, const ModalVector& rhs) noexcept {
    return lhs.data_ - rhs.data_;
  }
  template <typename VT, bool VF>
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator-(
      const blaze::Vector<VT, VF>& lhs, const ModalVector& rhs) noexcept {
    return ~lhs - rhs.data_;
  }
  template <typename VT, bool VF>
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator-(
      const ModalVector& lhs, const blaze::Vector<VT, VF>& rhs) noexcept {
    return lhs.data_ - ~rhs;
  }
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator-(
      const ModalVector& lhs, const double& rhs) noexcept {
    return lhs.data_ - rhs;
  }
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator-(
      const double& lhs, const ModalVector& rhs) noexcept {
    return lhs - rhs.data_;
  }
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator-(
      const ModalVector& rhs) noexcept {
    return -rhs.data_;
  };

  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator*(
      const ModalVector& lhs, const double& rhs) noexcept {
    return lhs.data_ * rhs;
  }
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator*(
      const double& lhs, const ModalVector& rhs) noexcept {
    return lhs * rhs.data_;
  }
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator*(
      const ModalVector& lhs, const ModalVector& rhs) noexcept {
    return lhs.data_ * rhs.data_;
  }
  template <typename VT, bool VF>
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator*(
      const blaze::Vector<VT, VF>& lhs, const ModalVector& rhs) noexcept {
    return ~lhs * rhs.data_;
  }
  template <typename VT, bool VF>
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator*(
      const ModalVector& lhs, const blaze::Vector<VT, VF>& rhs) noexcept {
    return lhs.data_ * ~rhs;
  }
  // FIXME PK: Can we avoid the instantiation of new DV objects in the
  // next 2 functions?
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator*(
      const ModalVector& lhs, const DataVector& rhs) noexcept {
    blaze::DynamicVector<double> _rhs_local(rhs.size(), rhs.data());
    return lhs * _rhs_local;
  }
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator*(
      const DataVector& lhs, const ModalVector& rhs) noexcept {
    blaze::DynamicVector<double> _lhs_local(lhs.size(), lhs.data());
    return rhs * _lhs_local;
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator/(
      const ModalVector& lhs, const double& rhs) noexcept {
    return lhs.data_ / rhs;
  }
  template <typename VT, bool VF>
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator/(
      const ModalVector& lhs, const blaze::Vector<VT, VF>& rhs) noexcept {
    return lhs.data_ / ~rhs;
  }
  // FIXME PK: Can we avoid the instantiation of new DV object in this function
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator/(
      const ModalVector& lhs, const DataVector& rhs) noexcept {
    const blaze::DynamicVector<double> _rhs_local(rhs.size(), rhs.data());
    return lhs / _rhs_local;
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) min(
      const ModalVector& t) noexcept {
    return min(t.data_);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) max(
      const ModalVector& t) noexcept {
    return max(t.data_);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) abs(
      const ModalVector& t) noexcept {
    return abs(t.data_);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) fabs(
      const ModalVector& t) noexcept {
    return abs(t.data_);
  }
  // @}


 private:
  /// \cond HIDDEN_SYMBOLS
  size_t size_ = 0;
  InternalStorage_t owned_data_;
  InternalModalVector_t data_;
  bool owning_{true};
  /// \endcond
};

/// Output operator for ModalVector
std::ostream& operator<<(std::ostream& os, const ModalVector& d);

/// Equivalence operator for ModalVector
bool operator==(const ModalVector& lhs, const ModalVector& rhs);

/// Inequivalence operator for ModalVector
bool operator!=(const ModalVector& lhs, const ModalVector& rhs);

template <typename T, size_t Dim>
std::array<ModalVector, Dim> operator+(
    const std::array<T, Dim>& lhs,
    const std::array<ModalVector, Dim>& rhs) noexcept {
  std::array<ModalVector, Dim> result;
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(result, i) = gsl::at(lhs, i) + gsl::at(rhs, i);
  }
  return result;
}
template <typename U, size_t Dim>
std::array<ModalVector, Dim> operator+(
    const std::array<ModalVector, Dim>& lhs,
    const std::array<U, Dim>& rhs) noexcept {
  return rhs + lhs;
}
template <size_t Dim>
std::array<ModalVector, Dim> operator+(
    const std::array<ModalVector, Dim>& lhs,
    const std::array<ModalVector, Dim>& rhs) noexcept {
  std::array<ModalVector, Dim> result;
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(result, i) = gsl::at(lhs, i) + gsl::at(rhs, i);
  }
  return result;
}
template <size_t Dim>
std::array<ModalVector, Dim>& operator+=(
    std::array<ModalVector, Dim>& lhs,
    const std::array<ModalVector, Dim>& rhs) noexcept {
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(lhs, i) += gsl::at(rhs, i);
  }
  return lhs;
}

template <typename T, size_t Dim>
std::array<ModalVector, Dim> operator-(
    const std::array<T, Dim>& lhs,
    const std::array<ModalVector, Dim>& rhs) noexcept {
  std::array<ModalVector, Dim> result;
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(result, i) = gsl::at(lhs, i) - gsl::at(rhs, i);
  }
  return result;
}
template <typename U, size_t Dim>
std::array<ModalVector, Dim> operator-(
    const std::array<ModalVector, Dim>& lhs,
    const std::array<U, Dim>& rhs) noexcept {
  std::array<ModalVector, Dim> result;
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(result, i) = gsl::at(lhs, i) - gsl::at(rhs, i);
  }
  return result;
}
template <size_t Dim>
std::array<ModalVector, Dim> operator-(
    const std::array<ModalVector, Dim>& lhs,
    const std::array<ModalVector, Dim>& rhs) noexcept {
  std::array<ModalVector, Dim> result;
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(result, i) = gsl::at(lhs, i) - gsl::at(rhs, i);
  }
  return result;
}
template <size_t Dim>
std::array<ModalVector, Dim>& operator-=(
    std::array<ModalVector, Dim>& lhs,
    const std::array<ModalVector, Dim>& rhs) noexcept {
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(lhs, i) -= gsl::at(rhs, i);
  }
  return lhs;
}

/// \cond HIDDEN_SYMBOLS
template <typename VT, bool VF>
ModalVector::ModalVector(const blaze::Vector<VT, VF>& expression)
    : size_((~expression).size()),
      owned_data_((~expression).size()),
      data_(owned_data_.data(), (~expression).size()) {
  data_ = expression;
}

template <typename VT, bool VF>
ModalVector& ModalVector::operator=(
    const blaze::Vector<VT, VF>& expression) {
  if (owning_ and (~expression).size() != size()) {
    size_ = (~expression).size();
    owned_data_ = InternalStorage_t(size_);
    data_ = decltype(data_){owned_data_.data(), size_};
  } else if (not owning_) {
    ASSERT((~expression).size() == size(), "Must copy into same size");
  }
  data_ = expression;
  return *this;
}
/// \endcond
