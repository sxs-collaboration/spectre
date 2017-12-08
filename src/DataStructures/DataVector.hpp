// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Data.

#pragma once

#include <blaze/math/typetraits/IsVector.h>
#include <cstddef>
#include <initializer_list>
#include <iosfwd>
#include <limits>
#include <memory>
#include <vector>

#include "DataStructures/MakeWithValue.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PointerVector.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits.hpp"

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
 * \brief A class for storing data on a mesh.
 *
 * A Data holds an array of contiguous data and can be either owning (the array
 * is deleted when the Data goes out of scope) or non-owning, meaning it just
 * has a pointer to an array.
 *
 * A variety of mathematical operations are supported with DataVectors. In
 * addition to the addition, subtraction, multiplication, division, etc. there
 * are the following element-wise operations:
 *
 * - abs
 * - acos
 * - acosh
 * - asin
 * - asinh
 * - atan
 * - atan2
 * - atanh
 * - cbrt
 * - cos
 * - cosh
 * - erf
 * - erfc
 * - exp
 * - exp2
 * - exp10
 * - fabs
 * - hypot
 * - invsqrt
 * - log
 * - log2
 * - log10
 * - max
 * - min
 * - pow
 * - sin
 * - sinh
 * - sqrt
 * - step_function: if less than zero returns zero, otherwise returns one
 * - tan
 * - tanh
 */
class DataVector {
  /// \cond HIDDEN_SYMBOLS
  static constexpr int private_asserts() {
    static_assert(std::is_nothrow_move_constructible<DataVector>::value,
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
  using InternalDataVector_t = PointerVector<double>;
  /// The type used to store the data in
  using InternalStorage_t = std::vector<double, allocator_type>;

 public:
  /// Create with the given size and value.
  ///
  /// \param size number of values
  /// \param value the value to initialize each element.
  explicit DataVector(
      size_t size, double value = std::numeric_limits<double>::signaling_NaN());

  /// Create a non-owning DataVector that points to `start`
  DataVector(double* start, size_t size);

  /// Create from an initializer list of doubles. All elements in the
  /// `std::initializer_list` must have decimal points
  template <class T, Requires<cpp17::is_same_v<T, double>> = nullptr>
  DataVector(std::initializer_list<T> list);

  /// Empty DataVector
  DataVector() = default;
  /// \cond HIDDEN_SYMBOLS
  ~DataVector() = default;

  DataVector(const DataVector& rhs);
  DataVector(DataVector&& rhs) noexcept;
  DataVector& operator=(const DataVector& rhs);
  DataVector& operator=(DataVector&& rhs) noexcept;

  // This is a converting constructor. clang-tidy complains that it's not
  // explicit, but we want it to allow conversion.
  // clang-tidy: mark as explicit (we want conversion to DataVector)
  template <typename VT, bool VF>
  DataVector(const blaze::Vector<VT, VF>& expression);  // NOLINT

  template <typename VT, bool VF>
  DataVector& operator=(const blaze::Vector<VT, VF>& expression);
  /// \endcond

  /// Number of values stored
  size_t size() const noexcept { return size_; }

  // @{
  /// Set the DataVector to be a reference to another DataVector object
  void set_data_ref(gsl::not_null<DataVector*> rhs) noexcept {
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
  DataVector& operator=(const double& rhs) noexcept {
    data_ = rhs;
    return *this;
  }

  DataVector& operator+=(const DataVector& rhs) noexcept {
    data_ += rhs.data_;
    return *this;
  }
  template <typename VT, bool VF>
  DataVector& operator+=(const blaze::Vector<VT, VF>& rhs) noexcept {
    data_ += rhs;
    return *this;
  }
  DataVector& operator+=(const double& rhs) noexcept {
    data_ += rhs;
    return *this;
  }

  DataVector& operator-=(const DataVector& rhs) noexcept {
    data_ -= rhs.data_;
    return *this;
  }
  template <typename VT, bool VF>
  DataVector& operator-=(const blaze::Vector<VT, VF>& rhs) noexcept {
    data_ -= rhs;
    return *this;
  }
  DataVector& operator-=(const double& rhs) noexcept {
    data_ -= rhs;
    return *this;
  }

  DataVector& operator*=(const DataVector& rhs) noexcept {
    data_ *= rhs.data_;
    return *this;
  }
  template <typename VT, bool VF>
  DataVector& operator*=(const blaze::Vector<VT, VF>& rhs) noexcept {
    data_ *= rhs;
    return *this;
  }
  DataVector& operator*=(const double& rhs) noexcept {
    data_ *= rhs;
    return *this;
  }

  DataVector& operator/=(const DataVector& rhs) noexcept {
    data_ /= rhs.data_;
    return *this;
  }
  template <typename VT, bool VF>
  DataVector& operator/=(const blaze::Vector<VT, VF>& rhs) noexcept {
    data_ /= rhs;
    return *this;
  }
  DataVector& operator/=(const double& rhs) noexcept {
    data_ /= rhs;
    return *this;
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator+(
      const DataVector& lhs, const DataVector& rhs) noexcept {
    return lhs.data_ + rhs.data_;
  }
  template <typename VT, bool VF>
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator+(
      const blaze::Vector<VT, VF>& lhs, const DataVector& rhs) noexcept {
    return ~lhs + rhs.data_;
  }
  template <typename VT, bool VF>
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator+(
      const DataVector& lhs, const blaze::Vector<VT, VF>& rhs) noexcept {
    return lhs.data_ + ~rhs;
  }
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator+(
      const DataVector& lhs, const double& rhs) noexcept {
    return lhs.data_ + rhs;
  }
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator+(
      const double& lhs, const DataVector& rhs) noexcept {
    return lhs + rhs.data_;
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator-(
      const DataVector& lhs, const DataVector& rhs) noexcept {
    return lhs.data_ - rhs.data_;
  }
  template <typename VT, bool VF>
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator-(
      const blaze::Vector<VT, VF>& lhs, const DataVector& rhs) noexcept {
    return ~lhs - rhs.data_;
  }
  template <typename VT, bool VF>
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator-(
      const DataVector& lhs, const blaze::Vector<VT, VF>& rhs) noexcept {
    return lhs.data_ - ~rhs;
  }
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator-(
      const DataVector& lhs, const double& rhs) noexcept {
    return lhs.data_ - rhs;
  }
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator-(
      const double& lhs, const DataVector& rhs) noexcept {
    return lhs - rhs.data_;
  }
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator-(
      const DataVector& rhs) noexcept {
    return -rhs.data_;
  };

  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator*(
      const DataVector& lhs, const double& rhs) noexcept {
    return lhs.data_ * rhs;
  }
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator*(
      const double& lhs, const DataVector& rhs) noexcept {
    return lhs * rhs.data_;
  }
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator*(
      const DataVector& lhs, const DataVector& rhs) noexcept {
    return lhs.data_ * rhs.data_;
  }
  template <typename VT, bool VF>
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator*(
      const blaze::Vector<VT, VF>& lhs, const DataVector& rhs) noexcept {
    return ~lhs * rhs.data_;
  }
  template <typename VT, bool VF>
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator*(
      const DataVector& lhs, const blaze::Vector<VT, VF>& rhs) noexcept {
    return lhs.data_ * ~rhs;
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator/(
      const double& lhs, const DataVector& rhs) noexcept {
    return lhs / rhs.data_;
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator/(
      const DataVector& lhs, const double& rhs) noexcept {
    return lhs.data_ / rhs;
  }
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator/(
      const DataVector& lhs, const DataVector& rhs) noexcept {
    return lhs.data_ / rhs.data_;
  }
  template <typename VT, bool VF>
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator/(
      const blaze::Vector<VT, VF>& lhs, const DataVector& rhs) noexcept {
    return ~lhs / rhs.data_;
  }
  template <typename VT, bool VF>
  SPECTRE_ALWAYS_INLINE friend decltype(auto) operator/(
      const DataVector& lhs, const blaze::Vector<VT, VF>& rhs) noexcept {
    return lhs.data_ / ~rhs;
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) min(
      const DataVector& t) noexcept {
    return min(t.data_);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) max(
      const DataVector& t) noexcept {
    return max(t.data_);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) abs(
      const DataVector& t) noexcept {
    return abs(t.data_);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) fabs(
      const DataVector& t) noexcept {
    return abs(t.data_);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) sqrt(
      const DataVector& t) noexcept {
    return sqrt(t.data_);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) invsqrt(
      const DataVector& t) noexcept {
    return invsqrt(t.data_);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) cbrt(
      const DataVector& t) noexcept {
    return cbrt(t.data_);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) invcbrt(
      const DataVector& t) noexcept {
    return invcbrt(t.data_);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) pow(
      const DataVector& t, const double exponent) noexcept {
    return pow(t.data_, exponent);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) exp(
      const DataVector& t) noexcept {
    return exp(t.data_);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) exp2(
      const DataVector& t) noexcept {
    return exp2(t.data_);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) exp10(
      const DataVector& t) noexcept {
    return exp10(t.data_);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) log(
      const DataVector& t) noexcept {
    return log(t.data_);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) log2(
      const DataVector& t) noexcept {
    return log2(t.data_);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) log10(
      const DataVector& t) noexcept {
    return log10(t.data_);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) hypot(
      const DataVector& x, const DataVector& y) noexcept {
    return hypot(x.data_, y.data_);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) sin(
      const DataVector& t) noexcept {
    return sin(t.data_);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) cos(
      const DataVector& t) noexcept {
    return cos(t.data_);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) tan(
      const DataVector& t) noexcept {
    return tan(t.data_);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) asin(
      const DataVector& t) noexcept {
    return asin(t.data_);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) acos(
      const DataVector& t) noexcept {
    return acos(t.data_);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) atan(
      const DataVector& t) noexcept {
    return atan(t.data_);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) atan2(
      const DataVector& y, const DataVector& x) noexcept {
    return atan2(~(y.data_), ~(x.data_));
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) sinh(
      const DataVector& t) noexcept {
    return sinh(t.data_);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) cosh(
      const DataVector& t) noexcept {
    return cosh(t.data_);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) tanh(
      const DataVector& t) noexcept {
    return tanh(t.data_);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) asinh(
      const DataVector& t) noexcept {
    return asinh(t.data_);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) acosh(
      const DataVector& t) noexcept {
    return acosh(t.data_);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) atanh(
      const DataVector& t) noexcept {
    return atanh(t.data_);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) erf(
      const DataVector& t) noexcept {
    return erf(t.data_);
  }

  SPECTRE_ALWAYS_INLINE friend decltype(auto) erfc(
      const DataVector& t) noexcept {
    return erfc(t.data_);
  }
  // @}

  /// If less than zero returns zero, otherwise returns one
  SPECTRE_ALWAYS_INLINE friend decltype(auto) step_function(
      const DataVector& t) noexcept {
    return step_function(t.data_);
  }

 private:
  /// \cond HIDDEN_SYMBOLS
  size_t size_ = 0;
  InternalStorage_t owned_data_;
  InternalDataVector_t data_;
  bool owning_{true};
  /// \endcond
};

/// Output operator for DataVector
std::ostream& operator<<(std::ostream& os, const DataVector& d);

/// Equivalence operator for DataVector
bool operator==(const DataVector& lhs, const DataVector& rhs);

/// Inequivalence operator for DataVector
bool operator!=(const DataVector& lhs, const DataVector& rhs);

template <typename T, size_t Dim>
std::array<DataVector, Dim> operator+(
    const std::array<T, Dim>& lhs,
    const std::array<DataVector, Dim>& rhs) noexcept {
  std::array<DataVector, Dim> result;
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(result, i) = gsl::at(lhs, i) + gsl::at(rhs, i);
  }
  return result;
}
template <typename U, size_t Dim>
std::array<DataVector, Dim> operator+(const std::array<DataVector, Dim>& lhs,
                                      const std::array<U, Dim>& rhs) noexcept {
  return rhs + lhs;
}
template <size_t Dim>
std::array<DataVector, Dim> operator+(
    const std::array<DataVector, Dim>& lhs,
    const std::array<DataVector, Dim>& rhs) noexcept {
  std::array<DataVector, Dim> result;
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(result, i) = gsl::at(lhs, i) + gsl::at(rhs, i);
  }
  return result;
}
template <size_t Dim>
std::array<DataVector, Dim>& operator+=(
    std::array<DataVector, Dim>& lhs,
    const std::array<DataVector, Dim>& rhs) noexcept {
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(lhs, i) += gsl::at(rhs, i);
  }
  return lhs;
}
template <typename T, size_t Dim>
std::array<DataVector, Dim> operator-(
    const std::array<T, Dim>& lhs,
    const std::array<DataVector, Dim>& rhs) noexcept {
  std::array<DataVector, Dim> result;
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(result, i) = gsl::at(lhs, i) - gsl::at(rhs, i);
  }
  return result;
}
template <typename U, size_t Dim>
std::array<DataVector, Dim> operator-(const std::array<DataVector, Dim>& lhs,
                                      const std::array<U, Dim>& rhs) noexcept {
  std::array<DataVector, Dim> result;
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(result, i) = gsl::at(lhs, i) - gsl::at(rhs, i);
  }
  return result;
}
template <size_t Dim>
std::array<DataVector, Dim> operator-(
    const std::array<DataVector, Dim>& lhs,
    const std::array<DataVector, Dim>& rhs) noexcept {
  std::array<DataVector, Dim> result;
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(result, i) = gsl::at(lhs, i) - gsl::at(rhs, i);
  }
  return result;
}
template <size_t Dim>
std::array<DataVector, Dim>& operator-=(
    std::array<DataVector, Dim>& lhs,
    const std::array<DataVector, Dim>& rhs) noexcept {
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(lhs, i) -= gsl::at(rhs, i);
  }
  return lhs;
}

/// \cond HIDDEN_SYMBOLS
template <typename VT, bool VF>
DataVector::DataVector(const blaze::Vector<VT, VF>& expression)
    : size_((~expression).size()),
      owned_data_((~expression).size()),
      data_(owned_data_.data(), (~expression).size()) {
  data_ = expression;
}

template <typename VT, bool VF>
DataVector& DataVector::operator=(const blaze::Vector<VT, VF>& expression) {
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

namespace MakeWithValueImpls {
/// \brief Returns a DataVector the same size as `input`, with each element
/// equal to `value`.
template <>
SPECTRE_ALWAYS_INLINE DataVector
MakeWithValueImpl<DataVector, DataVector>::apply(const DataVector& input,
                                                 const double value) {
  return DataVector(input.size(), value);
}
}  // namespace MakeWithValueImpls

namespace ConstantExpressions_detail {
template <>
struct pow<DataVector, 0, std::nullptr_t> {
  SPECTRE_ALWAYS_INLINE static constexpr double apply(const DataVector& /*t*/) {
    return 1.0;
  }
};
template <typename BlazeVector>
struct pow<BlazeVector, 0, Requires<blaze::IsVector<BlazeVector>::value>> {
  SPECTRE_ALWAYS_INLINE static constexpr double apply(
      const BlazeVector& /*t*/) {
    return 1.0;
  }
};

template <int N>
struct pow<DataVector, N, Requires<(N < 0)>> {
  static_assert(N > 0,
                "Cannot use pow on DataVectorStructures with a negative "
                "exponent. You must "
                "divide by a positive exponent instead.");
  SPECTRE_ALWAYS_INLINE static constexpr decltype(auto) apply(
      const DataVector& t) {
    return DataVector(t.size(), 1.0) / (t * pow<DataVector, -N - 1>::apply(t));
  }
};
}  // namespace ConstantExpressions_detail
