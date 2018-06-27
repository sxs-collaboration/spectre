// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Data.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>  // for std::reference_wrapper
#include <initializer_list>
#include <limits>
#include <ostream>
#include <type_traits>
#include <vector>

#include "ErrorHandling/Assert.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/PointerVector.hpp"
#include "Utilities/Requires.hpp"

/// \cond HIDDEN_SYMBOLS
// IWYU pragma: no_forward_declare ConstantExpressions_detail::pow
namespace PUP {
class er;
}  // namespace PUP

// clang-tidy: no using declarations in header files
//             We want the std::abs to be used
using std::abs;  // NOLINT
/// \endcond

// IWYU doesn't like that we want PointerVector.hpp to expose Blaze and also
// have DataVector.hpp to expose PointerVector.hpp without including Blaze
// directly in DataVector.hpp
//
// IWYU pragma: no_include <blaze/math/dense/DenseVector.h>
// IWYU pragma: no_include <blaze/math/expressions/DVecDVecAddExpr.h>
// IWYU pragma: no_include <blaze/math/expressions/DVecDVecDivExpr.h>
// IWYU pragma: no_include <blaze/math/expressions/DVecDVecMultExpr.h>
// IWYU pragma: no_include <blaze/math/expressions/DVecDVecSubExpr.h>
// IWYU pragma: no_include <blaze/math/expressions/DVecMapExpr.h>
// IWYU pragma: no_include <blaze/math/expressions/DVecScalarDivExpr.h>
// IWYU pragma: no_include <blaze/math/expressions/DVecScalarMultExpr.h>
// IWYU pragma: no_include <blaze/math/expressions/DenseVector.h>
// IWYU pragma: no_include <blaze/math/expressions/Vector.h>
// IWYU pragma: no_include <blaze/math/typetraits/IsVector.h>
// IWYU pragma: no_include <blaze/math/expressions/Forward.h>
// IWYU pragma: no_include <blaze/math/AlignmentFlag.h>
// IWYU pragma: no_include <blaze/math/PaddingFlag.h>
// IWYU pragma: no_include <blaze/math/traits/AddTrait.h>
// IWYU pragma: no_include <blaze/math/traits/DivTrait.h>
// IWYU pragma: no_include <blaze/math/traits/MultTrait.h>
// IWYU pragma: no_include <blaze/math/traits/SubTrait.h>
// IWYU pragma: no_include <blaze/system/TransposeFlag.h>
// IWYU pragma: no_include <blaze/math/traits/BinaryMapTrait.h>
// IWYU pragma: no_include <blaze/math/traits/UnaryMapTrait.h>
// IWYU pragma: no_include <blaze/math/typetraits/TransposeFlag.h>

// IWYU pragma: no_forward_declare blaze::DenseVector
// IWYU pragma: no_forward_declare blaze::UnaryMapTrait
// IWYU pragma: no_forward_declare blaze::BinaryMapTrait
// IWYU pragma: no_forward_declare blaze::IsVector
// IWYU pragma: no_forward_declare blaze::TransposeFlag

/*!
 * \ingroup DataStructuresGroup
 * \brief Stores a collection of function values.
 *
 * \details Use DataVector to represent function values on the computational
 * domain. Note that interpreting the data also requires knowledge of the points
 * that these function values correspond to.
 *
 * A DataVector holds an array of contiguous data. The DataVector can be owning,
 * meaning the array is deleted when the DataVector goes out of scope, or
 * non-owning, meaning it just has a pointer to an array.
 *
 * Refer to the \ref DataStructuresGroup documentation for a list of other
 * available types. In particular, to represent a generic vector that supports
 * common vector and matrix operations and whose meaning may not be of function
 * values at points, use DenseVector instead.
 *
 * DataVectors support a variety of mathematical operations that are applicable
 * to nodal coefficients. In addition to common arithmetic operations such as
 * elementwise addition, subtraction, multiplication and division the following
 * elementwise operations are implemented:
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
 * - invcbrt
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
class DataVector
    : public PointerVector<double, blaze::unaligned, blaze::unpadded,
                           blaze::defaultTransposeFlag, DataVector> {
  /// \cond HIDDEN_SYMBOLS
  static constexpr void private_asserts() noexcept {
    static_assert(std::is_nothrow_move_constructible<DataVector>::value,
                  "Missing move semantics");
  }
  /// \endcond
 public:
  using value_type = double;
  using allocator_type = std::allocator<value_type>;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;
  using BaseType = PointerVector<double, blaze::unaligned, blaze::unpadded,
                                 blaze::defaultTransposeFlag, DataVector>;
  static constexpr bool transpose_flag = blaze::defaultTransposeFlag;

  using BaseType::ElementType;
  using TransposeType = DataVector;
  using CompositeType = const DataVector&;

  using BaseType::operator[];
  using BaseType::begin;
  using BaseType::cbegin;
  using BaseType::cend;
  using BaseType::data;
  using BaseType::end;
  using BaseType::size;

  // @{
  // Upcast to `BaseType`
  const BaseType& operator~() const noexcept {
    return static_cast<const BaseType&>(*this);
  }
  BaseType& operator~() noexcept { return static_cast<BaseType&>(*this); }
  // @}

  /// Create with the given size and value.
  ///
  /// \param size number of values
  /// \param value the value to initialize each element.
  explicit DataVector(
      size_t size,
      double value = std::numeric_limits<double>::signaling_NaN()) noexcept;

  /// Create a non-owning DataVector that points to `start`
  DataVector(double* start, size_t size) noexcept;

  /// Create from an initializer list of doubles. All elements in the
  /// `std::initializer_list` must have decimal points
  template <class T, Requires<cpp17::is_same_v<T, double>> = nullptr>
  DataVector(std::initializer_list<T> list) noexcept;

  /// Empty DataVector
  DataVector() noexcept = default;
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
  DataVector(const blaze::DenseVector<VT, VF>& expression) noexcept;  // NOLINT

  template <typename VT, bool VF>
  DataVector& operator=(const blaze::DenseVector<VT, VF>& expression) noexcept;
  /// \endcond

  MAKE_EXPRESSION_MATH_ASSIGN_PV(+=, DataVector)
  MAKE_EXPRESSION_MATH_ASSIGN_PV(-=, DataVector)
  MAKE_EXPRESSION_MATH_ASSIGN_PV(*=, DataVector)
  MAKE_EXPRESSION_MATH_ASSIGN_PV(/=, DataVector)

  DataVector& operator=(const double& rhs) noexcept {
    ~*this = rhs;
    return *this;
  }

  // @{
  /// Set the DataVector to be a reference to another DataVector object
  void set_data_ref(gsl::not_null<DataVector*> rhs) noexcept {
    set_data_ref(rhs->data(), rhs->size());
  }
  void set_data_ref(double* start, size_t size) noexcept {
    owned_data_ = decltype(owned_data_){};
    (~*this).reset(start, size);
    owning_ = false;
  }
  // @}

  /// Returns true if the class owns the data
  bool is_owning() const noexcept { return owning_; }

  /// Serialization for Charm++
  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  SPECTRE_ALWAYS_INLINE void reset_pointer_vector() noexcept {
    reset(owned_data_.data(), owned_data_.size());
  }

  /// \cond HIDDEN_SYMBOLS
  std::vector<double, allocator_type> owned_data_;
  bool owning_{true};
  /// \endcond
};

/// Output operator for DataVector
std::ostream& operator<<(std::ostream& os, const DataVector& d);

/// Equivalence operator for DataVector
bool operator==(const DataVector& lhs, const DataVector& rhs) noexcept;

/// Inequivalence operator for DataVector
bool operator!=(const DataVector& lhs, const DataVector& rhs) noexcept;

/// \cond
// Used for comparing DataVector to an expression
template <typename VT, bool VF>
bool operator==(const DataVector& lhs,
                const blaze::DenseVector<VT, VF>& rhs) noexcept {
  return lhs == DataVector(rhs);
}

template <typename VT, bool VF>
bool operator!=(const DataVector& lhs,
                const blaze::DenseVector<VT, VF>& rhs) noexcept {
  return not(lhs == rhs);
}

template <typename VT, bool VF>
bool operator==(const blaze::DenseVector<VT, VF>& lhs,
                const DataVector& rhs) noexcept {
  return DataVector(lhs) == rhs;
}

template <typename VT, bool VF>
bool operator!=(const blaze::DenseVector<VT, VF>& lhs,
                const DataVector& rhs) noexcept {
  return not(lhs == rhs);
}
/// \endcond

// Specialize the Blaze type traits to correctly handle DataVector
namespace blaze {
template <>
struct IsVector<DataVector> : std::true_type {};

template <>
struct TransposeFlag<DataVector> : BoolConstant<DataVector::transpose_flag> {};

template <>
struct AddTrait<DataVector, DataVector> {
  using Type = DataVector;
};

template <>
struct AddTrait<DataVector, double> {
  using Type = DataVector;
};

template <>
struct AddTrait<double, DataVector> {
  using Type = DataVector;
};

template <>
struct SubTrait<DataVector, DataVector> {
  using Type = DataVector;
};

template <>
struct SubTrait<DataVector, double> {
  using Type = DataVector;
};

template <>
struct SubTrait<double, DataVector> {
  using Type = DataVector;
};

template <>
struct MultTrait<DataVector, DataVector> {
  using Type = DataVector;
};

template <>
struct MultTrait<DataVector, double> {
  using Type = DataVector;
};

template <>
struct MultTrait<double, DataVector> {
  using Type = DataVector;
};

template <>
struct DivTrait<DataVector, DataVector> {
  using Type = DataVector;
};

template <>
struct DivTrait<DataVector, double> {
  using Type = DataVector;
};

template <typename Operator>
struct UnaryMapTrait<DataVector, Operator> {
  using Type = DataVector;
};

template <typename Operator>
struct BinaryMapTrait<DataVector, DataVector, Operator> {
  using Type = DataVector;
};
}  // namespace blaze

SPECTRE_ALWAYS_INLINE decltype(auto) fabs(const DataVector& t) noexcept {
  return abs(~t);
}

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
DataVector::DataVector(const blaze::DenseVector<VT, VF>& expression) noexcept
    : owned_data_((~expression).size()) {
  static_assert(cpp17::is_same_v<typename VT::ResultType, DataVector>,
                "You are attempting to assign the result of an expression that "
                "is not a DataVector to a DataVector.");
  reset_pointer_vector();
  ~*this = expression;
}

template <typename VT, bool VF>
DataVector& DataVector::operator=(
    const blaze::DenseVector<VT, VF>& expression) noexcept {
  static_assert(cpp17::is_same_v<typename VT::ResultType, DataVector>,
                "You are attempting to assign the result of an expression that "
                "is not a DataVector to a DataVector.");
  if (owning_ and (~expression).size() != size()) {
    owned_data_.resize((~expression).size());
    reset_pointer_vector();
  } else if (not owning_) {
    ASSERT((~expression).size() == size(), "Must copy into same size, not "
                                               << (~expression).size()
                                               << " into " << size());
  }
  ~*this = expression;
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
template <>
struct pow<std::reference_wrapper<DataVector>, 0, std::nullptr_t> {
  SPECTRE_ALWAYS_INLINE static constexpr double apply(
      const std::reference_wrapper<DataVector>& /*t*/) {
    return 1.0;
  }
};
template <>
struct pow<std::reference_wrapper<const DataVector>, 0, std::nullptr_t> {
  SPECTRE_ALWAYS_INLINE static constexpr double apply(
      const std::reference_wrapper<const DataVector>& /*t*/) {
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
