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

#include "ErrorHandling/Assert.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/PointerVector.hpp" // IWYU pragma: keep
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"          // for list

/// \cond HIDDEN_SYMBOLS
namespace PUP {
class er;
}  // namespace PUP

// clang-tidy: no using declarations in header files
//             We want the std::abs to be used
using std::abs;  // NOLINT
/// \endcond

// IWYU doesn't like that we want PointerVector.hpp to expose Blaze and also
// have ModalVector.hpp to expose PointerVector.hpp without including Blaze
// directly in ModalVector.hpp
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
 * - abs/fabs/magnitude
 * - max
 * - min
 *
 * In order to allow filtering, multiplication (*, *=) and division (/, /=)
 * operations with a DenseVectors (holding filters) is supported.
 *
 */
class ModalVector
    : public PointerVector<double, blaze::unaligned, blaze::unpadded,
                           blaze::defaultTransposeFlag, ModalVector> {
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
  using BaseType = PointerVector<value_type, blaze::unaligned, blaze::unpadded,
                                 blaze::defaultTransposeFlag, ModalVector>;
  static constexpr bool transpose_flag = blaze::defaultTransposeFlag;

  using BaseType::ElementType;
  using TransposeType = ModalVector;
  using CompositeType = const ModalVector&;

  /// Iterators etc obtained from base class PointerVector
  using BaseType::operator[];
  using BaseType::begin;
  using BaseType::cbegin;
  using BaseType::cend;
  using BaseType::data;
  using BaseType::end;
  using BaseType::size;

  /// Do we need an upcast from ModalVector to PointerVector?
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
  explicit
  ModalVector(size_t size,
              double value = std::numeric_limits<double>::signaling_NaN());

  /// Create a non-owning ModalVector that points to `start`
  ModalVector(double* start, size_t size) noexcept;

  /// Create from an initializer list of doubles (must have decimal points)
  template <class T, Requires<cpp17::is_same_v<T, double>> = nullptr>
  ModalVector(std::initializer_list<T> list);

  /// Empty ModalVector
  ModalVector() noexcept = default;
  /// \cond HIDDEN_SYMBOLS
  ~ModalVector() = default;

  // Initialize ModalVector with expressions involving itself
  ModalVector(const ModalVector& rhs);
  ModalVector(ModalVector&& rhs) noexcept;
  ModalVector& operator=(const ModalVector& rhs);
  ModalVector& operator=(ModalVector&& rhs) noexcept;

  /// Catch DenseVector expressions into ModalVectors, only if their return
  /// types are defined as ModalVectors
  //
  // This is a converting constructor. clang-tidy complains that it's not
  // explicit, but we want it to allow conversion.
  // clang-tidy: mark as explicit (we want conversion to ModalVector)
  template <typename VT, bool VF>
  ModalVector(const blaze::DenseVector<VT, VF>& expression) noexcept;  // NOLINT

  template <typename VT, bool VF>
  ModalVector& operator=(const blaze::DenseVector<VT, VF>& expression) noexcept;
  /// \endcond

  ModalVector& operator=(const double& rhs) noexcept {
    ~*this = rhs;
    return *this;
  }

  // Using expression macro, defined in PointerVector.hpp, these lines overload
  // assignment operators { +=, -=, *=, /= } for LHS = ModalVector and RHS =
  // (i) ModalVector, (ii) blaze::DenseVector (to hold spectral filters), and
  // (iii) double (i.e. the ElementType)
  MAKE_EXPRESSION_MATH_ASSIGN_PV(+=, ModalVector)
  MAKE_EXPRESSION_MATH_ASSIGN_PV(-=, ModalVector)
  MAKE_EXPRESSION_MATH_ASSIGN_PV(*=, ModalVector)
  MAKE_EXPRESSION_MATH_ASSIGN_PV(/=, ModalVector)

  // @{
  /// Set ModalVector to be a reference to another ModalVector object
  void set_data_ref(gsl::not_null<ModalVector*> rhs) noexcept {
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
  std::vector<value_type, allocator_type> owned_data_;
  bool owning_{true};
  /// \endcond
};

/// Output operator for ModalVector
std::ostream& operator<<(std::ostream& os, const ModalVector& d);

/// Equivalence operator for ModalVector
bool operator==(const ModalVector& lhs, const ModalVector& rhs) noexcept;

/// Inequivalence operator for ModalVector
bool operator!=(const ModalVector& lhs, const ModalVector& rhs) noexcept;

/// \cond
// Used for comparing ModalVector to an expression
template <typename VT, bool VF>
bool operator==(const ModalVector& lhs,
                const blaze::DenseVector<VT, VF>& rhs) noexcept {
  return lhs == ModalVector(rhs);
}

template <typename VT, bool VF>
bool operator!=(const ModalVector& lhs,
                const blaze::DenseVector<VT, VF>& rhs) noexcept {
  return not(lhs == rhs);
}

template <typename VT, bool VF>
bool operator==(const blaze::DenseVector<VT, VF>& lhs,
                const ModalVector& rhs) noexcept {
  return ModalVector(lhs) == rhs;
}

template <typename VT, bool VF>
bool operator!=(const blaze::DenseVector<VT, VF>& lhs,
                const ModalVector& rhs) noexcept {
  return not(lhs == rhs);
}
/// \endcond

// Specialize the Blaze type traits to correctly handle ModalVector
namespace blaze {
template <>
struct IsVector<ModalVector> : std::true_type {};

template <>
struct TransposeFlag<ModalVector> : BoolConstant<
                ModalVector::transpose_flag> {};

template <>
struct AddTrait<ModalVector, ModalVector> {
  using Type = ModalVector;
};

template <>
struct AddTrait<ModalVector, double> {
  using Type = ModalVector;
};

template <>
struct AddTrait<double, ModalVector> {
  using Type = ModalVector;
};

template <>
struct SubTrait<ModalVector, ModalVector> {
  using Type = ModalVector;
};

template <>
struct SubTrait<ModalVector, double> {
  using Type = ModalVector;
};

template <>
struct SubTrait<double, ModalVector> {
  using Type = ModalVector;
};

template <>
struct MultTrait<ModalVector, ModalVector> {
  using Type = ModalVector;
};

template <>
struct MultTrait<ModalVector, double> {
  using Type = ModalVector;
};

template <>
struct MultTrait<double, ModalVector> {
  using Type = ModalVector;
};

template <>
struct DivTrait<ModalVector, ModalVector> {
  using Type = ModalVector;
};

template <>
struct DivTrait<ModalVector, double> {
  using Type = ModalVector;
};

// Forbid math operations in this specialization of UnaryMap traits for
// ModalVector that are unlikely to be used on spectral coefficients
template <typename Operator>
struct UnaryMapTrait<ModalVector, Operator> {
  static_assert(not tmpl::list_contains_v<tmpl::list<//blaze::Max, blaze::Min,
                //blaze::Sqrt,
                blaze::Cbrt,
                blaze::InvSqrt, blaze::InvCbrt,
                blaze::Acos, blaze::Acosh, blaze::Cos, blaze::Cosh,
                blaze::Asin, blaze::Asinh, blaze::Sin, blaze::Sinh,
                blaze::Atan, blaze::Atan2, blaze::Atanh,
                blaze::Tan, blaze::Tanh, blaze::Hypot,
                blaze::Exp, blaze::Exp2, blaze::Exp10,
                blaze::Log, blaze::Log2, blaze::Log10,
                blaze::Erf, blaze::Erfc, blaze::StepFunction
                >, Operator>,
                "This operation is not permitted on a ModalVector."
                "Only unary operation permitted are: abs, fabs, sqrt.");
  using Type = ModalVector;
};

// Forbid math operations in this specialization of BinaryMap traits for
// ModalVector that are unlikely to be used on spectral coefficients
template <typename Operator>
struct BinaryMapTrait<ModalVector, ModalVector, Operator> {
  static_assert(not tmpl::list_contains_v<tmpl::list<blaze::Max, blaze::Min
                >, Operator>,
                "This operation is not permitted on a ModalVector."
                "Only unary operation are permitted: abs, fabs, sqrt.");
  using Type = ModalVector;
};

}  // namespace blaze

SPECTRE_ALWAYS_INLINE decltype(auto) fabs(const ModalVector& t) noexcept {
  return abs(~t);
}

SPECTRE_ALWAYS_INLINE decltype(auto) abs(const ModalVector& t) noexcept {
  return abs(~t);
}

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
ModalVector::ModalVector(const blaze::DenseVector<VT, VF>& expression) noexcept
    : owned_data_((~expression).size()) {
  static_assert(cpp17::is_same_v<typename VT::ResultType, ModalVector>,
                "You are attempting to assign the result of an expression that "
                "is not a ModalVector to a ModalVector.");
  reset_pointer_vector();
  ~*this = expression;
}

template <typename VT, bool VF>
ModalVector& ModalVector::operator=(
    const blaze::DenseVector<VT, VF>& expression) noexcept {
  static_assert(cpp17::is_same_v<typename VT::ResultType, ModalVector>,
                "You are attempting to assign the result of an expression that "
                "is not a ModalVector to a ModalVector.");
  if (owning_ and (~expression).size() != size()) {
    owned_data_.resize((~expression).size());
    reset_pointer_vector();
  } else if (not owning_) {
    ASSERT((~expression).size() == size(), "Must copy into same size");
  }
  ~*this = expression;
  return *this;
}
/// \endcond

namespace MakeWithValueImpls {
/// \brief Returns a ModalVector the same size as `input`, with each element
/// equal to `value`.
template <>
SPECTRE_ALWAYS_INLINE ModalVector
MakeWithValueImpl<ModalVector, ModalVector>::apply(const ModalVector& input,
                                                 const double value) {
  return ModalVector(input.size(), value);
}
}  // namespace MakeWithValueImpls
