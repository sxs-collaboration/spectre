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
 * \brief Templatized storage for function values.
 *
 * \details DataVectorImpl gives the underlying implementation for the
 * DataVector class, which is a specialization to DataVectorImpl<double>. The
 * additional template structure is added to permit more general DataVector
 * specializations which do not necessarily store doubles.
 *
 * Refer to specializations of DataVectorImpl for additional features offered
 * which depend on the stored type.
 * \ref DataVector
 */
template <typename T>
class DataVectorImpl
    : public PointerVector<T, blaze::unaligned, blaze::unpadded,
                           blaze::defaultTransposeFlag, DataVectorImpl<T>> {
  /// \cond HIDDEN_SYMBOLS
  static constexpr void private_asserts() noexcept {
    static_assert(std::is_nothrow_move_constructible<DataVectorImpl<T>>::value,
                  "Missing move semantics");
  }
  /// \endcond
 public:
  using value_type = T;
  using allocator_type = std::allocator<value_type>;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;
  using BaseType = PointerVector<T, blaze::unaligned, blaze::unpadded,
                                 blaze::defaultTransposeFlag,
                                 DataVectorImpl<T>>;
  static constexpr bool transpose_flag = blaze::defaultTransposeFlag;

  using ElementType = T;
  using TransposeType = DataVectorImpl<T>;
  using CompositeType = const DataVectorImpl<T>&;

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
  explicit DataVectorImpl(
      size_t size,
      T value = std::numeric_limits<T>::signaling_NaN()) noexcept;

  /// Create a non-owning DataVector that points to `start`
  DataVectorImpl(double* start, size_t size) noexcept;

  /// Create from an initializer list of doubles. All elements in the
  /// `std::initializer_list` must have decimal points
  template <class C, Requires<cpp17::is_same_v<C, T>> = nullptr>
  DataVectorImpl(std::initializer_list<C> list) noexcept;

  /// Empty DataVector
  DataVectorImpl() noexcept = default;
  /// \cond HIDDEN_SYMBOLS
  ~DataVectorImpl() = default;

  DataVectorImpl(const DataVectorImpl<T>& rhs);
  DataVectorImpl(DataVectorImpl<T>&& rhs) noexcept;
  DataVectorImpl<T>& operator=(const DataVectorImpl<T>& rhs);
  DataVectorImpl<T>& operator=(DataVectorImpl<T>&& rhs) noexcept;

  // This is a converting constructor. clang-tidy complains that it's not
  // explicit, but we want it to allow conversion.
  // clang-tidy: mark as explicit (we want conversion to DataVector)
  template <typename VT, bool VF>
  DataVectorImpl(const blaze::DenseVector<VT, VF>& expression) noexcept //NOLINT
    : owned_data_((~expression).size()) {
    static_assert(cpp17::is_same_v<typename VT::ResultType, DataVectorImpl<T>>,
                  "You are attempting to assign the result of an expression "
                  "that is not a DataVectorImpl<T> to a DataVectorImpl<T>.");
    reset_pointer_vector();
    ~*this = expression;
  }

  template <typename VT, bool VF>
  DataVectorImpl<T>& operator=(
    const blaze::DenseVector<VT, VF>& expression) noexcept {
    static_assert(cpp17::is_same_v<typename VT::ResultType, DataVectorImpl<T>>,
                  "You are attempting to assign the result of an expression "
                  "that is not a DataVectorImpl<T> to a DataVectorImpl<T>.");
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

  MAKE_EXPRESSION_MATH_ASSIGN_PV(+=, DataVectorImpl<T>)
  MAKE_EXPRESSION_MATH_ASSIGN_PV(-=, DataVectorImpl<T>)
  MAKE_EXPRESSION_MATH_ASSIGN_PV(*=, DataVectorImpl<T>)
  MAKE_EXPRESSION_MATH_ASSIGN_PV(/=, DataVectorImpl<T>)


  DataVectorImpl<T>& operator=(const T& rhs) noexcept {
    ~*this = rhs;
    return *this;
  }

  // @{
  /// Set the DataVector to be a reference to another DataVector object
  template <typename VT>
  void set_data_ref(DataVectorImpl<VT>* rhs) noexcept {
    set_data_ref(rhs->data(), rhs->size());
  }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow" /*ignore shadowing of size*/
  void set_data_ref(double* start, size_t size) noexcept {
    owned_data_ = decltype(owned_data_){};
    (~*this).reset(start, size);
    owning_ = false;
  }
  // @}
#pragma GCC diagnostic pop

  /// Returns true if the class owns the data
  bool is_owning() const noexcept { return owning_; }

  /// Serialization for Charm++
  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  SPECTRE_ALWAYS_INLINE void reset_pointer_vector() noexcept {
    this->reset(owned_data_.data(), owned_data_.size());
  }

  /// \cond HIDDEN_SYMBOLS
  std::vector<T, allocator_type> owned_data_;
  bool owning_{true};
  /// \endcond
};


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
using DataVector = DataVectorImpl<double>;

/// Output operator for DataVector
std::ostream& operator<<(std::ostream& os, const DataVector& d);


/// Equivalence operator for DataVector
bool operator==(const DataVector& lhs, const DataVector& rhs) noexcept;

/// Inequivalence operator for DataVector
bool operator!=(const DataVector& lhs,
                const DataVector& rhs) noexcept;

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

// Specialize the Blaze type traits to correctly handle all DataVectorImpl<T>
namespace blaze {
template <typename T>
struct IsVector<DataVectorImpl<T>> : std::true_type {};

template <typename T>
struct TransposeFlag<DataVectorImpl<T>>
  : BoolConstant<DataVector::transpose_flag> {};

template <typename T>
struct AddTrait<DataVectorImpl<T>, DataVectorImpl<T>> {
  using Type = DataVectorImpl<T>;
};

template <typename T>
struct AddTrait<DataVectorImpl<T>, double> {
  using Type = DataVectorImpl<T>;
};

template <typename T>
struct AddTrait<T, DataVectorImpl<T>> {
  using Type = DataVectorImpl<T>;
};

template <typename T>
struct SubTrait<DataVectorImpl<T>, DataVectorImpl<T>> {
  using Type = DataVectorImpl<T>;
};

template <typename T>
struct SubTrait<DataVectorImpl<T>, T> {
  using Type = DataVectorImpl<T>;
};

template <typename T>
struct SubTrait<T, DataVectorImpl<T>> {
  using Type = DataVectorImpl<T>;
};

template <typename T>
struct MultTrait<DataVectorImpl<T>, DataVectorImpl<T>> {
  using Type = DataVectorImpl<T>;
};

template <typename T>
struct MultTrait<DataVectorImpl<T>, T> {
  using Type = DataVectorImpl<T>;
};

template <typename T>
struct MultTrait<T, DataVectorImpl<T>> {
  using Type = DataVectorImpl<T>;
};

template <typename T>
struct DivTrait<DataVectorImpl<T>, DataVectorImpl<T>> {
  using Type = DataVectorImpl<T>;
};

template <typename T>
struct DivTrait<DataVectorImpl<T>, T> {
  using Type = DataVectorImpl<T>;
};

template <typename T, typename Operator>
struct UnaryMapTrait<DataVectorImpl<T>, Operator> {
  using Type = DataVectorImpl<T>;
};

template <typename T, typename Operator>
struct BinaryMapTrait<DataVectorImpl<T>, DataVectorImpl<T>, Operator> {
  using Type = DataVectorImpl<T>;
};
}  // namespace blaze


SPECTRE_ALWAYS_INLINE decltype(auto) fabs(const DataVector& t) noexcept {
  return abs(~t);
}

template <typename T, size_t Dim>
std::array<DataVectorImpl<T>, Dim> operator+(
    const std::array<T, Dim>& lhs,
    const std::array<DataVectorImpl<T>, Dim>& rhs) noexcept {
  std::array<DataVectorImpl<T>, Dim> result;
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(result, i) = gsl::at(lhs, i) + gsl::at(rhs, i);
  }
  return result;
}
template <typename lT, typename rT,  size_t Dim>
std::array<DataVectorImpl<lT>, Dim> operator+(
    const std::array<DataVectorImpl<lT>, Dim>& lhs,
    const std::array<rT, Dim>& rhs) noexcept {
  return rhs + lhs;
}
template <typename T, size_t Dim>
std::array<DataVectorImpl<T>, Dim> operator+(
    const std::array<DataVectorImpl<T>, Dim>& lhs,
    const std::array<DataVectorImpl<T>, Dim>& rhs) noexcept {
  std::array<DataVectorImpl<T>, Dim> result;
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(result, i) = gsl::at(lhs, i) + gsl::at(rhs, i);
  }
  return result;
}
template <typename T, size_t Dim>
std::array<DataVectorImpl<T>, Dim>& operator+=(
    std::array<DataVectorImpl<T>, Dim>& lhs,
    const std::array<DataVectorImpl<T>, Dim>& rhs) noexcept {
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(lhs, i) += gsl::at(rhs, i);
  }
  return lhs;
}
template <typename T, size_t Dim>
std::array<DataVectorImpl<T>, Dim> operator-(
    const std::array<T, Dim>& lhs,
    const std::array<DataVectorImpl<T>, Dim>& rhs) noexcept {
  std::array<DataVectorImpl<T>, Dim> result;
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(result, i) = gsl::at(lhs, i) - gsl::at(rhs, i);
  }
  return result;
}
template <typename T,  size_t Dim>
std::array<DataVectorImpl<T>, Dim> operator-(
    const std::array<DataVectorImpl<T>, Dim>& lhs,
    const std::array<T, Dim>& rhs) noexcept {
  std::array<DataVectorImpl<T>, Dim> result;
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(result, i) = gsl::at(lhs, i) - gsl::at(rhs, i);
  }
  return result;
}
template <typename T, size_t Dim>
std::array<DataVectorImpl<T>, Dim> operator-(
    const std::array<DataVectorImpl<T>, Dim>& lhs,
    const std::array<DataVectorImpl<T>, Dim>& rhs) noexcept {
  std::array<DataVectorImpl<T>, Dim> result;
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(result, i) = gsl::at(lhs, i) - gsl::at(rhs, i);
  }
  return result;
}
template <typename T, size_t Dim>
std::array<DataVectorImpl<T>, Dim>& operator-=(
    std::array<DataVectorImpl<T>, Dim>& lhs,
    const std::array<DataVectorImpl<T>, Dim>& rhs) noexcept {
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(lhs, i) -= gsl::at(rhs, i);
  }
  return lhs;
}

template <>
DataVector::DataVectorImpl(const size_t size, const double value) noexcept;

template <>
DataVector::DataVectorImpl(double* start, size_t size) noexcept;

/// \cond HIDDEN_SYMBOLS
// clang-tidy: calling a base constructor other than the copy constructor.
//             We reset the base class in reset_pointer_vector after calling its
//             default constructor
template <>
DataVector::DataVectorImpl(const DataVector& rhs);

template <>
DataVector& DataVector::operator=(const DataVector& rhs);

template <>
DataVector::DataVectorImpl(DataVector&& rhs) noexcept;

template <>
DataVector& DataVector::operator=(DataVector&& rhs) noexcept;
/// \endcond

template <>
void DataVector::pup(PUP::er& p) noexcept;

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
