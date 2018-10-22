// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <functional>  // for std::reference_wrapper

#include "DataStructures/VectorImpl.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/PointerVector.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/StdArrayHelpers.hpp"

// IWYU doesn't like that we want PointerVector.hpp to expose Blaze and also
// have VectorImpl.hpp to expose PointerVector.hpp without including Blaze
// directly in VectorImpl.hpp
//
// IWYU pragma: no_include <blaze/math/expressions/DVecMapExpr.h>
// IWYU pragma: no_include <blaze/math/typetraits/IsVector.h>
// IWYU pragma: no_include <blaze/math/expressions/Forward.h>
// IWYU pragma: no_include <blaze/math/PaddingFlag.h>
// IWYU pragma: no_include <blaze/math/traits/AddTrait.h>
// IWYU pragma: no_include <blaze/math/traits/DivTrait.h>
// IWYU pragma: no_include <blaze/math/traits/MultTrait.h>
// IWYU pragma: no_include <blaze/math/traits/SubTrait.h>
// IWYU pragma: no_include <blaze/system/TransposeFlag.h>
#if ((BLAZE_MAJOR_VERSION == 3) && (BLAZE_MINOR_VERSION <= 3))
// IWYU pragma: no_include <blaze/math/traits/UnaryMapTrait.h>
// IWYU pragma: no_include <blaze/math/traits/BinaryMapTrait.h>
#else
// IWYU pragma: no_include <blaze/math/traits/MapTrait.h>
#endif  // ((BLAZE_MAJOR_VERSION == 3) && (BLAZE_MINOR_VERSION <= 3))
// IWYU pragma: no_include <blaze/math/typetraits/TransposeFlag.h>

// IWYU pragma: no_forward_declare blaze::DenseVector
#if ((BLAZE_MAJOR_VERSION == 3) && (BLAZE_MINOR_VERSION <= 3))
// IWYU pragma: no_forward_declare blaze::UnaryMapTrait
// IWYU pragma: no_forward_declare blaze::BinaryMapTrait
#else
// IWYU pragma: no_forward_declare blaze::MapTrait
#endif  // ((BLAZE_MAJOR_VERSION == 3) && (BLAZE_MINOR_VERSION <= 3))
// IWYU pragma: no_forward_declare blaze::IsVector
// IWYU pragma: no_forward_declare blaze::TransposeFlag
// IWYU pragma: no_forward_declare ConstantExpressions_detail::pow

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
 * elementwise addition, subtraction, multiplication and division the
 * elementwise operations on blaze vectors of doubles are supported. See
 * [blaze-wiki/Vector_Operations]
 * (https://bitbucket.org/blaze-lib/blaze/wiki/Vector%20Operations).
 *
 * In addition the Heaviside step function is supported for DataVectors.
 */
class DataVector : public VectorImpl<double, DataVector> {
 public:
  using VectorImpl<double, DataVector>::operator=;
  using VectorImpl<double, DataVector>::VectorImpl;

  MAKE_EXPRESSION_MATH_ASSIGN_PV(+=, DataVector)
  MAKE_EXPRESSION_MATH_ASSIGN_PV(-=, DataVector)
  MAKE_EXPRESSION_MATH_ASSIGN_PV(*=, DataVector)
  MAKE_EXPRESSION_MATH_ASSIGN_PV(/=, DataVector)
};

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

#if ((BLAZE_MAJOR_VERSION == 3) && (BLAZE_MINOR_VERSION <= 3))
template <typename Operator>
struct UnaryMapTrait<DataVector, Operator> {
  using Type = DataVector;
};

template <typename Operator>
struct BinaryMapTrait<DataVector, DataVector, Operator> {
  using Type = DataVector;
};
#else
template <typename Operator>
struct MapTrait<DataVector, Operator> {
  using Type = DataVector;
};

template <typename Operator>
struct MapTrait<DataVector, DataVector, Operator> {
  using Type = DataVector;
};
#endif  // ((BLAZE_MAJOR_VERSION == 3) && (BLAZE_MINOR_VERSION <= 3))
}  // namespace blaze

SPECTRE_ALWAYS_INLINE decltype(auto) fabs(const DataVector& t) noexcept {
  return abs(~t);
}

DEFINE_ARRAY_BINOP(DataVector, double, DataVector, operator+, std::plus<>())
DEFINE_ARRAY_BINOP(DataVector, DataVector, double, operator+, std::plus<>())
DEFINE_ARRAY_BINOP(DataVector, DataVector, DataVector, operator+, std::plus<>())

DEFINE_ARRAY_BINOP(DataVector, double, DataVector, operator-, std::minus<>())
DEFINE_ARRAY_BINOP(DataVector, DataVector, double, operator-, std::minus<>())
DEFINE_ARRAY_BINOP(DataVector, DataVector, DataVector, operator-,
                   std::minus<>())

DEFINE_ARRAY_INPLACE_BINOP(DataVector, DataVector, operator-=, std::minus<>())
DEFINE_ARRAY_INPLACE_BINOP(DataVector, double, operator-=, std::minus<>())
DEFINE_ARRAY_INPLACE_BINOP(DataVector, DataVector, operator+=, std::plus<>())
DEFINE_ARRAY_INPLACE_BINOP(DataVector, double, operator+=, std::plus<>())

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
  SPECTRE_ALWAYS_INLINE static constexpr double apply(
      const DataVector& /*t*/) noexcept {
    return 1.0;
  }
};
template <>
struct pow<std::reference_wrapper<DataVector>, 0, std::nullptr_t> {
  SPECTRE_ALWAYS_INLINE static constexpr double apply(
      const std::reference_wrapper<DataVector>& /*t*/) noexcept {
    return 1.0;
  }
};
template <>
struct pow<std::reference_wrapper<const DataVector>, 0, std::nullptr_t> {
  SPECTRE_ALWAYS_INLINE static constexpr double apply(
      const std::reference_wrapper<const DataVector>& /*t*/) noexcept {
    return 1.0;
  }
};
template <typename BlazeVector>
struct pow<BlazeVector, 0, Requires<blaze::IsVector<BlazeVector>::value>> {
  SPECTRE_ALWAYS_INLINE static constexpr double apply(
      const BlazeVector& /*t*/) noexcept {
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
      const DataVector& t) noexcept {
    return DataVector(t.size(), 1.0) / (t * pow<DataVector, -N - 1>::apply(t));
  }
};
}  // namespace ConstantExpressions_detail
