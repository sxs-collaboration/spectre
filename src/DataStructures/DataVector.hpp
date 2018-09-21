// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Data.

#pragma once

#include <array>                // IWYU pragma: keep
#include <cmath>
#include <cstddef>              // IWYU pragma: keep
#include <functional>  // for std::reference_wrapper
#include <initializer_list>     // IWYU pragma: keep
#include <limits>               // IWYU pragma: keep
#include <ostream>              // IWYU pragma: keep
#include <type_traits>          // IWYU pragma: keep
#include <vector>               // IWYU pragma: keep

#include "DataStructures/VectorMacros.hpp"
#include "ErrorHandling/Assert.hpp"           // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"                  // IWYU pragma: keep
#include "Utilities/MakeWithValue.hpp"        // IWYU pragma: keep
#include "Utilities/PointerVector.hpp"
#include "Utilities/Requires.hpp"

/// \cond HIDDEN_SYMBOLS
// IWYU pragma: no_forward_declare ConstantExpressions_detail::pow
namespace PUP {       // IWYU pragma: keep
class er;             // IWYU pragma: keep
} // namespace PUP    // IWYU pragma: keep

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
// IWYU pragma: no_include "DataStructures/DataVector.hpp"

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
/// DataVector class
MAKE_EXPRESSION_DATA_MODAL_VECTOR_CLASSES(DataVector)

/// Declare shift and (in)equivalence operators for DataVector with itself
MAKE_EXPRESSION_VECMATH_OP_COMP_SELF(DataVector)

/// \cond
/// Define shift and (in)equivalence operators for DataVector with
/// blaze::DenseVector
MAKE_EXPRESSION_VECMATH_OP_COMP_DV(DataVector)
/// \endcond

// Specialize the Blaze type traits to correctly handle DataVector
MAKE_EXPRESSION_VECMATH_SPECIALIZE_BLAZE_ARITHMETIC_TRAITS(DataVector)

// Specialize the Blaze {Unary,Binary}Map traits to correctly handle DataVector
MAKE_EXPRESSION_VECMATH_SPECIALIZE_BLAZE_MAP_TRAITS(DataVector)

SPECTRE_ALWAYS_INLINE decltype(auto) fabs(const DataVector& t) noexcept {
  return abs(~t);
}

/// Define +, +=, -, -= operations between std::array's of DataVectors
MAKE_EXPRESSION_VECMATH_OP_ADD_ARRAYS_OF_VEC(DataVector)
MAKE_EXPRESSION_VECMATH_OP_SUB_ARRAYS_OF_VEC(DataVector)

/// \cond HIDDEN_SYMBOLS
/// Forbid assignment of blaze::DenseVector<VT,VF>'s to DataVector, if its
/// result type VT::ResultType is not DataVector
MAKE_EXPRESSION_VEC_OP_ASSIGNMENT_RESTRICT_TYPE(DataVector)
/// \endcond

/// Construct a DataVector with value(s)
MAKE_EXPRESSION_VEC_OP_MAKE_WITH_VALUE(DataVector)

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
