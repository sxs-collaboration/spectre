// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class ModalVector.

#pragma once

#include <array>                // IWYU pragma: keep
#include <cmath>
#include <cstddef>              // IWYU pragma: keep
#include <initializer_list>     // IWYU pragma: keep
#include <limits>               // IWYU pragma: keep
#include <ostream>              // IWYU pragma: keep
#include <type_traits>          // IWYU pragma: keep
#include <vector>               // IWYU pragma: keep

#include "DataStructures/VectorMacros.hpp"
#include "ErrorHandling/Assert.hpp"    // IWYU pragma: keep
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"           // IWYU pragma: keep
#include "Utilities/MakeWithValue.hpp" // IWYU pragma: keep
#include "Utilities/PointerVector.hpp" // IWYU pragma: keep
#include "Utilities/Requires.hpp"      // IWYU pragma: keep
#include "Utilities/TMPL.hpp"          // for list

/// \cond HIDDEN_SYMBOLS
namespace PUP {       // IWYU pragma: keep
class er;             // IWYU pragma: keep
} // namespace PUP    // IWYU pragma: keep

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
// IWYU pragma: no_include "DataStructures/DataVector.hpp"
// IWYU pragma: no_include "DataStructures/ModalVector.hpp"

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
 * - abs/fabs
 * - max
 * - min
 *
 * In order to allow filtering, multiplication (*, *=) and division (/, /=)
 * operations with a DenseVectors (holding filters) is supported.
 *
 */
/// ModalVector class
MAKE_EXPRESSION_DATA_MODAL_VECTOR_CLASSES(ModalVector)

/// Declare shift and (in)equivalence operators for ModalVector with itself
MAKE_EXPRESSION_VECMATH_OP_COMP_SELF(ModalVector)

/// \cond
/// Define shift and (in)equivalence operators for ModalVector with
/// blaze::DenseVector
MAKE_EXPRESSION_VECMATH_OP_COMP_DV(ModalVector)
/// \endcond

// Specialize Blaze type traits to correctly handle ModalVector
MAKE_EXPRESSION_VECMATH_SPECIALIZE_BLAZE_ARITHMETIC_TRAITS(ModalVector)

// Specialize the Blaze {Unary,Binary}Map traits to correctly handle ModalVector
namespace blaze {
using ::ModalVector; // for ModalVector::ElementType
template <typename Operator>
struct UnaryMapTrait<ModalVector, Operator> {
  // Forbid math operations in this specialization of UnaryMap traits for
  // ModalVector that are inapplicable to spectral coefficients
  static_assert(not tmpl::list_contains_v<tmpl::list<
                blaze::Sqrt, blaze::Cbrt,
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
                "Only unary operation permitted are: max, min, abs, fabs.");
  // Selectively allow unary operations for spectral coefficients
  static_assert(tmpl::list_contains_v<tmpl::list<
                blaze::Abs,
                blaze::Max, blaze::Min,
                // Following 3 reqd. by operator(+,+=), (-,-=), (-) w/doubles
                blaze::AddScalar<ModalVector::ElementType>,
                blaze::SubScalarRhs<ModalVector::ElementType>,
                blaze::SubScalarLhs<ModalVector::ElementType>
                >, Operator>,
                "Only unary operation permitted on a ModalVector are:"
                " max, min, abs, fabs.");
  using Type = ModalVector;
};

// Specialize Blaze UnaryMap traits to handle ModalVector
template <typename Operator>
struct BinaryMapTrait<ModalVector, ModalVector, Operator> {
  // Forbid math operations in this specialization of BinaryMap traits for
  // ModalVector that are unlikely to be used on spectral coefficients
  static_assert(not tmpl::list_contains_v<tmpl::list<
                blaze::Max, blaze::Min
                >, Operator>,
                "This binary operation is not permitted on a ModalVector.");
  using Type = ModalVector;
};
}  // namespace blaze


SPECTRE_ALWAYS_INLINE decltype(auto) fabs(const ModalVector& t) noexcept {
  return abs(~t);
}


/// Define +, +=, -, -= operations between std::array's of ModalVectors
MAKE_EXPRESSION_VECMATH_OP_ADD_ARRAYS_OF_VEC(ModalVector)
MAKE_EXPRESSION_VECMATH_OP_SUB_ARRAYS_OF_VEC(ModalVector)

/// \cond HIDDEN_SYMBOLS
/// Forbid assignment of blaze::DenseVector<VT,VF>'s to ModalVector, if its
/// result type VT::ResultType is not ModalVector
MAKE_EXPRESSION_VEC_OP_ASSIGNMENT_RESTRICT_TYPE(ModalVector)
/// \endcond

MAKE_EXPRESSION_VEC_OP_MAKE_WITH_VALUE(ModalVector)
