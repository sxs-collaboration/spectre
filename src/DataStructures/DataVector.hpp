// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>

#include "DataStructures/VectorImpl.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/PointerVector.hpp"

// IWYU pragma: no_include <blaze/math/expressions/DVecMapExpr.h>
// IWYU pragma: no_include <blaze/math/typetraits/IsVector.h>

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
 * elementwise addition, subtraction, multiplication and division, the
 * elementwise operations on blaze vectors of doubles are supported. See
 * [blaze-wiki/Vector_Operations]
 * (https://bitbucket.org/blaze-lib/blaze/wiki/Vector%20Operations).
 *
 * In addition, the Heaviside step function `step_function` is supported for
 * DataVectors.
 */
class DataVector : public VectorImpl<double, DataVector> {
 public:
  using VectorImpl<double, DataVector>::operator=;
  using VectorImpl<double, DataVector>::VectorImpl;

  MAKE_MATH_ASSIGN_EXPRESSION_ARITHMETIC(DataVector)
};

// Specialize the Blaze type traits to correctly handle DataVector
namespace blaze {
VECTOR_BLAZE_TRAIT_SPECIALIZE_ARITHMETIC_TRAITS(DataVector);
VECTOR_BLAZE_TRAIT_SPECIALIZE_ALL_MAP_TRAITS(DataVector);
}  // namespace blaze

SPECTRE_ALWAYS_INLINE decltype(auto) fabs(const DataVector& t) noexcept {
  return abs(~t);
}

MAKE_STD_ARRAY_VECTOR_BINOPS(DataVector)

MAKE_WITH_VALUE_IMPL_DEFINITION_FOR(DataVector)
