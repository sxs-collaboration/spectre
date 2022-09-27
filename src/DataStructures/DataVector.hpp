// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>

#include "DataStructures/VectorImpl.hpp"
#include "Utilities/ForceInline.hpp"

/// \cond
class DataVector;
/// \endcond

namespace blaze {
DECLARE_GENERAL_VECTOR_BLAZE_TRAITS(DataVector);
}  // namespace blaze

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
 * values at points, use any of the
 * [Blaze vector types](https://bitbucket.org/blaze-lib/blaze/wiki/Vector%20Types)
 * instead.
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
  DataVector() = default;
  DataVector(const DataVector&) = default;
  DataVector(DataVector&&) = default;
  DataVector& operator=(const DataVector&) = default;
  DataVector& operator=(DataVector&&) = default;
  ~DataVector() = default;

  using BaseType = VectorImpl<double, DataVector>;

  using BaseType::operator=;
  using BaseType::VectorImpl;
};

// Specialize the Blaze type traits to correctly handle DataVector
namespace blaze {
VECTOR_BLAZE_TRAIT_SPECIALIZE_ARITHMETIC_TRAITS(DataVector);
VECTOR_BLAZE_TRAIT_SPECIALIZE_ALL_MAP_TRAITS(DataVector);
// Only specialize cross product for DataVector because it is unclear what a
// cross product of other vector types is. This is why this is here and not in
// VectorImpl.hpp
BLAZE_TRAIT_SPECIALIZE_BINARY_TRAIT(DataVector, CrossTrait);
}  // namespace blaze

SPECTRE_ALWAYS_INLINE auto fabs(const DataVector& t) { return abs(*t); }

MAKE_STD_ARRAY_VECTOR_BINOPS(DataVector)

MAKE_WITH_VALUE_IMPL_DEFINITION_FOR(DataVector)
