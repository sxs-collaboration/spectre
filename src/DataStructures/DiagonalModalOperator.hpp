// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/VectorImpl.hpp"
#include "Utilities/PointerVector.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

/// \cond
class ModalVector;
/// \endcond

// IWYU pragma: no_include <blaze/math/expressions/DVecMapExpr.h>
// IWYU pragma: no_include <blaze/math/typetraits/IsVector.h>

/*!
 * \ingroup DataStructuresGroup
 * \brief A class for an element-wise multiplier of modal coefficients.
 *
 * \details A `DiagonalModalOperator` holds an array of factors to multiply by
 * spectral coefficients, and can be either owning (the array is deleted when
 * the `DiagonalModalOperator` goes out of scope) or non-owning, meaning it just
 * has a pointer to an array.
 *
 * `DiagonalModalOperator`s are intended to represent a diagonal matrix that can
 * operate (via the `*` operator) on spectral coefficients represented by
 * `ModalVector`s easily. Only basic mathematical operations are supported with
 * `DiagonalModalOperator`s. `DiagonalModalOperator`s may be added, subtracted,
 * multiplied, or divided, and may be multiplied with a `ModalVector`, which
 * results in a new `ModalVector`. This multiplication is commutative,
 * supporting the interpretation of the `ModalVector` as either a 'row' or a
 * 'column' vector.
 *
 * Also, addition, subtraction, multiplication and division of
 * `DiagonalModalOperator`s with doubles is supported.
 */
class DiagonalModalOperator : public VectorImpl<double, DiagonalModalOperator> {
 public:
  using VectorImpl<double, DiagonalModalOperator>::operator=;
  using VectorImpl<double, DiagonalModalOperator>::VectorImpl;

  MAKE_MATH_ASSIGN_EXPRESSION_ARITHMETIC(DiagonalModalOperator)
};

namespace blaze {
VECTOR_BLAZE_TRAIT_SPECIALIZE_ARITHMETIC_TRAITS(DiagonalModalOperator);
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ModalVector,
                                               DiagonalModalOperator,
                                               MultTrait);

#if ((BLAZE_MAJOR_VERSION == 3) && (BLAZE_MINOR_VERSION <= 3))
template <typename Operator>
struct UnaryMapTrait<DiagonalModalOperator, Operator> {
  // Selectively allow unary operations for spectral coefficient operators
  static_assert(
      tmpl::list_contains_v<
          tmpl::list<
              // these traits are required for operators acting with doubles
              blaze::AddScalar<DiagonalModalOperator::ElementType>,
              blaze::SubScalarRhs<DiagonalModalOperator::ElementType>,
              blaze::SubScalarLhs<DiagonalModalOperator::ElementType>,
              blaze::DivideScalarByVector<DiagonalModalOperator::ElementType>>,
          Operator>,
      "This unary operation is not permitted on a DiagonalModalOperator");
  using Type = DiagonalModalOperator;
};

template <typename Operator>
struct BinaryMapTrait<DiagonalModalOperator, DiagonalModalOperator, Operator> {
  // Forbid math operations in this specialization of BinaryMap traits for
  // DiagonalModalOperator that are unlikely to be used on spectral
  // coefficients. Currently no non-arithmetic binary operations are supported.
  static_assert(
      tmpl::list_contains_v<tmpl::list<>, Operator>,
      "This binary operation is not permitted on a DiagonalModalOperator.");
  using Type = DiagonalModalOperator;
};
#else
template <typename Operator>
struct MapTrait<DiagonalModalOperator, Operator> {
  // Selectively allow unary operations for spectral coefficient operators
  static_assert(
      tmpl::list_contains_v<
          tmpl::list<
              // these traits are required for operators acting with doubles
              blaze::AddScalar<DiagonalModalOperator::ElementType>,
              blaze::SubScalarRhs<DiagonalModalOperator::ElementType>,
              blaze::SubScalarLhs<DiagonalModalOperator::ElementType>,
              blaze::DivideScalarByVector<DiagonalModalOperator::ElementType>>,
          Operator>,
      "This unary operation is not permitted on a DiagonalModalOperator");
  using Type = DiagonalModalOperator;
};

template <typename Operator>
struct MapTrait<DiagonalModalOperator, DiagonalModalOperator, Operator> {
  // Forbid math operations in this specialization of BinaryMap traits for
  // DiagonalModalOperator that are unlikely to be used on spectral
  // coefficients. Currently no non-arithmetic binary operations are supported.
  static_assert(
      tmpl::list_contains_v<tmpl::list<>, Operator>,
      "This binary operation is not permitted on a DiagonalModalOperator.");
  using Type = DiagonalModalOperator;
};
#endif  // ((BLAZE_MAJOR_VERSION == 3) && (BLAZE_MINOR_VERSION <= 3))
}  // namespace blaze

MAKE_STD_ARRAY_VECTOR_BINOPS(DiagonalModalOperator)

MAKE_WITH_VALUE_IMPL_DEFINITION_FOR(DiagonalModalOperator)
