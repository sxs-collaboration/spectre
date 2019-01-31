// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/VectorImpl.hpp"
#include "Utilities/PointerVector.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include <blaze/math/expressions/DVecMapExpr.h>
// IWYU pragma: no_include <blaze/math/typetraits/IsVector.h>

/// \cond
class ModalVector;
/// \endcond

/*!
 * \ingroup DataStructuresGroup
 * \brief A class for storing complex spectral coefficients on a spectral grid.
 *
 * \details A ComplexModalVector holds an array of spectral coefficients
 * represented as `std::complex<double>`s, and can be either owning (the array
 * is deleted when the ComplexModalVector goes out of scope) or non-owning,
 * meaning it just has a pointer to an array.
 *
 * Only basic mathematical operations are supported with
 * `ComplexModalVector`s. `ComplexModalVector`s may be added or subtracted and
 * may be added or subtracted with `ModalVector`s, and the following unary
 * operations are supported:
 * - conj
 * - imag (which returns a `ModalVector`)
 * - real (which returns a `ModalVector`)
 *
 * Also multiplication is supported with `std::complex<double>`s or doubles and
 * `ComplexModalVector`s, and `ComplexModalVector`s can be divided by
 * `std::complex<double>`s or `double`s. Multiplication is supported with
 * `ComplexDiagonalModalOperator`s and `ComplexModalVector`s.
 */
class ComplexModalVector
    : public VectorImpl<std::complex<double>, ComplexModalVector> {
 public:
  using VectorImpl<std::complex<double>, ComplexModalVector>::operator=;
  using VectorImpl<std::complex<double>, ComplexModalVector>::VectorImpl;

  MAKE_MATH_ASSIGN_EXPRESSION_POINTERVECTOR(+=, ComplexModalVector)
  MAKE_MATH_ASSIGN_EXPRESSION_POINTERVECTOR(-=, ComplexModalVector)
};

namespace blaze {
template <>
struct IsVector<ComplexModalVector> : std::true_type {};
template <>
struct TransposeFlag<ComplexModalVector>
    : BoolConstant<ComplexModalVector::transpose_flag> {};
template <>
struct AddTrait<ComplexModalVector, ComplexModalVector> {
  using Type = ComplexModalVector;
};
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexModalVector, ModalVector,
                                               AddTrait);
template <>
struct SubTrait<ComplexModalVector, ComplexModalVector> {
  using Type = ComplexModalVector;
};
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexModalVector, ModalVector,
                                               SubTrait);
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexModalVector,
                                               std::complex<double>, MultTrait);
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexModalVector, double,
                                               MultTrait);
template <>
struct DivTrait<ComplexModalVector, std::complex<double>> {
  using Type = ComplexModalVector;
};
template <>
struct DivTrait<ComplexModalVector, double> {
  using Type = ComplexModalVector;
};

#if ((BLAZE_MAJOR_VERSION == 3) && (BLAZE_MINOR_VERSION <= 3))
template <typename Operator>
struct UnaryMapTrait<ComplexModalVector, Operator> {
  // Selectively allow unary operations for spectral coefficients
  static_assert(
      tmpl::list_contains_v<
          tmpl::list<blaze::Conj,
                     // Following 3 reqd. by operator(+,+=), (-,-=),
                     // (-) w/doubles
                     blaze::AddScalar<ComplexModalVector::ElementType>,
                     blaze::SubScalarRhs<ComplexModalVector::ElementType>,
                     blaze::SubScalarLhs<ComplexModalVector::ElementType>>,
          Operator>,
      "Only unary operations permitted on a ComplexModalVector are:"
      " conj, imag, and real");
  using Type = ComplexModalVector;
};
template <>
struct UnaryMapTrait<ComplexModalVector, blaze::Imag> {
  using Type = ModalVector;
};
template <>
struct UnaryMapTrait<ComplexModalVector, blaze::Real> {
  using Type = ModalVector;
};
// for consistency with the std::imag and std::real, these operations should be
// supported on real types
template <>
struct UnaryMapTrait<ModalVector, blaze::Imag> {
  using Type = ModalVector;
};
template <>
struct UnaryMapTrait<ModalVector, blaze::Real> {
  using Type = ModalVector;
};

template <typename Operator>
struct BinaryMapTrait<ComplexModalVector, ComplexModalVector, Operator> {
  // Forbid math operations in this specialization of BinaryMap traits for
  // ComplexModalVector that are unlikely to be used on spectral coefficients.
  // Currently no non-arithmetic binary operations are supported.
  static_assert(
      tmpl::list_contains_v<tmpl::list<>, Operator>,
      "This binary operation is not permitted on a ComplexModalVector.");
  using Type = ComplexModalVector;
};
#else
template <typename Operator>
struct MapTrait<ComplexModalVector, Operator> {
  // Selectively allow unary operations for spectral coefficients
  static_assert(
      tmpl::list_contains_v<
          tmpl::list<blaze::Conj,
                     // Following 3 reqd. by operator(+,+=), (-,-=),
                     // (-) w/doubles
                     blaze::AddScalar<ComplexModalVector::ElementType>,
                     blaze::SubScalarRhs<ComplexModalVector::ElementType>,
                     blaze::SubScalarLhs<ComplexModalVector::ElementType>>,
          Operator>,
      "Only unary operations permitted on a ComplexModalVector are:"
      " conj, imag, and real");
  using Type = ComplexModalVector;
};
template <>
struct MapTrait<ComplexModalVector, blaze::Imag> {
  using Type = ModalVector;
};
template <>
struct MapTrait<ComplexModalVector, blaze::Real> {
  using Type = ModalVector;
};
template <>
struct MapTrait<ModalVector, blaze::Imag> {
  using Type = ModalVector;
};
template <>
struct MapTrait<ModalVector, blaze::Real> {
  using Type = ModalVector;
};

template <typename Operator>
struct MapTrait<ComplexModalVector, ComplexModalVector, Operator> {
  // Forbid math operations in this specialization of BinaryMap traits for
  // ComplexModalVector that are unlikely to be used on spectral coefficients.
  // Currently no non-arithmetic binary operations are supported.
  static_assert(
      tmpl::list_contains_v<tmpl::list<>, Operator>,
      "This binary operation is not permitted on a ComplexModalVector.");
  using Type = ComplexModalVector;
};
#endif  // ((BLAZE_MAJOR_VERSION == 3) && (BLAZE_MINOR_VERSION <= 3))
}  // namespace blaze

/// \cond
DEFINE_STD_ARRAY_BINOP(ComplexModalVector, ComplexModalVector,
                       ComplexModalVector, operator+, std::plus<>())
DEFINE_STD_ARRAY_BINOP(ComplexModalVector, ComplexModalVector,
                       ComplexModalVector, operator-, std::minus<>())
DEFINE_STD_ARRAY_INPLACE_BINOP(ComplexModalVector,
                               ComplexModalVector, operator+=, std::plus<>())
DEFINE_STD_ARRAY_INPLACE_BINOP(ComplexModalVector,
                               ComplexModalVector, operator-=, std::minus<>())
/// \endcond
MAKE_WITH_VALUE_IMPL_DEFINITION_FOR(ComplexModalVector)
