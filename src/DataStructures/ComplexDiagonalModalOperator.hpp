// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/DiagonalModalOperator.hpp"
#include "DataStructures/VectorImpl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DiagonalModalOperator;
class ModalVector;
class ComplexDiagonalModalOperator;
/// \endcond

namespace blaze {
DECLARE_GENERAL_VECTOR_BLAZE_TRAITS(ComplexDiagonalModalOperator);
}  // namespace blaze

/*!
 * \ingroup DataStructuresGroup
 * \brief A class for an element-wise complex multiplier of modal coefficients.
 *
 * \details A `ComplexDiagonalModalOperator` holds an array of factors to
 * multiply by spectral coefficients, and can be either owning (the array is
 * deleted when the `ComplexDiagonalModalOperator` goes out of scope) or
 * non-owning, meaning it just has a pointer to an array.
 *
 * `ComplexDiagonalModalOperator`s are intended to represent a diagonal matrix
 * that can operate (via the `*` operator) on spectral coefficients represented
 * by `ComplexModalVector`s easily. Only basic mathematical operations are
 * supported with `ComplexDiagonalModalOperator`s.
 * `ComplexDiagonalModalOperator`s may be added, subtracted, multiplied, or
 * divided, and all arithmetic operations are similarly supported between
 * `ComplexDiagonalModalOperator`s and `DiagonalModalOperator`s. In addition,
 * the following operations with modal data structures are supported:
 * - multiplication of a `ComplexDiagonalModalOperator` and a
 *   `ComplexModalVector` resulting in a `ComplexModalVector`
 * - multiplication of a `ComplexDiagonalModalOperator` and a `ModalVector`
 *   resulting in a `ComplexModalVector`
 * - multiplication of a `DiagonalModalOperator` and a `ComplexModalVector`
 *   resulting in a `ComplexModalVector`
 * All of these multiplication operations are commutative, supporting the
 * interpretation of the modal data structure as either a 'row' or a 'column'
 * vector.
 *
 * The following unary operations are supported with
 * `ComplexDiagonalModalOperator`s:
 * - conj
 * - imag (results in a `DiagonalModalOperator`)
 * - real (results in a `DiagonalModalOperator`)
 *
 * Also, addition, subtraction, multiplication and division of
 * `DiagonalModalOperator`s with `std::complex<double>`s or `double`s is
 * supported.
 */
class ComplexDiagonalModalOperator
    : public VectorImpl<std::complex<double>, ComplexDiagonalModalOperator> {
 public:
  ComplexDiagonalModalOperator() = default;
  ComplexDiagonalModalOperator(const ComplexDiagonalModalOperator&) = default;
  ComplexDiagonalModalOperator(ComplexDiagonalModalOperator&&) = default;
  ComplexDiagonalModalOperator& operator=(const ComplexDiagonalModalOperator&) =
      default;
  ComplexDiagonalModalOperator& operator=(ComplexDiagonalModalOperator&&) =
      default;
  ~ComplexDiagonalModalOperator() = default;

  using VectorImpl<std::complex<double>, ComplexDiagonalModalOperator>::
  operator=;
  using VectorImpl<std::complex<double>,
                   ComplexDiagonalModalOperator>::VectorImpl;
};

namespace blaze {
VECTOR_BLAZE_TRAIT_SPECIALIZE_ARITHMETIC_TRAITS(ComplexDiagonalModalOperator);

BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexDiagonalModalOperator,
                                               DiagonalModalOperator, AddTrait,
                                               ComplexDiagonalModalOperator);
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexDiagonalModalOperator,
                                               DiagonalModalOperator, DivTrait,
                                               ComplexDiagonalModalOperator);
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexDiagonalModalOperator,
                                               DiagonalModalOperator, MultTrait,
                                               ComplexDiagonalModalOperator);
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexDiagonalModalOperator,
                                               DiagonalModalOperator, SubTrait,
                                               ComplexDiagonalModalOperator);

BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexDiagonalModalOperator,
                                               double, AddTrait,
                                               ComplexDiagonalModalOperator);
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexDiagonalModalOperator,
                                               double, DivTrait,
                                               ComplexDiagonalModalOperator);
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexDiagonalModalOperator,
                                               double, MultTrait,
                                               ComplexDiagonalModalOperator);
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexDiagonalModalOperator,
                                               double, SubTrait,
                                               ComplexDiagonalModalOperator);

BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(DiagonalModalOperator,
                                               std::complex<double>, AddTrait,
                                               ComplexDiagonalModalOperator);
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(DiagonalModalOperator,
                                               std::complex<double>, DivTrait,
                                               ComplexDiagonalModalOperator);
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(DiagonalModalOperator,
                                               std::complex<double>, MultTrait,
                                               ComplexDiagonalModalOperator);
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(DiagonalModalOperator,
                                               std::complex<double>, SubTrait,
                                               ComplexDiagonalModalOperator);

BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexModalVector,
                                               DiagonalModalOperator,
                                               MultTrait, ComplexModalVector);
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexModalVector,
                                               ComplexDiagonalModalOperator,
                                               MultTrait, ComplexModalVector);
template <>
struct MultTrait<ModalVector, ComplexDiagonalModalOperator> {
  using Type = ComplexModalVector;
};
template <>
struct MultTrait<ComplexDiagonalModalOperator, ModalVector> {
  using Type = ComplexModalVector;
};

template <typename Operator>
struct MapTrait<ComplexDiagonalModalOperator, Operator> {
  // Selectively allow unary operations for spectral coefficient operators
  static_assert(tmpl::list_contains_v<
                    tmpl::list<blaze::Conj, blaze::Sqrt,
                               blaze::Bind1st<blaze::Add, std::complex<double>>,
                               blaze::Bind2nd<blaze::Add, std::complex<double>>,
                               blaze::Bind1st<blaze::Div, std::complex<double>>,
                               blaze::Bind2nd<blaze::Div, std::complex<double>>,
                               blaze::Bind1st<blaze::Sub, std::complex<double>>,
                               blaze::Bind2nd<blaze::Sub, std::complex<double>>,
                               blaze::Bind1st<blaze::Add, double>,
                               blaze::Bind2nd<blaze::Add, double>,
                               blaze::Bind1st<blaze::Div, double>,
                               blaze::Bind2nd<blaze::Div, double>,
                               blaze::Bind1st<blaze::Sub, double>,
                               blaze::Bind2nd<blaze::Sub, double>>,
                    Operator>,
                "This unary operation is not permitted on a "
                "ComplexDiagonalModalOperator");
  using Type = ComplexDiagonalModalOperator;
};
template <>
struct MapTrait<ComplexDiagonalModalOperator, blaze::Imag> {
  using Type = DiagonalModalOperator;
};
template <>
struct MapTrait<ComplexDiagonalModalOperator, blaze::Real> {
  using Type = DiagonalModalOperator;
};
template <>
struct MapTrait<DiagonalModalOperator, blaze::Imag> {
  using Type = DiagonalModalOperator;
};
template <>
struct MapTrait<DiagonalModalOperator, blaze::Real> {
  using Type = DiagonalModalOperator;
};

template <typename Operator>
struct MapTrait<ComplexDiagonalModalOperator, ComplexDiagonalModalOperator,
                Operator> {
  // Forbid math operations in this specialization of BinaryMap traits for
  // ComplexDiagonalModalOperator that are unlikely to be used on spectral
  // coefficients. Currently no non-arithmetic binary operations are supported.
  static_assert(tmpl::list_contains_v<tmpl::list<>, Operator>,
                "This binary operation is not permitted on a "
                "ComplexDiagonalModalOperator.");
  using Type = ComplexDiagonalModalOperator;
};
}  // namespace blaze

MAKE_STD_ARRAY_VECTOR_BINOPS(ComplexDiagonalModalOperator)

MAKE_WITH_VALUE_IMPL_DEFINITION_FOR(ComplexDiagonalModalOperator)
