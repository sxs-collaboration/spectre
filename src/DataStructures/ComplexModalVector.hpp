// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/VectorImpl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class ModalVector;
class ComplexModalVector;
/// \endcond

namespace blaze {
DECLARE_GENERAL_VECTOR_BLAZE_TRAITS(ComplexModalVector);
}  // namespace blaze

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
  ComplexModalVector() = default;
  ComplexModalVector(const ComplexModalVector&) = default;
  ComplexModalVector(ComplexModalVector&&) = default;
  ComplexModalVector& operator=(const ComplexModalVector&) = default;
  ComplexModalVector& operator=(ComplexModalVector&&) = default;
  ~ComplexModalVector() = default;

  using VectorImpl<std::complex<double>, ComplexModalVector>::operator=;
  using VectorImpl<std::complex<double>, ComplexModalVector>::VectorImpl;
};

namespace blaze {
template <>
struct TransposeFlag<ComplexModalVector>
    : BoolConstant<ComplexModalVector::transpose_flag> {};
template <>
struct AddTrait<ComplexModalVector, ComplexModalVector> {
  using Type = ComplexModalVector;
};
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexModalVector, ModalVector,
                                               AddTrait, ComplexModalVector);
template <>
struct SubTrait<ComplexModalVector, ComplexModalVector> {
  using Type = ComplexModalVector;
};
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexModalVector, ModalVector,
                                               SubTrait, ComplexModalVector);
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexModalVector,
                                               std::complex<double>, MultTrait,
                                               ComplexModalVector);
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ModalVector,
                                               std::complex<double>, MultTrait,
                                               ComplexModalVector);
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexModalVector, double,
                                               MultTrait, ComplexModalVector);
template <>
struct DivTrait<ComplexModalVector, std::complex<double>> {
  using Type = ComplexModalVector;
};
template <>
struct DivTrait<ComplexModalVector, double> {
  using Type = ComplexModalVector;
};

template <typename Operator>
struct MapTrait<ComplexModalVector, Operator> {
  // Selectively allow unary operations for spectral coefficients
  static_assert(tmpl::list_contains_v<
                    tmpl::list<blaze::Conj,
                               // Following 3 reqd. by operator(+,+=), (-,-=),
                               // (-) w/doubles
                               blaze::Bind1st<blaze::Add, std::complex<double>>,
                               blaze::Bind2nd<blaze::Add, std::complex<double>>,
                               blaze::Bind1st<blaze::Sub, std::complex<double>>,
                               blaze::Bind2nd<blaze::Sub, std::complex<double>>,
                               blaze::Bind1st<blaze::Add, double>,
                               blaze::Bind2nd<blaze::Add, double>,
                               blaze::Bind1st<blaze::Sub, double>,
                               blaze::Bind2nd<blaze::Sub, double>>,
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

namespace blaze {
// Partial specialization to disable being able to take the l?Norm of a
// ComplexModalVector. This does *not* prevent taking the norm of the square (or
// some other math expression) of a ComplexModalVector.
template <typename Abs, typename Power>
struct DVecNormHelper<
    blaze::CustomVector<std::complex<double>, blaze::AlignmentFlag::unaligned,
                        blaze::PaddingFlag::unpadded,
                        blaze::defaultTransposeFlag, blaze_default_group,
                        ComplexModalVector>,
    Abs, Power> {};
}  // namespace blaze
/// \endcond
MAKE_WITH_VALUE_IMPL_DEFINITION_FOR(ComplexModalVector)
