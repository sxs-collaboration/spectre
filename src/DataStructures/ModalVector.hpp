// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/VectorImpl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class ModalVector;
/// \endcond

namespace blaze {
DECLARE_GENERAL_VECTOR_BLAZE_TRAITS(ModalVector);
}  // namespace blaze

/*!
 * \ingroup DataStructuresGroup
 * \brief A class for storing spectral coefficients on a spectral grid.
 *
 * \details A ModalVector holds an array of spectral coefficients, and can be
 * either owning (the array is deleted when the ModalVector goes out of scope)
 * or non-owning, meaning it just has a pointer to an array.
 *
 * Only basic mathematical operations are supported with
 * ModalVectors. ModalVectors may be added or subtracted, and the following
 * unary operations are supported:
 * - abs
 *
 * Also multiplication is supported with doubles and `ModalVector`s, and
 * `ModalVector`s can be divided by doubles. Multiplication is supported with
 * `DiagonalModalOperator`s and `ModalVector`s.
 */
class ModalVector : public VectorImpl<double, ModalVector> {
 public:
  ModalVector() = default;
  ModalVector(const ModalVector&) = default;
  ModalVector(ModalVector&&) = default;
  ModalVector& operator=(const ModalVector&) = default;
  ModalVector& operator=(ModalVector&&) = default;
  ~ModalVector() = default;

  using VectorImpl<double, ModalVector>::operator=;
  using VectorImpl<double, ModalVector>::VectorImpl;
};

namespace blaze {
template <>
struct TransposeFlag<ModalVector> : BoolConstant<ModalVector::transpose_flag> {
};
template <>
struct AddTrait<ModalVector, ModalVector> {
  using Type = ModalVector;
};
template <>
struct SubTrait<ModalVector, ModalVector> {
  using Type = ModalVector;
};
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ModalVector, double, MultTrait,
                                               ModalVector);
template <>
struct DivTrait<ModalVector, double> {
  using Type = ModalVector;
};

template <typename Operator>
struct MapTrait<ModalVector, Operator> {
  // Selectively allow unary operations for spectral coefficients
  static_assert(
      tmpl::list_contains_v<
          tmpl::list<blaze::Abs, blaze::Bind1st<blaze::Add, double>,
                     blaze::Bind2nd<blaze::Add, double>,
                     blaze::Bind1st<blaze::Sub, double>,
                     blaze::Bind2nd<blaze::Sub, double>,
                     blaze::Bind1st<blaze::Add, std::complex<double>>,
                     blaze::Bind2nd<blaze::Add, std::complex<double>>,
                     blaze::Bind1st<blaze::Sub, std::complex<double>>,
                     blaze::Bind2nd<blaze::Sub, std::complex<double>>>,
          Operator>,
      "The only unary operation permitted on a ModalVector is:"
      " abs.");
  using Type = ModalVector;
};

template <typename Operator>
struct MapTrait<ModalVector, ModalVector, Operator> {
  // Forbid math operations in this specialization of BinaryMap traits for
  // ModalVector that are unlikely to be used on spectral coefficients.
  // Currently no non-arithmetic binary operations are supported.
  static_assert(tmpl::list_contains_v<tmpl::list<>, Operator>,
                "This binary operation is not permitted on a ModalVector.");
  using Type = ModalVector;
};
}  // namespace blaze

/// \cond
DEFINE_STD_ARRAY_BINOP(ModalVector, ModalVector, ModalVector, operator+,
                       std::plus<>())
DEFINE_STD_ARRAY_BINOP(ModalVector, ModalVector, ModalVector, operator-,
                       std::minus<>())
DEFINE_STD_ARRAY_INPLACE_BINOP(ModalVector, ModalVector, operator+=,
                               std::plus<>())
DEFINE_STD_ARRAY_INPLACE_BINOP(ModalVector, ModalVector, operator-=,
                               std::minus<>())
/// \endcond
MAKE_WITH_VALUE_IMPL_DEFINITION_FOR(ModalVector)
