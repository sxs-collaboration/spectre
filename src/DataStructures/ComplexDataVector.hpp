// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <cstddef>
#include <functional>  // for std::reference_wrapper

#include "DataStructures/DataVector.hpp"
#include "DataStructures/VectorImpl.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/StdArrayHelpers.hpp"

/// \cond
class ComplexDataVector;
/// \endcond

namespace blaze {
DECLARE_GENERAL_VECTOR_BLAZE_TRAITS(ComplexDataVector);
}  // namespace blaze

/*!
 * \ingroup DataStructuresGroup
 * \brief Stores a collection of complex function values.
 *
 * \details Use ComplexDataVector to represent function values on the
 * computational domain. Note that interpreting the data also requires knowledge
 * of the points that these function values correspond to.
 *
 * A ComplexDataVector holds an array of contiguous data. The ComplexDataVector
 * can be owning, meaning the array is deleted when the ComplexDataVector goes
 * out of scope, or non-owning, meaning it just has a pointer to an array.
 *
 * Refer to the \ref DataStructuresGroup documentation for a list of other
 * available types.
 *
 * ComplexDataVectors support a variety of mathematical operations that are
 * applicable contiguous data. In addition to common arithmetic operations such
 * as element-wise addition, subtraction, multiplication and division,
 * element-wise operations between `ComplexDataVector` and `DataVector` are
 * supported. See [blaze-wiki/Vector_Operations]
 * (https://bitbucket.org/blaze-lib/blaze/wiki/Vector%20Operations).
 *
 * in addition, support is provided for:
 * - abs  (returns DataVector, as the result is real)
 * - conj
 * - imag  (returns DataVector, as the result is real)
 * - real  (returns DataVector, as the result is real)
 */
class ComplexDataVector
    : public VectorImpl<std::complex<double>, ComplexDataVector> {
 public:
  ComplexDataVector() = default;
  ComplexDataVector(const ComplexDataVector&) = default;
  ComplexDataVector(ComplexDataVector&&) = default;
  ComplexDataVector& operator=(const ComplexDataVector&) = default;
  ComplexDataVector& operator=(ComplexDataVector&&) = default;
  ~ComplexDataVector() = default;

  using VectorImpl<std::complex<double>, ComplexDataVector>::operator=;
  using VectorImpl<std::complex<double>, ComplexDataVector>::VectorImpl;
};

namespace blaze {
VECTOR_BLAZE_TRAIT_SPECIALIZE_ARITHMETIC_TRAITS(ComplexDataVector);
VECTOR_BLAZE_TRAIT_SPECIALIZE_ALL_MAP_TRAITS(ComplexDataVector);
template <>
struct MapTrait<ComplexDataVector, blaze::Real> {
  using Type = DataVector;
};
template <>
struct MapTrait<ComplexDataVector, blaze::Imag> {
  using Type = DataVector;
};
template <>
struct MapTrait<DataVector, blaze::Real> {
  using Type = DataVector;
};
template <>
struct MapTrait<DataVector, blaze::Imag> {
  using Type = DataVector;
};
template <>
struct MapTrait<ComplexDataVector, blaze::Abs> {
  using Type = DataVector;
};

BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexDataVector, DataVector,
                                               AddTrait, ComplexDataVector);
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexDataVector, DataVector,
                                               DivTrait, ComplexDataVector);
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexDataVector, DataVector,
                                               MultTrait, ComplexDataVector);
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexDataVector, DataVector,
                                               SubTrait, ComplexDataVector);

BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexDataVector, double,
                                               AddTrait, ComplexDataVector);
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexDataVector, double,
                                               DivTrait, ComplexDataVector);
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexDataVector, double,
                                               MultTrait, ComplexDataVector);
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexDataVector, double,
                                               SubTrait, ComplexDataVector);

BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(DataVector, std::complex<double>,
                                               AddTrait, ComplexDataVector);
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(DataVector, std::complex<double>,
                                               DivTrait, ComplexDataVector);
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(DataVector, std::complex<double>,
                                               MultTrait, ComplexDataVector);
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(DataVector, std::complex<double>,
                                               SubTrait, ComplexDataVector);

template <typename Operator>
struct MapTrait<DataVector, ComplexDataVector, Operator> {
  using Type = ComplexDataVector;
};
template <typename Operator>
struct MapTrait<ComplexDataVector, DataVector, Operator> {
  using Type = ComplexDataVector;
};
}  // namespace blaze

MAKE_STD_ARRAY_VECTOR_BINOPS(ComplexDataVector)

/// \cond HIDDEN_SYMBOLS
DEFINE_STD_ARRAY_BINOP(ComplexDataVector, ComplexDataVector,
                       DataVector, operator+, std::plus<>())
DEFINE_STD_ARRAY_BINOP(ComplexDataVector, DataVector,
                       ComplexDataVector, operator+, std::plus<>())
DEFINE_STD_ARRAY_BINOP(ComplexDataVector, ComplexDataVector, double, operator+,
                       std::plus<>())
DEFINE_STD_ARRAY_BINOP(ComplexDataVector, double, ComplexDataVector, operator+,
                       std::plus<>())

DEFINE_STD_ARRAY_BINOP(ComplexDataVector, ComplexDataVector,
                       DataVector, operator-, std::minus<>())
DEFINE_STD_ARRAY_BINOP(ComplexDataVector, DataVector,
                       ComplexDataVector, operator-, std::minus<>())
DEFINE_STD_ARRAY_BINOP(ComplexDataVector, ComplexDataVector, double, operator-,
                       std::minus<>())
DEFINE_STD_ARRAY_BINOP(ComplexDataVector, double, ComplexDataVector, operator-,
                       std::minus<>())

DEFINE_STD_ARRAY_INPLACE_BINOP(ComplexDataVector, DataVector, operator+=,
                               std::plus<>())
DEFINE_STD_ARRAY_INPLACE_BINOP(ComplexDataVector, DataVector, operator-=,
                               std::minus<>())
DEFINE_STD_ARRAY_INPLACE_BINOP(ComplexDataVector, double, operator+=,
                               std::plus<>())
DEFINE_STD_ARRAY_INPLACE_BINOP(ComplexDataVector, double, operator-=,
                               std::minus<>())

namespace blaze {
// Partial specialization to disable being able to take the l?Norm of a
// ComplexDataVector. This does *not* prevent taking the norm of the square (or
// some other math expression) of a ComplexDataVector.
template <typename Abs, typename Power>
struct DVecNormHelper<
    blaze::CustomVector<std::complex<double>, blaze_unaligned, blaze_unpadded,
                        blaze::defaultTransposeFlag, blaze::GroupTag<0>,
                        ComplexDataVector>,
    Abs, Power> {};
}  // namespace blaze
/// \endcond

MAKE_WITH_VALUE_IMPL_DEFINITION_FOR(ComplexDataVector)
