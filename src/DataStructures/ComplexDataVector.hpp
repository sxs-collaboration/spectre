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

// IWYU pragma: no_forward_declare ConstantExpressions_detail::pow

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
  using VectorImpl<std::complex<double>, ComplexDataVector>::operator=;
  using VectorImpl<std::complex<double>, ComplexDataVector>::VectorImpl;

  MAKE_MATH_ASSIGN_EXPRESSION_ARITHMETIC(ComplexDataVector)
};

namespace blaze {
VECTOR_BLAZE_TRAIT_SPECIALIZE_ARITHMETIC_TRAITS(ComplexDataVector);
VECTOR_BLAZE_TRAIT_SPECIALIZE_ALL_MAP_TRAITS(ComplexDataVector);
#if ((BLAZE_MAJOR_VERSION == 3) && (BLAZE_MINOR_VERSION <= 3))
template <>
struct UnaryMapTrait<ComplexDataVector, blaze::Real> {
  using Type = DataVector;
};
template <>
struct UnaryMapTrait<ComplexDataVector, blaze::Imag> {
  using Type = DataVector;
};
template <>
struct UnaryMapTrait<DataVector, blaze::Real> {
  using Type = DataVector;
};
template <>
struct UnaryMapTrait<DataVector, blaze::Imag> {
  using Type = DataVector;
};
template <>
struct UnaryMapTrait<ComplexDataVector, blaze::Abs> {
  using Type = DataVector;
};
#else
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
#endif  // ((BLAZE_MAJOR_VERSION == 3) && (BLAZE_MINOR_VERSION <= 3))

BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexDataVector, DataVector,
                                               AddTrait);
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexDataVector, DataVector,
                                               DivTrait);
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexDataVector, DataVector,
                                               MultTrait);
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexDataVector, DataVector,
                                               SubTrait);

BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexDataVector, double,
                                               AddTrait);
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexDataVector, double,
                                               DivTrait);
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexDataVector, double,
                                               MultTrait);
BLAZE_TRAIT_SPECIALIZE_COMPATIBLE_BINARY_TRAIT(ComplexDataVector, double,
                                               SubTrait);

#if ((BLAZE_MAJOR_VERSION == 3) && (BLAZE_MINOR_VERSION <= 3))
template <typename Operator>
struct BinaryMapTrait<DataVector, ComplexDataVector, Operator> {
  using Type = ComplexDataVector;
};
template <typename Operator>
struct BinaryMapTrait<ComplexDataVector, DataVector, Operator> {
  using Type = ComplexDataVector;
};
#else
template <typename Operator>
struct MapTrait<DataVector, ComplexDataVector, Operator> {
  using Type = ComplexDataVector;
};
template <typename Operator>
struct MapTrait<ComplexDataVector, DataVector, Operator> {
  using Type = ComplexDataVector;
};
#endif  // ((BLAZE_MAJOR_VERSION == 3) && (BLAZE_MINOR_VERSION <= 3))
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
/// \endcond

MAKE_WITH_VALUE_IMPL_DEFINITION_FOR(ComplexDataVector)
