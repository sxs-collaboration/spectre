// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>

#include "DataStructures/VectorImpl.hpp"

// A VectorImpl whose static size limit is different than the default value
class CustomStaticSizeVector;
namespace blaze {
DECLARE_GENERAL_VECTOR_BLAZE_TRAITS(CustomStaticSizeVector);
}  // namespace blaze
class CustomStaticSizeVector
    : public VectorImpl<double, CustomStaticSizeVector, 3> {
 public:
  CustomStaticSizeVector() = default;
  CustomStaticSizeVector(const CustomStaticSizeVector&) = default;
  CustomStaticSizeVector(CustomStaticSizeVector&&) = default;
  CustomStaticSizeVector& operator=(const CustomStaticSizeVector&) = default;
  CustomStaticSizeVector& operator=(CustomStaticSizeVector&&) = default;
  ~CustomStaticSizeVector() = default;

  using BaseType = VectorImpl<double, CustomStaticSizeVector, static_size>;

  using BaseType::operator=;
  using BaseType::VectorImpl;
};

namespace blaze {
VECTOR_BLAZE_TRAIT_SPECIALIZE_ARITHMETIC_TRAITS(CustomStaticSizeVector);
VECTOR_BLAZE_TRAIT_SPECIALIZE_ALL_MAP_TRAITS(CustomStaticSizeVector);
}  // namespace blaze

SPECTRE_ALWAYS_INLINE auto fabs(const CustomStaticSizeVector& t) {
  return abs(*t);
}

MAKE_STD_ARRAY_VECTOR_BINOPS(CustomStaticSizeVector)

MAKE_WITH_VALUE_IMPL_DEFINITION_FOR(CustomStaticSizeVector)

class CustomComplexStaticSizeVector;
namespace blaze {
DECLARE_GENERAL_VECTOR_BLAZE_TRAITS(CustomComplexStaticSizeVector);
}  // namespace blaze
class CustomComplexStaticSizeVector
    : public VectorImpl<std::complex<double>, CustomComplexStaticSizeVector,
                        3> {
 public:
  CustomComplexStaticSizeVector() = default;
  CustomComplexStaticSizeVector(const CustomComplexStaticSizeVector&) = default;
  CustomComplexStaticSizeVector(CustomComplexStaticSizeVector&&) = default;
  CustomComplexStaticSizeVector& operator=(
      const CustomComplexStaticSizeVector&) = default;
  CustomComplexStaticSizeVector& operator=(CustomComplexStaticSizeVector&&) =
      default;
  ~CustomComplexStaticSizeVector() = default;

  using BaseType = VectorImpl<std::complex<double>,
                              CustomComplexStaticSizeVector, static_size>;

  using BaseType::operator=;
  using BaseType::VectorImpl;
};

namespace blaze {
VECTOR_BLAZE_TRAIT_SPECIALIZE_ARITHMETIC_TRAITS(CustomComplexStaticSizeVector);
VECTOR_BLAZE_TRAIT_SPECIALIZE_ALL_MAP_TRAITS(CustomComplexStaticSizeVector);
}  // namespace blaze

SPECTRE_ALWAYS_INLINE auto fabs(const CustomComplexStaticSizeVector& t) {
  return abs(*t);
}

MAKE_STD_ARRAY_VECTOR_BINOPS(CustomComplexStaticSizeVector)

MAKE_WITH_VALUE_IMPL_DEFINITION_FOR(CustomComplexStaticSizeVector)
