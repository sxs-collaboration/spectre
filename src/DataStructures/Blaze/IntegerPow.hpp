// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <blaze/math/DenseVector.h>
#include <blaze/math/constraints/SIMDPack.h>
#include <blaze/math/simd/BasicTypes.h>
#include <blaze/system/Inline.h>
#include <blaze/system/Vectorization.h>

#include "Utilities/Math.hpp"

namespace blaze {

namespace detail {
template <typename T>
BLAZE_ALWAYS_INLINE SIMDdouble integer_pow_impl(const SIMDf64<T>& x,
                                                const int e) {
  ASSERT(e >= 0, "Negative powers are not implemented");
  int ecount = e;
  int bitcount = 1;
  while (ecount >>= 1) {
    ++bitcount;
  }
  SIMDdouble result = blaze::set(1.0);
  while (bitcount) {
    result *= result;
    if ((e >> --bitcount) & 0x1) {
      result *= x;
    }
  }
  return result;
}
}  // namespace detail
// This vectorized implementation of the integer pow function is necessary
// because blaze does not offer its own version of a integer pow function.
template <typename T>
BLAZE_ALWAYS_INLINE SIMDdouble integer_pow(const SIMDf64<T>& b, const int e) {
  switch (e) {
    case 0:
      return blaze::set(1.0);
    case 1:
      return b;
    case 2:
      return b * b;
    case 3:
      return b * b * b;
    case 4: {
      const SIMDdouble b2 = b * b;
      return b2 * b2;
    }
    case 5: {
      const SIMDdouble b2 = b * b;
      return b2 * b2 * b;
    }
    case 6: {
      const SIMDdouble b2 = b * b;
      return b2 * b2 * b2;
    }
    case 7: {
      const SIMDdouble b2 = b * b;
      return b2 * b2 * b2 * b;
    }
    case 8: {
      const SIMDdouble b2 = b * b;
      const SIMDdouble b4 = b2 * b2;
      return b4 * b4;
    }
    default:
      return detail::integer_pow_impl(b, e);
  }
}

struct IntegerPow {
  int exponent;
  explicit inline IntegerPow(const int e) : exponent(e) {}

  BLAZE_ALWAYS_INLINE double operator()(const double a) const {
    return ::integer_pow(a, exponent);
  }

  template <typename T>
  BLAZE_ALWAYS_INLINE decltype(auto) load(const T& a) const {
    BLAZE_CONSTRAINT_MUST_BE_SIMD_PACK(T);
    return integer_pow(a, exponent);
  }
};
}  // namespace blaze

template <typename VT, bool TF>
BLAZE_ALWAYS_INLINE decltype(auto) integer_pow(
    const blaze::DenseVector<VT, TF>& vec, const int e) {
  return map(*vec, blaze::IntegerPow{e});
}

template <typename VT, bool TF>
BLAZE_ALWAYS_INLINE decltype(auto) IntegerPow(
    const blaze::DenseVector<VT, TF>& vec, const int e) {
  return map(*vec, blaze::IntegerPow{e});
}
