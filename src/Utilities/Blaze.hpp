// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Includes Blaze library with specific configs

#pragma once

#include <blaze/math/CustomVector.h>
#include <blaze/math/DenseVector.h>
#include <blaze/math/GroupTag.h>

using blaze_default_group = blaze::GroupTag<0>;

namespace blaze {
// This vectorized implementation of the step function is necessary because
// blaze does not offer its own version of a vectorized step function.
template <typename T>
BLAZE_ALWAYS_INLINE SIMDdouble step_function(const SIMDf64<T>& v) noexcept
#if BLAZE_AVX512F_MODE || BLAZE_MIC_MODE
{
  return _mm512_set_pd((*v).eval().value[7] < 0.0 ? 0.0 : 1.0,
                       (*v).eval().value[6] < 0.0 ? 0.0 : 1.0,
                       (*v).eval().value[5] < 0.0 ? 0.0 : 1.0,
                       (*v).eval().value[4] < 0.0 ? 0.0 : 1.0,
                       (*v).eval().value[3] < 0.0 ? 0.0 : 1.0,
                       (*v).eval().value[2] < 0.0 ? 0.0 : 1.0,
                       (*v).eval().value[1] < 0.0 ? 0.0 : 1.0,
                       (*v).eval().value[0] < 0.0 ? 0.0 : 1.0);
}
#elif BLAZE_AVX_MODE
{
  return _mm256_set_pd((*v).eval().value[3] < 0.0 ? 0.0 : 1.0,
                       (*v).eval().value[2] < 0.0 ? 0.0 : 1.0,
                       (*v).eval().value[1] < 0.0 ? 0.0 : 1.0,
                       (*v).eval().value[0] < 0.0 ? 0.0 : 1.0);
}
#elif BLAZE_SSE2_MODE
{
  return _mm_set_pd((*v).eval().value[1] < 0.0 ? 0.0 : 1.0,
                    (*v).eval().value[0] < 0.0 ? 0.0 : 1.0);
}
#else
{
  return SIMDdouble{(*v).value < 0.0 ? 0.0 : 1.0};
}
#endif

BLAZE_ALWAYS_INLINE double step_function(const double v) noexcept {
  return v < 0.0 ? 0.0 : 1.0;
}

struct StepFunction {
  explicit inline StepFunction() = default;

  template <typename T>
  BLAZE_ALWAYS_INLINE decltype(auto) operator()(const T& a) const noexcept {
    return step_function(a);
  }

  template <typename T>
  BLAZE_ALWAYS_INLINE decltype(auto) load(const T& a) const noexcept {
    BLAZE_CONSTRAINT_MUST_BE_SIMD_PACK(T);
    return step_function(a);
  }
};
}  // namespace blaze

template <typename VT, bool TF>
BLAZE_ALWAYS_INLINE decltype(auto) step_function(
    const blaze::DenseVector<VT, TF>& vec) noexcept {
  return map(*vec, blaze::StepFunction{});
}

template <typename VT, bool TF>
BLAZE_ALWAYS_INLINE decltype(auto) StepFunction(
    const blaze::DenseVector<VT, TF>& vec) noexcept {
  return map(*vec, blaze::StepFunction{});
}
