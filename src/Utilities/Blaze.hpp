// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Includes Blaze library with specific configs

#include <blaze/math/CustomVector.h>
#include <blaze/system/Optimizations.h>

#pragma once

#ifdef __GNUC__
#pragma GCC system_header
#endif

/// \cond

// Override cache size
//#define _BLAZE_SYSTEM_CACHESIZE_H_
// constexpr size_t cacheSize = 6291456UL;

// Override padding, streaming and kernel options
#define _BLAZE_SYSTEM_OPTIMIZATIONS_H_

const blaze::AlignmentFlag blaze_unaligned = blaze::AlignmentFlag::unaligned;
const blaze::PaddingFlag blaze_unpadded = blaze::PaddingFlag::unpadded;

// Override SMP configurations
#define _BLAZE_SYSTEM_SMP_H_
#define BLAZE_USE_SHARED_MEMORY_PARALLELIZATION 0
#define BLAZE_OPENMP_PARALLEL_MODE 0
#define BLAZE_CPP_THREADS_PARALLEL_MODE 0
#define BLAZE_BOOST_THREADS_PARALLEL_MODE 0

// Disable MPI parallelization
#define _BLAZE_SYSTEM_MPI_H_
#define BLAZE_MPI_PARALLEL_MODE 0

// Disable HPX parallelization
#define BLAZE_HPX_PARALLEL_MODE 0

// Disable all padding
#define BLAZE_USE_PADDING 0

#define BLAZE_USE_STREAMING 1
#define BLAZE_USE_OPTIMIZED_KERNELS 1
#define BLAZE_USE_DEFAULT_INITIALIZATON 0
/// \endcond

namespace blaze {
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
