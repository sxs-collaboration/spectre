// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Includes Blaze library with specific configs

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
namespace blaze {
constexpr bool usePadding = false;
constexpr bool useStreaming = true;
constexpr bool useOptimizedKernels = true;
}

// Override SMP configurations
#define _BLAZE_SYSTEM_SMP_H_
#define BLAZE_USE_SHARED_MEMORY_PARALLELIZATION 0
#define BLAZE_OPENMP_PARALLEL_MODE 0
#define BLAZE_CPP_THREADS_PARALLEL_MODE 0

// Disable MPI parallelization
#define _BLAZE_SYSTEM_MPI_H_
#define BLAZE_MPI_PARALLEL_MODE 0

#include <blaze/Blaze.h>

#define SPECTRE_BLAZE_ALLOCATOR(_TYPE_T, _SIZE_V) new _TYPE_T[_SIZE_V]
#define SPECTRE_BLAZE_DEALLOCATOR blaze::ArrayDelete()

namespace blaze {
template <typename ST>
struct AddScalar {
 public:
  explicit inline AddScalar(ST scalar) : scalar_(scalar) {}

  template <typename T>
  BLAZE_ALWAYS_INLINE decltype(auto) operator()(const T& a) const {
    return a + scalar_;
  }

  template <typename T>
  static constexpr bool simdEnabled() {
    return blaze::HasSIMDAdd<T, ST>::value;
  }

  template <typename T>
  BLAZE_ALWAYS_INLINE decltype(auto) load(const T& a) const {
    BLAZE_CONSTRAINT_MUST_BE_SIMD_PACK(T);
    return a + set(scalar_);
  }

 private:
  ST scalar_;
};

template <typename VT, bool TF, typename Scalar,
          typename = blaze::EnableIf_<blaze::IsNumeric<Scalar>>>
decltype(auto) operator+(const blaze::DenseVector<VT, TF>& vec, Scalar scalar) {
  return forEach(~vec, AddScalar<Scalar>(scalar));
}

template <typename Scalar, typename VT, bool TF,
          typename = blaze::EnableIf_<blaze::IsNumeric<Scalar>>>
decltype(auto) operator+(Scalar scalar, const blaze::DenseVector<VT, TF>& vec) {
  return forEach(~vec, AddScalar<Scalar>(scalar));
}

template <typename VT, bool TF, typename Scalar,
          typename = blaze::EnableIf_<blaze::IsNumeric<Scalar>>>
VT& operator+=(blaze::DenseVector<VT, TF>& vec, Scalar scalar) {
  (~vec) = (~vec) + scalar;
  return ~vec;
}

template <typename ST>
struct SubScalarRhs {
 public:
  explicit inline SubScalarRhs(ST scalar) : scalar_(scalar) {}

  template <typename T>
  BLAZE_ALWAYS_INLINE decltype(auto) operator()(const T& a) const {
    return a - scalar_;
  }

  template <typename T>
  static constexpr bool simdEnabled() {
    return blaze::HasSIMDSub<T, ST>::value;
  }

  template <typename T>
  BLAZE_ALWAYS_INLINE decltype(auto) load(const T& a) const {
    BLAZE_CONSTRAINT_MUST_BE_SIMD_PACK(T);
    return a - set(scalar_);
  }

 private:
  ST scalar_;
};

template <typename ST>
struct SubScalarLhs {
 public:
  explicit inline SubScalarLhs(ST scalar) : scalar_(scalar) {}

  template <typename T>
  BLAZE_ALWAYS_INLINE decltype(auto) operator()(const T& a) const {
    return scalar_ - a;
  }

  template <typename T>
  static constexpr bool simdEnabled() {
    return blaze::HasSIMDSub<T, ST>::value;
  }

  template <typename T>
  BLAZE_ALWAYS_INLINE decltype(auto) load(const T& a) const {
    BLAZE_CONSTRAINT_MUST_BE_SIMD_PACK(T);
    return set(scalar_) - a;
  }

 private:
  ST scalar_;
};

template <typename VT, bool TF, typename Scalar,
          typename = blaze::EnableIf_<blaze::IsNumeric<Scalar>>>
decltype(auto) operator-(const blaze::DenseVector<VT, TF>& vec, Scalar scalar) {
  return forEach(~vec, SubScalarRhs<Scalar>(scalar));
}

template <typename VT, bool TF, typename Scalar,
          typename = blaze::EnableIf_<blaze::IsNumeric<Scalar>>>
decltype(auto) operator-(Scalar scalar, const blaze::DenseVector<VT, TF>& vec) {
  return forEach(~vec, SubScalarLhs<Scalar>(scalar));
}

template <typename VT, bool TF, typename Scalar,
          typename = blaze::EnableIf_<blaze::IsNumeric<Scalar>>>
VT& operator-=(blaze::DenseVector<VT, TF>& vec, Scalar scalar) {
  (~vec) = (~vec) - scalar;
  return ~vec;
}
}  // namespace blaze
/// \endcond
