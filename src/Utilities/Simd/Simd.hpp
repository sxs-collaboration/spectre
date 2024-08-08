// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#ifdef SPECTRE_USE_XSIMD
#include <limits>
#include <xsimd/xsimd.hpp>

#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TypeTraits/CreateGetTypeAliasOrDefault.hpp"

/// Namespace containing SIMD functions based on XSIMD.
namespace simd = xsimd;

namespace MakeWithValueImpls {
template <typename U, typename T, typename Arch>
struct MakeWithValueImpl<xsimd::batch<U, Arch>, T> {
  static SPECTRE_ALWAYS_INLINE xsimd::batch<U, Arch> apply(const T& /* input */,
                                                           const U value) {
    return xsimd::batch<U, Arch>(value);
  }
};
}  // namespace MakeWithValueImpls

namespace xsimd {
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(value_type)

namespace detail {
template <typename T>
struct size_impl : std::integral_constant<size_t, 1> {};

template <typename T, typename A>
struct size_impl<batch<T, A>>
    : std::integral_constant<size_t, batch<T, A>::size> {};
}  // namespace detail

template <typename T>
constexpr size_t size() {
  return detail::size_impl<T>::value;
}

namespace detail {
template <typename T, size_t... Is>
T make_sequence_impl(std::index_sequence<Is...> /*meta*/) {
  return T{static_cast<typename T::value_type>(Is)...};
}
}  // namespace detail

template <typename T>
T make_sequence() {
  return detail::make_sequence_impl<T>(std::make_index_sequence<size<T>()>{});
}
}  // namespace xsimd

namespace std {
template <typename T, typename Arch>
class numeric_limits<::xsimd::batch<T, Arch>> : public numeric_limits<T> {
 public:
  static constexpr bool is_iec559 = false;

  static ::xsimd::batch<T, Arch> min() {
    return ::xsimd::batch<T, Arch>(numeric_limits<T>::min());
  }
  static ::xsimd::batch<T, Arch> lowest() {
    return ::xsimd::batch<T, Arch>(numeric_limits<T>::lowest());
  }
  static ::xsimd::batch<T, Arch> max() {
    return ::xsimd::batch<T, Arch>(numeric_limits<T>::max());
  }
  static ::xsimd::batch<T, Arch> epsilon() {
    return ::xsimd::batch<T, Arch>(numeric_limits<T>::epsilon());
  }
  static ::xsimd::batch<T, Arch> round_error() {
    return ::xsimd::batch<T, Arch>(numeric_limits<T>::round_error());
  }
  static ::xsimd::batch<T, Arch> infinity() {
    return ::xsimd::batch<T, Arch>(numeric_limits<T>::infinity());
  }
  static ::xsimd::batch<T, Arch> quiet_NaN() {
    return ::xsimd::batch<T, Arch>(numeric_limits<T>::quiet_NaN());
  }
  static ::xsimd::batch<T, Arch> signaling_NaN() {
    return ::xsimd::batch<T, Arch>(numeric_limits<T>::signaling_NaN());
  }
  static ::xsimd::batch<T, Arch> denorm_min() {
    return ::xsimd::batch<T, Arch>(numeric_limits<T>::denorm_min());
  }
};
}  // namespace std
#else  // no xsimd

#include <cmath>
#include <complex>
#include <type_traits>

#include "Utilities/Requires.hpp"

namespace simd {
template <typename T, typename A = void>
class batch;
template <typename T, typename A = void>
class batch_bool;

template <typename T>
struct scalar_type {
  using type = T;
};
template <typename T>
using scalar_type_t = typename scalar_type<T>::type;

template <typename T>
struct mask_type {
  using type = bool;
};
template <typename T>
using mask_type_t = typename mask_type<T>::type;

template <typename T>
struct is_batch : std::false_type {};

template <typename T, typename A>
struct is_batch<batch<T, A>> : std::true_type {};

namespace detail {
template <typename T>
struct size_impl : std::integral_constant<size_t, 1> {};
}  // namespace detail

template <typename T>
constexpr size_t size() {
  return detail::size_impl<T>::value;
}

namespace detail {
template <typename T, size_t... Is>
T make_sequence_impl(std::index_sequence<Is...> /*meta*/) {
  return T{static_cast<typename T::value_type>(Is)...};
}
}  // namespace detail

template <typename T>
T make_sequence() {
  return detail::make_sequence_impl<T>(std::make_index_sequence<size<T>()>{});
}

// NOLINTBEGIN(misc-unused-using-decls)
using std::abs;
using std::acos;
using std::acosh;
using std::arg;
using std::asin;
using std::asinh;
using std::atan;
using std::atan2;
using std::atanh;
using std::cbrt;
using std::ceil;
using std::conj;
using std::copysign;
using std::cos;
using std::cosh;
using std::erf;
using std::erfc;
using std::exp;
using std::exp2;
using std::expm1;
using std::fabs;
using std::fdim;
using std::floor;
using std::fmax;
using std::fmin;
using std::fmod;
using std::hypot;
using std::isfinite;
using std::isinf;
using std::isnan;
using std::ldexp;
using std::lgamma;
using std::log;
using std::log10;
using std::log1p;
using std::log2;
using std::max;
using std::min;
using std::modf;
using std::nearbyint;
using std::nextafter;
using std::norm;
using std::polar;
using std::proj;
using std::remainder;
using std::rint;
using std::round;
using std::sin;
using std::sinh;
using std::sqrt;
using std::tan;
using std::tanh;
using std::tgamma;
using std::trunc;
// NOLINTEND(misc-unused-using-decls)

inline bool all(const bool mask) { return mask; }

inline bool any(const bool mask) { return mask; }

inline bool none(const bool mask) { return not mask; }

template <typename T, Requires<std::is_scalar_v<T>> = nullptr>
T clip(const T& val, const T& low, const T& hi) {
  assert(low <= hi && "ordered clipping bounds");
  return low > val ? low : (hi < val ? hi : val);
}

#if defined(__GLIBC__)
inline float exp10(const float& x) { return ::exp10f(x); }
inline double exp10(const double& x) { return ::exp10(x); }
#else
inline float exp10(const float& x) {
  const float ln10 = std::log(10.f);
  return std::exp(ln10 * x);
}
inline double exp10(const double& x) {
  const double ln10 = std::log(10.);
  return std::exp(ln10 * x);
}
#endif

inline double sign(const bool& v) { return static_cast<double>(v); }

template <typename T>
T sign(const T& v) {
  return v < static_cast<T>(0)    ? static_cast<T>(-1.)
         : v == static_cast<T>(0) ? static_cast<T>(0.)
                                  : static_cast<T>(1.);
}

template <typename T>
T select(const bool cond, const T true_branch, const T false_branch) {
  return cond ? true_branch : false_branch;
}

inline std::pair<float, float> sincos(const float val) {
  // The nvcc compiler's built-in __sincos is for GPU code, not CPU code. In
  // the case that we are running on a GPU (__CUDA_ARCH__ is defined) or we
  // are not using nvcc then use the builtin, otherwise call sin and cos
  // separately.
#if (defined(__CUDACC__) && defined(__CUDA_ARCH__)) or (not defined(__CUDACC__))
  float result_sin{};
  float result_cos{};
  __sincosf(val, &result_sin, &result_cos);
  return std::pair{result_sin, result_cos};
#else
  return std::pair{sin(val), cos(val)};
#endif
}

inline std::pair<double, double> sincos(const double val) {
  // The nvcc compiler's built-in __sincos is for GPU code, not CPU code. In
  // the case that we are running on a GPU (__CUDA_ARCH__ is defined) or we
  // are not using nvcc then use the builtin, otherwise call sin and cos
  // separately.
#if (defined(__CUDACC__) && defined(__CUDA_ARCH__)) or (not defined(__CUDACC__))
  double result_sin{};
  double result_cos{};
  __sincos(val, &result_sin, &result_cos);
  return std::pair{result_sin, result_cos};
#else
  return std::pair{sin(val), cos(val)};
#endif
}

template <typename T, Requires<std::is_integral_v<T>> = nullptr>
T fma(const T a, const T b, const T c) {
  return a * b + c;
}

template <typename T, Requires<std::is_floating_point_v<T>> = nullptr>
T fma(const T a, const T b, const T c) {
  return std::fma(a, b, c);
}

template <typename T, Requires<std::is_integral_v<T>> = nullptr>
T fms(const T& a, const T& b, const T& c) {
  return a * b - c;
}

template <typename T, Requires<std::is_floating_point_v<T>> = nullptr>
T fms(const T a, const T b, const T c) {
  return std::fma(a, b, -c);
}

template <typename T, Requires<std::is_integral_v<T>> = nullptr>
T fnma(const T a, const T b, const T c) {
  return -(a * b) + c;
}

template <typename T, Requires<std::is_floating_point_v<T>> = nullptr>
T fnma(const T a, const T b, const T c) {
  return std::fma(-a, b, c);
}

template <typename T, Requires<std::is_integral_v<T>> = nullptr>
T fnms(const T a, const T b, const T c) {
  return -(a * b) - c;
}

template <typename T, Requires<std::is_floating_point_v<T>> = nullptr>
T fnms(const T a, const T b, const T c) {
  return -std::fma(a, b, c);
}

template <typename Arch = void, typename From>
From load(From* mem) {
  static_assert(std::is_arithmetic_v<From>);
  return *mem;
}

template <typename Arch = void, typename T>
void store(T* mem, const T& val) {
  static_assert(std::is_arithmetic_v<T>);
  *mem = val;
}

template <typename Arch = void, typename From>
From load_unaligned(From* mem) {
  static_assert(std::is_arithmetic_v<From>);
  return *mem;
}

template <typename Arch = void, typename T>
void store_unaligned(T* mem, const T& val) {
  static_assert(std::is_arithmetic_v<T>);
  *mem = val;
}
}  // namespace simd
#endif
