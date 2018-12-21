// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function dereference_wrapper

#pragma once

#include <complex>
#include <functional>
#include <utility>

#include "Utilities/ForceInline.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits.hpp"

/// \ingroup UtilitiesGroup
/// \brief Returns the reference object held by a reference wrapper, if a
/// non-reference_wrapper type is passed in then the object is returned
template <typename T>
decltype(auto) dereference_wrapper(T&& t) {
  return std::forward<T>(t);
}

/// \cond
template <typename T>
T& dereference_wrapper(const std::reference_wrapper<T>& t) {
  return t.get();
}
template <typename T>
T& dereference_wrapper(std::reference_wrapper<T>& t) {
  return t.get();
}
template <typename T>
T&& dereference_wrapper(const std::reference_wrapper<T>&& t) {
  return t.get();
}
template <typename T>
T&& dereference_wrapper(std::reference_wrapper<T>&& t) {
  return t.get();
}
/// \endcond

/// \cond
// Add overloads of math functions for reference_wrapper.
// This is necessary because if a class, say DataVector, inherits from
// PointerVector and does not specify the math operators specifically for
// DataVector then the implicit cast from reference_wrapper<DataVector> to
// DataVector does not result in finding the math operators.
//
// We use forwarding references to resolve ambiguity errors with
// DVecScalarMultExpr and DVecScalarDivExpr, with a std::reference_wrapper
// The forwarding references match everything perfectly and so the functions
// here will (almost) always win in overload selection.
#define UNARY_REF_WRAP_OP(OP)                        \
  template <typename T>                              \
  SPECTRE_ALWAYS_INLINE decltype(auto) OP(           \
      const std::reference_wrapper<T>& t) noexcept { \
    return OP(t.get());                              \
  }
#define BINARY_REF_WRAP_FUNCTION_OP(OP)                                    \
  template <                                                               \
      typename T0, typename T1,                                            \
      Requires<not tt::is_a_v<std::reference_wrapper, std::decay_t<T1>>> = \
          nullptr>                                                         \
  SPECTRE_ALWAYS_INLINE decltype(auto) OP(                                 \
      const std::reference_wrapper<T0>& t0, T1&& t1) noexcept {            \
    return OP(t0.get(), t1);                                               \
  }                                                                        \
  template <                                                               \
      typename T0, typename T1,                                            \
      Requires<not tt::is_a_v<std::reference_wrapper, std::decay_t<T0>>> = \
          nullptr>                                                         \
  SPECTRE_ALWAYS_INLINE decltype(auto) OP(                                 \
      T0&& t0, const std::reference_wrapper<T1>& t1) noexcept {            \
    return OP(t0, t1.get());                                               \
  }                                                                        \
  template <typename T0, typename T1>                                      \
  SPECTRE_ALWAYS_INLINE decltype(auto) OP(                                 \
      const std::reference_wrapper<T0>& t0,                                \
      const std::reference_wrapper<T1>& t1) noexcept {                     \
    return OP(t0.get(), t1.get());                                         \
  }

#define BINARY_REF_WRAP_OP(OP)                                             \
  template <                                                               \
      typename T0, typename T1,                                            \
      Requires<not tt::is_a_v<std::reference_wrapper, std::decay_t<T1>>> = \
          nullptr>                                                         \
  SPECTRE_ALWAYS_INLINE decltype(auto) operator OP(                        \
      const std::reference_wrapper<T0>& t0, T1&& t1) noexcept {            \
    return t0.get() OP t1;                                                 \
  }                                                                        \
  template <                                                               \
      typename T0, typename T1,                                            \
      Requires<not tt::is_a_v<std::reference_wrapper, std::decay_t<T0>>> = \
          nullptr>                                                         \
  SPECTRE_ALWAYS_INLINE decltype(auto) operator OP(                        \
      T0&& t0, const std::reference_wrapper<T1>& t1) noexcept {            \
    return t0 OP t1.get();                                                 \
  }                                                                        \
  template <typename T0, typename T1>                                      \
  SPECTRE_ALWAYS_INLINE decltype(auto) operator OP(                        \
      const std::reference_wrapper<T0>& t0,                                \
      const std::reference_wrapper<T1>& t1) noexcept {                     \
    return t0.get() OP t1.get();                                           \
  }

UNARY_REF_WRAP_OP(abs)
UNARY_REF_WRAP_OP(acos)
UNARY_REF_WRAP_OP(acosh)
UNARY_REF_WRAP_OP(asin)
UNARY_REF_WRAP_OP(asinh)
UNARY_REF_WRAP_OP(atan)
BINARY_REF_WRAP_FUNCTION_OP(atan2)
UNARY_REF_WRAP_OP(atanh)
UNARY_REF_WRAP_OP(cbrt)
UNARY_REF_WRAP_OP(conj)
UNARY_REF_WRAP_OP(cos)
UNARY_REF_WRAP_OP(cosh)
UNARY_REF_WRAP_OP(erf)
UNARY_REF_WRAP_OP(erfc)
UNARY_REF_WRAP_OP(exp)
UNARY_REF_WRAP_OP(exp2)
UNARY_REF_WRAP_OP(exp10)
UNARY_REF_WRAP_OP(fabs)
BINARY_REF_WRAP_FUNCTION_OP(hypot)
UNARY_REF_WRAP_OP(imag)
UNARY_REF_WRAP_OP(invcbrt)
UNARY_REF_WRAP_OP(invsqrt)
UNARY_REF_WRAP_OP(log)
UNARY_REF_WRAP_OP(log2)
UNARY_REF_WRAP_OP(log10)
UNARY_REF_WRAP_OP(max)
UNARY_REF_WRAP_OP(min)
BINARY_REF_WRAP_FUNCTION_OP(pow)
UNARY_REF_WRAP_OP(real)
UNARY_REF_WRAP_OP(sin)
UNARY_REF_WRAP_OP(sinh)
UNARY_REF_WRAP_OP(sqrt)
UNARY_REF_WRAP_OP(step_function)
UNARY_REF_WRAP_OP(tan)
UNARY_REF_WRAP_OP(tanh)

BINARY_REF_WRAP_OP(+)
BINARY_REF_WRAP_OP(-)
BINARY_REF_WRAP_OP(*)
BINARY_REF_WRAP_OP(/)
BINARY_REF_WRAP_OP(==)

template <typename T>
SPECTRE_ALWAYS_INLINE decltype(auto) operator-(
    const std::reference_wrapper<T>& t) noexcept {
  return -t.get();
}

#undef UNARY_REF_WRAP_OP
#undef BINARY_REF_WRAP_OP
#undef BINARY_REF_WRAP_FUNCTION_OP
/// \endcond
