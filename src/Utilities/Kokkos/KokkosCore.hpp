// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Utilities/ErrorHandling/Assert.hpp"

#if __has_include(<Kokkos_Core.hpp>)
#include <Kokkos_Core.hpp>

/// \brief If defined then SpECTRE is using Kokkos
#define SPECTRE_KOKKOS 1

/*!
 * \brief Defined while compiling device code for a GPU backend.
 *
 * GPU compilers compile each source file twice: once for the host and once for
 * the device. This macro is defined only in the device pass, so code that is
 * never called on the device (e.g. explicit instantiations of `KOKKOS_FUNCTION`
 * templates for `DataVector`) can be excluded from device compilation:
 *
 * \code
 * GENERATE_INSTANTIATIONS(INSTANTIATE, (double))
 * // The `DataVector` versions are never called on the device
 * #ifndef SPECTRE_KOKKOS_DEVICE_PASS
 * GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector))
 * #endif
 * \endcode
 *
 * Only use this for code at namespace scope. Inside functions use Kokkos'
 * `KOKKOS_IF_ON_HOST` and `KOKKOS_IF_ON_DEVICE` instead.
 */
#if defined(__CUDA_ARCH__) or defined(__HIP_DEVICE_COMPILE__) or \
    defined(__SYCL_DEVICE_ONLY__)
#define SPECTRE_KOKKOS_DEVICE_PASS 1
#endif

#else  // #if __has_include(<Kokkos_Core.hpp>)
#define KOKKOS_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
// Without Kokkos all code runs on the host. The argument is wrapped in
// parentheses, e.g. `KOKKOS_IF_ON_HOST((ASSERT(...);))`, like in Kokkos.
#define SPECTRE_KOKKOS_STRIP_PARENS(...) __VA_ARGS__
#define KOKKOS_IF_ON_HOST(CODE) {SPECTRE_KOKKOS_STRIP_PARENS CODE}
#define KOKKOS_IF_ON_DEVICE(CODE) \
  {                               \
  }
#endif  // #if __has_include(<Kokkos_Core.hpp>)

/// \cond
#define SPECTRE_KOKKOS_STRINGIFY_IMPL(x) #x
#define SPECTRE_KOKKOS_STRINGIFY(x) SPECTRE_KOKKOS_STRINGIFY_IMPL(x)
#define SPECTRE_KOKKOS_ASSERT_DEVICE_MESSAGE(condition)                     \
  "SPECTRE_KOKKOS_ASSERT failed on the device: " condition "\nat " __FILE__ \
  ":" SPECTRE_KOKKOS_STRINGIFY(                                             \
      __LINE__) "\nRun on the host for the full error message.\n"
/// \endcond

/*!
 * \ingroup ErrorHandlingGroup
 * \brief `ASSERT` that can be used in `KOKKOS_FUNCTION`s.
 *
 * `ASSERT` can't be compiled for the device because it streams its message.
 * On the host this macro is `ASSERT(a, m)`. On the device the message `m`
 * can't be formatted, so if `SPECTRE_DEBUG` is defined and `a` is false it
 * calls `Kokkos::abort` with the condition, file and line instead. Like
 * `ASSERT`, it does nothing in release builds.
 *
 * Use `KOKKOS_IF_ON_HOST` and `KOKKOS_IF_ON_DEVICE` directly for host- or
 * device-specific code that isn't an assertion. Note that they take their
 * argument in an extra pair of parentheses, e.g. `KOKKOS_IF_ON_HOST((code;))`,
 * so that commas in the code don't split it into several macro arguments.
 */
#ifdef SPECTRE_DEBUG
#define SPECTRE_KOKKOS_ASSERT(a, m)                            \
  do {                                                         \
    KOKKOS_IF_ON_HOST((ASSERT(a, m);))                         \
    KOKKOS_IF_ON_DEVICE((if (not(a)) {                         \
      Kokkos::abort(SPECTRE_KOKKOS_ASSERT_DEVICE_MESSAGE(#a)); \
    }))                                                        \
  } while (false)
#else
#define SPECTRE_KOKKOS_ASSERT(a, m) KOKKOS_IF_ON_HOST((ASSERT(a, m);))
#endif
