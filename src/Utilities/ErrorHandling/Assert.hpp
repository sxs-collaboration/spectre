// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines macro ASSERT.

#pragma once

#include <iomanip>
#include <sstream>
#include <string>

#include "Utilities/ErrorHandling/AbortWithErrorMessage.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Kokkos/KokkosMacros.hpp"

/*!
 * \ingroup ErrorHandlingGroup
 * \brief Assert that an expression should be true.
 *
 * If the preprocessor macro SPECTRE_DEBUG is defined and the expression is
 * false, an error message is printed to the standard error stream, and the
 * program aborts. ASSERT should be used to catch coding errors as it does
 * nothing in production code.
 *
 * ASSERT can also be used in functions that run on a GPU (`KOKKOS_FUNCTION`s).
 * On the GPU the error message can't be printed, so a failed assertion only
 * reports the expression, file and line.
 *
 * \param a the expression that must be true
 * \param m the error message as an ostream
 */
#ifdef SPECTRE_DEBUG
// isocpp.org recommends using an `if (true)` instead of a `do
// while(false)` for macros because the latter can mess with inlining
// in some (old?) compilers:
// https://isocpp.org/wiki/faq/misc-technical-issues#macros-with-multi-stmts
// https://isocpp.org/wiki/faq/misc-technical-issues#macros-with-if
// However, Intel's reachability analyzer (as of version 16.0.3
// 20160415) can't figure out that the else branch and everything
// after it is unreachable, causing warnings (and possibly suboptimal
// code generation).
#define SPECTRE_ASSERT_ON_HOST(a, m)                                           \
  do {                                                                         \
    if (!(a)) { /* NOLINT(readability-simplify-boolean-expr) */                \
      const ScopedFpeState disable_fpes_ASSERT(false);                         \
      std::ostringstream avoid_name_collisions_ASSERT;                         \
      /* clang-tidy: macro arg in parentheses */                               \
      avoid_name_collisions_ASSERT << std::setprecision(18) << std::scientific \
                                   << m; /* NOLINT */                          \
      abort_with_error_message(#a, __FILE__, __LINE__,                         \
                               static_cast<const char*>(__PRETTY_FUNCTION__),  \
                               avoid_name_collisions_ASSERT.str());            \
    }                                                                          \
  } while (false)
#else
#define SPECTRE_ASSERT_ON_HOST(a, m)                                           \
  do {                                                                         \
    if (false) {                                                               \
      static_cast<void>(a);                                                    \
      const ScopedFpeState disable_fpes_ASSERT(false);                         \
      std::ostringstream avoid_name_collisions_ASSERT;                         \
      /* clang-tidy: macro arg in parentheses */                               \
      avoid_name_collisions_ASSERT << std::setprecision(18) << std::scientific \
                                   << m; /* NOLINT */                          \
      static_cast<void>(avoid_name_collisions_ASSERT);                         \
    }                                                                          \
  } while (false)
#endif

#ifdef SPECTRE_KOKKOS
// In code that is compiled for a GPU, the message can't be streamed and
// printed, so only the expression, file and line are reported there. Kokkos
// compiles only one of the branches for the host and the GPU, respectively.
#define SPECTRE_ASSERT_STRINGIFY_IMPL(x) #x
#define SPECTRE_ASSERT_STRINGIFY(x) SPECTRE_ASSERT_STRINGIFY_IMPL(x)
#define SPECTRE_ASSERT_DEVICE_MESSAGE(a) \
  "ASSERT failed on the device: " a "\nat " __FILE__                  \
  ":" SPECTRE_ASSERT_STRINGIFY(__LINE__) "\nRun on the host for the " \
                                         "full error message.\n"
#ifdef SPECTRE_DEBUG
#define ASSERT(a, m)                                                           \
  do {                                                                         \
    KOKKOS_IF_ON_HOST((SPECTRE_ASSERT_ON_HOST(a, m);))                         \
    KOKKOS_IF_ON_DEVICE(                                                       \
        (if (not(a)) { ::Kokkos::abort(SPECTRE_ASSERT_DEVICE_MESSAGE(#a)); })) \
  } while (false)
#else
#define ASSERT(a, m)                                   \
  do {                                                 \
    KOKKOS_IF_ON_HOST((SPECTRE_ASSERT_ON_HOST(a, m);)) \
  } while (false)
#endif
#else
#define ASSERT(a, m) SPECTRE_ASSERT_ON_HOST(a, m)
#endif
