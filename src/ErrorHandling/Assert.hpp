// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines macro ASSERT.

#pragma once

#include <sstream>
#include <string>

#include "ErrorHandling/AbortWithErrorMessage.hpp"
#include "Parallel/Abort.hpp"
#include "Utilities/Literals.hpp"

/*!
 * \ingroup ErrorHandlingGroup
 * \brief Assert that an expression should be true.
 *
 * If the preprocessor macro SPECTRE_DEBUG is defined and the expression is
 * false, an error message is printed to the standard error stream, and the
 * program aborts. ASSERT should be used to catch coding errors as it does
 * nothing in production code.
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
#define ASSERT(a, m)                                                          \
  do {                                                                        \
    if (!(a)) {                                                               \
      std::ostringstream avoid_name_collisions_ASSERT;                        \
      /* clang-tidy: macro arg in parentheses */                              \
      avoid_name_collisions_ASSERT << m; /* NOLINT */                         \
      abort_with_error_message(#a, __FILE__, __LINE__,                        \
                               static_cast<const char*>(__PRETTY_FUNCTION__), \
                               avoid_name_collisions_ASSERT.str());           \
    }                                                                         \
  } while (false)
#else
#define ASSERT(a, m)                                   \
  do {                                                 \
    if (false) {                                       \
      static_cast<void>(a);                            \
      std::ostringstream avoid_name_collisions_ASSERT; \
      /* clang-tidy: macro arg in parentheses */       \
      avoid_name_collisions_ASSERT << m;  /* NOLINT */ \
      static_cast<void>(avoid_name_collisions_ASSERT); \
    }                                                  \
  } while (false)
#endif
