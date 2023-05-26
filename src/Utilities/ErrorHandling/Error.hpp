// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines macro ERROR.

#pragma once

#include <iomanip>
#include <string>

#include "Utilities/ErrorHandling/AbortWithErrorMessage.hpp"
#include "Utilities/ErrorHandling/Breakpoint.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/System/Abort.hpp"

namespace Error_detail {
// You can't use ScopedFpeState (a non-literal type) in a constexpr
// function, but you can call another function that uses it.
template <typename F>
[[noreturn]] SPECTRE_ALWAYS_INLINE void abort_without_fpes(
    const char* file, const int line, const char* const pretty_function,
    F&& message) {
  const ScopedFpeState disable_fpes(false);
  abort_with_error_message(file, line, pretty_function, message());
}
}  // namespace Error_detail

/*!
 * \ingroup ErrorHandlingGroup
 * \brief prints an error message to the standard error stream and aborts the
 * program.
 *
 * ERROR should not be used for coding errors, but instead for user errors
 * or failure modes of numerical algorithms. An acceptable use for error is also
 * in the default case of a switch statement.
 *
 * \details
 * The implementation is specialized so that in compile time contexts, a short
 * error message will be thrown, but in runtime contexts, a more verbose error
 * will be printed. This specialization of throwing a short error at compile
 * time greatly reduces the compile time and memory consumption during debug
 * builds of deep and heavily inlined `TensorExpression` tree traversals.
 *
 * To accomplish this, `__builtin_is_constant_evaluated()` is used directly
 * instead of calling a wrapper function because calling a wrapper was found to
 * slightly increase the compile time and memory usage of large
 * `TensorExpression`s when compiling in debug mode.
 *
 * \param m an arbitrary output stream.
 */
// isocpp.org recommends using an `if (true)` instead of a `do
// while(false)` for macros because the latter can mess with inlining
// in some (old?) compilers:
// https://isocpp.org/wiki/faq/misc-technical-issues#macros-with-multi-stmts
// https://isocpp.org/wiki/faq/misc-technical-issues#macros-with-if
// However, Intel's reachability analyzer (as of version 16.0.3
// 20160415) can't figure out that the else branch and everything
// after it is unreachable, causing warnings (and possibly suboptimal
// code generation).
#define ERROR(m)                                                             \
  do {                                                                       \
    if (__builtin_is_constant_evaluated()) {                                 \
      throw std::runtime_error("Failed");                                    \
    } else {                                                                 \
      Error_detail::abort_without_fpes(                                      \
          __FILE__, __LINE__, static_cast<const char*>(__PRETTY_FUNCTION__), \
          [&]() -> std::string {                                             \
            return MakeString{} << std::setprecision(18) << std::scientific  \
                                << m;                                        \
          });                                                                \
    }                                                                        \
  } while (false)

/*!
 * \ingroup ErrorHandlingGroup
 * \brief Same as ERROR but does not print a backtrace. Intended to be used for
 * user errors, such as incorrect values in an input file.
 */
#define ERROR_NO_TRACE(m)                                                    \
  do {                                                                       \
    if (__builtin_is_constant_evaluated()) {                                 \
      throw std::runtime_error("Failed");                                    \
    } else {                                                                 \
      const ScopedFpeState disable_fpes_ERROR(false);                        \
      abort_with_error_message_no_trace(                                     \
          __FILE__, __LINE__, static_cast<const char*>(__PRETTY_FUNCTION__), \
          MakeString{} << std::setprecision(18) << std::scientific << m);    \
    }                                                                        \
  } while (false)
