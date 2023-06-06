// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Declares function abort_with_error_message

#pragma once

#include <string>

/// \ingroup ErrorHandlingGroup
/// Compose an error message with an expression and a backtrace, then abort the
/// program.
///
/// We try to demangle and format the backtrace. Long symbol names are
/// abbreviated, unless you set the `SPECTRE_SHOW_FULL_BACKTRACE_SYMBOLS`
/// environment variable to a non-empty value (e.g. "1").
[[noreturn]] void abort_with_error_message(const char* expression,
                                           const char* file, int line,
                                           const char* pretty_function,
                                           const std::string& message);

/// \ingroup ErrorHandlingGroup
/// Compose an error message including a backtrace and abort the program.
///
/// We try to demangle and format the backtrace. Long symbol names are
/// abbreviated, unless you set the `SPECTRE_SHOW_FULL_BACKTRACE_SYMBOLS`
/// environment variable to a non-empty value (e.g. "1").
[[noreturn]] void abort_with_error_message(const char* file, int line,
                                           const char* pretty_function,
                                           const std::string& message);

/// \ingroup ErrorHandlingGroup
/// Compose an error message without a backtrace and abort the program.
[[noreturn]] void abort_with_error_message_no_trace(const char* file, int line,
                                                    const char* pretty_function,
                                                    const std::string& message);
