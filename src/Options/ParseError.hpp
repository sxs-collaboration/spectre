// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <exception>
#include <sstream>
#include <string>
#include <utility>

#include "Options/Context.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace Options {
/// \cond
namespace detail {
class propagate_context : public std::exception {
 public:
  // cppcheck-suppress passedByValue
  explicit propagate_context(std::string message)
      : message_(std::move(message)) {}

  const char* what() const noexcept(true) override { return message_.c_str(); }
  const std::string& message() const { return message_; }

 private:
  std::string message_;
};
}  // namespace detail
/// \endcond

/// \ingroup OptionParsingGroup
/// Like ERROR("\n" << (context) << m), but instead throws an
/// exception that will be caught in a higher level Options if not
/// passed a top-level context.  This is used to print a parsing
/// "backtrace" since we can't pass any extra data through the
/// yaml-cpp code.
///
/// \param context Context used to print a parsing traceback
/// \param m error message, as for ERROR
#define PARSE_ERROR(context, m)                                         \
  do {                                                                  \
    if ((context).top_level) {                                          \
      /* clang-tidy: macro arg in parentheses */                        \
      ERROR_NO_TRACE("\n" << (context) << m); /* NOLINT */              \
    } else {                                                            \
      std::ostringstream avoid_name_collisions_PARSE_ERROR;             \
      /* clang-tidy: macro arg in parentheses */                        \
      avoid_name_collisions_PARSE_ERROR << (context) << m; /* NOLINT */ \
      throw ::Options::detail::propagate_context(                       \
          avoid_name_collisions_PARSE_ERROR.str());                     \
    }                                                                   \
  } while (false)
}  // namespace Options
