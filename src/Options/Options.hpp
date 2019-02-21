// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines classes and functions for making classes creatable from
/// input files.

#pragma once

#include <exception>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>

#include "ErrorHandling/Error.hpp"
#include "Utilities/NoSuchType.hpp"

/// \cond
namespace YAML {
class Node;
}  // namespace YAML
/// \endcond

/// The string used in option structs
using OptionString = const char* const;

/// \ingroup OptionParsingGroup
/// Information about the nested operations being performed by the
/// parser, for use in printing errors.  A default-constructed
/// OptionContext is printed as an empty string.  This struct is
/// primarily used as an argument to PARSE_ERROR for reporting input
/// file parsing errors.  Users outside of the core option parsing
/// code should not need to manipulate the contents.
struct OptionContext {
  bool top_level{true};
  /// (Part of) the parsing "backtrace" printed with an error
  std::string context;
  /// File line number (0 based)
  int line{-1};
  /// File column number (0 based)
  int column{-1};

  /// Append a line to the context.  Automatically appends a colon.
  void append(const std::string& c) noexcept { context += c + ":\n"; }
};

inline std::ostream& operator<<(std::ostream& s,
                                const OptionContext& c) noexcept {
  s << c.context;
  if (c.line >= 0 and c.column >= 0) {
    s << "At line " << c.line + 1 << " column " << c.column + 1 << ":\n";
  }
  return s;
}

/// \ingroup OptionParsingGroup
/// Like ERROR("\n" << (context) << m), but instead throws an
/// exception that will be caught in a higher level Options if not
/// passed a top-level context.  This is used to print a parsing
/// "backtrace" since we can't pass any extra data through the
/// yaml-cpp code.
///
/// \param context OptionContext used to print a parsing traceback
/// \param m error message, as for ERROR
#define PARSE_ERROR(context, m)                                         \
  do {                                                                  \
    if ((context).top_level) {                                          \
      /* clang-tidy: macro arg in parentheses */                        \
      ERROR("\n" << (context) << m); /* NOLINT */                       \
    } else {                                                            \
      std::ostringstream avoid_name_collisions_PARSE_ERROR;             \
      /* clang-tidy: macro arg in parentheses */                        \
      avoid_name_collisions_PARSE_ERROR << (context) << m; /* NOLINT */ \
      throw Options_detail::propagate_context(                          \
          avoid_name_collisions_PARSE_ERROR.str());                     \
    }                                                                   \
  } while (false)

namespace Options_detail {
class propagate_context : public std::exception {
 public:
  // cppcheck-suppress passedByValue
  explicit propagate_context(std::string message) noexcept
      : message_(std::move(message)) {}

  const char* what() const noexcept override { return message_.c_str(); }
  const std::string& message() const noexcept { return message_; }

 private:
  std::string message_;
};
}  // namespace Options_detail

/// \ingroup OptionParsingGroup
/// The type that options are passed around as.  Contains YAML node
/// data and an OptionContext.
///
/// \note To use any methods on this class in a concrete function you
/// must include ParseOptions.hpp, but you do *not* need to include
/// that header to use this in an uninstantiated
/// `create_from_yaml::create` function.
class Option {
 public:
  const OptionContext& context() const noexcept;

  /// Append a line to the contained context.
  void append_context(const std::string& context) noexcept;

  /// Convert to an object of type `T`.
  template <typename T, typename Metavariables = NoSuchType>
  T parse_as() const;

  /// \note This constructor overwrites the mark data in the supplied
  /// context with the one from the node.
  ///
  /// \warning This method is for internal use of the option parser.
  explicit Option(YAML::Node node, OptionContext context = {}) noexcept;

  /// \warning This method is for internal use of the option parser.
  explicit Option(OptionContext context) noexcept;

  /// \warning This method is for internal use of the option parser.
  const YAML::Node& node() const noexcept;

  /// Sets the node and updates the context's mark to correspond to it.
  ///
  /// \warning This method is for internal use of the option parser.
  void set_node(YAML::Node node) noexcept;

 private:
  std::unique_ptr<YAML::Node> node_;
  OptionContext context_;
};

/// \ingroup OptionParsingGroup
/// Used by the parser to create an object.  The default action is to
/// parse options using `T::options`.  This struct may be specialized
/// to change that behavior for specific types.
///
/// Do not call create directly.  Use Option::parse_as instead.
template <typename T>
struct create_from_yaml {
  template <typename Metavariables>
  static T create(const Option& options);
};
