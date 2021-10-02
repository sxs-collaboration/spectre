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

#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
namespace YAML {
class Node;
}  // namespace YAML
/// \endcond

/// Utilities for parsing input files.
namespace Options {
/// The string used in option structs
using String = const char* const;

/// \ingroup OptionParsingGroup
/// Information about the nested operations being performed by the
/// parser, for use in printing errors.  A default-constructed
/// Context is printed as an empty string.  This struct is
/// primarily used as an argument to PARSE_ERROR for reporting input
/// file parsing errors.  Users outside of the core option parsing
/// code should not need to manipulate the contents.
struct Context {
  bool top_level{true};
  /// (Part of) the parsing "backtrace" printed with an error
  std::string context;
  /// File line number (0 based)
  int line{-1};
  /// File column number (0 based)
  int column{-1};

  /// Append a line to the context.  Automatically appends a colon.
  void append(const std::string& c) { context += c + ":\n"; }
};

inline std::ostream& operator<<(std::ostream& s, const Context& c) {
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
      throw ::Options::Options_detail::propagate_context(               \
          avoid_name_collisions_PARSE_ERROR.str());                     \
    }                                                                   \
  } while (false)

namespace Options_detail {
class propagate_context : public std::exception {
 public:
  // cppcheck-suppress passedByValue
  explicit propagate_context(std::string message)
      : message_(std::move(message)) {}

  const char* what() const noexcept override { return message_.c_str(); }
  const std::string& message() const { return message_; }

 private:
  std::string message_;
};
}  // namespace Options_detail

/// \ingroup OptionParsingGroup
/// The type that options are passed around as.  Contains YAML node
/// data and an Context.
///
/// \note To use any methods on this class in a concrete function you
/// must include ParseOptions.hpp, but you do *not* need to include
/// that header to use this in an uninstantiated
/// `create_from_yaml::create` function.
class Option {
 public:
  const Context& context() const;

  /// Append a line to the contained context.
  void append_context(const std::string& context);

  /// Convert to an object of type `T`.
  template <typename T, typename Metavariables = NoSuchType>
  T parse_as() const;

  /// \note This constructor overwrites the mark data in the supplied
  /// context with the one from the node.
  ///
  /// \warning This method is for internal use of the option parser.
  explicit Option(YAML::Node node, Context context = {});

  /// \warning This method is for internal use of the option parser.
  explicit Option(Context context);

  /// \warning This method is for internal use of the option parser.
  const YAML::Node& node() const;

  /// Sets the node and updates the context's mark to correspond to it.
  ///
  /// \warning This method is for internal use of the option parser.
  void set_node(YAML::Node node);

 private:
  std::unique_ptr<YAML::Node> node_;
  Context context_;
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

namespace Options_detail {
template <typename T, typename = std::void_t<>>
struct name_helper {
  static std::string name() { return pretty_type::short_name<T>(); }
};

template <typename T>
struct name_helper<T, std::void_t<decltype(T::name())>> {
  static std::string name() { return T::name(); }
};
}  // namespace Options_detail

// The name in the YAML file for a struct.
template <typename T>
std::string name() {
  return Options_detail::name_helper<T>::name();
}

/// Provide multiple ways to construct a class.
///
/// This type may be included in an option list along with option
/// tags.  When creating the class, the parser will choose one of the
/// lists of options to use, depending on the user input.
///
/// The class must be prepared to accept any of the possible
/// alternatives as arguments to its constructor.  To disambiguate
/// multiple alternatives with the same types, a constructor may take
/// the full list of option tags expected as its first argument.
///
/// \snippet Test_Options.cpp alternatives
template <typename... AlternativeLists>
struct Alternatives {
  static_assert(sizeof...(AlternativeLists) >= 2,
                "Option alternatives must provide at least two alternatives.");
  static_assert(
      tmpl::all<tmpl::list<tt::is_a<tmpl::list, AlternativeLists>...>>::value,
      "Option alternatives must be given as tmpl::lists.");
  static_assert(
      tmpl::none<
          tmpl::list<std::is_same<tmpl::list<>, AlternativeLists>...>>::value,
      "All option alternatives must have at least one option.");
};
}  // namespace Options
