// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines classes and functions that handle parsing of input parameters.

#pragma once

#include <cstring>
#include <exception>
#include <iterator>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "Options/OptionsDetails.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TypeTraits.hpp"

/// \ingroup OptionParsing
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

/// \ingroup OptionParsing
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
      throw Options_details::propagate_context(                         \
          avoid_name_collisions_PARSE_ERROR.str());                     \
    }                                                                   \
  } while (false)

/// \ingroup OptionParsing
/// The type that options are passed around as.  Contains YAML node
/// data and an OptionContext.
class Option_t {
 public:
  /// NOTE: This constructor overwrites the mark data in the supplied
  /// context with the one from the node.
  explicit Option_t(YAML::Node node,
                    OptionContext context = OptionContext()) noexcept
      // clang-tidy: YAML::Node not movable (as of yaml-cpp-0.5.3)
      : node_(std::move(node)), context_(std::move(context)) {  // NOLINT
    context_.line = node.Mark().line;
    context_.column = node.Mark().column;
  }

  explicit Option_t(OptionContext context) noexcept
      : context_(std::move(context)) {}

  const YAML::Node& node() const noexcept { return node_; }
  const OptionContext& context() const noexcept { return context_; }

  /// Append a line to the contained context.
  void append_context(const std::string& context) noexcept {
    context_.append(context);
  }

  /// Sets the node and updates the context's mark to correspond to it.
  void set_node(YAML::Node node) noexcept {
    // clang-tidy: YAML::Node not movable (as of yaml-cpp-0.5.3)
    node_ = std::move(node);  // NOLINT
    context_.line = node_.Mark().line;
    context_.column = node_.Mark().column;
  }

  /// Convert to an object of type `T`.
  template <typename T>
  T parse_as() const;

 private:
  YAML::Node node_;
  OptionContext context_;
};

template <typename T>
T Option_t::parse_as() const {
  try {
    return node().template as<T>();
  } catch (const YAML::BadConversion& e) {
    // This happens when trying to parse an empty value as a container
    // with no entries.
    if ((tt::is_a_v<std::vector, T> or
         tt::is_std_array_of_size_v<0, T> or
         tt::is_maplike_v<T>) and
        node().IsNull()) {
      return T{};
    }
    OptionContext error_context = context();
    error_context.line = e.mark.line;
    error_context.column = e.mark.column;
    std::ostringstream ss;
    ss << "Failed to convert value to type " << pretty_type::get_name<T>()
       << ":";

    const std::string value_text = YAML::Dump(node());
    if (value_text.find('\n') == std::string::npos) {
      ss << " " << value_text;
    } else {
      // Indent each line of the value by two spaces and start on a new line
      ss << "\n  ";
      for (char c : value_text) {
        ss << c;
        if (c == '\n') {
          ss << "  ";
        }
      }
    }

    if (tt::is_a_v<std::vector, T> or tt::is_std_array_v<T>) {
      ss << "\n\nNote: For sequences this can happen because the length of the "
          "sequence specified\nin the input file is not equal to the length "
          "expected by the code. Sequences in\nfiles can be denoted either as "
          "a bracket enclosed list ([foo, bar]) or with each\nentry on a "
          "separate line, indented and preceeded by a dash (  - foo).";
    }
    PARSE_ERROR(error_context, ss.str());
  } catch (const Options_details::propagate_context& e) {
    OptionContext error_context = context();
    // Avoid line numbers in the middle of the trace
    error_context.line = -1;
    error_context.column = -1;
    PARSE_ERROR(error_context, e.message());
  } catch (std::exception& e) {
    ERROR("Unexpected exception: " << e.what());
  }
}

/// \ingroup OptionParsing
/// \brief Class that handles parsing an input file
///
/// \details Options is templated on `OptionList`, which must be a
/// typelist of structs.  Each of these structs corresponds to a
/// parameter to be read from an input file.  These structs must
/// define the type to be parsed and a help string, and may also
/// define defaults and limits:
/// \snippet Test_Options.cpp options_example_scalar_struct
/// \snippet Test_Options.cpp options_example_vector_struct
///
/// Options must be given YAML data to parse before output can be
/// extracted.  This can be done either from a file (parse_file
/// method), from a string (parse method), or, in the case of
/// recursive parsing, from an Option_t (parse method).  The options
/// can then be extracted using the get method.
///
/// \example
/// Using the above structs:
/// \snippet Test_Options.cpp options_example_scalar_parse
template <typename OptionList>
class Options {
 public:
  /// \param help_text an overall description of the options
  explicit Options(std::string help_text) noexcept;

  /// Parse a string to obtain options and their values.
  ///
  /// \param options the string holding the YAML formatted options
  void parse(const std::string& options) noexcept;

  /// Parse an Option_t to obtain options and their values.
  void parse(const Option_t& options);

  /// Parse a file containing options
  ///
  /// \param file_name the path to the file to parse
  void parse_file(const std::string& file_name) noexcept;

  /// Get the value of the specified option
  ///
  /// \tparam T the option to retrieve
  /// \return the value of the option
  template <typename T>
  typename T::type get() const;

  /// Call a function with the specified options as arguments.
  ///
  /// \tparam TagList a typelist of options to pass
  /// \return the result of the function call
  template <typename TagList, typename F>
  decltype(auto) apply(F&& func) const;

  /// Get the help string
  std::string help() const noexcept;

 private:
  static_assert(tt::is_a<tmpl::list, OptionList>::value,
                "The OptionList template parameter to Options must be a "
                "tmpl::list<...>.");
  using opts_t = OptionList;
  static constexpr int max_label_size_ = 20;
  static constexpr size_t max_help_size_ = 56;

  void parse(const YAML::Node& node);

  //@{
  /// Check that the size is not smaller than the lower bound
  ///
  /// \tparam T the option struct
  /// \param t the value of the read in option
  template <typename T,
            Requires<Options_details::has_lower_bound_on_size<T>::value> =
                nullptr>
  void check_lower_bound_on_size(const typename T::type& t,
                                 const OptionContext& context) const;
  template <typename T,
            Requires<not Options_details::has_lower_bound_on_size<T>::value> =
                nullptr>
  constexpr void check_lower_bound_on_size(
      const typename T::type& /*t*/,
      const OptionContext& /*context*/) const noexcept {}
  //@}

  //@{
  /// Check that the size is not larger than the upper bound
  ///
  /// \tparam T the option struct
  /// \param t the value of the read in option
  template <typename T,
            Requires<Options_details::has_upper_bound_on_size<T>::value> =
                nullptr>
  void check_upper_bound_on_size(const typename T::type& t,
                                 const OptionContext& context) const;
  template <typename T,
            Requires<not Options_details::has_upper_bound_on_size<T>::value> =
                nullptr>
  constexpr void check_upper_bound_on_size(
      const typename T::type& /*t*/,
      const OptionContext& /*context*/) const noexcept {}
  //@}

  //@{
  /// Returns the default value or errors if there is no default.
  ///
  /// \tparam T the option struct
  template <typename T,
            Requires<Options_details::has_default<T>::value> = nullptr>
  typename T::type get_default() const noexcept {
    static_assert(
        cpp17::is_same_v<decltype(T::default_value()), typename T::type>,
        "Default value is not of the same type as the option.");
    return T::default_value();
  }
  template <typename T,
            Requires<not Options_details::has_default<T>::value> = nullptr>
  [[noreturn]] typename T::type get_default() const {
    PARSE_ERROR(context_,
                "You did not specify the option '"
                << pretty_type::short_name<T>() << "'.\n"
                << help());
  }
  //@}

  //@{
  /// If the options has a lower bound, check it is satisfied.
  ///
  /// Note: Lower bounds are >=, not just >.
  /// \tparam T the option struct
  /// \param t the value of the read in option
  template <typename T,
            Requires<Options_details::has_lower_bound<T>::value> = nullptr>
  void check_lower_bound(const typename T::type& t,
                         const OptionContext& context) const;
  template <typename T,
            Requires<not Options_details::has_lower_bound<T>::value> = nullptr>
  constexpr void check_lower_bound(
      const typename T::type& /*t*/,
      const OptionContext& /*context*/) const noexcept {}
  //@}

  //@{
  /// If the options has a upper bound, check it is satisfied.
  ///
  /// Note: Upper bounds are <=, not just <.
  /// \tparam T the option struct
  /// \param t the value of the read in option
  template <
      typename T,
      Requires<Options_details::has_upper_bound<T>::value> = nullptr>
  void check_upper_bound(const typename T::type& t,
                         const OptionContext& context) const;
  template <typename T,
            Requires<not Options_details::has_upper_bound<T>::value> = nullptr>
  constexpr void check_upper_bound(
      const typename T::type& /*t*/,
      const OptionContext& /*context*/) const noexcept {}
  //@}

  //@{
  /// Check that the default (if any) satisfies the bounds
  ///
  /// \tparam T the option struct
  template <typename T,
            Requires<Options_details::has_default<T>::value> = nullptr>
  void validate_default() const {
    OptionContext context;
    context.append("Checking DEFAULT value for " +
                   pretty_type::short_name<T>());
    const auto default_value = T::default_value();
    check_lower_bound_on_size<T>(default_value, context);
    check_upper_bound_on_size<T>(default_value, context);
    check_lower_bound<T>(default_value, context);
    check_upper_bound<T>(default_value, context);
  }
  template <typename T,
            Requires<not Options_details::has_default<T>::value> = nullptr>
  constexpr void validate_default() const noexcept {}
  //@}

  /// Get the help string for parsing errors
  std::string parsing_help(const YAML::Node& options) const noexcept;

  /// Error message when failed to parse an input file.
  std::string parser_error(const YAML::ParserException& e) const noexcept;

  std::string help_text_{};
  OptionContext context_{};
  std::unordered_set<std::string> valid_names_{};
  std::unordered_map<std::string, YAML::Node> parsed_options_{};
};

template <typename OptionList>
Options<OptionList>::Options(std::string help_text) noexcept
    : help_text_(std::move(help_text)),
      valid_names_(
          tmpl::for_each<opts_t>(Options_details::create_valid_names{}).value) {
  tmpl::for_each<opts_t>([](auto t) noexcept {
    using T = typename decltype(t)::type;
    const std::string label = pretty_type::short_name<T>();
    // These could be checked at compile time, but we couldn't print
    // a useful error message.
    ASSERT(label.size() < max_label_size_,
           "The option name " << label << " is too long for nice formatting, "
           "please shorten the name to be under " << max_label_size_
           << " characters");
    ASSERT(std::strlen(T::help) > 0,
           "You must supply a help string of non-zero length for " << label);
    ASSERT(std::strlen(T::help) < max_help_size_,
           "Option help strings should be short and to the point.  "
           "The help string for " << label << " should be less than "
           << max_help_size_ << " characters long.");
  });
}

template <typename OptionList>
void Options<OptionList>::parse_file(const std::string& file_name) noexcept {
  context_.append("In " + file_name);
  try {
    parse(YAML::LoadFile(file_name));
  } catch (const YAML::ParserException& e) {
    ERROR(parser_error(e));
  } catch (YAML::BadFile& /*e*/) {
    ERROR("Could not open the input file " << file_name);
  }
}

template <typename OptionList>
void Options<OptionList>::parse(const std::string& options) noexcept {
  context_.append("In string");
  try {
    parse(YAML::Load(options));
  } catch (YAML::ParserException& e) {
    ERROR(parser_error(e));
  }
}

template <typename OptionList>
void Options<OptionList>::parse(const Option_t& options) {
  context_ = options.context();
  parse(options.node());
}

template <typename OptionList>
void Options<OptionList>::parse(const YAML::Node& node) {
  if (not(node.IsMap() or node.IsNull())) {
    PARSE_ERROR(context_,
                "'" << node << "' does not look like options.\n"
                << help());
  }
  for (const auto& name_and_value : node) {
    const auto& name = name_and_value.first.as<std::string>();
    const auto& value = name_and_value.second;
    auto context = context_;
    context.line = name_and_value.first.Mark().line;
    context.column = name_and_value.first.Mark().column;

    // Check for invalid key
    if (1 != valid_names_.count(name)) {
      PARSE_ERROR(context,
                  "Option '" << name << "' is not a valid option.\n"
                  << parsing_help(node));
    }

    // Check for duplicate key
    if (0 != parsed_options_.count(name)) {
      PARSE_ERROR(context,
                  "Option '" << name << "' specified twice.\n"
                  << parsing_help(node));
    }
    parsed_options_.emplace(name, value);
  }
}

template <typename OptionList>
template <typename T>
typename T::type Options<OptionList>::get() const {
  static_assert(
      not cpp17::is_same_v<tmpl::index_of<opts_t, T>, tmpl::no_such_type_>,
      "Could not find requested option in the list of options provided. Did "
      "you forget to add the option struct to the OptionList?");
  const std::string label = pretty_type::short_name<T>();

  validate_default<T>();
  if (0 == parsed_options_.count(label)) {
    return get_default<T>();
  }

  Option_t option(parsed_options_.find(label)->second, context_);
  option.append_context("While parsing option " + label);

  auto t = option.parse_as<typename T::type>();

  check_lower_bound_on_size<T>(t, option.context());
  check_upper_bound_on_size<T>(t, option.context());
  check_lower_bound<T>(t, option.context());
  check_upper_bound<T>(t, option.context());
  return t;
}

namespace Options_details {
template <typename>
struct apply_helper;

template <typename... Tags>
struct apply_helper<tmpl::list<Tags...>> {
  template <typename Options, typename F>
  static decltype(auto) apply(const Options& opts, F&& func) {
    return func(opts.template get<Tags>()...);
  }
};
}  // namespace Options_details

// \cond
// Doxygen is confused by decltype(auto)
template <typename OptionList>
template <typename TagList, typename F>
decltype(auto) Options<OptionList>::apply(F&& func) const {
  return Options_details::apply_helper<TagList>::apply(*this,
                                                       std::forward<F>(func));
}
// \endcond

template <typename OptionList>
std::string Options<OptionList>::help() const noexcept {
  std::ostringstream ss;
  ss << "\n==== Description of expected options:\n" << help_text_;
  if (tmpl::size<opts_t>::value > 0) {
    ss << "\n\nOptions:\n"
       << tmpl::for_each<opts_t>(Options_details::print{max_label_size_}).value;
  } else {
    ss << "\n\n<No options>\n";
  }
  return ss.str();
}

template <typename OptionList>
template <typename T,
          Requires<Options_details::has_lower_bound_on_size<T>::value>>
void Options<OptionList>::check_lower_bound_on_size(
    const typename T::type& t, const OptionContext& context) const {
  static_assert(cpp17::is_same_v<decltype(T::lower_bound_on_size()), size_t>,
                "lower_bound_on_size() is not a size_t.");
  if (t.size() < T::lower_bound_on_size()) {
    PARSE_ERROR(context,
                "Value must have at least " << T::lower_bound_on_size()
                << " entries, but " << t.size() << " were given.\n"
                << help());
  }
}

template <typename OptionList>
template <typename T,
          Requires<Options_details::has_upper_bound_on_size<T>::value>>
void Options<OptionList>::check_upper_bound_on_size(
    const typename T::type& t, const OptionContext& context) const {
  static_assert(cpp17::is_same_v<decltype(T::upper_bound_on_size()), size_t>,
                "upper_bound_on_size() is not a size_t.");
  if (t.size() > T::upper_bound_on_size()) {
    PARSE_ERROR(context,
                "Value must have at most " << T::upper_bound_on_size()
                << " entries, but " << t.size() << " were given.\n"
                << help());
  }
}

template <typename OptionList>
template <typename T, Requires<Options_details::has_lower_bound<T>::value>>
inline void Options<OptionList>::check_lower_bound(
    const typename T::type& t, const OptionContext& context) const {
  static_assert(cpp17::is_same_v<decltype(T::lower_bound()), typename T::type>,
                "Lower bound is not of the same type as the option.");
  static_assert(not cpp17::is_same_v<typename T::type, bool>,
                "Cannot set a lower bound for a bool.");
  if (t < T::lower_bound()) {
    PARSE_ERROR(context,
                "Value " << t << " is below the lower bound of "
                << T::lower_bound() << ".\n"
                << help());
  }
}

template <typename OptionList>
template <typename T, Requires<Options_details::has_upper_bound<T>::value>>
inline void Options<OptionList>::check_upper_bound(
    const typename T::type& t, const OptionContext& context) const {
  static_assert(cpp17::is_same_v<decltype(T::upper_bound()), typename T::type>,
                "Upper bound is not of the same type as the option.");
  static_assert(not cpp17::is_same_v<typename T::type, bool>,
                "Cannot set an upper bound for a bool.");
  if (t > T::upper_bound()) {
    PARSE_ERROR(context,
                "Value " << t << " is above the upper bound of "
                << T::upper_bound() << ".\n"
                << help());
  }
}

template <typename OptionList>
std::string Options<OptionList>::parsing_help(
    const YAML::Node& options) const noexcept {
  std::ostringstream os;
  // At top level this would dump the entire input file, which is very
  // verbose and not very informative.  At lower levels the result
  // should be much shorter and may actually give useful context for
  // what part of the file is being parsed.
  if (not context_.top_level) {
    os << "\n==== Parsing the option string:\n" << options << "\n";
  }
  os << help();
  return os.str();
}

template <typename OptionList>
std::string Options<OptionList>::parser_error(
    const YAML::ParserException& e) const noexcept {
  std::ostringstream ss;
  if (not e.mark.is_null()) {
    ss << "At line " << e.mark.line + 1 << " column " << e.mark.column + 1
       << ":\n";
  }
  ss << "Unable to correctly parse the input file because of a syntax error.\n"
      "This is typically due to placing a suboption on the same line as an "
      "option, e.g.:\nDomainCreator: CreateInterval:\n  IsPeriodicIn: "
      "[false]\n\nShould be:\nDomainCreator:\n  CreateInterval:\n    "
      "IsPeriodicIn: [true]\n\nSee an example input file for help.";
  return ss.str();
}

// yaml-cpp doesn't handle C++11 types yet
namespace YAML {
template <typename K, typename V, typename H, typename P>
struct convert<std::unordered_map<K, V, H, P>> {
  static bool decode(const Node& node, std::unordered_map<K, V, H, P>& rhs) {
    std::map<K, V> ordered;
    if (not convert<std::map<K, V>>::decode(node, ordered)) {
      return false;
    }
    rhs.clear();
    rhs.insert(std::make_move_iterator(ordered.begin()),
               std::make_move_iterator(ordered.end()));
    return true;
  }
};
}  // namespace YAML

#include "Options/Factory.hpp"
