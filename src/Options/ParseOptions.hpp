// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines classes and functions that handle parsing of input parameters.

#pragma once

#include <cstring>
#include <exception>
#include <ios>
#include <iterator>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "Options/Options.hpp"
#include "Options/OptionsDetails.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TypeTraits.hpp"
#include "Utilities/TypeTraits/IsA.hpp"
#include "Utilities/TypeTraits/IsMaplike.hpp"
#include "Utilities/TypeTraits/IsStdArray.hpp"
#include "Utilities/TypeTraits/IsStdArrayOfSize.hpp"

namespace Options {
// Defining methods as inline in a different header from the class
// definition is somewhat strange.  It is done here to minimize the
// amount of code in the frequently-included Options.hpp file.  The
// only external consumers of Option should be create_from_yaml
// specializations, and they should only be instantiated by code in
// this file.  (Or explicitly instantiated in cpp files, which can
// include this file.)

// clang-tidy: YAML::Node not movable (as of yaml-cpp-0.5.3)
// NOLINTNEXTLINE(performance-unnecessary-value-param)
inline Option::Option(YAML::Node node, Context context) noexcept
    : node_(std::make_unique<YAML::Node>(std::move(node))),
      context_(std::move(context)) {  // NOLINT
  context_.line = node.Mark().line;
  context_.column = node.Mark().column;
}

inline Option::Option(Context context) noexcept
    : node_(std::make_unique<YAML::Node>()), context_(std::move(context)) {}

inline const YAML::Node& Option::node() const noexcept { return *node_; }
inline const Context& Option::context() const noexcept { return context_; }

/// Append a line to the contained context.
inline void Option::append_context(const std::string& context) noexcept {
  context_.append(context);
}

// NOLINTNEXTLINE(performance-unnecessary-value-param)
inline void Option::set_node(YAML::Node node) noexcept {
  // clang-tidy: YAML::Node not movable (as of yaml-cpp-0.5.3)
  *node_ = std::move(node);  // NOLINT
  context_.line = node_->Mark().line;
  context_.column = node_->Mark().column;
}

template <typename T, typename Metavariables>
T Option::parse_as() const {
  try {
    // yaml-cpp's `as` method won't parse empty nodes, so we need to
    // inline a bit of its logic.
    Options_detail::wrap_create_types<T, Metavariables> result{};
    if (YAML::convert<decltype(result)>::decode(node(), result)) {
      return Options_detail::unwrap_create_types(std::move(result));
    }
    // clang-tidy: thrown exception is not nothrow copy constructible
    throw YAML::BadConversion(node().Mark());  // NOLINT
  } catch (const YAML::BadConversion& e) {
    // This happens when trying to parse an empty value as a container
    // with no entries.
    if ((tt::is_a_v<std::vector, T> or tt::is_std_array_of_size_v<0, T> or
         tt::is_maplike_v<T>) and node().IsNull()) {
      return T{};
    }
    Context error_context = context();
    error_context.line = e.mark.line;
    error_context.column = e.mark.column;
    std::ostringstream ss;
    ss << "Failed to convert value to type "
       << Options_detail::yaml_type<T>::value() << ":";

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
            "expected by the code. Sequences in\nfiles can be denoted either "
            "as a bracket enclosed list ([foo, bar]) or with each\nentry on a "
            "separate line, indented and preceeded by a dash (  - foo).";
    }
    PARSE_ERROR(error_context, ss.str());
  } catch (const Options_detail::propagate_context& e) {
    Context error_context = context();
    // Avoid line numbers in the middle of the trace
    error_context.line = -1;
    error_context.column = -1;
    PARSE_ERROR(error_context, e.message());
  } catch (std::exception& e) {
    ERROR("Unexpected exception: " << e.what());
  }
}

namespace Options_detail {
template <typename T, typename Metavariables, typename Subgroup>
struct get_impl;
}  // namespace Options_detail

/// \ingroup OptionParsingGroup
/// \brief Class that handles parsing an input file
///
/// Options must be given YAML data to parse before output can be
/// extracted.  This can be done either from a file (parse_file
/// method), from a string (parse method), or, in the case of
/// recursive parsing, from an Option (parse method).  The options
/// can then be extracted using the get method.
///
/// \example
/// \snippet Test_Options.cpp options_example_scalar_struct
/// \snippet Test_Options.cpp options_example_scalar_parse
///
/// \see the \ref dev_guide_option_parsing tutorial
///
/// \tparam OptionList the list of option structs to parse
/// \tparam Group the option group with a group hierarchy
template <typename OptionList, typename Group = NoSuchType>
class Parser {
 public:
  /// \param help_text an overall description of the options
  explicit Parser(std::string help_text) noexcept;

  /// Parse a string to obtain options and their values.
  ///
  /// \param options the string holding the YAML formatted options
  void parse(const std::string& options) noexcept;

  /// Parse an Option to obtain options and their values.
  void parse(const Option& options);

  /// Parse a file containing options
  ///
  /// \param file_name the path to the file to parse
  void parse_file(const std::string& file_name) noexcept;

  /// Parse a YAML node containing options
  void parse(const YAML::Node& node);

  /// Get the value of the specified option
  ///
  /// \tparam T the option to retrieve
  /// \return the value of the option
  template <typename T, typename Metavariables = NoSuchType>
  typename T::type get() const;

  /// Call a function with the specified options as arguments.
  ///
  /// \tparam TagList a typelist of options to pass
  /// \return the result of the function call
  template <typename TagList, typename Metavariables = NoSuchType, typename F>
  decltype(auto) apply(F&& func) const;

  /// Get the help string
  std::string help() const noexcept;

 private:
  template <typename, typename, typename>
  friend struct Options_detail::get_impl;

  static_assert(tt::is_a<tmpl::list, OptionList>::value,
                "The OptionList template parameter to Options must be a "
                "tmpl::list<...>.");

  /// All top-level options and top-level groups of options. Every option in
  /// `OptionList` is either in this list or in the hierarchy of one of the
  /// groups in this list.
  using tags_and_subgroups_list = tmpl::remove_duplicates<tmpl::transform<
      OptionList, Options_detail::find_subgroup<tmpl::_1, Group>>>;

  // The maximum length of an option label.
  static constexpr int max_label_size_ = 70;

  /// Check that the size is not smaller than the lower bound
  ///
  /// \tparam T the option struct
  /// \param t the value of the read in option
  template <typename T>
  void check_lower_bound_on_size(const typename T::type& t,
                                 const Context& context) const;

  /// Check that the size is not larger than the upper bound
  ///
  /// \tparam T the option struct
  /// \param t the value of the read in option
  template <typename T>
  void check_upper_bound_on_size(const typename T::type& t,
                                 const Context& context) const;

  /// If the options has a lower bound, check it is satisfied.
  ///
  /// Note: Lower bounds are >=, not just >.
  /// \tparam T the option struct
  /// \param t the value of the read in option
  template <typename T>
  void check_lower_bound(const typename T::type& t,
                         const Context& context) const;

  /// If the options has a upper bound, check it is satisfied.
  ///
  /// Note: Upper bounds are <=, not just <.
  /// \tparam T the option struct
  /// \param t the value of the read in option
  template <typename T>
  void check_upper_bound(const typename T::type& t,
                         const Context& context) const;

  /// Get the help string for parsing errors
  std::string parsing_help(const YAML::Node& options) const noexcept;

  /// Error message when failed to parse an input file.
  [[noreturn]] void parser_error(const YAML::Exception& e) const noexcept;

  std::string help_text_{};
  Context context_{};
  std::unordered_map<std::string, YAML::Node> parsed_options_{};
};

template <typename OptionList, typename Group>
Parser<OptionList, Group>::Parser(std::string help_text) noexcept
    : help_text_(std::move(help_text)) {
  tmpl::for_each<tags_and_subgroups_list>([](auto t) noexcept {
    using T = typename decltype(t)::type;
    const std::string label = name<T>();
    ASSERT(label.size() <= max_label_size_,
           "The option name " << label
                              << " is too long for nice formatting, "
                                 "please shorten the name to "
                              << max_label_size_ << " characters or fewer");
    ASSERT(std::strlen(T::help) > 0,
           "You must supply a help string of non-zero length for " << label);
  });
}

template <typename OptionList, typename Group>
void Parser<OptionList, Group>::parse_file(
    const std::string& file_name) noexcept {
  context_.append("In " + file_name);
  try {
    parse(YAML::LoadFile(file_name));
  } catch (const YAML::BadFile& /*e*/) {
    ERROR("Could not open the input file " << file_name);
  } catch (const YAML::Exception& e) {
    parser_error(e);
  } catch (const std::ios_base::failure& e) {
    ERROR("I/O error reading " << file_name << ": " << e.what());
  }
}

template <typename OptionList, typename Group>
void Parser<OptionList, Group>::parse(const std::string& options) noexcept {
  context_.append("In string");
  try {
    parse(YAML::Load(options));
  } catch (YAML::Exception& e) {
    parser_error(e);
  }
}

template <typename OptionList, typename Group>
void Parser<OptionList, Group>::parse(const Option& options) {
  context_ = options.context();
  parse(options.node());
}

template <typename OptionList, typename Group>
void Parser<OptionList, Group>::parse(const YAML::Node& node) {
  if (not(node.IsMap() or node.IsNull())) {
    PARSE_ERROR(context_, "'" << node << "' does not look like options.\n"
                              << help());
  }

  // Use an ordered container so the missing options are reported in
  // the order they are given in the help string.
  std::vector<std::string> valid_names;
  valid_names.reserve(tmpl::size<tags_and_subgroups_list>{});
  tmpl::for_each<tags_and_subgroups_list>([&valid_names](auto opt) noexcept {
    using Opt = tmpl::type_from<decltype(opt)>;
    const std::string label = name<Opt>();
    ASSERT(alg::find(valid_names, label) == valid_names.end(),
           "Duplicate option name: " << label);
    valid_names.push_back(label);
  });

  for (const auto& name_and_value : node) {
    const auto& name = name_and_value.first.as<std::string>();
    const auto& value = name_and_value.second;
    auto context = context_;
    context.line = name_and_value.first.Mark().line;
    context.column = name_and_value.first.Mark().column;

    // Check for duplicate key
    if (0 != parsed_options_.count(name)) {
      PARSE_ERROR(context, "Option '" << name << "' specified twice.\n"
                                      << parsing_help(node));
    }

    // Check for invalid key
    const auto name_it = alg::find(valid_names, name);
    if (name_it == valid_names.end()) {
      PARSE_ERROR(context, "Option '" << name << "' is not a valid option.\n"
                                      << parsing_help(node));
    }

    parsed_options_.emplace(name, value);
    valid_names.erase(name_it);
  }

  if (not valid_names.empty()) {
    PARSE_ERROR(context_, "You did not specify the option"
                << (valid_names.size() == 1 ? " " : "s ")
                << (MakeString{} << valid_names) << "\n" << parsing_help(node));
  }

  // Any actual warnings will be printed by later calls to get or
  // apply, but it is not clear how to determine in those functions
  // whether this message should be printed.
  if (std::is_same_v<Group, NoSuchType> and context_.top_level) {
    Parallel::printf_error(
        "The following options differ from their suggested values:\n");
  }
}

namespace Options_detail {
template <typename Tag, typename Metavariables, typename Subgroup>
struct get_impl {
  template <typename OptionList, typename Group>
  static typename Tag::type apply(const Parser<OptionList, Group>& opts) {
    static_assert(
        tmpl::list_contains_v<OptionList, Tag>,
        "Could not find requested option in the list of options provided. Did "
        "you forget to add the option tag to the OptionList?");
    const std::string subgroup_label = name<Subgroup>();
    Parser<options_in_group<OptionList, Subgroup>, Subgroup> subgroup_options(
        Subgroup::help);
    subgroup_options.context_ = opts.context_;
    subgroup_options.context_.append("In group " + subgroup_label);
    subgroup_options.parse(opts.parsed_options_.find(subgroup_label)->second);
    return subgroup_options.template get<Tag, Metavariables>();
  }
};

template <typename Tag, typename Metavariables>
struct get_impl<Tag, Metavariables, Tag> {
  template <typename OptionList, typename Group>
  static typename Tag::type apply(const Parser<OptionList, Group>& opts) {
    static_assert(
        tmpl::list_contains_v<OptionList, Tag>,
        "Could not find requested option in the list of options provided. Did "
        "you forget to add the option tag to the OptionList?");
    const std::string label = name<Tag>();

    Option option(opts.parsed_options_.find(label)->second, opts.context_);
    option.append_context("While parsing option " + label);

    auto t = option.parse_as<typename Tag::type, Metavariables>();

    if constexpr (Options_detail::has_suggested<Tag>::value) {
      static_assert(
          std::is_same_v<decltype(Tag::suggested_value()), typename Tag::type>,
          "Suggested value is not of the same type as the option.");

      // This can be easily relaxed, but using it would require
      // writing comparison operators for abstract base classes.  If
      // someone wants this enough to go though the effort of doing
      // that, it would just require comparing the dereferenced
      // pointers below to decide whether the suggestion was followed.
      static_assert(not tt::is_a_v<std::unique_ptr, typename Tag::type>,
                    "Suggestions are not supported for pointer types.");

      const auto suggested_value = Tag::suggested_value();
      {
        Context context;
        context.append("Checking SUGGESTED value for " + name<Tag>());
        opts.template check_lower_bound_on_size<Tag>(suggested_value, context);
        opts.template check_upper_bound_on_size<Tag>(suggested_value, context);
        opts.template check_lower_bound<Tag>(suggested_value, context);
        opts.template check_upper_bound<Tag>(suggested_value, context);
      }

      if (t != suggested_value) {
        Parallel::printf_error(
            "%s, line %d:\n  Specified: %s\n  Suggested: %s\n",
            label, option.context().line + 1,
            (MakeString{} << std::boolalpha << t),
            (MakeString{} << std::boolalpha << suggested_value));
      }
    }

    opts.template check_lower_bound_on_size<Tag>(t, option.context());
    opts.template check_upper_bound_on_size<Tag>(t, option.context());
    opts.template check_lower_bound<Tag>(t, option.context());
    opts.template check_upper_bound<Tag>(t, option.context());
    return t;
  }
};
}  // namespace Options_detail

template <typename OptionList, typename Group>
template <typename Tag, typename Metavariables>
typename Tag::type Parser<OptionList, Group>::get() const {
  return Options_detail::get_impl<
      Tag, Metavariables,
      typename Options_detail::find_subgroup<Tag, Group>::type>::apply(*this);
}

namespace Options_detail {
template <typename>
struct apply_helper;

template <typename... Tags>
struct apply_helper<tmpl::list<Tags...>> {
  template <typename Metavariables, typename Options, typename F>
  static decltype(auto) apply(const Options& opts, F&& func) {
    return func(opts.template get<Tags, Metavariables>()...);
  }
};
}  // namespace Options_detail

/// \cond
// Doxygen is confused by decltype(auto)
template <typename OptionList, typename Group>
template <typename TagList, typename Metavariables, typename F>
decltype(auto) Parser<OptionList, Group>::apply(F&& func) const {
  return Options_detail::apply_helper<TagList>::template apply<Metavariables>(
      *this, std::forward<F>(func));
}
/// \endcond

template <typename OptionList, typename Group>
std::string Parser<OptionList, Group>::help() const noexcept {
  std::ostringstream ss;
  ss << "\n==== Description of expected options:\n" << help_text_;
  if (tmpl::size<tags_and_subgroups_list>::value > 0) {
    ss << "\n\nOptions:\n"
       << tmpl::for_each<tags_and_subgroups_list>(
              Options_detail::print<OptionList>{})
              .value;
  } else {
    ss << "\n\n<No options>\n";
  }
  return ss.str();
}

template <typename OptionList, typename Group>
template <typename T>
void Parser<OptionList, Group>::check_lower_bound_on_size(
    const typename T::type& t, const Context& context) const {
  if constexpr (Options_detail::has_lower_bound_on_size<T>::value) {
    static_assert(std::is_same_v<decltype(T::lower_bound_on_size()), size_t>,
                  "lower_bound_on_size() is not a size_t.");
    if (t.size() < T::lower_bound_on_size()) {
      PARSE_ERROR(context, "Value must have at least "
                               << T::lower_bound_on_size() << " entries, but "
                               << t.size() << " were given.\n"
                               << help());
    }
  }
}

template <typename OptionList, typename Group>
template <typename T>
void Parser<OptionList, Group>::check_upper_bound_on_size(
    const typename T::type& t, const Context& context) const {
  if constexpr (Options_detail::has_upper_bound_on_size<T>::value) {
    static_assert(std::is_same_v<decltype(T::upper_bound_on_size()), size_t>,
                  "upper_bound_on_size() is not a size_t.");
    if (t.size() > T::upper_bound_on_size()) {
      PARSE_ERROR(context, "Value must have at most "
                               << T::upper_bound_on_size() << " entries, but "
                               << t.size() << " were given.\n"
                               << help());
    }
  }
}

template <typename OptionList, typename Group>
template <typename T>
inline void Parser<OptionList, Group>::check_lower_bound(
    const typename T::type& t, const Context& context) const {
  if constexpr (Options_detail::has_lower_bound<T>::value) {
    static_assert(std::is_same_v<decltype(T::lower_bound()), typename T::type>,
                  "Lower bound is not of the same type as the option.");
    static_assert(not std::is_same_v<typename T::type, bool>,
                  "Cannot set a lower bound for a bool.");
    if (t < T::lower_bound()) {
      PARSE_ERROR(context, "Value " << (MakeString{} << t)
                                    << " is below the lower bound of "
                                    << (MakeString{} << T::lower_bound())
                                    << ".\n" << help());
    }
  }
}

template <typename OptionList, typename Group>
template <typename T>
inline void Parser<OptionList, Group>::check_upper_bound(
    const typename T::type& t, const Context& context) const {
  if constexpr (Options_detail::has_upper_bound<T>::value) {
    static_assert(std::is_same_v<decltype(T::upper_bound()), typename T::type>,
                  "Upper bound is not of the same type as the option.");
    static_assert(not std::is_same_v<typename T::type, bool>,
                  "Cannot set an upper bound for a bool.");
    if (t > T::upper_bound()) {
      PARSE_ERROR(context, "Value " << (MakeString{} << t)
                                    << " is above the upper bound of "
                                    << (MakeString{} << T::upper_bound())
                                    << ".\n" << help());
    }
  }
}

template <typename OptionList, typename Group>
std::string Parser<OptionList, Group>::parsing_help(
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

template <typename OptionList, typename Group>
[[noreturn]] void Parser<OptionList, Group>::parser_error(
    const YAML::Exception& e) const noexcept {
  auto context = context_;
  context.line = e.mark.line;
  context.column = e.mark.column;
  // Inline the top_level branch of PARSE_ERROR to avoid warning that
  // the other branch would call terminate.  (Parser errors can only
  // be generated at top level.)
  ERROR(
      "\n"
      << context
      << "Unable to correctly parse the input file because of a syntax error.\n"
         "This is often due to placing a suboption on the same line as an "
         "option, e.g.:\nDomainCreator: CreateInterval:\n  IsPeriodicIn: "
         "[false]\n\nShould be:\nDomainCreator:\n  CreateInterval:\n    "
         "IsPeriodicIn: [true]\n\nSee an example input file for help.");
}

namespace Options_detail {
template <typename T, typename Metavariables, typename = std::void_t<>>
struct get_options_list {
  using type = typename T::template options<Metavariables>;
};

template <typename T, typename Metavariables>
struct get_options_list<T, Metavariables, std::void_t<typename T::options>> {
  using type = typename T::options;
};
}  // namespace Options_detail

template <typename T>
template <typename Metavariables>
T create_from_yaml<T>::create(const Option& options) {
  Parser<typename Options_detail::get_options_list<T, Metavariables>::type>
      parser(T::help);
  parser.parse(options);
  return parser.template apply<typename Options_detail::get_options_list<
      T, Metavariables>::type>([&options](auto&&... args) {
    if constexpr (std::is_constructible<T, decltype(std::move(args))...,
                                        const Context&, Metavariables>{}) {
      return T(std::move(args)..., options.context(), Metavariables{});
    } else if constexpr (std::is_constructible<T, decltype(std::move(args))...,
                                               const Context&>{}) {
      return T(std::move(args)..., options.context());
    } else {
      return T(std::move(args)...);
    }
  });
}

// yaml-cpp doesn't handle C++11 types yet
template <typename K, typename V, typename H, typename P>
struct create_from_yaml<std::unordered_map<K, V, H, P>> {
  template <typename Metavariables>
  static std::unordered_map<K, V, H, P> create(const Option& options) {
    auto ordered = options.parse_as<std::map<K, V>, Metavariables>();
    std::unordered_map<K, V, H, P> result;
    for (auto it = ordered.begin(); it != ordered.end();) {
      auto node = ordered.extract(it++);
      result.emplace(std::move(node.key()), std::move(node.mapped()));
    }
    return result;
  }
};
}  // namespace Options

template <typename T, typename Metavariables>
struct YAML::convert<Options::Options_detail::CreateWrapper<T, Metavariables>> {
  static bool decode(
      const Node& node,
      Options::Options_detail::CreateWrapper<T, Metavariables>& rhs) {
    Options::Context context;
    context.top_level = false;
    context.append("While creating a " + pretty_type::short_name<T>());
    Options::Option options(node, std::move(context));
    rhs = Options::Options_detail::CreateWrapper<T, Metavariables>{
        Options::create_from_yaml<T>::template create<Metavariables>(options)};
    return true;
  }
};

#include "Options/Factory.hpp"
