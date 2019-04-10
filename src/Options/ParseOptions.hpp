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
#include "Options/Options.hpp"
#include "Options/OptionsDetails.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TypeTraits.hpp"

// Defining methods as inline in a different header from the class
// definition is somewhat strange.  It is done here to minimize the
// amount of code in the frequently-included Options.hpp file.  The
// only external consumers of Option should be create_from_yaml
// specializations, and they should only be instantiated by code in
// this file.  (Or explicitly instantiated in cpp files, which can
// include this file.)

inline Option::Option(YAML::Node node, OptionContext context) noexcept
    // clang-tidy: YAML::Node not movable (as of yaml-cpp-0.5.3)
    : node_(std::make_unique<YAML::Node>(std::move(node))),
      context_(std::move(context)) {  // NOLINT
  context_.line = node.Mark().line;
  context_.column = node.Mark().column;
}

inline Option::Option(OptionContext context) noexcept
    : node_(std::make_unique<YAML::Node>()), context_(std::move(context)) {}

inline const YAML::Node& Option::node() const noexcept { return *node_; }
inline const OptionContext& Option::context() const noexcept {
  return context_;
}

/// Append a line to the contained context.
inline void Option::append_context(const std::string& context) noexcept {
  context_.append(context);
}

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
    OptionContext error_context = context();
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
    OptionContext error_context = context();
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
class Options {
 public:
  /// \param help_text an overall description of the options
  explicit Options(std::string help_text) noexcept;

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

  // The maximum length of an option label. 21 characters fits
  // "DiscontinuousGalerkin".
  static constexpr int max_label_size_ = 21;
  // The maximum length of option help strings.
  static constexpr size_t max_help_size_ = 55;

  //@{
  /// Check that the size is not smaller than the lower bound
  ///
  /// \tparam T the option struct
  /// \param t the value of the read in option
  template <
      typename T,
      Requires<Options_detail::has_lower_bound_on_size<T>::value> = nullptr>
  void check_lower_bound_on_size(const typename T::type& t,
                                 const OptionContext& context) const;
  template <
      typename T,
      Requires<not Options_detail::has_lower_bound_on_size<T>::value> = nullptr>
  constexpr void check_lower_bound_on_size(
      const typename T::type& /*t*/, const OptionContext& /*context*/) const
      noexcept {}
  //@}

  //@{
  /// Check that the size is not larger than the upper bound
  ///
  /// \tparam T the option struct
  /// \param t the value of the read in option
  template <
      typename T,
      Requires<Options_detail::has_upper_bound_on_size<T>::value> = nullptr>
  void check_upper_bound_on_size(const typename T::type& t,
                                 const OptionContext& context) const;
  template <
      typename T,
      Requires<not Options_detail::has_upper_bound_on_size<T>::value> = nullptr>
  constexpr void check_upper_bound_on_size(
      const typename T::type& /*t*/, const OptionContext& /*context*/) const
      noexcept {}
  //@}

  //@{
  /// Returns the default value or errors if there is no default.
  ///
  /// \tparam T the option struct
  template <typename T,
            Requires<Options_detail::has_default<T>::value> = nullptr>
  typename T::type get_default() const noexcept {
    static_assert(
        cpp17::is_same_v<decltype(T::default_value()), typename T::type>,
        "Default value is not of the same type as the option.");
    return T::default_value();
  }
  template <typename T,
            Requires<not Options_detail::has_default<T>::value> = nullptr>
  [[noreturn]] typename T::type get_default() const {
    PARSE_ERROR(context_, "You did not specify the option '"
                              << Options_detail::name<T>() << "'.\n"
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
            Requires<Options_detail::has_lower_bound<T>::value> = nullptr>
  void check_lower_bound(const typename T::type& t,
                         const OptionContext& context) const;
  template <typename T,
            Requires<not Options_detail::has_lower_bound<T>::value> = nullptr>
  constexpr void check_lower_bound(const typename T::type& /*t*/,
                                   const OptionContext& /*context*/) const
      noexcept {}
  //@}

  //@{
  /// If the options has a upper bound, check it is satisfied.
  ///
  /// Note: Upper bounds are <=, not just <.
  /// \tparam T the option struct
  /// \param t the value of the read in option
  template <typename T,
            Requires<Options_detail::has_upper_bound<T>::value> = nullptr>
  void check_upper_bound(const typename T::type& t,
                         const OptionContext& context) const;
  template <typename T,
            Requires<not Options_detail::has_upper_bound<T>::value> = nullptr>
  constexpr void check_upper_bound(const typename T::type& /*t*/,
                                   const OptionContext& /*context*/) const
      noexcept {}
  //@}

  //@{
  /// Check that the default (if any) satisfies the bounds
  ///
  /// \tparam T the option struct
  template <typename T,
            Requires<Options_detail::has_default<T>::value> = nullptr>
  void validate_default() const {
    OptionContext context;
    context.append("Checking DEFAULT value for " + Options_detail::name<T>());
    const auto default_value = T::default_value();
    check_lower_bound_on_size<T>(default_value, context);
    check_upper_bound_on_size<T>(default_value, context);
    check_lower_bound<T>(default_value, context);
    check_upper_bound<T>(default_value, context);
  }
  template <typename T,
            Requires<not Options_detail::has_default<T>::value> = nullptr>
  constexpr void validate_default() const noexcept {}
  //@}

  /// Get the help string for parsing errors
  std::string parsing_help(const YAML::Node& options) const noexcept;

  /// Error message when failed to parse an input file.
  [[noreturn]] void parser_error(const YAML::Exception& e) const noexcept;

  std::string help_text_{};
  OptionContext context_{};
  std::unordered_set<std::string> valid_names_{};
  std::unordered_map<std::string, YAML::Node> parsed_options_{};
};

template <typename OptionList, typename Group>
Options<OptionList, Group>::Options(std::string help_text) noexcept
    : help_text_(std::move(help_text)),
      valid_names_(tmpl::for_each<tags_and_subgroups_list>(
                       Options_detail::create_valid_names{})
                       .value) {
  tmpl::for_each<tags_and_subgroups_list>([](auto t) noexcept {
    using T = typename decltype(t)::type;
    const std::string label = Options_detail::name<T>();
    ASSERT(label.size() <= max_label_size_,
           "The option name " << label
                              << " is too long for nice formatting, "
                                 "please shorten the name to "
                              << max_label_size_ << " characters or fewer");
    ASSERT(std::strlen(T::help) > 0,
           "You must supply a help string of non-zero length for " << label);
    ASSERT(std::strlen(T::help) <= max_help_size_,
           "Option help strings should be short and to the point.  "
           "The help string for "
               << label << " should have " << max_help_size_
               << " characters or fewer.");
  });
}

template <typename OptionList, typename Group>
void Options<OptionList, Group>::parse_file(
    const std::string& file_name) noexcept {
  context_.append("In " + file_name);
  try {
    parse(YAML::LoadFile(file_name));
  } catch (YAML::BadFile& /*e*/) {
    ERROR("Could not open the input file " << file_name);
  } catch (const YAML::Exception& e) {
    parser_error(e);
  }
}

template <typename OptionList, typename Group>
void Options<OptionList, Group>::parse(const std::string& options) noexcept {
  context_.append("In string");
  try {
    parse(YAML::Load(options));
  } catch (YAML::Exception& e) {
    parser_error(e);
  }
}

template <typename OptionList, typename Group>
void Options<OptionList, Group>::parse(const Option& options) {
  context_ = options.context();
  parse(options.node());
}

template <typename OptionList, typename Group>
void Options<OptionList, Group>::parse(const YAML::Node& node) {
  if (not(node.IsMap() or node.IsNull())) {
    PARSE_ERROR(context_, "'" << node << "' does not look like options.\n"
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
      PARSE_ERROR(context, "Option '" << name << "' is not a valid option.\n"
                                      << parsing_help(node));
    }

    // Check for duplicate key
    if (0 != parsed_options_.count(name)) {
      PARSE_ERROR(context, "Option '" << name << "' specified twice.\n"
                                      << parsing_help(node));
    }
    parsed_options_.emplace(name, value);
  }
}

namespace Options_detail {
template <typename Tag, typename Metavariables, typename Subgroup>
struct get_impl {
  template <typename OptionList, typename Group>
  static typename Tag::type apply(const Options<OptionList, Group>& opts) {
    static_assert(
        tmpl::list_contains_v<OptionList, Tag>,
        "Could not find requested option in the list of options provided. Did "
        "you forget to add the option tag to the OptionList?");
    const std::string subgroup_label = name<Subgroup>();
    Options<options_in_group<OptionList, Subgroup>, Subgroup> subgroup_options(
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
  static typename Tag::type apply(const Options<OptionList, Group>& opts) {
    static_assert(
        tmpl::list_contains_v<OptionList, Tag>,
        "Could not find requested option in the list of options provided. Did "
        "you forget to add the option tag to the OptionList?");
    const std::string label = name<Tag>();

    opts.template validate_default<Tag>();
    if (0 == opts.parsed_options_.count(label)) {
      return opts.template get_default<Tag>();
    }

    Option option(opts.parsed_options_.find(label)->second, opts.context_);
    option.append_context("While parsing option " + label);

    auto t = option.parse_as<typename Tag::type, Metavariables>();

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
typename Tag::type Options<OptionList, Group>::get() const {
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
    return func(opts.template get<Tags>()...);
  }
};
}  // namespace Options_detail

// \cond
// Doxygen is confused by decltype(auto)
template <typename OptionList, typename Group>
template <typename TagList, typename Metavariables, typename F>
decltype(auto) Options<OptionList, Group>::apply(F&& func) const {
  return Options_detail::apply_helper<TagList>::template apply<Metavariables>(
      *this, std::forward<F>(func));
}
// \endcond

template <typename OptionList, typename Group>
std::string Options<OptionList, Group>::help() const noexcept {
  std::ostringstream ss;
  ss << "\n==== Description of expected options:\n" << help_text_;
  if (tmpl::size<tags_and_subgroups_list>::value > 0) {
    ss << "\n\nOptions:\n"
       << tmpl::for_each<tags_and_subgroups_list>(
              Options_detail::print<OptionList>{max_label_size_})
              .value;
  } else {
    ss << "\n\n<No options>\n";
  }
  return ss.str();
}

template <typename OptionList, typename Group>
template <typename T,
          Requires<Options_detail::has_lower_bound_on_size<T>::value>>
void Options<OptionList, Group>::check_lower_bound_on_size(
    const typename T::type& t, const OptionContext& context) const {
  static_assert(cpp17::is_same_v<decltype(T::lower_bound_on_size()), size_t>,
                "lower_bound_on_size() is not a size_t.");
  if (t.size() < T::lower_bound_on_size()) {
    PARSE_ERROR(context, "Value must have at least "
                             << T::lower_bound_on_size() << " entries, but "
                             << t.size() << " were given.\n"
                             << help());
  }
}

template <typename OptionList, typename Group>
template <typename T,
          Requires<Options_detail::has_upper_bound_on_size<T>::value>>
void Options<OptionList, Group>::check_upper_bound_on_size(
    const typename T::type& t, const OptionContext& context) const {
  static_assert(cpp17::is_same_v<decltype(T::upper_bound_on_size()), size_t>,
                "upper_bound_on_size() is not a size_t.");
  if (t.size() > T::upper_bound_on_size()) {
    PARSE_ERROR(context, "Value must have at most "
                             << T::upper_bound_on_size() << " entries, but "
                             << t.size() << " were given.\n"
                             << help());
  }
}

template <typename OptionList, typename Group>
template <typename T, Requires<Options_detail::has_lower_bound<T>::value>>
inline void Options<OptionList, Group>::check_lower_bound(
    const typename T::type& t, const OptionContext& context) const {
  static_assert(cpp17::is_same_v<decltype(T::lower_bound()), typename T::type>,
                "Lower bound is not of the same type as the option.");
  static_assert(not cpp17::is_same_v<typename T::type, bool>,
                "Cannot set a lower bound for a bool.");
  if (t < T::lower_bound()) {
    PARSE_ERROR(context, "Value " << t << " is below the lower bound of "
                                  << T::lower_bound() << ".\n"
                                  << help());
  }
}

template <typename OptionList, typename Group>
template <typename T, Requires<Options_detail::has_upper_bound<T>::value>>
inline void Options<OptionList, Group>::check_upper_bound(
    const typename T::type& t, const OptionContext& context) const {
  static_assert(cpp17::is_same_v<decltype(T::upper_bound()), typename T::type>,
                "Upper bound is not of the same type as the option.");
  static_assert(not cpp17::is_same_v<typename T::type, bool>,
                "Cannot set an upper bound for a bool.");
  if (t > T::upper_bound()) {
    PARSE_ERROR(context, "Value " << t << " is above the upper bound of "
                                  << T::upper_bound() << ".\n"
                                  << help());
  }
}

template <typename OptionList, typename Group>
std::string Options<OptionList, Group>::parsing_help(
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
[[noreturn]] void Options<OptionList, Group>::parser_error(
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

template <typename T>
template <typename Metavariables>
T create_from_yaml<T>::create(const Option& options) {
  Options<typename T::options> parser(T::help);
  parser.parse(options);
  return parser.template apply<typename T::options>([&options](auto&&... args) {
    return make_overloader(
        [&options](std::false_type /*option_context_no_metavars*/,
                   std::true_type /*option_context_with_metavars*/,
                   auto&&... args2) {
          return T(std::move(args2)..., options.context(), Metavariables{});
        },
        [&options](std::true_type /*option_context_no_metavars*/,
                   std::false_type /*option_context_with_metavars*/,
                   auto&&... args2) {
          return T(std::move(args2)..., options.context());
        },
        [](std::false_type /*option_context_no_metavars*/,
           std::false_type /*option_context_with_metavars*/, auto&&... args2) {
          return T(std::move(args2)...);
        })(cpp17::is_constructible_t<T, decltype(std::move(args))...,
                                     const OptionContext&>{},
           cpp17::is_constructible_t<T, decltype(std::move(args))...,
                                     const OptionContext&, Metavariables>{},
           std::move(args)...);
  });
}

namespace YAML {
template <typename T, typename Metavariables>
struct convert<Options_detail::CreateWrapper<T, Metavariables>> {
  /* clang-tidy: non-const reference parameter */
  static bool decode(
      const Node& node,
      Options_detail::CreateWrapper<T, Metavariables>& rhs) { /* NOLINT */
    OptionContext context;
    context.top_level = false;
    context.append("While creating a " + pretty_type::short_name<T>());
    Option options(node, std::move(context));
    rhs = Options_detail::CreateWrapper<T, Metavariables>{
        create_from_yaml<T>::template create<Metavariables>(options)};
    return true;
  }
};
}  // namespace YAML

// yaml-cpp doesn't handle C++11 types yet
template <typename K, typename V, typename H, typename P>
struct create_from_yaml<std::unordered_map<K, V, H, P>> {
  template <typename Metavariables>
  static std::unordered_map<K, V, H, P> create(const Option& options) {
    // This shared_ptr stuff is a hack to work around the inability to
    // extract keys from maps before C++17.  Once we require C++17
    // this function and the conversion code for maps in
    // OptionsDetails.hpp can be updated to use the map `extract`
    // method and the shared_ptr conversion below can be removed.
    std::map<std::shared_ptr<K>, V> ordered =
        options.parse_as<std::map<std::shared_ptr<K>, V>, Metavariables>();
    std::unordered_map<K, V, H, P> result;
    for (auto& kv : ordered) {
      result.emplace(std::move(*kv.first), std::move(kv.second));
    }
    return result;
  }
};

// This is more of the hack for pre-C++17 unordered_maps
template <typename T>
struct create_from_yaml<std::shared_ptr<T>> {
  template <typename Metavariables>
  static std::shared_ptr<T> create(const Option& options) {
    return std::make_shared<T>(options.parse_as<T, Metavariables>());
  }
};

// This is more of the hack for pre-C++17 unordered_maps
namespace Options_detail {
template <typename T>
struct yaml_type<std::shared_ptr<T>> {
  static std::string value() noexcept { return yaml_type<T>::value(); }
};
}  // namespace Options_detail

#include "Options/Factory.hpp"
