// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class template Factory.

#pragma once

#include <algorithm>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "ErrorHandling/Assert.hpp"
#include "Options/Options.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup OptionParsing
/// \brief Allows creating a pointer to an abstract base class from an
/// input file.
///
/// To use a factory, the base class (here `BaseClass`) should inherit
/// from Factory<BaseClass> and define a type
/// \code
///  using creatable_classes = tmpl::list<Derived1, ...>;
/// \endcode
///
/// Derived classes should:
/// -# define static OptionString_t help containing class-specfic help
///    text
/// -# define a type `options` as a typelist of option structs
///    required to create the class (see Options for details)
/// -# define a constructor taking those options and an OptionContext
///
/// \tparam BaseClass the base class of the objects to be created.
template <typename BaseClass>
class Factory {
 public:
  using Base_t = BaseClass;

  /// Create a derived object.
  static std::unique_ptr<BaseClass> create(const Option_t& options);

 private:
  static std::string help_derived() noexcept;

  template <typename CreateList,
            Requires<(tmpl::size<CreateList>::value != 0)> = nullptr>
  static std::unique_ptr<BaseClass> create_derived(const std::string& id,
                                                   const Option_t& opts);

  template <typename CreateList,
            Requires<(tmpl::size<CreateList>::value == 0)> = nullptr>
  static std::unique_ptr<BaseClass> create_derived(
      const std::string& /*id*/, const Option_t& /*opts*/) noexcept {
    return std::unique_ptr<BaseClass>{};
  }
};

template <typename BaseClass>
template <typename CreateList, Requires<(tmpl::size<CreateList>::value != 0)>>
std::unique_ptr<BaseClass> Factory<BaseClass>::create_derived(
    const std::string& id, const Option_t& opts) {
  using derived = tmpl::front<CreateList>;

  if (pretty_type::short_name<derived>() != id) {
    return create_derived<tmpl::pop_front<CreateList>>(id, opts);
  }

  ASSERT(not create_derived<tmpl::pop_front<CreateList>>(id, opts),
         "Duplicate factory id: " << id);

  Options<typename derived::options> options(derived::help);
  options.parse(opts);
  return options.template apply<typename derived::options>(
      [&opts](auto&&... args) {
        return std::make_unique<derived>(std::move(args)..., opts.context());
      });
}

template <typename BaseClass>
std::unique_ptr<BaseClass> Factory<BaseClass>::create(const Option_t& options) {
  const auto& node = options.node();
  Option_t derived_opts(options.context());
  std::string id;
  if (node.IsScalar()) {
    id = node.as<std::string>();
  } else if (node.IsMap()) {
    if (node.size() != 1) {
      PARSE_ERROR(options.context(),
                  "Expected a single class to create, got "
                  << node.size() << ":\n" << node);
    }
    id = node.begin()->first.as<std::string>();
    derived_opts.set_node(node.begin()->second);
  } else if (node.IsNull()) {
    PARSE_ERROR(options.context(),
                "Expected a class to create:\n" << help_derived());
  } else {
    PARSE_ERROR(options.context(),
                "Expected a class or a class with options, got:\n"
                << node);
  }
  derived_opts.append_context("While creating type " + id);
  auto derived =
      create_derived<typename BaseClass::creatable_classes>(id, derived_opts);
  if (derived != nullptr) {
    return derived;
  }
  PARSE_ERROR(options.context(),
              "Unknown Id '" << id << "'\n" << help_derived());
}

namespace Factory_detail {
struct print_derived {
  // Not a stream because brigand requires the functor to be copyable.
  std::string value;
  template <typename T>
  void operator()(tmpl::type_<T> /*meta*/) noexcept {
    // These are zero-based
    const size_t name_col = 2;
    const size_t help_col = 22;
    const size_t end_col = 80;

    std::ostringstream ss;
    ss << std::left
       << std::setw(name_col) << ""
       << std::setw(help_col - name_col - 1) << pretty_type::short_name<T>();
    if (ss.str().size() >= help_col) {
      ss << "\n" << std::setw(help_col - 1) << "";
    }

    std::string help_snippet(T::help);
    if (help_snippet.size() > end_col - help_col) {
      help_snippet.resize(end_col - help_col - 3);
      help_snippet += "...";
    }
    std::replace(help_snippet.begin(), help_snippet.end(), '\n', ' ');
    ss << " " << help_snippet << "\n";

    value += ss.str();
  }
};
}  // namespace Factory_detail

template <typename BaseClass>
std::string Factory<BaseClass>::help_derived() noexcept {
  return "Known Ids:\n" + tmpl::for_each<typename BaseClass::creatable_classes>(
                              Factory_detail::print_derived{})
                              .value;
}

namespace YAML {
template <typename T>
struct convert<std::unique_ptr<T>> {
  static bool decode(const Node& node, std::unique_ptr<T>& rhs) {
    OptionContext context;
    context.top_level = false;
    context.append("While creating a " + pretty_type::short_name<T>());
    Option_t options(node, std::move(context));
    rhs = T::create(options);
    return true;
  }
};
}  // namespace YAML
