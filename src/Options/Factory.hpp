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
#include <yaml-cpp/yaml.h>

#include "Options/Options.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/CreateGetStaticMemberVariableOrDefault.hpp"

namespace Options {
namespace Factory_detail {
struct print_derived {
  // Not a stream because brigand requires the functor to be copyable.
  std::string value;
  template <typename T>
  void operator()(tmpl::type_<T> /*meta*/) {
    // These are zero-based
    const size_t name_col = 2;
    const size_t help_col = 22;
    const size_t end_col = 80;

    std::ostringstream ss;
    ss << std::left
       << std::setw(name_col) << ""
       << std::setw(help_col - name_col - 1) << name<T>();
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

template <typename CreatableClasses>
std::string help_derived() {
  return "Known Ids:\n" +
         tmpl::for_each<CreatableClasses>(
             Factory_detail::print_derived{})
             .value;
}

// This is for handling legacy code that still uses creatable_classes.
// It should be inlined once everything is converted.
template <typename BaseClass, typename Metavariables, typename = std::void_t<>>
struct get_creatable_classes {
  using factory_creation = typename Metavariables::factory_creation;
  // This assertion is normally done in tests, but the executable
  // metavariables don't have tests, so we do it here.
  static_assert(tt::assert_conforms_to<factory_creation,
                                       Options::protocols::FactoryCreation>);
  using type = tmpl::at<typename factory_creation::factory_classes, BaseClass>;
};

template <typename BaseClass, typename Metavariables>
struct get_creatable_classes<
  BaseClass, Metavariables,
  std::void_t<typename BaseClass::creatable_classes>> {
  using type = typename BaseClass::creatable_classes;
};

CREATE_GET_STATIC_MEMBER_VARIABLE_OR_DEFAULT(factory_creatable)

template <typename T>
struct is_factory_creatable
    : std::bool_constant<get_factory_creatable_or_default_v<T, true>> {};

template <typename BaseClass, typename Metavariables>
std::unique_ptr<BaseClass> create(const Option& options) {
  using creatable_classes = tmpl::filter<
      typename get_creatable_classes<BaseClass, Metavariables>::type,
      is_factory_creatable<tmpl::_1>>;
  static_assert(not std::is_same_v<creatable_classes, tmpl::no_such_type_>,
                "List of creatable derived types for this class is missing "
                "from Metavariables::factory_classes.");

  const auto& node = options.node();
  Option derived_opts(options.context());
  derived_opts.append_context("While operating factory for " +
                              name<BaseClass>());
  std::string id;
  if (node.IsScalar()) {
    id = node.as<std::string>();
  } else if (node.IsMap()) {
    if (node.size() != 1) {
      PARSE_ERROR(derived_opts.context(),
                  "Expected a single class to create, got "
                  << node.size() << ":\n" << node);
    }
    id = node.begin()->first.as<std::string>();
    derived_opts.set_node(node.begin()->second);
  } else if (node.IsNull()) {
    PARSE_ERROR(derived_opts.context(),
                "Expected a class to create:\n"
                << help_derived<creatable_classes>());
  } else {
    PARSE_ERROR(derived_opts.context(),
                "Expected a class or a class with options, got:\n"
                << node);
  }

  std::unique_ptr<BaseClass> result;
  tmpl::for_each<creatable_classes>(
      [&id, &derived_opts, &result](auto derived_v) {
        using Derived = tmpl::type_from<decltype(derived_v)>;
        if (name<Derived>() == id) {
          ASSERT(result == nullptr, "Duplicate factory id: " << id);
          result = std::make_unique<Derived>(
              derived_opts.parse_as<Derived, Metavariables>());
        }
      });
  if (result != nullptr) {
    return result;
  }
  PARSE_ERROR(derived_opts.context(),
              "Unknown Id '" << id << "'\n"
              << help_derived<creatable_classes>());
}
}  // namespace Factory_detail

template <typename T>
struct create_from_yaml<std::unique_ptr<T>> {
  template <typename Metavariables>
  static std::unique_ptr<T> create(const Option& options) {
    return Factory_detail::create<T, Metavariables>(options);
  }
};
}  // namespace Options
