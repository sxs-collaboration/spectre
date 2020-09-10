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

#include "ErrorHandling/Assert.hpp"
#include "Options/Options.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Options {
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

template <typename BaseClass>
std::string help_derived() noexcept {
  return "Known Ids:\n" +
         tmpl::for_each<typename BaseClass::creatable_classes>(
             Factory_detail::print_derived{})
             .value;
}

template <typename BaseClass, typename Metavariables>
std::unique_ptr<BaseClass> create(const Option& options) {
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
                "Expected a class to create:\n" << help_derived<BaseClass>());
  } else {
    PARSE_ERROR(derived_opts.context(),
                "Expected a class or a class with options, got:\n"
                << node);
  }

  std::unique_ptr<BaseClass> result;
  tmpl::for_each<typename BaseClass::creatable_classes>(
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
              "Unknown Id '" << id << "'\n" << help_derived<BaseClass>());
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
