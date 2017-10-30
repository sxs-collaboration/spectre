// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines macro MAKE_CREATABLE_FROM_YAML

#pragma once

#include <yaml-cpp/yaml.h>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup OptionParsing
/// Used by MAKE_CREATABLE_FROM_YAML to create an object.  The default
/// action is to parse options using `T::options`.  This function may
/// be specialized to change that behavior for specific types.
///
/// Do not call this function directly.  Use Option_t::parse_as
/// instead.
///
/// \see MAKE_CREATABLE_FROM_YAML
template <typename T>
T create_from_yaml(const Option_t& options) {
  Options<typename T::options> parser(T::help);
  parser.parse(options);
  return parser.template apply<typename T::options>(
      [&](auto&&... args) { return T(args...); });
}

/// \ingroup OptionParsing
/// \brief Makes a concrete class constructible during option parsing.
///
/// See the \ref tuts_option_parsing tutorial.
#define MAKE_CREATABLE_FROM_YAML(templates, klass)                         \
  namespace YAML {                                                         \
  template <templates>                                                     \
  struct convert<klass> {                                                  \
    using type = klass;                                                    \
    /* clang-tidy: non-const reference parameter */                        \
    static bool decode(const Node& node, type& rhs) { /* NOLINT */         \
      OptionContext context;                                               \
      context.top_level = false;                                           \
      context.append("While creating a " + pretty_type::get_name<type>()); \
      Option_t options(node, std::move(context));                          \
      rhs = create_from_yaml<type>(options);                               \
      return true;                                                         \
    }                                                                      \
  };                                                                       \
  }
