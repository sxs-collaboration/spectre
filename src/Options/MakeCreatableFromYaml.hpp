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
/// When an object of the passed form is requested from the option
/// parser, the function `create_from_yaml(const Option_t&)` is
/// called.  The default implementation of that function creates a
/// `Options<T::options>` using the help string `T::help` and passes
/// the parsed options to the class constructor.  Other behavior can
/// be obtained by specializing the create_from_yaml function.
///
/// A class to be created must be default constructible and move
/// assignable.  If create_from_yaml is not specialized for the class
/// it must additionally declare a tmpl::list of Options-style option
/// structs `options`, a static OptionString_t `help`, and a
/// constructor taking arguments corresponding to the structs in
/// `options`.
///
/// The arguments to this macro should be chosen to make the following
/// a template specialization with `klass` the constructible class:
/// \code
///   template <templates>
///   struct convert<klass> { /* ... */ };
/// \endcode
///
/// \example
/// \snippet Test_MakeCreatableFromYaml.cpp MCFY_example
///
/// Using a specialization to parse an enum:
/// \snippet Test_MakeCreatableFromYaml.cpp MCFY_enum_example
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
