// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <string>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/NoSuchType.hpp"

namespace TestCreation_detail {
template <typename T>
struct Opt {
  using type = T;
  static constexpr OptionString help = {"halp"};
};
}  // namespace TestCreation_detail

/// \ingroup TestingFrameworkGroup
/// Construct an object from a given string.  Each line in the string
/// must be indented.
template <typename T, typename Metavariables = NoSuchType>
T test_creation(const std::string& construction_string) noexcept {
  Options<tmpl::list<TestCreation_detail::Opt<T>>> options("");
  options.parse("Opt:\n" + construction_string);
  return options.template get<TestCreation_detail::Opt<T>, Metavariables>();
}

/// \ingroup TestingFrameworkGroup
/// Construct a factory object from a given string.  Each line in the
/// string must be indented.
template <typename BaseClass, typename Metavariables = NoSuchType>
std::unique_ptr<BaseClass> test_factory_creation(
    const std::string& construction_string) noexcept {
  return test_creation<std::unique_ptr<BaseClass>, Metavariables>(
      construction_string);
}
