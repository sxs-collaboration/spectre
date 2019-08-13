// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <string>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/NoSuchType.hpp"

namespace TestHelpers {
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
  return TestHelpers::test_creation<std::unique_ptr<BaseClass>, Metavariables>(
      construction_string);
}

/// \ingroup TestingFrameworkGroup
/// Construct an enum from a given string.
///
/// Whereas `test_creation` creates a class with options, this creates an enum.
/// The enum is created from a simple string with no newlines or indents.
template <typename T, typename Metavariables = NoSuchType,
          Requires<std::is_enum<T>::value> = nullptr>
T test_enum_creation(const std::string& enum_string) noexcept {
  Options<tmpl::list<TestCreation_detail::Opt<T>>> options("");
  options.parse("Opt: " + enum_string);
  return options.template get<TestCreation_detail::Opt<T>, Metavariables>();
}
}  // namespace TestHelpers
