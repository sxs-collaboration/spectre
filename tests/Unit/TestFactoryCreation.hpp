// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <string>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"

namespace TestFactoryCreation_detail {
template <typename T>
struct Opt {
  using type = std::unique_ptr<T>;
  static constexpr OptionString_t help = {"halp"};
};
}  // namespace TestFactoryCreation_detail

/// \ingroup TestingFramework
/// Construct a factory object from a given string.  Each line in the
/// string must be indented.
template <typename BaseClass>
std::unique_ptr<BaseClass> test_factory_creation(
    const std::string& construction_string) {
  Options<tmpl::list<TestFactoryCreation_detail::Opt<BaseClass>>> options("");
  options.parse("Opt:\n" + construction_string);
  return options.template get<TestFactoryCreation_detail::Opt<BaseClass>>();
}
