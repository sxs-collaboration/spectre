// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/algorithm/string.hpp>
#include <memory>
#include <string>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/NoSuchType.hpp"

namespace TestHelpers {
namespace TestCreation_detail {
template <typename Tag, typename = void>
struct AddGroups {
  template <typename OptionTag = Tag>
  static std::string apply(std::string construction_string) noexcept {
    construction_string.insert(0, 2, ' ');
    construction_string =
        boost::algorithm::replace_all_copy(construction_string, "\n", "\n  ");
    construction_string.insert(0, Options::name<OptionTag>() + ":\n");
    return construction_string;
  }
};

template <typename Tag>
struct AddGroups<Tag, std::void_t<typename Tag::group>> {
  static std::string apply(const std::string& construction_string) noexcept {
    return AddGroups<typename Tag::group>::apply(
        AddGroups<void>::template apply<Tag>(construction_string));
  }
};
}  // namespace TestCreation_detail

/// \ingroup TestingFrameworkGroup
/// The default option tag for `TestHelpers::test_creation()`.
template <typename T>
struct TestCreationOpt {
  using type = T;
  static constexpr Options::String help = {"halp"};
};

/// \ingroup TestingFrameworkGroup
/// Construct an object or enum from a given string.
///
/// When creating a non-enum option the string must not contain the name of the
/// option (specifically, the struct being created from options, which is the
/// name of the struct by default if no `name()` function is present). For
/// example, to create a struct named `ClassWithoutMetavariables` with an option
/// tag named `SizeT` the following would be used:
///
/// \snippet Test_TestCreation.cpp size_t_argument
///
/// When creating an enum option the string must not contain the name of the
/// enum being constructed. The following is an example of an enum named `Color`
/// with a member `Purple` being created from options.
///
/// \snippet Test_TestCreation.cpp enum_purple
///
/// Option tags can be tested by passing them as the second template parameter.
/// If the metavariables are required to create the class then the metavariables
/// must be passed as the third template parameter. The default option tag is
/// `TestCreationOpt<T>`.
template <typename T, typename OptionTag = TestCreationOpt<T>,
          typename Metavariables = NoSuchType>
T test_creation(const std::string& construction_string) noexcept {
  Options::Parser<tmpl::list<OptionTag>> options("");
  options.parse(
      TestCreation_detail::AddGroups<OptionTag>::apply(construction_string));
  return options.template get<OptionTag, Metavariables>();
}
}  // namespace TestHelpers
