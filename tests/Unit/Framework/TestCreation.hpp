// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/algorithm/string.hpp>
#include <memory>
#include <string>
#include <type_traits>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers {
namespace TestCreation_detail {
template <typename T, typename = std::void_t<>>
struct looks_like_tag : std::false_type {};

template <typename T>
struct looks_like_tag<T, std::void_t<typename T::type>> : std::true_type {};

template <typename T>
struct TestCreationOpt {
  using type = T;
  static constexpr Options::String help = {"halp"};
};

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

template <typename BaseClass, typename DerivedClass>
struct SingleFactoryMetavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<BaseClass, tmpl::list<DerivedClass>>>;
  };
};
}  // namespace TestCreation_detail

/// \ingroup TestingFrameworkGroup
/// Creates an instance of a given option-creatable type.
///
/// This is a wrapper around Options::Parser constructing a single,
/// specified type \p T from the supplied string.  If necessary,
/// metavariables can be supplied as the second template argument.
///
/// A class can be explicitly created through a factory by passing
/// `std::unique_ptr<BaseClass>` as the type.  This will require
/// metavariables to be passed.  For testing basic factory creation,
/// the simpler TestHelpers::test_factory_creation() can be used
/// instead.
///
/// \snippet Test_TestCreation.cpp class_without_metavariables
/// \snippet Test_TestCreation.cpp class_without_metavariables_create
///
/// \see TestHelpers::test_option_tag()
template <typename T, typename Metavariables = NoSuchType>
T test_creation(const std::string& construction_string) noexcept {
  static_assert(not TestCreation_detail::looks_like_tag<Metavariables>::value,
                "test_creation no longer allows overriding the default "
                "option tag.  To test a particular option tag use "
                "test_option_tag.");
  using tag = TestCreation_detail::TestCreationOpt<T>;
  Options::Parser<tmpl::list<tag>> options("");
  options.parse(
      TestCreation_detail::AddGroups<tag>::apply(construction_string));
  return options.template get<tag, Metavariables>();
}

/// \ingroup TestingFrameworkGroup
/// Runs the option parser on a given tag
///
/// Runs the option parser with the supplied input on a given tag.
/// The tag name and any groups are handled by this function and
/// should not be supplied in the argument string.  If necessary,
/// metavariables can be supplied as the second template argument.
///
/// \snippet Test_TestCreation.cpp class_without_metavariables
/// \snippet Test_TestCreation.cpp class_without_metavariables_tag
/// \snippet Test_TestCreation.cpp class_without_metavariables_create_tag
///
/// \see TestHelpers::test_creation()
template <typename OptionTag, typename Metavariables = NoSuchType>
typename OptionTag::type test_option_tag(
    const std::string& construction_string) noexcept {
  Options::Parser<tmpl::list<OptionTag>> options("");
  options.parse(
      TestCreation_detail::AddGroups<OptionTag>::apply(construction_string));
  return options.template get<OptionTag, Metavariables>();
}

/// \ingroup TestingFrameworkGroup
/// Creates a class of a known derived type using a factory.
///
/// This is a shorthand for creating a \p DerivedClass through a \p
/// BaseClass factory, saving the caller from having to explicitly
/// write metavariables with the appropriate `factory_classes` alias.
/// The name of the type should be supplied as the first line of the
/// passed string, just as for normal use of a factory.
///
/// If multiple factory creatable types must be handled or if
/// metavariables must be passed for some other reason, then the more
/// general TestHelpers::test_creation() must be used instead.
///
/// \snippet Test_TestCreation.cpp test_factory_creation
template <typename BaseClass, typename DerivedClass>
std::unique_ptr<BaseClass> test_factory_creation(
    const std::string& construction_string) noexcept {
  return TestHelpers::test_creation<
      std::unique_ptr<BaseClass>,
      TestCreation_detail::SingleFactoryMetavariables<BaseClass, DerivedClass>>(
      construction_string);
}
}  // namespace TestHelpers
