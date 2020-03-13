// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TypeTraits.hpp"
#include "Utilities/TypeTraits/CreateHasTypeAlias.hpp"
#include "tests/Unit/ProtocolTestHelpers.hpp"

namespace {

/// [named_protocol]
namespace protocols {
namespace detail {
CREATE_IS_CALLABLE(name)
}  // namespace detail
/*!
 * \brief Has a name.
 *
 * Requires the class has these member functions:
 * - `name`: Returns the name of the object as a `std::string`.
 */
template <typename ConformingType>
using Named = detail::is_name_callable_r<std::string, ConformingType>;
}  // namespace protocols
/// [named_protocol]

/// [named_conformance]
class Person : public tt::ConformsTo<protocols::Named> {
 public:
  // Function required to conform to the protocol
  std::string name() const { return first_name_ + " " + last_name_; }

 private:
  // Implementation details of the class that are irrelevant to the protocol
  std::string first_name_;
  std::string last_name_;

 public:
  Person(std::string first_name, std::string last_name)
      : first_name_(std::move(first_name)), last_name_(std::move(last_name)) {}
};
/// [named_conformance]

/// [using_named_protocol]
template <typename NamedThing>
std::string greet(const NamedThing& named_thing) {
  // Make sure the template parameter conforms to the protocol
  static_assert(tt::conforms_to_v<NamedThing, protocols::Named>,
                "NamedThing must be Named.");
  // Now we can rely on the interface that the protocol defines
  return "Hello, " + named_thing.name() + "!";
}
/// [using_named_protocol]

/// [protocol_sfinae]
template <typename Thing,
          Requires<not tt::conforms_to_v<Thing, protocols::Named>> = nullptr>
std::string greet_anything(const Thing& /*anything*/) {
  return "Hello!";
}
template <typename NamedThing,
          Requires<tt::conforms_to_v<NamedThing, protocols::Named>> = nullptr>
std::string greet_anything(const NamedThing& named_thing) {
  return greet(named_thing);
}
/// [protocol_sfinae]
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.ProtocolHelpers", "[Utilities][Unit]") {
  Person person{"Alice", "Bob"};
  CHECK(greet(person) == "Hello, Alice Bob!");
  CHECK(greet_anything<int>(0) == "Hello!");
  CHECK(greet_anything(person) == "Hello, Alice Bob!");
}

// Test tt::conforms_to metafunction
/// [conforms_to]
static_assert(tt::conforms_to_v<Person, protocols::Named>,
              "The class does not conform to the protocol.");
/// [conforms_to]
namespace {
struct NotNamed {};
class DerivedNonConformingClass : private Person {};
class DerivedConformingClass : public Person {};
}  // namespace
static_assert(not tt::conforms_to_v<NotNamed, protocols::Named>,
              "Failed testing tt::conforms_to_v");
static_assert(
    not tt::conforms_to_v<DerivedNonConformingClass, protocols::Named>,
    "Failed testing tt::conforms_to_v");
static_assert(tt::conforms_to_v<DerivedConformingClass, protocols::Named>,
              "Failed testing tt::conforms_to_v");

// Give examples about protocol antipatterns
namespace {
namespace protocols {
/// [named_antipattern]
// Don't do this. Protocols should be _unary_ type traits.
template <typename ConformingType, typename NameType>
using NamedAntipattern =
    // Check that the `name` function exists _and_ its return type
    detail::is_name_callable_r<NameType, ConformingType>;
/// [named_antipattern]
// Just making sure the protocol works correctly, even though this is an
// antipattern
static_assert(NamedAntipattern<Person, std::string>::value,
              "Failed testing is_conforming_v");
static_assert(not NamedAntipattern<Person, int>::value,
              "Failed testing is_conforming_v");
/// [named_with_type]
// Instead, do this.
namespace detail {
CREATE_HAS_TYPE_ALIAS(NameType)
CREATE_HAS_TYPE_ALIAS_V(NameType)
// Lazily evaluated so we can use `ConformingType::NameType`
template <typename ConformingType>
struct IsNameCallableWithType
    : is_name_callable_r_t<typename ConformingType::NameType, ConformingType> {
};
}  // namespace detail
template <typename ConformingType>
using NamedWithType =
    // First check the class has a `NameType`, then use it to check the return
    // type of the `name` function.
    std::conditional_t<detail::has_NameType_v<ConformingType>,
                       detail::IsNameCallableWithType<ConformingType>,
                       std::false_type>;
/// [named_with_type]
}  // namespace protocols
/// [person_with_name_type]
struct PersonWithNameType : tt::ConformsTo<protocols::NamedWithType> {
  using NameType = std::string;
  std::string name() const;
};
/// [person_with_name_type]
// Make sure the protocol is implemented correctly
static_assert(not protocols::NamedWithType<Person>::value,
              "Failed testing is_conforming_v");
static_assert(protocols::NamedWithType<PersonWithNameType>::value,
              "Failed testing is_conforming_v");
/// [example_check_name_type]
static_assert(tt::conforms_to_v<PersonWithNameType, protocols::NamedWithType>,
              "The class does not conform to the protocol.");
static_assert(
    cpp17::is_same_v<typename PersonWithNameType::NameType, std::string>,
    "The `NameType` isn't a `std::string`!");
/// [example_check_name_type]
}  // namespace

// Give an example how a protocol author should test a new protocol
/// [testing_a_protocol]
static_assert(protocols::Named<Person>::value, "Failed testing the protocol");
static_assert(not protocols::Named<NotNamed>::value,
              "Failed testing the protocol");
/// [testing_a_protocol]

// Give an example how protocol consumers should test protocol conformance
/// [test_protocol_conformance]
static_assert(test_protocol_conformance<Person, protocols::Named>,
              "Failed testing protocol conformance");
/// [test_protocol_conformance]
