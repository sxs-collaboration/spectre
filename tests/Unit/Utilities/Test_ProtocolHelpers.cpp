// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TypeTraits.hpp"
#include "Utilities/TypeTraits/CreateHasTypeAlias.hpp"
#include "Utilities/TypeTraits/CreateIsCallable.hpp"

namespace {

/// [named_protocol]
namespace protocols {
/*
 * \brief Has a name.
 *
 * Requires the class has these member functions:
 * - `name`: Returns the name of the object as a `std::string`.
 */
struct Named {
  template <typename ConformingType>
  struct test {
    // Try calling the `ConformingType::name` member function
    using name_return_type = decltype(std::declval<ConformingType>().name());
    // Check the return type of the `name` member function
    static_assert(std::is_same_v<name_return_type, std::string>,
                  "The 'name' function must return a 'std::string'.");
  };
};
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
  static_assert(tt::assert_conforms_to<NamedThing, protocols::Named>);
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
// SFINAE-friendly version:
constexpr bool person_class_is_named =
    tt::conforms_to_v<Person, protocols::Named>;
// Assert-friendly version with more diagnostics:
static_assert(tt::assert_conforms_to<Person, protocols::Named>);
/// [conforms_to]
static_assert(person_class_is_named,
              "The 'Person' class does not conform to the 'Named' protocol.");

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
// Don't do this. Protocols should not have template parameters.
template <typename NameType>
struct NamedAntipattern {
  template <typename ConformingType>
  struct test {
    // Check that the `name` function exists _and_ its return type
    using name_return_type = decltype(std::declval<ConformingType>().name());
    static_assert(std::is_same_v<name_return_type, NameType>,
                  "The 'name' function must return a 'NameType'.");
  };
};
/// [named_antipattern]
/// [named_with_type]
// Instead, do this.
struct NamedWithType {
  template <typename ConformingType>
  struct test {
    // Use the `ConformingType::NameType` to check the return type of the `name`
    // function.
    using name_type = typename ConformingType::NameType;
    using name_return_type = decltype(std::declval<ConformingType>().name());
    static_assert(std::is_same_v<name_return_type, name_type>,
                  "The 'name' function must return a 'NameType'.");
  };
};
/// [named_with_type]
}  // namespace protocols
/// [person_with_name_type]
struct PersonWithNameType : tt::ConformsTo<protocols::NamedWithType> {
  using NameType = std::string;
  std::string name() const;
};
/// [person_with_name_type]
/// [example_check_name_type]
static_assert(
    tt::assert_conforms_to<PersonWithNameType, protocols::NamedWithType>);
static_assert(
    std::is_same_v<typename PersonWithNameType::NameType, std::string>,
    "The `NameType` isn't a `std::string`!");
/// [example_check_name_type]
}  // namespace

// Give an example how protocol consumers should test protocol conformance
/// [test_protocol_conformance]
static_assert(tt::assert_conforms_to<Person, protocols::Named>);
/// [test_protocol_conformance]
