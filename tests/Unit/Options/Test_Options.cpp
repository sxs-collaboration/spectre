// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <list>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TMPL.hpp"

namespace {
void test_options_empty_success() {
  Options::Parser<tmpl::list<>> opts("");
  opts.parse("");
  // Catch requires us to have at least one CHECK in each test
  // The Unit.Options.Empty.success does not need to check anything
  CHECK(true);
}

// [[OutputRegex, In string:.*At line 1 column 1:.Option 'Option' is not a valid
// option.]]
SPECTRE_TEST_CASE("Unit.Options.Empty.extra", "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<>> opts("");
  opts.parse("Option:");
}

// [[OutputRegex, In string:.*'4' does not look like options]]
SPECTRE_TEST_CASE("Unit.Options.Empty.not_map", "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<>> opts("");
  opts.parse("4");
}

// [[OutputRegex, In string:.*At line 1 column 30:.Unable to correctly parse
// the input file because of a syntax error]]
SPECTRE_TEST_CASE("Unit.Options.syntax_error", "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<>> opts("");
  opts.parse(
      "DomainCreator: CreateInterval:\n"
      "  IsPeriodicIn: [false]");
}

struct Simple {
  using type = int;
  static constexpr Options::String help = {"halp"};
};
struct NamedSimple {
  using type = int;
  static std::string name() noexcept { return "SomeName"; }
  static constexpr Options::String help = {
      "halp halp halp halp halp halp halp halp halp halp halp halp\n"
      "halp halp halp halp halp halp halp halp halp halp halp halp"
      "halp halp halp halp halp halp halp halp halp halp halp halp"
      "halp halp halp halp halp halp halp halp halp halp halp halp"
      "halp halp halp halp halp halp halp halp halp halp halp halp"
      "halp halp halp halp halp halp halp halp halp halp halp halp"};
};

void test_options_simple_success() {
  {
    Options::Parser<tmpl::list<Simple>> opts("");
    opts.parse("Simple: -4");
    CHECK(opts.get<Simple>() == -4);
  }
  {
    Options::Parser<tmpl::list<NamedSimple>> opts("");
    opts.parse("SomeName: -4");
    CHECK(opts.get<NamedSimple>() == -4);
  }
}

void test_options_print_long_help() {
  Options::Parser<tmpl::list<NamedSimple>> opts("");
  CHECK(opts.help() ==
        R"(
==== Description of expected options:


Options:
  SomeName:
    type=int
    halp halp halp halp halp halp halp halp halp halp halp halp
    halp halp halp halp halp halp halp halp halp halp halp halphalp halp halp
    halp halp halp halp halp halp halp halp halphalp halp halp halp halp halp
    halp halp halp halp halp halphalp halp halp halp halp halp halp halp halp
    halp halp halphalp halp halp halp halp halp halp halp halp halp halp halp

)");
}

// [[OutputRegex, In string:.*At line 2 column 1:.Option 'Simple' specified
// twice.]]
SPECTRE_TEST_CASE("Unit.Options.Simple.duplicate", "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<Simple>> opts("");
  opts.parse(
      "Simple: -4\n"
      "Simple: -3");
}

// [[OutputRegex, In string:.*At line 2 column 1:.Option 'SomeName' specified
// twice.]]
SPECTRE_TEST_CASE("Unit.Options.NamedSimple.duplicate", "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<NamedSimple>> opts("");
  opts.parse(
      "SomeName: -4\n"
      "SomeName: -3");
}

// [[OutputRegex, In string:.*You did not specify the option \(Simple\)]]
SPECTRE_TEST_CASE("Unit.Options.Simple.missing", "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<Simple>> opts("");
  opts.parse("");
}

// [[OutputRegex, In string:.*You did not specify the option \(SomeName\)]]
SPECTRE_TEST_CASE("Unit.Options.NamedSimple.missing", "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<NamedSimple>> opts("");
  opts.parse("");
}

// [[OutputRegex, In string:.*You did not specify the options
// \(SomeName,Simple\)]]
SPECTRE_TEST_CASE("Unit.Options.multiple_missing", "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<NamedSimple, Simple>> opts("");
  opts.parse("");
}

// [[OutputRegex, In string:.*While parsing option Simple:.At line 1 column
// 1:.Failed to convert value to type int:]]
SPECTRE_TEST_CASE("Unit.Options.Simple.missing_arg", "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<Simple>> opts("");
  opts.parse("Simple:");
  opts.get<Simple>();
}

// [[OutputRegex, In string:.*While parsing option SomeName:.At line 1 column
// 1:.Failed to convert value to type int:]]
SPECTRE_TEST_CASE("Unit.Options.NamedSimple.missing_arg", "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<NamedSimple>> opts("");
  opts.parse("SomeName:");
  opts.get<NamedSimple>();
}

// [[OutputRegex, In string:.*While parsing option Simple:.At line 1 column
// 9:.Failed to convert value to type int: 2.3]]
SPECTRE_TEST_CASE("Unit.Options.Simple.invalid", "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<Simple>> opts("");
  opts.parse("Simple: 2.3");
  opts.get<Simple>();
}

// [[OutputRegex, In string:.*While parsing option SomeName:.At line 1 column
// 11:.Failed to convert value to type int: 2.3]]
SPECTRE_TEST_CASE("Unit.Options.NamedSimple.invalid", "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<NamedSimple>> opts("");
  opts.parse("SomeName: 2.3");
  opts.get<NamedSimple>();
}

namespace {
// [options_example_group]
struct Group {
  static constexpr Options::String help = {"Group halp"};
};

struct GroupedTag {
  using type = int;
  static constexpr Options::String help = {"Tag halp"};
  using group = Group;
};
// [options_example_group]

struct OuterGroup {
  static constexpr Options::String help = {"Outer group halp"};
};

struct InnerGroup {
  static constexpr Options::String help = {"Inner group halp"};
  using group = OuterGroup;
};

struct InnerGroupedTag {
  using type = int;
  static constexpr Options::String help = {"Inner tag halp"};
  using group = InnerGroup;
};

struct OuterGroupedTag {
  using type = int;
  static constexpr Options::String help = {"Outer tag halp"};
  using group = OuterGroup;
};

void test_options_grouped() {
  {
    INFO("Option groups");
    Options::Parser<tmpl::list<GroupedTag, Simple>> opts("Overall help text");
    opts.parse(
        "Group:\n"
        "  GroupedTag: 3\n"
        "Simple: 2");
    CHECK(opts.get<GroupedTag>() == 3);
    CHECK(opts.get<Simple>() == 2);
  }
  {
    INFO("Nested option groups");
    Options::Parser<tmpl::list<InnerGroupedTag, OuterGroupedTag, Simple>> opts(
        "Overall help text");
    opts.parse(
        "OuterGroup:\n"
        "  InnerGroup:\n"
        "    InnerGroupedTag: 3\n"
        "  OuterGroupedTag: 1\n"
        "Simple: 2\n");
    CHECK(opts.get<InnerGroupedTag>() == 3);
    CHECK(opts.get<OuterGroupedTag>() == 1);
    CHECK(opts.get<Simple>() == 2);
  }
}
}  // namespace

// [[OutputRegex, In string:.*You did not specify the option \(OuterGroup\)]]
SPECTRE_TEST_CASE("Unit.Options.Grouped.missing_outer_group",
                  "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<InnerGroupedTag>> opts("");
  opts.parse("");
  opts.get<InnerGroupedTag>();
}

// [[OutputRegex, In string:.*In group OuterGroup:.You did not specify the
// option \(InnerGroup\)]]
SPECTRE_TEST_CASE("Unit.Options.Grouped.missing_inner_group",
                  "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<InnerGroupedTag>> opts("");
  opts.parse("OuterGroup:");
  opts.get<InnerGroupedTag>();
}

// [options_example_scalar_struct]
struct Bounded {
  using type = int;
  static constexpr Options::String help = {
      "Option with bounds and a suggested value"};
  // These are optional
  static type suggested_value() noexcept { return 3; }
  static type lower_bound() noexcept { return 2; }
  static type upper_bound() noexcept { return 10; }
};
// [options_example_scalar_struct]

void test_options_suggested_followed() {
  Options::Parser<tmpl::list<Bounded>> opts("Overall help text");
  opts.parse("Bounded: 8");
  CHECK(opts.get<Bounded>() == 8);
}

void test_options_suggested_not_followed() {
  // [options_example_scalar_parse]
  Options::Parser<tmpl::list<Bounded>> opts("Overall help text");
  opts.parse("Bounded: 3");
  CHECK(opts.get<Bounded>() == 3);
  // [options_example_scalar_parse]
}

// [[OutputRegex, In string:.*While parsing option Bounded:.At line 1 column
// 10:.Value 1 is below the lower bound of 2]]
SPECTRE_TEST_CASE("Unit.Options.Bounded.below", "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<Bounded>> opts("");
  opts.parse("Bounded: 1");
  opts.get<Bounded>();
}

void test_options_bounded_lower_bound() {
  Options::Parser<tmpl::list<Bounded>> opts("");
  opts.parse("Bounded: 2");
  CHECK(opts.get<Bounded>() == 2);
}

void test_options_bounded_upper_bound() {
  Options::Parser<tmpl::list<Bounded>> opts("");
  opts.parse("Bounded: 10");
  CHECK(opts.get<Bounded>() == 10);
}

// [[OutputRegex, In string:.*While parsing option Bounded:.At line 1 column
// 10:.Value 11 is above the upper bound of 10]]
SPECTRE_TEST_CASE("Unit.Options.Bounded.above", "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<Bounded>> opts("");
  opts.parse("Bounded: 11");
  opts.get<Bounded>();
}

// [[OutputRegex, Bounded, line 1:.  Specified: 5.  Suggested: 3]]
SPECTRE_TEST_CASE("Unit.Options.suggestion_warning", "[Unit][Options]") {
  OUTPUT_TEST();
  Options::Parser<tmpl::list<Bounded>> opts("");
  opts.parse("Bounded: 5");
  opts.get<Bounded>();
}

struct BadSuggestion {
  using type = int;
  static constexpr Options::String help = {"halp"};
  static type suggested_value() noexcept { return 3; }
  static type lower_bound() noexcept { return 4; }
};
struct NamedBadSuggestion {
  using type = int;
  static std::string name() noexcept { return "SomeName"; }
  static constexpr Options::String help = {"halp"};
  static type suggested_value() noexcept { return 3; }
  static type lower_bound() noexcept { return 4; }
};

// [[OutputRegex, Checking SUGGESTED value for BadSuggestion:.Value 3 is below
// the lower bound of 4]]
SPECTRE_TEST_CASE("Unit.Options.BadSuggestion", "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<BadSuggestion>> opts("");
  opts.parse("BadSuggestion: 5");
  opts.get<BadSuggestion>();
}

// [[OutputRegex, Checking SUGGESTED value for SomeName:.Value 3 is below the
// lower bound of 4]]
SPECTRE_TEST_CASE("Unit.Options.NamedBadSuggestion", "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<NamedBadSuggestion>> opts("");
  opts.parse("SomeName: 5");
  opts.get<NamedBadSuggestion>();
}

// [options_example_vector_struct]
struct VectorOption {
  using type = std::vector<int>;
  static constexpr Options::String help = {"A vector with length limits"};
  // These are optional
  static std::string name() noexcept {
    return "Vector";  // defaults to "VectorOption"
  }
  static size_t lower_bound_on_size() { return 2; }
  static size_t upper_bound_on_size() { return 5; }
};
// [options_example_vector_struct]

// [[OutputRegex, In string:.*While parsing option Vector:.At line 1 column
// 9:.Value must have at least 2 entries, but 1 were given.]]
SPECTRE_TEST_CASE("Unit.Options.Vector.too_short", "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<VectorOption>> opts("");
  opts.parse("Vector: [2]");
  opts.get<VectorOption>();
}

void test_options_vector_lower_bound() {
  Options::Parser<tmpl::list<VectorOption>> opts("");
  opts.parse("Vector: [2,3]");
  CHECK(opts.get<VectorOption>() == (std::vector<int>{2, 3}));
}

void test_options_vector_upper_bound() {
  Options::Parser<tmpl::list<VectorOption>> opts("");
  opts.parse("Vector: [2, 3, 3, 3, 5]");
  CHECK(opts.get<VectorOption>() == (std::vector<int>{2, 3, 3, 3, 5}));
}

// [[OutputRegex, In string:.*While parsing option Vector:.At line 1 column
// 9:.Value must have at most 5 entries, but 6 were given.]]
SPECTRE_TEST_CASE("Unit.Options.Vector.too_long", "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<VectorOption>> opts("");
  opts.parse("Vector: [2, 3, 3, 3, 5, 6]");
  opts.get<VectorOption>();
}

// [[OutputRegex, In string:.*While parsing option Vector:.At line 1 column
// 1:.Value must have at least 2 entries, but 0 were given.]]
SPECTRE_TEST_CASE("Unit.Options.Vector.empty_too_short", "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<VectorOption>> opts("");
  opts.parse("Vector:");
  opts.get<VectorOption>();
}

struct Array {
  using type = std::array<int, 3>;
  static constexpr Options::String help = {"halp"};
};

// [[OutputRegex, In string:.*While parsing option Array:.At line 1 column
// 8:.Failed to convert value to type \[int x3\]:]]
SPECTRE_TEST_CASE("Unit.Options.Array.too_short", "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<Array>> opts("");
  opts.parse("Array: [1, 2]");
  opts.get<Array>();
}

void test_options_array_success() {
  Options::Parser<tmpl::list<Array>> opts("");
  opts.parse("Array: [1,2,3]");
  CHECK(opts.get<Array>() == (std::array<int, 3>{{1, 2, 3}}));
}

// [[OutputRegex, In string:.*While parsing option Array:.At line 1 column
// 8:.Failed to convert value to type \[int x3\]: .1, 2, 3, 4.]]
SPECTRE_TEST_CASE("Unit.Options.Array.too_long", "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<Array>> opts("");
  opts.parse("Array: [1, 2, 3, 4]");
  opts.get<Array>();
}

// [[OutputRegex, In string:.*While parsing option Array:.At line 2 column
// 3:.Failed to convert value to type \[int x3\]:.  - 1.  - 2.  -
// 3.  - 4]]
SPECTRE_TEST_CASE("Unit.Options.Array.too_long.formatting", "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<Array>> opts("");
  opts.parse(
      "Array:\n"
      "  - 1\n"
      "  - 2\n"
      "  - 3\n"
      "  - 4");
  opts.get<Array>();
}

// [[OutputRegex, In string:.*While parsing option Array:.At line 1 column
// 1:.Failed to convert value to type \[int x3\]:]]
SPECTRE_TEST_CASE("Unit.Options.Array.missing", "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<Array>> opts("");
  opts.parse("Array:");
  opts.get<Array>();
}

struct ZeroArray {
  using type = std::array<int, 0>;
  static constexpr Options::String help = {"halp"};
};

void test_options_zero_array_missing() {
  Options::Parser<tmpl::list<ZeroArray>> opts("");
  opts.parse("ZeroArray:");
  opts.get<ZeroArray>();
  // Catch requires us to have at least one CHECK in each test
  // The Unit.Options.ZeroArray.missing does not need to check anything
  CHECK(true);
}

struct Map {
  using type = std::map<std::string, int>;
  static constexpr Options::String help = {"halp"};
};

void test_options_map_success() {
  Options::Parser<tmpl::list<Map>> opts("");
  opts.parse(
      "Map:\n"
      "  A: 3\n"
      "  Z: 2");
  std::map<std::string, int> expected;
  expected.emplace("A", 3);
  expected.emplace("Z", 2);
  CHECK(opts.get<Map>() == expected);
}

void test_options_map_empty() {
  Options::Parser<tmpl::list<Map>> opts("");
  opts.parse("Map:");
  CHECK(opts.get<Map>().empty());
}

// [[OutputRegex, In string:.*While parsing option Map:.At line 1 column
// 6:.Failed to convert value to type {string: int}: string]]
SPECTRE_TEST_CASE("Unit.Options.Map.invalid", "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<Map>> opts("");
  opts.parse("Map: string");
  opts.get<Map>();
}

// [[OutputRegex, In string:.*While parsing option Map:.At line 2 column
// 6:.Failed to convert value to type {string: int}: A: string]]
SPECTRE_TEST_CASE("Unit.Options.Map.invalid_entry", "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<Map>> opts("");
  opts.parse(
      "Map:\n"
      "  A: string");
  opts.get<Map>();
}

struct UnorderedMap {
  using type = std::unordered_map<std::string, int>;
  static constexpr Options::String help = {"halp"};
};

void test_options_unordered_map_success() {
  Options::Parser<tmpl::list<UnorderedMap>> opts("");
  opts.parse(
      "UnorderedMap:\n"
      "  A: 3\n"
      "  Z: 2");
  std::unordered_map<std::string, int> expected;
  expected.emplace("A", 3);
  expected.emplace("Z", 2);
  CHECK(opts.get<UnorderedMap>() == expected);
}

template <typename... T>
struct VariantTag {
  using type = std::variant<T...>;
  static constexpr Options::String help = {"halp"};
};

void test_options_variant() {
  const auto check = [](const std::string& input,
                        const auto& expected) noexcept {
    using Tag = tmpl::wrap<std::decay_t<decltype(expected)>, VariantTag>;
    Options::Parser<tmpl::list<Tag>> opts("");
    opts.parse("VariantTag: " + input);
    CHECK(opts.template get<Tag>() == expected);
  };
  check("3", std::variant<int>(std::in_place_type_t<int>{}, 3));
  check("Hello", std::variant<std::string>(std::in_place_type_t<std::string>{},
                                           "Hello"));
  check("3", std::variant<int, std::string>(std::in_place_type_t<int>{}, 3));
  check("Hello", std::variant<int, std::string>(
                     std::in_place_type_t<std::string>{}, "Hello"));
  check("3", std::variant<std::string, int>(std::in_place_type_t<std::string>{},
                                            "3"));
  check("Hello", std::variant<std::string, int>(
                     std::in_place_type_t<std::string>{}, "Hello"));

  {
    const std::string help_text =
        Options::Parser<tmpl::list<VariantTag<int, std::string>>>("").help();
    CAPTURE(help_text);
    CHECK(help_text.find("type=int or string\n") != std::string::npos);
  }
}

// [[OutputRegex, While creating a variant:.At line 1 column 13:.Failed to
// convert value to type int or string]]
SPECTRE_TEST_CASE("Unit.Options.Format.variant.Error", "[Unit][Options]") {
  ERROR_TEST();
  using tag = VariantTag<int, std::string>;
  Options::Parser<tmpl::list<tag>> opts("");
  opts.parse("VariantTag: []");
  opts.get<tag>();
}

template <typename T>
struct Wrapped {
  T data;
};

#define FORWARD_OP(op)                                         \
  template <typename T>                                        \
  bool operator op(const Wrapped<T>& a, const Wrapped<T>& b) { \
    return a.data op b.data;                                   \
  }
FORWARD_OP(==)
FORWARD_OP(!=)
FORWARD_OP(<)
FORWARD_OP(>)
FORWARD_OP(<=)
FORWARD_OP(>=)
}  // namespace

namespace std {
template <typename T>
struct hash<Wrapped<T>> {
  size_t operator()(const Wrapped<T>& x) const { return hash<T>{}(x.data); }
};
}  // namespace std

template <typename T>
struct Options::create_from_yaml<Wrapped<T>> {
  template <typename Metavariables>
  static Wrapped<T> create(const Options::Option& options) {
    return Wrapped<T>{options.parse_as<T>()};
  }
};

namespace {
struct WrapMap {
  using type = std::map<Wrapped<int>, Wrapped<std::string>>;
  static constexpr Options::String help = {"halp"};
};
struct WrapVector {
  using type = std::vector<Wrapped<int>>;
  static constexpr Options::String help = {"halp"};
};
struct WrapList {
  using type = std::list<Wrapped<int>>;
  static constexpr Options::String help = {"halp"};
};
struct WrapArray {
  using type = std::array<Wrapped<int>, 2>;
  static constexpr Options::String help = {"halp"};
};
struct WrapPair {
  using type = std::pair<Wrapped<int>, Wrapped<std::string>>;
  static constexpr Options::String help = {"halp"};
};
struct WrapUnorderedMap {
  using type = std::unordered_map<Wrapped<int>, Wrapped<std::string>>;
  static constexpr Options::String help = {"halp"};
};

void test_options_complex_containers() {
  Options::Parser<tmpl::list<WrapMap, WrapVector, WrapList, WrapArray, WrapPair,
                             WrapUnorderedMap>>
      opts("");
  opts.parse(
      "WrapMap: {1: A, 2: B}\n"
      "WrapVector: [1, 2, 3]\n"
      "WrapList: [1, 2, 3]\n"
      "WrapArray: [1, 2]\n"
      "WrapPair: [1, X]\n"
      "WrapUnorderedMap: {1: A, 2: B}\n");
  CHECK(opts.get<WrapMap>() == (std::map<Wrapped<int>, Wrapped<std::string>>{
                                   {{1}, {"A"}}, {{2}, {"B"}}}));
  CHECK(opts.get<WrapVector>() == (std::vector<Wrapped<int>>{{1}, {2}, {3}}));
  CHECK(opts.get<WrapList>() == (std::list<Wrapped<int>>{{1}, {2}, {3}}));
  CHECK(opts.get<WrapArray>() == (std::array<Wrapped<int>, 2>{{{1}, {2}}}));
  CHECK(opts.get<WrapPair>() ==
        (std::pair<Wrapped<int>, Wrapped<std::string>>{{1}, {"X"}}));
  CHECK(opts.get<WrapUnorderedMap>() ==
        (std::unordered_map<Wrapped<int>, Wrapped<std::string>>{{{1}, {"A"}},
                                                                {{2}, {"B"}}}));
}

#ifdef SPECTRE_DEBUG
struct Duplicate {
  using type = int;
  static constexpr Options::String help = {"halp"};
};
struct NamedDuplicate {
  using type = int;
  static std::string name() noexcept { return "Duplicate"; }
  static constexpr Options::String help = {"halp"};
};
#endif  // SPECTRE_DEBUG

// [[OutputRegex, Duplicate option name: Duplicate]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Options.Duplicate", "[Unit][Options]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Options::Parser<tmpl::list<Duplicate, NamedDuplicate>> opts("");
  opts.parse("");
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

#ifdef SPECTRE_DEBUG
struct
    ToooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooLong {
  using type = int;
  static constexpr Options::String help = {"halp"};
};
struct NamedTooLong {
  using type = int;
  static std::string name() noexcept {
    return "Toooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo"
           "oLong";
  }
  static constexpr Options::String help = {"halp"};
};
struct NoHelp {
  using type = int;
  static constexpr Options::String help = {""};
};
struct NamedNoHelp {
  using type = int;
  static std::string name() noexcept { return "NoHelp"; }
  static constexpr Options::String help = {""};
};
#endif  // SPECTRE_DEBUG

// [[OutputRegex, The option name
// ToooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooLong is
// too long for nice formatting]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Options.TooLong", "[Unit][Options]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Options::Parser<tmpl::list<
      ToooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooLong>>
      opts("");
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The option name
// ToooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooLong is
// too long for nice formatting]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Options.NamedTooLong", "[Unit][Options]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Options::Parser<tmpl::list<NamedTooLong>> opts("");
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, You must supply a help string of non-zero length for NoHelp]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Options.NoHelp", "[Unit][Options]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Options::Parser<tmpl::list<NoHelp>> opts("");
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, You must supply a help string of non-zero length for NoHelp]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Options.NamedNoHelp", "[Unit][Options]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Options::Parser<tmpl::list<NamedNoHelp>> opts("");
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

struct Apply1 {
  using type = int;
  static constexpr Options::String help = {"halp"};
};
struct Apply2 {
  using type = std::string;
  static constexpr Options::String help = {"halp"};
};
struct Apply3 {
  using type = std::vector<int>;
  static constexpr Options::String help = {"halp"};
};

void test_options_apply() {
  Options::Parser<tmpl::list<Apply1, Apply2, Apply3>> opts("");
  opts.parse(
      "Apply1: 2\n"
      "Apply2: str\n"
      "Apply3: [1, 2, 3]");
  // We do the checks outside the lambda to make sure it actually gets called.
  std::vector<int> arg1;
  int arg2 = 0;
  opts.apply<tmpl::list<Apply3, Apply1>>([&](const auto& a, auto b) {
    arg1 = a;
    arg2 = b;
  });
  CHECK(arg1 == (std::vector<int>{1, 2, 3}));
  CHECK(arg2 == 2);
}

void test_options_option_context_default_stream() {
  CHECK(get_output(Options::Context{}).empty());
}

// Use formatted inner types to make sure nested formatting works
template <typename T>
using In = std::array<T, 0>;
struct FormatMap {
  using type = std::map<In<int>, In<double>>;
  static constexpr Options::String help = {"halp"};
  static constexpr const char* const expected = "{[int x0]: [double x0]}";
};
struct FormatVector {
  using type = std::vector<In<int>>;
  static constexpr Options::String help = {"halp"};
  static constexpr const char* const expected = "[[int x0], ...]";
};
struct FormatList {
  using type = std::list<In<int>>;
  static constexpr Options::String help = {"halp"};
  static constexpr const char* const expected = "[[int x0], ...]";
};
struct FormatArray {
  using type = std::array<In<int>, 3>;
  static constexpr Options::String help = {"halp"};
  static constexpr const char* const expected = "[[int x0] x3]";
};
struct FormatPair {
  using type = std::pair<In<int>, In<double>>;
  static constexpr Options::String help = {"halp"};
  static constexpr const char* const expected = "[[int x0], [double x0]]";
};
struct ArrayHash {
  size_t operator()(const std::array<int, 0>& /*unused*/) const noexcept {
    return 0;
  }
};
struct FormatUnorderedMap {
  using type = std::unordered_map<In<int>, In<double>, ArrayHash>;
  static constexpr Options::String help = {"halp"};
  static constexpr const char* const expected = "{[int x0]: [double x0]}";
};

struct ScalarWithLimits {
  using type = int;
  static constexpr Options::String help = "ScalarHelp";
  static type suggested_value() noexcept { return 7; }
  static type lower_bound() noexcept { return 2; }
  static type upper_bound() noexcept { return 8; }
};

struct VectorWithLimits {
  using type = std::vector<int>;
  static constexpr Options::String help = "VectorHelp";
  static size_t lower_bound_on_size() noexcept { return 5; }
  static size_t upper_bound_on_size() noexcept { return 9; }
};

void test_options_format() {
  const auto check = [](auto opt) noexcept {
    using Opt = decltype(opt);
    Options::Parser<tmpl::list<Opt>> opts("");
    INFO("Help string:\n"
         << opts.help() << "\n\nExpected to find:\n"
         << "  type="s + Opt::expected + "\n");
    // Add whitespace to check that we've got the entire type
    CHECK(opts.help().find("type="s + Opt::expected + "\n"s) !=
          std::string::npos);
  };
  check(FormatMap{});
  check(FormatVector{});
  check(FormatList{});
  check(FormatArray{});
  check(FormatPair{});
  check(FormatUnorderedMap{});

  CHECK(Options::Parser<tmpl::list<ScalarWithLimits>>("").help() == R"(
==== Description of expected options:


Options:
  ScalarWithLimits:
    type=int
    suggested=7
    min=2
    max=8
    ScalarHelp

)");
  CHECK(Options::Parser<tmpl::list<VectorWithLimits>>("").help() == R"(
==== Description of expected options:


Options:
  VectorWithLimits:
    type=[int, ...]
    min size=5
    max size=9
    VectorHelp

)");
}

// [[OutputRegex, Failed to convert value to type
// {\[int x0\]: \[double x0\]}:]]
SPECTRE_TEST_CASE("Unit.Options.Format.UnorderedMap.Error", "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<FormatUnorderedMap>> opts("");
  opts.parse("FormatUnorderedMap: X");
  opts.get<FormatUnorderedMap>();
}

// [[OutputRegex, At line 3 column 1:.Unable to correctly parse the
// input file because of a syntax error]]
SPECTRE_TEST_CASE("Unit.Options.bad_colon", "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<>> opts("");
  opts.parse("\n\n:");
}

struct ExplicitObject {
  explicit ExplicitObject() = default;
};

struct ExplicitObjectTag {
  using type = ExplicitObject;
  static constexpr Options::String help = {"halp"};
};
}  // namespace

template <>
struct Options::create_from_yaml<ExplicitObject> {
  template <typename Metavariables>
  static ExplicitObject create(const Options::Option& /*options*/) {
    return ExplicitObject{};
  }
};

namespace {
void test_options_explicit_constructor() {
  Options::Parser<tmpl::list<ExplicitObjectTag>> opts("");
  opts.parse("ExplicitObjectTag:");
  opts.get<ExplicitObjectTag>();
}

struct SuggestedBool {
  using type = bool;
  static type suggested_value() noexcept { return false; }
  constexpr static Options::String help = "halp";
};

void test_options_format_bool() noexcept {
  const auto help = Options::Parser<tmpl::list<SuggestedBool>>("").help();
  CAPTURE(help);
  CHECK(help.find("suggested=false") != std::string::npos);
}

void test_options_input_source() noexcept {
  Options::Parser<tmpl::list<Simple>> parser("");
  const std::string source = "Simple: 3";
  parser.parse(source);
  CHECK(parser.get<Options::InputSource>() == source);
}

void check_for_lines(const std::string& text,
                     const std::vector<std::string>& lines) noexcept {
  CAPTURE(text);
  const std::string search_text = "\n" + text + "\n";
  size_t pos = 0;
  for (const auto& line : lines) {
    CAPTURE(line);
    pos = search_text.find("\n" + line + "\n");
    if (pos == search_text.npos) {
      CHECK(false);
      // After failing to find a line none of the remaining lines will
      // be found after it (because we are at the end of the string),
      // so avoid printing an error for each of them.
      break;
    }
  }
}

struct Alternatives {
  struct A {
    using type = double;
    static constexpr Options::String help = "halp";
  };

  struct B {
    using type = int;
    static constexpr Options::String help = "halp";
  };

  struct C {
    using type = int;
    static constexpr Options::String help = "halp";
  };

  struct D {
    using type = std::vector<int>;
    static constexpr Options::String help = "halp";
  };

  struct E {
    using type = bool;
    static constexpr Options::String help = "halp";
  };

  struct F {
    using type = std::string;
    static constexpr Options::String help = "halp";
  };

  static constexpr Options::String help = "halp";
  Alternatives() = default;

  // [alternatives]
  using options = tmpl::list<
      A, Options::Alternatives<tmpl::list<B>, tmpl::list<C>, tmpl::list<D, E>>,
      F>;
  Alternatives(tmpl::list<A, B, F> /*meta*/, double a, int b, std::string f)
      : a_(a), b_(b), f_(std::move(f)) {}
  Alternatives(tmpl::list<A, C, F> /*meta*/, double a, int c, std::string f)
      : a_(a), c_(c), f_(std::move(f)) {}
  Alternatives(double a, std::vector<int> d, bool e, std::string f)
      : a_(a), d_(std::move(d)), e_(e), f_(std::move(f)) {}
  // [alternatives]

  double a_{-1.0};
  int b_{-1};
  int c_{-1};
  std::vector<int> d_{};
  bool e_{false};
  std::string f_{};
};

struct AlternativesTag {
  using type = Alternatives;
  static constexpr Options::String help = "halp";
};

struct NestedAlternatives {
  struct A {
    using type = int;
    static constexpr Options::String help = "halp";
  };

  struct B {
    using type = double;
    static constexpr Options::String help = "halp";
  };

  struct C {
    using type = std::string;
    static constexpr Options::String help = "halp";
  };

  struct D {
    using type = std::vector<int>;
    static constexpr Options::String help = "halp";
  };

  struct E {
    using type = bool;
    static constexpr Options::String help = "halp";
  };

  static constexpr Options::String help = "halp";
  using options = tmpl::list<Options::Alternatives<
      tmpl::list<A, Options::Alternatives<tmpl::list<B>, tmpl::list<C>>>,
      tmpl::list<D, E>>>;

  NestedAlternatives() = default;
  NestedAlternatives(int a, double b) : a_(a), b_(b) {}
  NestedAlternatives(int a, std::string c) : a_(a), c_(std::move(c)) {}
  NestedAlternatives(std::vector<int> d, bool e) : d_(std::move(d)), e_(e) {}

  int a_{-1};
  double b_{-1.0};
  std::string c_{};
  std::vector<int> d_{};
  bool e_{false};
};

struct NestedAlternativesTag {
  using type = NestedAlternatives;
  static constexpr Options::String help = "halp";
};

void test_options_alternatives() noexcept {
  {
    Options::Parser<tmpl::list<AlternativesTag>> parser("");
    parser.parse(
        "AlternativesTag:\n"
        "  A: 7.8\n"
        "  B: 4\n"
        "  F: first\n");
    const auto result = parser.get<AlternativesTag>();
    CHECK(result.a_ == 7.8);
    CHECK(result.b_ == 4);
    CHECK(result.c_ == -1);
    CHECK(result.d_.empty());
    CHECK(result.e_ == false);
    CHECK(result.f_ == "first");
  }
  {
    Options::Parser<tmpl::list<AlternativesTag>> parser("");
    parser.parse(
        "AlternativesTag:\n"
        "  C: 5\n"
        "  A: 7.8\n"
        "  F: second\n");
    const auto result = parser.get<AlternativesTag>();
    CHECK(result.a_ == 7.8);
    CHECK(result.b_ == -1);
    CHECK(result.c_ == 5);
    CHECK(result.d_.empty());
    CHECK(result.e_ == false);
    CHECK(result.f_ == "second");
  }
  {
    Options::Parser<tmpl::list<AlternativesTag>> parser("");
    parser.parse(
        "AlternativesTag:\n"
        "  A: 7.8\n"
        "  D: [2, 3, 5, 7, 11]\n"
        "  E: true\n"
        "  F: third\n");
    const auto result = parser.get<AlternativesTag>();
    CHECK(result.a_ == 7.8);
    CHECK(result.b_ == -1);
    CHECK(result.c_ == -1);
    CHECK(result.d_ == std::vector<int>{2, 3, 5, 7, 11});
    CHECK(result.e_ == true);
    CHECK(result.f_ == "third");
  }

  {
    Options::Parser<tmpl::list<NestedAlternativesTag>> parser("");
    parser.parse(
        "NestedAlternativesTag:\n"
        "  A: 3\n"
        "  B: 7.8\n");
    const auto result = parser.get<NestedAlternativesTag>();
    CHECK(result.a_ == 3);
    CHECK(result.b_ == 7.8);
    CHECK(result.c_.empty());
    CHECK(result.d_.empty());
    CHECK(result.e_ == false);
  }
  {
    Options::Parser<tmpl::list<NestedAlternativesTag>> parser("");
    parser.parse(
        "NestedAlternativesTag:\n"
        "  C: hi\n"
        "  A: 3\n");
    const auto result = parser.get<NestedAlternativesTag>();
    CHECK(result.a_ == 3);
    CHECK(result.b_ == -1.0);
    CHECK(result.c_ == "hi");
    CHECK(result.d_.empty());
    CHECK(result.e_ == false);
  }
  {
    Options::Parser<tmpl::list<NestedAlternativesTag>> parser("");
    parser.parse(
        "NestedAlternativesTag:\n"
        "  D: [2, 3, 5, 7, 11]\n"
        "  E: true\n");
    const auto result = parser.get<NestedAlternativesTag>();
    CHECK(result.a_ == -1);
    CHECK(result.b_ == -1.0);
    CHECK(result.c_.empty());
    CHECK(result.d_ == std::vector<int>{2, 3, 5, 7, 11});
    CHECK(result.e_ == true);
  }

  // clang-format off
  check_for_lines(Options::Parser<Alternatives::options>("").help(),
                  {"Options:",
                   "  A:",
                   "  EITHER",
                   "    B:",
                   "  OR",
                   "    C:",
                   "  OR",
                   "    D:",
                   "    E:",
                   "  F:"});
  check_for_lines(Options::Parser<NestedAlternatives::options>("").help(),
                  {"Options:",
                   "  EITHER",
                   "    A:",
                   "    EITHER",
                   "      B:",
                   "    OR",
                   "      C:",
                   "  OR",
                   "    D:",
                   "    E:"});
  // clang-format on
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Options", "[Unit][Options]") {
  test_options_empty_success();
  test_options_simple_success();
  test_options_print_long_help();
  test_options_grouped();
  test_options_suggested_followed();
  test_options_suggested_not_followed();
  test_options_bounded_lower_bound();
  test_options_bounded_upper_bound();
  test_options_vector_lower_bound();
  test_options_vector_upper_bound();
  test_options_array_success();
  test_options_zero_array_missing();
  test_options_map_success();
  test_options_map_empty();
  test_options_unordered_map_success();
  test_options_variant();
  test_options_complex_containers();
  test_options_apply();
  test_options_option_context_default_stream();
  test_options_format();
  test_options_explicit_constructor();
  test_options_format_bool();
  test_options_input_source();
  test_options_alternatives();
}
