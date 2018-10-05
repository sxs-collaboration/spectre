// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <list>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ErrorHandling/Error.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TMPL.hpp"

SPECTRE_TEST_CASE("Unit.Options.Empty.success", "[Unit][Options]") {
  Options<tmpl::list<>> opts("");
  opts.parse("");
  // Catch requires us to have at least one CHECK in each test
  // The Unit.Options.Empty.success does not need to check anything
  CHECK(true);
}

// [[OutputRegex, In string:.*At line 1 column 1:.Option 'Option' is not a valid
// option.]]
SPECTRE_TEST_CASE("Unit.Options.Empty.extra", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<>> opts("");
  opts.parse("Option:");
}

// [[OutputRegex, In string:.*'4' does not look like options]]
SPECTRE_TEST_CASE("Unit.Options.Empty.not_map", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<>> opts("");
  opts.parse("4");
}

// [[OutputRegex, In string:.*At line 1 column 30:.Unable to correctly parse
// the input file because of a syntax error]]
SPECTRE_TEST_CASE("Unit.Options.syntax_error", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<>> opts("");
  opts.parse("DomainCreator: CreateInterval:\n"
             "  IsPeriodicIn: [false]");
}

namespace {
struct Simple {
  using type = int;
  static constexpr OptionString help = {"halp"};
};
struct NamedSimple {
  using type = int;
  static std::string name() noexcept { return "SomeName"; }
  static constexpr OptionString help = {"halp"};
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Options.Simple.success", "[Unit][Options]") {
  {
    Options<tmpl::list<Simple>> opts("");
    opts.parse("Simple: -4");
    CHECK(opts.get<Simple>() == -4);
  }
  {
    Options<tmpl::list<NamedSimple>> opts("");
    opts.parse("SomeName: -4");
    CHECK(opts.get<NamedSimple>() == -4);
  }
}

// [[OutputRegex, In string:.*At line 2 column 1:.Option 'Simple' specified
// twice.]]
SPECTRE_TEST_CASE("Unit.Options.Simple.duplicate", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<Simple>> opts("");
  opts.parse("Simple: -4\n"
             "Simple: -3");
  opts.get<Simple>();
}

// [[OutputRegex, In string:.*At line 2 column 1:.Option 'SomeName' specified
// twice.]]
SPECTRE_TEST_CASE("Unit.Options.NamedSimple.duplicate", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<NamedSimple>> opts("");
  opts.parse("SomeName: -4\n"
             "SomeName: -3");
  opts.get<NamedSimple>();
}

// [[OutputRegex, In string:.*You did not specify the option 'Simple']]
SPECTRE_TEST_CASE("Unit.Options.Simple.missing", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<Simple>> opts("");
  opts.parse("");
  opts.get<Simple>();
}

// [[OutputRegex, In string:.*You did not specify the option 'SomeName']]
SPECTRE_TEST_CASE("Unit.Options.NamedSimple.missing", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<NamedSimple>> opts("");
  opts.parse("");
  opts.get<NamedSimple>();
}

// [[OutputRegex, In string:.*While parsing option Simple:.At line 1 column
// 1:.Failed to convert value to type int:]]
SPECTRE_TEST_CASE("Unit.Options.Simple.missing_arg", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<Simple>> opts("");
  opts.parse("Simple:");
  opts.get<Simple>();
}

// [[OutputRegex, In string:.*While parsing option SomeName:.At line 1 column
// 1:.Failed to convert value to type int:]]
SPECTRE_TEST_CASE("Unit.Options.NamedSimple.missing_arg", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<NamedSimple>> opts("");
  opts.parse("SomeName:");
  opts.get<NamedSimple>();
}

// [[OutputRegex, In string:.*While parsing option Simple:.At line 1 column
// 9:.Failed to convert value to type int: 2.3]]
SPECTRE_TEST_CASE("Unit.Options.Simple.invalid", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<Simple>> opts("");
  opts.parse("Simple: 2.3");
  opts.get<Simple>();
}

// [[OutputRegex, In string:.*While parsing option SomeName:.At line 1 column
// 11:.Failed to convert value to type int: 2.3]]
SPECTRE_TEST_CASE("Unit.Options.NamedSimple.invalid", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<NamedSimple>> opts("");
  opts.parse("SomeName: 2.3");
  opts.get<NamedSimple>();
}

namespace {
/// [options_example_scalar_struct]
struct Bounded {
  using type = int;
  static constexpr OptionString help = {
    "Option with bounds and a default value"};
  // These are optional
  static type default_value() { return 3; }
  static type lower_bound() { return 2; }
  static type upper_bound() { return 10; }
};
/// [options_example_scalar_struct]
}  // namespace

SPECTRE_TEST_CASE("Unit.Options.Defaulted.specified", "[Unit][Options]") {
/// [options_example_scalar_parse]
  Options<tmpl::list<Bounded>> opts("Overall help text");
  opts.parse("Bounded: 8");
  CHECK(opts.get<Bounded>() == 8);
/// [options_example_scalar_parse]
}

SPECTRE_TEST_CASE("Unit.Options.Defaulted.defaulted", "[Unit][Options]") {
  Options<tmpl::list<Bounded>> opts("");
  opts.parse("");
  CHECK(opts.get<Bounded>() == 3);
}

// [[OutputRegex, In string:.*While parsing option Bounded:.At line 1 column
// 10:.Value 1 is below the lower bound of 2]]
SPECTRE_TEST_CASE("Unit.Options.Bounded.below", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<Bounded>> opts("");
  opts.parse("Bounded: 1");
  opts.get<Bounded>();
}

SPECTRE_TEST_CASE("Unit.Options.Bounded.lower_bound", "[Unit][Options]") {
  Options<tmpl::list<Bounded>> opts("");
  opts.parse("Bounded: 2");
  CHECK(opts.get<Bounded>() == 2);
}

SPECTRE_TEST_CASE("Unit.Options.Bounded.upper_bound", "[Unit][Options]") {
  Options<tmpl::list<Bounded>> opts("");
  opts.parse("Bounded: 10");
  CHECK(opts.get<Bounded>() == 10);
}

// [[OutputRegex, In string:.*While parsing option Bounded:.At line 1 column
// 10:.Value 11 is above the upper bound of 10]]
SPECTRE_TEST_CASE("Unit.Options.Bounded.above", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<Bounded>> opts("");
  opts.parse("Bounded: 11");
  opts.get<Bounded>();
}

namespace {
struct BadDefault {
  using type = int;
  static constexpr OptionString help = {"halp"};
  static type default_value() { return 3; }
  static type lower_bound() { return 4; }
};
struct NamedBadDefault {
  using type = int;
  static std::string name() noexcept { return "SomeName"; }
  static constexpr OptionString help = {"halp"};
  static type default_value() { return 3; }
  static type lower_bound() { return 4; }
};
}  // namespace

// [[OutputRegex, Checking DEFAULT value for BadDefault:.Value 3 is below the
// lower bound of 4]]
SPECTRE_TEST_CASE("Unit.Options.BadDefault", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<BadDefault>> opts("");
  opts.parse("");
  opts.get<BadDefault>();
}

// [[OutputRegex, Checking DEFAULT value for SomeName:.Value 3 is below the
// lower bound of 4]]
SPECTRE_TEST_CASE("Unit.Options.NamedBadDefault", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<NamedBadDefault>> opts("");
  opts.parse("");
  opts.get<NamedBadDefault>();
}

namespace {
/// [options_example_vector_struct]
struct VectorOption {
  using type = std::vector<int>;
  static constexpr OptionString help = {"A vector with length limits"};
  // These are optional
  static std::string name() noexcept {
    return "Vector";  // defaults to "VectorOption"
  }
  static size_t lower_bound_on_size() { return 2; }
  static size_t upper_bound_on_size() { return 5; }
};
/// [options_example_vector_struct]
}  // namespace

// [[OutputRegex, In string:.*While parsing option Vector:.At line 1 column
// 9:.Value must have at least 2 entries, but 1 were given.]]
SPECTRE_TEST_CASE("Unit.Options.Vector.too_short", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<VectorOption>> opts("");
  opts.parse("Vector: [2]");
  opts.get<VectorOption>();
}

SPECTRE_TEST_CASE("Unit.Options.Vector.lower_bound", "[Unit][Options]") {
  Options<tmpl::list<VectorOption>> opts("");
  opts.parse("Vector: [2,3]");
  CHECK(opts.get<VectorOption>() == (std::vector<int>{2, 3}));
}

SPECTRE_TEST_CASE("Unit.Options.Vector.upper_bound", "[Unit][Options]") {
  Options<tmpl::list<VectorOption>> opts("");
  opts.parse("Vector: [2, 3, 3, 3, 5]");
  CHECK(opts.get<VectorOption>() == (std::vector<int>{2, 3, 3, 3, 5}));
}

// [[OutputRegex, In string:.*While parsing option Vector:.At line 1 column
// 9:.Value must have at most 5 entries, but 6 were given.]]
SPECTRE_TEST_CASE("Unit.Options.Vector.too_long", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<VectorOption>> opts("");
  opts.parse("Vector: [2, 3, 3, 3, 5, 6]");
  opts.get<VectorOption>();
}

// [[OutputRegex, In string:.*While parsing option Vector:.At line 1 column
// 1:.Value must have at least 2 entries, but 0 were given.]]
SPECTRE_TEST_CASE("Unit.Options.Vector.empty_too_short", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<VectorOption>> opts("");
  opts.parse("Vector:");
  opts.get<VectorOption>();
}

namespace {
struct Array {
  using type = std::array<int, 3>;
  static constexpr OptionString help = {"halp"};
};
}  // namespace

// [[OutputRegex, In string:.*While parsing option Array:.At line 1 column
// 8:.Failed to convert value to type \[int x3\]:]]
SPECTRE_TEST_CASE("Unit.Options.Array.too_short", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<Array>> opts("");
  opts.parse("Array: [1, 2]");
  opts.get<Array>();
}

SPECTRE_TEST_CASE("Unit.Options.Array.success", "[Unit][Options]") {
  Options<tmpl::list<Array>> opts("");
  opts.parse("Array: [1,2,3]");
  CHECK(opts.get<Array>() == (std::array<int, 3>{{1, 2, 3}}));
}

// [[OutputRegex, In string:.*While parsing option Array:.At line 1 column
// 8:.Failed to convert value to type \[int x3\]: .1, 2, 3, 4.]]
SPECTRE_TEST_CASE("Unit.Options.Array.too_long", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<Array>> opts("");
  opts.parse("Array: [1, 2, 3, 4]");
  opts.get<Array>();
}

// [[OutputRegex, In string:.*While parsing option Array:.At line 2 column
// 3:.Failed to convert value to type \[int x3\]:.  - 1.  - 2.  -
// 3.  - 4]]
SPECTRE_TEST_CASE("Unit.Options.Array.too_long.formatting", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<Array>> opts("");
  opts.parse("Array:\n"
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
  Options<tmpl::list<Array>> opts("");
  opts.parse("Array:");
  opts.get<Array>();
}

namespace {
struct ZeroArray {
  using type = std::array<int, 0>;
  static constexpr OptionString help = {"halp"};
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Options.ZeroArray.missing", "[Unit][Options]") {
  Options<tmpl::list<ZeroArray>> opts("");
  opts.parse("ZeroArray:");
  opts.get<ZeroArray>();
  // Catch requires us to have at least one CHECK in each test
  // The Unit.Options.ZeroArray.missing does not need to check anything
  CHECK(true);
}

namespace {
struct Map {
  using type = std::map<std::string, int>;
  static constexpr OptionString help = {"halp"};
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Options.Map.success", "[Unit][Options]") {
  Options<tmpl::list<Map>> opts("");
  opts.parse("Map:\n"
             "  A: 3\n"
             "  Z: 2");
  std::map<std::string, int> expected;
  expected.emplace("A", 3);
  expected.emplace("Z", 2);
  CHECK(opts.get<Map>() == expected);
}

SPECTRE_TEST_CASE("Unit.Options.Map.empty", "[Unit][Options]") {
  Options<tmpl::list<Map>> opts("");
  opts.parse("Map:");
  CHECK(opts.get<Map>().empty());
}

// [[OutputRegex, In string:.*While parsing option Map:.At line 1 column
// 6:.Failed to convert value to type {std::string: int}: string]]
SPECTRE_TEST_CASE("Unit.Options.Map.invalid", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<Map>> opts("");
  opts.parse("Map: string");
  opts.get<Map>();
}

// [[OutputRegex, In string:.*While parsing option Map:.At line 2 column
// 6:.Failed to convert value to type {std::string: int}: A: string]]
SPECTRE_TEST_CASE("Unit.Options.Map.invalid_entry", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<Map>> opts("");
  opts.parse("Map:\n"
             "  A: string");
  opts.get<Map>();
}

namespace {
struct UnorderedMap {
  using type = std::unordered_map<std::string, int>;
  static constexpr OptionString help = {"halp"};
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Options.UnorderedMap.success", "[Unit][Options]") {
  Options<tmpl::list<UnorderedMap>> opts("");
  opts.parse("UnorderedMap:\n"
             "  A: 3\n"
             "  Z: 2");
  std::unordered_map<std::string, int> expected;
  expected.emplace("A", 3);
  expected.emplace("Z", 2);
  CHECK(opts.get<UnorderedMap>() == expected);
}

namespace {
template <typename T>
struct Wrapped {
  T data;
};

#define FORWARD_OP(op)                                          \
  template <typename T>                                         \
  bool operator op(const Wrapped<T>& a, const Wrapped<T>& b) {  \
    return a.data op b.data;                                    \
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
struct create_from_yaml<Wrapped<T>> {
  static Wrapped<T> create(const Option& options) {
    return Wrapped<T>{options.parse_as<T>()};
  }
};

namespace {
struct WrapMap {
  using type = std::map<Wrapped<int>, Wrapped<std::string>>;
  static constexpr OptionString help = {"halp"};
};
struct WrapVector {
  using type = std::vector<Wrapped<int>>;
  static constexpr OptionString help = {"halp"};
};
struct WrapList {
  using type = std::list<Wrapped<int>>;
  static constexpr OptionString help = {"halp"};
};
struct WrapArray {
  using type = std::array<Wrapped<int>, 2>;
  static constexpr OptionString help = {"halp"};
};
struct WrapPair {
  using type = std::pair<Wrapped<int>, Wrapped<std::string>>;
  static constexpr OptionString help = {"halp"};
};
struct WrapUnorderedMap {
  using type = std::unordered_map<Wrapped<int>, Wrapped<std::string>>;
  static constexpr OptionString help = {"halp"};
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Options.ComplexContainers", "[Unit][Options]") {
  Options<tmpl::list<WrapMap, WrapVector, WrapList, WrapArray, WrapPair,
                     WrapUnorderedMap>> opts("");
  opts.parse("WrapMap: {1: A, 2: B}\n"
             "WrapVector: [1, 2, 3]\n"
             "WrapList: [1, 2, 3]\n"
             "WrapArray: [1, 2]\n"
             "WrapPair: [1, X]\n"
             "WrapUnorderedMap: {1: A, 2: B}\n");
  CHECK(opts.get<WrapMap>() ==
        (std::map<Wrapped<int>, Wrapped<std::string>>{
          {{1}, {"A"}}, {{2}, {"B"}}}));
  CHECK(opts.get<WrapVector>() == (std::vector<Wrapped<int>>{{1}, {2}, {3}}));
  CHECK(opts.get<WrapList>() == (std::list<Wrapped<int>>{{1}, {2}, {3}}));
  CHECK(opts.get<WrapArray>() == (std::array<Wrapped<int>, 2>{{{1}, {2}}}));
  CHECK(opts.get<WrapPair>() ==
        (std::pair<Wrapped<int>, Wrapped<std::string>>{{1}, {"X"}}));
  CHECK(opts.get<WrapUnorderedMap>() ==
        (std::unordered_map<Wrapped<int>, Wrapped<std::string>>{
          {{1}, {"A"}}, {{2}, {"B"}}}));
}

#ifdef SPECTRE_DEBUG
namespace {
struct Duplicate {
  using type = int;
  static constexpr OptionString help = {"halp"};
};
struct NamedDuplicate {
    using type = int;
  static std::string name() noexcept { return "Duplicate"; }
  static constexpr OptionString help = {"halp"};
};
}  // namespace
#endif  // SPECTRE_DEBUG

// [[OutputRegex, Duplicate option name: Duplicate]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Options.Duplicate", "[Unit][Options]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Options<tmpl::list<Duplicate, NamedDuplicate>> opts("");
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

#ifdef SPECTRE_DEBUG
namespace {
struct TooooooooooooooooooooLong {
  using type = int;
  static constexpr OptionString help = {"halp"};
};
struct NamedTooLong {
  using type = int;
  static std::string name() noexcept { return "TooooooooooooooooooooLong"; }
  static constexpr OptionString help = {"halp"};
};
struct NoHelp {
  using type = int;
  static constexpr OptionString help = {""};
};
struct NamedNoHelp {
  using type = int;
  static std::string name() noexcept { return "NoHelp"; }
  static constexpr OptionString help = {""};
};
struct TooLongHelp {
  using type = int;
  static constexpr OptionString help = {
    "halp halp halp halp halp halp halp halp halp halp halp halp"};
};
struct NamedTooLongHelp {
  using type = int;
  static std::string name() noexcept { return "TooLongHelp"; }
  static constexpr OptionString help = {
    "halp halp halp halp halp halp halp halp halp halp halp halp"};
};
}  // namespace
#endif  // SPECTRE_DEBUG

// [[OutputRegex, The option name TooooooooooooooooooooLong is too long for
// nice formatting]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Options.TooLong", "[Unit][Options]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Options<tmpl::list<TooooooooooooooooooooLong>> opts("");
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The option name TooooooooooooooooooooLong is too long for
// nice formatting]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Options.NamedTooLong", "[Unit][Options]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Options<tmpl::list<NamedTooLong>> opts("");
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, You must supply a help string of non-zero length for NoHelp]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Options.NoHelp", "[Unit][Options]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Options<tmpl::list<NoHelp>> opts("");
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, You must supply a help string of non-zero length for NoHelp]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Options.NamedNoHelp", "[Unit][Options]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Options<tmpl::list<NamedNoHelp>> opts("");
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The help string for TooLongHelp should be less than]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Options.TooLongHelp", "[Unit][Options]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Options<tmpl::list<TooLongHelp>> opts("");
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The help string for TooLongHelp should be less than]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Options.NamedTooLongHelp",
                               "[Unit][Options]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  Options<tmpl::list<NamedTooLongHelp>> opts("");
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

namespace {
struct Apply1 {
  using type = int;
  static constexpr OptionString help = {"halp"};
};
struct Apply2 {
  using type = std::string;
  static constexpr OptionString help = {"halp"};
};
struct Apply3 {
  using type = std::vector<int>;
  static constexpr OptionString help = {"halp"};
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Options.Apply", "[Unit][Options]") {
  Options<tmpl::list<Apply1, Apply2, Apply3>> opts("");
  opts.parse("Apply1: 2\n"
             "Apply2: str\n"
             "Apply3: [1, 2, 3]");
  // We do the checks outside the lambda to make sure it actually gets called.
  std::vector<int> arg1;
  int arg2;
  opts.apply<tmpl::list<Apply3, Apply1>>(
      [&](const auto& a, auto b) {
        arg1 = a;
        arg2 = b;
      });
  CHECK(arg1 == (std::vector<int>{1, 2, 3}));
  CHECK(arg2 == 2);
}

SPECTRE_TEST_CASE("Unit.Options.OptionContext.default_stream",
                  "[Unit][Options]") {
  CHECK(get_output(OptionContext{}).empty());
}

namespace {
// Use formatted inner types to make sure nested formatting works
template <typename T>
using In = std::array<T, 0>;
struct FormatMap {
  using type = std::map<In<int>, In<double>>;
  static constexpr OptionString help = {"halp"};
  static constexpr const char* const expected = "{[int x0]: [double x0]}";
};
struct FormatVector {
  using type = std::vector<In<int>>;
  static constexpr OptionString help = {"halp"};
  static constexpr const char* const expected = "[[int x0], ...]";
};
struct FormatList {
  using type = std::list<In<int>>;
  static constexpr OptionString help = {"halp"};
  static constexpr const char* const expected = "[[int x0], ...]";
};
struct FormatArray {
  using type = std::array<In<int>, 3>;
  static constexpr OptionString help = {"halp"};
  static constexpr const char* const expected = "[[int x0] x3]";
};
struct FormatPair {
  using type = std::pair<In<int>, In<double>>;
  static constexpr OptionString help = {"halp"};
  static constexpr const char* const expected = "[[int x0], [double x0]]";
};
struct ArrayHash {
  size_t operator()(const std::array<int, 0>& /*unused*/) const noexcept {
    return 0;
  }
};
struct FormatUnorderedMap {
  using type = std::unordered_map<In<int>, In<double>, ArrayHash>;
  static constexpr OptionString help = {"halp"};
  static constexpr const char* const expected = "{[int x0]: [double x0]}";
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Options.Format", "[Unit][Options]") {
  const auto check = [](auto opt) noexcept {
    using Opt = decltype(opt);
    Options<tmpl::list<Opt>> opts("");
    INFO(opts.help());
    // Add whitespace to check that we've got the entire type
    CHECK(opts.help().find("  "s + Opt::expected + "\n") != std::string::npos);
  };
  check(FormatMap{});
  check(FormatVector{});
  check(FormatList{});
  check(FormatArray{});
  check(FormatPair{});
  check(FormatUnorderedMap{});
}

// [[OutputRegex, Failed to convert value to type
// {\[int x0\]: \[double x0\]}:]]
SPECTRE_TEST_CASE("Unit.Options.Format.UnorderedMap.Error", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<FormatUnorderedMap>> opts("");
  opts.parse("FormatUnorderedMap: X");
  opts.get<FormatUnorderedMap>();
}

// [[OutputRegex, At line 3 column 1:.Unable to correctly parse the
// input file because of a syntax error]]
SPECTRE_TEST_CASE("Unit.Options.bad_colon", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<>> opts("");
  opts.parse("\n\n:");
}
