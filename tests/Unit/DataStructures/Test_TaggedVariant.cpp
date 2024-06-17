// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <variant>

#include "DataStructures/TaggedVariant.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

#if (not defined(__clang__) and __GNUC__ < 12) or \
    (defined(__clang__) and __clang__ < 15)
#define LIMITED_CONSTEXPR
#endif

namespace {
namespace Tags {
struct Int {
  using type = int;
};

template <typename T>
struct Templated {
  using type = T;
};

struct MoveOnly {
  using type = std::unique_ptr<int>;
};
}  // namespace Tags

struct CreatedStruct {
  struct Opt1 {
    static constexpr Options::String help = "First option";
    using type = int;
  };
  struct Opt2 {
    static constexpr Options::String help = "Second option";
    using type = int;
  };
  static constexpr Options::String help = "The struct";
  using options = tmpl::list<Opt1, Opt2>;
  CreatedStruct() = default;
  CreatedStruct(const int opt1_in, const int opt2_in)
      : opt1(opt1_in), opt2(opt2_in) {}
  int opt1{};
  int opt2{};
};

bool operator==(const CreatedStruct& a, const CreatedStruct& b) {
  return a.opt1 == b.opt1 and a.opt2 == b.opt2;
}

namespace OptionTags {
struct Integer {
  static constexpr Options::String help = "An integer";
  using type = int;
};

struct Struct {
  static constexpr Options::String help = "A struct";
  using type = CreatedStruct;
};
}  // namespace OptionTags

constexpr variants::TaggedVariant<Tags::Int> simple_variant(
    std::in_place_type<Tags::Int>, 4);
static_assert(simple_variant.index() == 0);
static_assert(not simple_variant.valueless_by_exception());
static_assert(holds_alternative<Tags::Int>(simple_variant));
static_assert(get<Tags::Int>(simple_variant) == 4);

// [construct single]
constexpr variants::TaggedVariant<Tags::Int> simple_variant2(4);
// [construct single]
static_assert(simple_variant2.index() == 0);
static_assert(not simple_variant2.valueless_by_exception());
static_assert(holds_alternative<Tags::Int>(simple_variant2));
static_assert(get<Tags::Int>(simple_variant2) == 4);
static_assert(simple_variant == simple_variant2);

static_assert(visit([](const std::pair<tmpl::type_<Tags::Int>, const int&>&
                           entry) { return entry.second; },
                    simple_variant) == 4);

// [construct in_place_type]
constexpr variants::TaggedVariant<Tags::Int, Tags::Templated<double>>
    int_variant(std::in_place_type<Tags::Int>, 1);
// [construct in_place_type]
static_assert(int_variant.index() == 0);
static_assert(not int_variant.valueless_by_exception());
static_assert(holds_alternative<Tags::Int>(int_variant));
static_assert(not holds_alternative<Tags::Templated<double>>(int_variant));
static_assert(get<Tags::Int>(int_variant) == 1);
static_assert(visit(
    [](const auto& entry) {
      return std::is_same_v<typename std::decay_t<decltype(entry)>::first_type,
                            tmpl::type_<Tags::Int>> and
             entry.second == 1;
    },
    int_variant));

constexpr variants::TaggedVariant<Tags::Int, Tags::Templated<double>>
    double_variant(std::in_place_type<Tags::Templated<double>>, 4.56);
static_assert(double_variant.index() == 1);
static_assert(not double_variant.valueless_by_exception());
static_assert(holds_alternative<Tags::Templated<double>>(double_variant));
static_assert(not holds_alternative<Tags::Int>(double_variant));
static_assert(get<Tags::Templated<double>>(double_variant) == 4.56);
static_assert(visit(
    [](const auto& entry) {
      return std::is_same_v<typename std::decay_t<decltype(entry)>::first_type,
                            tmpl::type_<Tags::Templated<double>>> and
             entry.second == 4.56;
    },
    double_variant));

static_assert(visit(
    [](const auto& entry1, const auto& entry2) {
      return std::is_same_v<typename std::decay_t<decltype(entry1)>::first_type,
                            tmpl::type_<Tags::Int>> and
             std::is_same_v<typename std::decay_t<decltype(entry2)>::first_type,
                            tmpl::type_<Tags::Templated<double>>> and
             entry1.second == 1 and entry2.second == 4.56;
    },
    int_variant, double_variant));

constexpr variants::TaggedVariant<Tags::Int, Tags::Templated<double>>
    double_variant2(std::in_place_type<Tags::Templated<double>>, -1.23);

// NOLINTBEGIN(misc-redundant-expression)
static_assert(int_variant == int_variant);
static_assert(not(int_variant != int_variant));
static_assert(not(int_variant < int_variant));
static_assert(not(int_variant > int_variant));
static_assert(int_variant <= int_variant);
static_assert(int_variant >= int_variant);
// NOLINTEND(misc-redundant-expression)

static_assert(not(int_variant == double_variant));
static_assert(int_variant != double_variant);
static_assert(int_variant < double_variant);
static_assert(not(int_variant > double_variant));
static_assert(int_variant <= double_variant);
static_assert(not(int_variant >= double_variant));

static_assert(not(int_variant == double_variant2));
static_assert(int_variant != double_variant2);
static_assert(int_variant < double_variant2);
static_assert(not(int_variant > double_variant2));
static_assert(int_variant <= double_variant2);
static_assert(not(int_variant >= double_variant2));

static_assert(not(double_variant == double_variant2));
static_assert(double_variant != double_variant2);
static_assert(not(double_variant < double_variant2));
static_assert(double_variant > double_variant2);
static_assert(not(double_variant <= double_variant2));
static_assert(double_variant >= double_variant2);

constexpr variants::TaggedVariant<Tags::Int, Tags::Templated<double>>
    default_variant{};
static_assert(holds_alternative<Tags::Int>(default_variant));

constexpr variants::TaggedVariant<Tags::Int, Tags::Templated<double>>
    converted_variant = variants::TaggedVariant<Tags::Templated<double>>(4.56);
static_assert(converted_variant == double_variant);
static_assert([]() {
  variants::TaggedVariant<Tags::Int, Tags::Templated<double>> variant(
      std::in_place_type<Tags::Templated<double>>, 4.56);
  const variants::TaggedVariant<Tags::Int, Tags::Templated<double>>
      assigned_variant = std::move(variant);
  return converted_variant == assigned_variant;
}());
static_assert([]() {
  variants::TaggedVariant<Tags::Templated<double>> variant(4.56);
  const variants::TaggedVariant<Tags::Int, Tags::Templated<double>>
      converted_variant2 = std::move(variant);
  return converted_variant == converted_variant2;
}());

static_assert([]() {
  // [convert]
  variants::TaggedVariant<Tags::Int, Tags::Templated<double>> input(
      std::in_place_type<Tags::Int>, 4);
  variants::TaggedVariant<Tags::Templated<double>, Tags::Templated<int>,
                          Tags::Int>
      complex_converted(std::move(input));
  return get<Tags::Int>(complex_converted) == 4;
  // [convert]
}());

static_assert([]() {
  variants::TaggedVariant<Tags::Int, Tags::Templated<double>> variant(
      std::in_place_type<Tags::Templated<double>>, 1.23);
  decltype(auto) lvalue_mut = get<Tags::Templated<double>>(variant);
  decltype(auto) lvalue_const =
      get<Tags::Templated<double>>(std::as_const(variant));
  decltype(auto) rvalue_mut = get<Tags::Templated<double>>(std::move(variant));
  decltype(auto) rvalue_const =
  // NOLINTNEXTLINE(performance-move-const-arg)
      get<Tags::Templated<double>>(std::move(std::as_const(variant)));
  decltype(auto) if_mut = get_if<Tags::Templated<double>>(&variant);
  decltype(auto) if_const =
      get_if<Tags::Templated<double>>(&std::as_const(variant));

  static_assert(std::is_same_v<decltype(lvalue_mut), double&>);
  static_assert(std::is_same_v<decltype(lvalue_const), const double&>);
  static_assert(std::is_same_v<decltype(rvalue_mut), double&&>);
  static_assert(std::is_same_v<decltype(rvalue_const), const double&&>);
  static_assert(std::is_same_v<decltype(if_mut), double*>);
  static_assert(std::is_same_v<decltype(if_const), const double*>);

  variants::TaggedVariant<Tags::Int, Tags::Templated<double>>* const null_mut =
      nullptr;
  const variants::TaggedVariant<Tags::Int, Tags::Templated<double>>* const
      null_const = nullptr;

  return lvalue_mut == 1.23 and &lvalue_mut == &lvalue_const and
         &lvalue_mut == &rvalue_mut and &lvalue_mut == &rvalue_const and
         &lvalue_mut == if_mut and &lvalue_mut == if_const and
         get_if<Tags::Int>(&variant) == nullptr and
         get_if<Tags::Int>(&std::as_const(variant)) == nullptr and
         get_if<Tags::Int>(null_mut) == nullptr and
         get_if<Tags::Int>(null_const) == nullptr;
}());

constexpr bool test_emplace() {
  // [emplace]
  variants::TaggedVariant<Tags::Int, Tags::Templated<double>> variant{};
  decltype(auto) emplace_result =
      variant.emplace<Tags::Templated<double>>(1.23);
  // [emplace]
  static_assert(std::is_same_v<decltype(emplace_result), double&>);
  return variant.index() == 1 and emplace_result == 1.23 and
         &emplace_result == &get<Tags::Templated<double>>(variant);
}
#ifndef LIMITED_CONSTEXPR
// Tested at runtime below
static_assert(test_emplace());
#endif

static_assert([]() {
  variants::TaggedVariant<Tags::Int> v{};
  visit(
      []<typename Entry>(const Entry& /*entry*/) {
        static_assert(
            std::is_same_v<Entry, std::pair<tmpl::type_<Tags::Int>, int&>>);
      },
      v);
  return true;
}());
static_assert([]() {
  const variants::TaggedVariant<Tags::Int> v{};
  visit(
      []<typename Entry>(const Entry& /*entry*/) {
        static_assert(
            std::is_same_v<Entry,
                           std::pair<tmpl::type_<Tags::Int>, const int&>>);
      },
      v);
  return true;
}());
static_assert([]() {
  variants::TaggedVariant<Tags::Int> v{};
  visit(
      []<typename Entry>(const Entry& /*entry*/) {
        static_assert(
            std::is_same_v<Entry, std::pair<tmpl::type_<Tags::Int>, int&&>>);
      },
      std::move(v));
  return true;
}());
static_assert([]() {
  const variants::TaggedVariant<Tags::Int> v{};
  visit(
      []<typename Entry>(const Entry& /*entry*/) {
        static_assert(
            std::is_same_v<Entry,
                           std::pair<tmpl::type_<Tags::Int>, const int&&>>);
      },
      std::move(v));  // NOLINT(performance-move-const-arg)
  return true;
}());

static_assert([]() {
  variants::TaggedVariant<Tags::Int> v1{};
  variants::TaggedVariant<Tags::Templated<double>> v2{};
  visit(
      []<typename Entry1, typename Entry2>(const Entry1& /*entry1*/,
                                           const Entry2& /*entry2*/) {
        static_assert(
            std::is_same_v<Entry1, std::pair<tmpl::type_<Tags::Int>, int&>>);
        static_assert(
            std::is_same_v<
                Entry2,
                std::pair<tmpl::type_<Tags::Templated<double>>, double&&>>);
      },
      v1, std::move(v2));
  return true;
}());

static_assert(
    visit<double>([](const auto& entry) { return entry.second; },
                  variants::TaggedVariant<Tags::Int, Tags::Templated<double>>(
                      std::in_place_type<Tags::Templated<double>>, 1.23)) ==
    1.23);
// Check this compiles
static_assert(
    (visit<void>([](const auto& entry) { return entry.second; },
                 variants::TaggedVariant<Tags::Int, Tags::Templated<double>>(
                     std::in_place_type<Tags::Templated<double>>, 1.23)),
     true));

static_assert(variants::visit([]() { return true; }));
static_assert(variants::visit<bool>([]() { return true; }));

struct DerivedFromVariant : variants::TaggedVariant<Tags::Int> {};
static_assert(variants::visit([](const auto& /*unused*/) { return true; },
                              DerivedFromVariant{}));

constexpr bool test_swap() {
  variants::TaggedVariant<Tags::Int, Tags::Templated<double>> v1(
      std::in_place_type<Tags::Int>, 5);
  variants::TaggedVariant<Tags::Int, Tags::Templated<double>> v2(
      std::in_place_type<Tags::Templated<double>>, 1.23);
  v1.swap(v2);
  if (get<Tags::Templated<double>>(v1) != 1.23 or get<Tags::Int>(v2) != 5) {
    return false;
  }
  using std::swap;
  swap(v1, v2);
  return get<Tags::Int>(v1) == 5 and get<Tags::Templated<double>>(v2) == 1.23;
}
#ifndef LIMITED_CONSTEXPR
// Tested at runtime below
static_assert(test_swap());
#endif

struct ValuelessCausingType {
  ValuelessCausingType() = default;
  [[noreturn]] ValuelessCausingType(const ValuelessCausingType& /*other*/) {
    throw 0;
  }
  ValuelessCausingType& operator=(const ValuelessCausingType&) = delete;
  ~ValuelessCausingType() = default;
};

bool operator==(const ValuelessCausingType& /*a*/,
                const ValuelessCausingType& /*b*/) {
  return true;
}
bool operator<(const ValuelessCausingType& /*a*/,
               const ValuelessCausingType& /*b*/) {
  return false;
}

struct ValuelessCausingTag {
  using type = ValuelessCausingType;
};

SPECTRE_TEST_CASE("Unit.DataStructures.TaggedVariant",
                  "[DataStructures][Unit]") {
  // Test stuff that can't be done in constexpr

  // errors
  {
    const variants::TaggedVariant<Tags::Int, Tags::Templated<double>> variant(
        std::in_place_type<Tags::Int>, 3);
    CHECK_THROWS_AS(get<Tags::Templated<double>>(variant),
                    std::bad_variant_access);
  }

  // Repeated type
  {
    const variants::TaggedVariant<Tags::Int, Tags::Templated<int>>
        repeated_int_variant(std::in_place_type<Tags::Templated<int>>, 4);
    CHECK(repeated_int_variant.index() == 1);
    CHECK(not repeated_int_variant.valueless_by_exception());
    CHECK(holds_alternative<Tags::Templated<int>>(repeated_int_variant));
    CHECK(not holds_alternative<Tags::Int>(repeated_int_variant));
    CHECK(get<Tags::Templated<int>>(repeated_int_variant) == 4);
    CHECK_THROWS_AS(get<Tags::Int>(repeated_int_variant),
                    std::bad_variant_access);
  }

  // move-only types (easier at runtime)
  {
    using Variant = variants::TaggedVariant<Tags::Int, Tags::MoveOnly>;
    const Variant v1(std::in_place_type<Tags::MoveOnly>,
                     std::make_unique<int>(1));
    CHECK(*get<Tags::MoveOnly>(v1) == 1);
    Variant v2(
        variants::TaggedVariant<Tags::MoveOnly>(std::make_unique<int>(2)));
    CHECK(*get<Tags::MoveOnly>(v2) == 2);
    Variant v3{};
    v3 = variants::TaggedVariant<Tags::MoveOnly>(std::make_unique<int>(3));
    CHECK(*get<Tags::MoveOnly>(v3) == 3);
    Variant v4(std::move(v3));
    CHECK(*get<Tags::MoveOnly>(v4) == 3);
    Variant v5{};
    v5 = std::move(v4);
    CHECK(*get<Tags::MoveOnly>(v5) == 3);
    Variant v6{};
    v6.emplace<Tags::MoveOnly>(std::make_unique<int>(6));
    CHECK(*get<Tags::MoveOnly>(v6) == 6);

    visit(
        []<typename Tag>(const std::pair<tmpl::type_<Tag>,
                                         const typename Tag::type&>& entry) {
          CHECK(std::is_same_v<Tag, Tags::MoveOnly>);
          if constexpr (std::is_same_v<Tag, Tags::MoveOnly>) {
            CHECK(*entry.second == 1);
          }
        },
        v1);
    visit(
        []<typename Tag>(
            const std::pair<tmpl::type_<Tag>, typename Tag::type&&>& entry) {
          CHECK(std::is_same_v<Tag, Tags::MoveOnly>);
          if constexpr (std::is_same_v<Tag, Tags::MoveOnly>) {
            CHECK(*entry.second == 6);
          }
        },
        std::move(v6));
  }

  // valueless_by_exception
  {
    variants::TaggedVariant<Tags::Int, ValuelessCausingTag> valueless_variant{};
    CHECK_THROWS_AS(
        (valueless_variant =
             variants::TaggedVariant<Tags::Int, ValuelessCausingTag>(
                 std::in_place_type<ValuelessCausingTag>)),
        int);
    CHECK(valueless_variant.valueless_by_exception());
    CHECK(valueless_variant.index() == std::variant_npos);
    CHECK(not holds_alternative<Tags::Int>(valueless_variant));
    CHECK(not holds_alternative<ValuelessCausingTag>(valueless_variant));
    CHECK_THROWS_AS(get<Tags::Int>(valueless_variant), std::bad_variant_access);
    CHECK_THROWS_AS(visit([](const auto& /*unused*/) {}, valueless_variant),
                    std::bad_variant_access);

    variants::TaggedVariant<Tags::Int, ValuelessCausingTag> good_variant(
        std::in_place_type<Tags::Int>, 5);

    CHECK(valueless_variant == valueless_variant);
    CHECK_FALSE(valueless_variant != valueless_variant);
    CHECK_FALSE(valueless_variant < valueless_variant);
    CHECK_FALSE(valueless_variant > valueless_variant);
    CHECK(valueless_variant <= valueless_variant);
    CHECK(valueless_variant >= valueless_variant);

    CHECK_FALSE(good_variant == valueless_variant);
    CHECK(good_variant != valueless_variant);
    CHECK_FALSE(good_variant < valueless_variant);
    CHECK(good_variant > valueless_variant);
    CHECK_FALSE(good_variant <= valueless_variant);
    CHECK(good_variant >= valueless_variant);

    CHECK_FALSE(valueless_variant == good_variant);
    CHECK(valueless_variant != good_variant);
    CHECK(valueless_variant < good_variant);
    CHECK_FALSE(valueless_variant > good_variant);
    CHECK(valueless_variant <= good_variant);
    CHECK_FALSE(valueless_variant >= good_variant);
  }

  // hashing
  {
    using Variant = variants::TaggedVariant<Tags::Int, Tags::Templated<double>>;
    const Variant v1(std::in_place_type<Tags::Int>, 1);
    const Variant v2(std::in_place_type<Tags::Templated<double>>, 2.0);
    const std::hash<Variant> h{};
    CHECK(h(v1) != h(v2));
  }

  // visit example
  {
    // clang-tidy is complaining about a move in a loop.  Presumably
    // the "do while (false)" in the CHECK macro.
    // NOLINTBEGIN(bugprone-use-after-move)
    // [visit]
    const variants::TaggedVariant<Tags::Int, Tags::Templated<double>>
        visit_example(std::in_place_type<Tags::Int>, 5);
    CHECK(visit(
              []<typename Tag>(
                  const std::pair<tmpl::type_<Tag>, const typename Tag::type&>&
                      entry) {
                if constexpr (std::is_same_v<Tag, Tags::Int>) {
                  // Deduced return types must match for all tags if
                  // no template argument is given to visit.
                  return static_cast<double>(entry.second + 1);
                } else {
                  static_assert(std::is_same_v<Tag, Tags::Templated<double>>);
                  return entry.second;
                }
              },
              visit_example) == 6.0);
    variants::TaggedVariant<Tags::Int, Tags::Templated<double>> visit_example2(
        std::in_place_type<Tags::Templated<double>>, 1.23);
    CHECK(visit<double>(
              []<typename Tag>(const std::pair<tmpl::type_<Tag>,
                                               typename Tag::type&&>& entry) {
                if constexpr (std::is_same_v<Tag, Tags::Int>) {
                  // Implicitly converted to double because of
                  // template argument of visit.
                  return entry.second + 1;
                } else {
                  static_assert(std::is_same_v<Tag, Tags::Templated<double>>);
                  return entry.second;
                }
              },
              std::move(visit_example2)) == 1.23);
    // [visit]
    // NOLINTEND(bugprone-use-after-move)
  }

  test_serialization(
      variants::TaggedVariant<Tags::Int, Tags::Templated<double>>(
          std::in_place_type<Tags::Templated<double>>, 1.23));

  // parsing
  {
    using Variant =
        variants::TaggedVariant<OptionTags::Integer, OptionTags::Struct>;
    CHECK(TestHelpers::test_creation<Variant>("Integer: 5") ==
          Variant(std::in_place_type<OptionTags::Integer>, 5));
    CHECK(TestHelpers::test_creation<Variant>("Struct:\n"
                                              "  Opt1: 1\n"
                                              "  Opt2: 3\n") ==
          Variant(std::in_place_type<OptionTags::Struct>, 1, 3));
  }

#ifdef LIMITED_CONSTEXPR
  // Tested in static_asserts for other compilers.
  CHECK(test_emplace());
  CHECK(test_swap());
#endif
}
}  // namespace
