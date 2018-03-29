// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace PUP {
class er;
}

namespace {
struct name {
  using type = std::string;
};

struct age {
  using type = int;
};
struct email {
  using type = std::string;
};

struct not_streamable {
  explicit not_streamable(int /*unused*/) {}
  not_streamable() = default;
  // clang-tidy: google-runtime-references
  void pup(PUP::er& /*p*/) {}  // NOLINT
};

constexpr bool operator==(const not_streamable& /*unused*/,
                          const not_streamable& /*unused*/) noexcept {
  return true;
}

struct not_streamable_tag {
  using type = not_streamable;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.TaggedTuple", "[Utilities][Unit]") {
  /// [construction_example]
  tuples::TaggedTuple<name, age, email, not_streamable_tag> test(
      "bla", 17, "bla@bla.bla", 0);
  /// [construction_example]
  static_assert(tuples::TaggedTuple<name, age, email>::size() == 3,
                "Failed to test size of TaggedTuple");
  {
    std::stringstream ss;
    ss << test;
    CHECK(ss.str() == "(bla, 17, bla@bla.bla, NOT STREAMABLE)");
  }
  CHECK(test.size() == 4);
  /// [get_example]
  CHECK("bla" == tuples::get<name>(test));
  CHECK(17 == tuples::get<age>(test));
  CHECK("bla@bla.bla" == tuples::get<email>(test));
  /// [get_example]
  auto& name_temp = tuples::get<name>(test);
  name_temp = "Dennis";
  CHECK(tuples::get<name>(test) == "Dennis");
  const auto& name_temp2 = tuples::get<name>(test);
  CHECK(name_temp2 == "Dennis");
  auto name_temp3 = std::move(tuples::get<name>(test));
  CHECK(name_temp3 == "Dennis");

  tuples::get<name>(test) = "Eamonn";
  const auto test2 = test;
  CHECK(17 == tuples::get<age>(test2));

  {
    std::stringstream ss;
    ss << tuples::TaggedTuple<>{};
    CHECK(ss.str() == "()");
  }

  test_serialization(test2);
}

namespace {

int global_of_no_default = 0;
int global_time_mock = 0;

struct empty_base {};
bool operator==(const empty_base& /*unused*/,
                const empty_base& /*unused*/) noexcept {
  return true;
}
struct no_default {
  no_default() = delete;
  explicit no_default(int i) { global_of_no_default = i; }
};
bool operator==(const no_default& /*unused*/,
                const no_default& /*unused*/) noexcept {
  return true;
}

namespace tags {
struct empty_base {
  using type = ::empty_base;
};
struct no_default {
  using type = ::no_default;
};
struct no_default2 {
  using type = ::no_default;
};
}  // namespace tags

namespace tags {

struct Int {
  using type = int;
};
struct Int1 {
  using type = int;
};
struct Int2 {
  using type = int;
};

struct Short0 {
  using type = short;
};
}  // namespace tags

namespace {
template <class Tuple, class... Value>
Tuple test_impl(Value&&... value) {
  Tuple tuple(std::forward<Value>(value)...);
  return tuple;
}

template <class Tag, class Tuple, class... Values>
void check_get(Tuple& t, Values&&... values) {
  CHECK((tuples::get<Tag>(static_cast<Tuple&>(t)) ==
         typename Tag::type{values...}));
  CHECK(tuples::get<Tag>(static_cast<const Tuple&>(t)) ==
        typename Tag::type{values...});
  CHECK(tuples::get<Tag>(static_cast<Tuple&&>(t)) ==
        typename Tag::type{values...});
  CHECK(tuples::get<Tag>(static_cast<const Tuple&&>(t)) ==
        typename Tag::type{values...});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.TaggedTuple.NoTag", "[Unit][Utilities]") {
  tuples::TaggedTuple<> dummy;
  tuples::TaggedTuple<> dummy1;
  dummy.swap(dummy1);
  // Catch requires us to have at least one CHECK in each test
  // The Unit.Utilities.TaggedTuple.NoTag does not need to check anything
  CHECK(true);
}

SPECTRE_TEST_CASE("Unit.Utilities.TaggedTuple.SingleTag", "[Unit][Utilities]") {
  const int i = 0;
  {
    auto t = test_impl<tuples::TaggedTuple<tags::Int>>(1);
    CHECK(tuples::get<tags::Int>(t) == 1);
  }
  {
    const auto t = test_impl<tuples::TaggedTuple<tags::Int>>(1);
    CHECK(tuples::get<tags::Int>(t) == 1);
  }

  {
    auto t = test_impl<tuples::TaggedTuple<tags::Int>>(1);
    auto t2 = tuples::get<tags::Int>(std::move(t));
    CHECK(t2 == 1);
  }
  {
    const auto t = test_impl<tuples::TaggedTuple<tags::Int>>(1);
    CHECK(tuples::get<tags::Int>(
              static_cast<const tuples::TaggedTuple<tags::Int>&&>(t)) == 1);
  }

  {
    auto t = test_impl<tuples::TaggedTuple<tags::Int>>(i);
    CHECK(tuples::get<tags::Int>(t) == i);
  }
  {
    const auto t = test_impl<tuples::TaggedTuple<tags::Int>>(i);
    CHECK(tuples::get<tags::Int>(t) == i);
  }

  {
    auto t = test_impl<tuples::TaggedTuple<tags::Int>>(i);
    auto t2 = tuples::get<tags::Int>(std::move(t));
    CHECK(t2 == i);
  }
  {
    const auto t = test_impl<tuples::TaggedTuple<tags::Int>>(i);
    CHECK(tuples::get<tags::Int>(
              static_cast<const tuples::TaggedTuple<tags::Int>&&>(t)) == i);
  }
}

SPECTRE_TEST_CASE("Unit.Utilities.TaggedTuple.Ebo", "[Unit][Utilities]") {
  const int i = 1;
  test_impl<tuples::TaggedTuple<tags::no_default>>(i);
  CHECK(global_of_no_default == 1);

  auto t =
      test_impl<tuples::TaggedTuple<tags::no_default, tags::no_default2,
                                    tags::empty_base>>(10, 100, empty_base{});
  CHECK((global_of_no_default == 10 or global_of_no_default == 100));
  check_get<tags::no_default>(t, 0);
  CHECK(global_of_no_default == 0);
  check_get<tags::no_default2>(t, 0);
  CHECK(global_of_no_default == 0);

  auto t2 = test_impl<tuples::TaggedTuple<tags::empty_base>>();
  check_get<tags::empty_base>(t2);
}

// C++17 Draft 23.5.3.1 Construction
struct non_copyable {
  explicit non_copyable() = default;
  // clang-tidy: copy constructor should not be explicit
  explicit non_copyable(non_copyable const&) = default;  // NOLINT
  non_copyable& operator=(non_copyable const&) = delete;
  // clang-tidy: move constructor should not be explicit
  explicit non_copyable(non_copyable&&) = default;  // NOLINT
  non_copyable& operator=(non_copyable&&) = default;
  ~non_copyable() = default;
};

namespace tags {
struct non_copyable {
  using type = ::non_copyable;
};
}  // namespace tags

SPECTRE_TEST_CASE("Unit.Utilities.TaggedTuple.construction",
                  "[Unit][Utilities]") {
  {
    // Test copy and move constructors
    tuples::TaggedTuple<tags::Int, tags::Short0> t0{2, 9};
    CHECK(tuples::get<tags::Int>(t0) == 2);
    CHECK(tuples::get<tags::Short0>(t0) == 9);
    tuples::TaggedTuple<tags::Int, tags::Short0> t1(t0);
    CHECK(tuples::get<tags::Int>(t1) == 2);
    CHECK(tuples::get<tags::Short0>(t1) == 9);
    CHECK(t1 == t0);
    tuples::TaggedTuple<tags::Int, tags::Short0> t2(std::move(t0));
    CHECK(tuples::get<tags::Int>(t2) == 2);
    CHECK(tuples::get<tags::Short0>(t2) == 9);
    CHECK(t2 == t1);
  }
  {
    // Construct one tuple from another where the types are convertible
    // implicit conversion constructors
    tuples::TaggedTuple<tags::Int, tags::Short0> t0{2, 9};
    CHECK(tuples::get<tags::Int>(t0) == 2);
    CHECK(tuples::get<tags::Short0>(t0) == 9);
    tuples::TaggedTuple<tags::Int2, tags::Int1> t1(t0);
    CHECK(tuples::get<tags::Int2>(t1) == 2);
    CHECK(tuples::get<tags::Int1>(t1) == 9);
    CHECK(tuples::get<tags::Int>(t0) == 2);
    CHECK(tuples::get<tags::Short0>(t0) == 9);
    tuples::TaggedTuple<tags::Int2, tags::Int1> t2(std::move(t0));
    CHECK(tuples::get<tags::Int2>(t2) == 2);
    CHECK(tuples::get<tags::Int1>(t2) == 9);
    // explicit constructors
    tuples::TaggedTuple<tags::non_copyable, tags::Int, tags::Short0> t3{
        non_copyable{}, 2, 9};
    CHECK(tuples::get<tags::Int>(t3) == 2);
    CHECK(tuples::get<tags::Short0>(t3) == 9);
    tuples::TaggedTuple<tags::non_copyable, tags::Int, tags::Int1> t4{t3};
    CHECK(tuples::get<tags::Int>(t4) == 2);
    CHECK(tuples::get<tags::Int1>(t4) == 9);
    tuples::TaggedTuple<tags::non_copyable, tags::Int, tags::Int1> t5{
        std::move(t3)};
    CHECK(tuples::get<tags::Int>(t5) == 2);
    CHECK(tuples::get<tags::Int1>(t5) == 9);
  }
}

// C++17 Draft 23.5.3.2 Assignment
SPECTRE_TEST_CASE("Unit.Utilities.TaggedTuple.assignment",
                  "[Unit][Utilities]") {
  {
    // Assignment from same type of TaggedTuple
    tuples::TaggedTuple<tags::Int, tags::Short0> t0{2, 9};
    CHECK(tuples::get<tags::Int>(t0) == 2);
    CHECK(tuples::get<tags::Short0>(t0) == 9);
    tuples::TaggedTuple<tags::Int, tags::Short0> t1{3, 4};
    CHECK(tuples::get<tags::Int>(t1) == 3);
    CHECK(tuples::get<tags::Short0>(t1) == 4);
    t1 = t0;
    CHECK(tuples::get<tags::Int>(t1) == 2);
    CHECK(tuples::get<tags::Short0>(t1) == 9);
    tuples::TaggedTuple<tags::Int, tags::Short0> t2{3, 4};
    CHECK(tuples::get<tags::Int>(t2) == 3);
    CHECK(tuples::get<tags::Short0>(t2) == 4);
    t2 = std::move(t0);
    CHECK(tuples::get<tags::Int>(t2) == 2);
    CHECK(tuples::get<tags::Short0>(t2) == 9);
    tuples::TaggedTuple<> t3{};
    tuples::TaggedTuple<> t4{};
    t3 = t4;
  }
  {
    // Assignment from different type of TaggedTuple
    tuples::TaggedTuple<tags::Int, tags::Short0> t0{2, 9};
    CHECK(tuples::get<tags::Int>(t0) == 2);
    CHECK(tuples::get<tags::Short0>(t0) == 9);
    tuples::TaggedTuple<tags::Int2, tags::Int1> t1(3, 4);
    CHECK(tuples::get<tags::Int2>(t1) == 3);
    CHECK(tuples::get<tags::Int1>(t1) == 4);
    t1 = t0;
    CHECK(tuples::get<tags::Int2>(t1) == 2);
    CHECK(tuples::get<tags::Int1>(t1) == 9);
    tuples::TaggedTuple<tags::Int2, tags::Int1> t2(3, 4);
    CHECK(tuples::get<tags::Int2>(t2) == 3);
    CHECK(tuples::get<tags::Int1>(t2) == 4);
    t2 = std::move(t0);
    CHECK(tuples::get<tags::Int2>(t2) == 2);
    CHECK(tuples::get<tags::Int1>(t2) == 9);
  }
}

// C++17 Draft 23.5.3.8 Relational operators
struct NotNoExceptCompare {
  explicit NotNoExceptCompare(int v) : v_(v) {}

  int v_{0};
};
bool operator==(NotNoExceptCompare const& lhs,
                NotNoExceptCompare const& rhs) noexcept(false) {
  return lhs.v_ == rhs.v_;
}
bool operator!=(NotNoExceptCompare const& lhs,
                NotNoExceptCompare const& rhs) noexcept(noexcept(lhs == rhs)) {
  return not(lhs == rhs);
}

struct timed_compare {
  explicit timed_compare(int v) : v_(v) {}
  int v_{0};
};
bool operator==(timed_compare const& lhs, timed_compare const& rhs) noexcept {
  global_time_mock++;
  return lhs.v_ == rhs.v_;
}

namespace relational_tags {
struct Int0 {
  using type = int;
};
struct Int1 {
  using type = int;
};
struct Int2 {
  using type = int;
};

struct NotNoExceptCompare {
  using type = ::NotNoExceptCompare;
};

struct TimedCompare0 {
  using type = ::timed_compare;
};
struct TimedCompare1 {
  using type = ::timed_compare;
};
struct TimedCompare2 {
  using type = ::timed_compare;
};
struct TimedCompare3 {
  using type = ::timed_compare;
};
struct TimedCompare4 {
  using type = ::timed_compare;
};
struct TimedCompare5 {
  using type = ::timed_compare;
};
struct TimedCompare6 {
  using type = ::timed_compare;
};
struct TimedCompare7 {
  using type = ::timed_compare;
};
struct TimedCompare8 {
  using type = ::timed_compare;
};
struct TimedCompare9 {
  using type = ::timed_compare;
};
struct TimedCompare10 {
  using type = ::timed_compare;
};
struct TimedCompare11 {
  using type = ::timed_compare;
};
struct TimedCompare12 {
  using type = ::timed_compare;
};
struct TimedCompare13 {
  using type = ::timed_compare;
};
}  // namespace relational_tags

SPECTRE_TEST_CASE("Unit.Utilities.TaggedTuple.equivalence",
                  "[Unit][Utilities]") {
  {
    constexpr tuples::TaggedTuple<relational_tags::Int0, relational_tags::Int1,
                                  relational_tags::Int2>
        t0{1, 4, 5};
    constexpr tuples::TaggedTuple<relational_tags::Int0, relational_tags::Int1,
                                  relational_tags::Int2>
        t1{1, 4, 5};
    constexpr tuples::TaggedTuple<relational_tags::Int0, relational_tags::Int1,
                                  relational_tags::Int2>
        t2{1, 3, 5};
    constexpr tuples::TaggedTuple<relational_tags::Int0, relational_tags::Int1,
                                  relational_tags::Int2>
        t3{2, 4, 5};
    constexpr tuples::TaggedTuple<relational_tags::Int0, relational_tags::Int1,
                                  relational_tags::Int2>
        t4{1, 4, 6};
    CHECK(t0 == t1);
    CHECK(t0 == t0);
    CHECK_FALSE(t0 != t1);
    CHECK_FALSE(t0 == t2);
    CHECK(t0 != t2);
    CHECK_FALSE(t0 == t3);
    CHECK(t0 != t3);
    CHECK_FALSE(t0 == t4);
    CHECK(t0 != t4);
    // Check that short circuiting in comparisons works correctly
    tuples::TaggedTuple<
        relational_tags::TimedCompare0, relational_tags::TimedCompare1,
        relational_tags::TimedCompare2, relational_tags::TimedCompare3,
        relational_tags::TimedCompare4, relational_tags::TimedCompare5,
        relational_tags::TimedCompare6, relational_tags::TimedCompare7,
        relational_tags::TimedCompare8, relational_tags::TimedCompare9,
        relational_tags::TimedCompare10, relational_tags::TimedCompare11,
        relational_tags::TimedCompare12, relational_tags::TimedCompare13>
        t5{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    tuples::TaggedTuple<
        relational_tags::TimedCompare0, relational_tags::TimedCompare1,
        relational_tags::TimedCompare2, relational_tags::TimedCompare3,
        relational_tags::TimedCompare4, relational_tags::TimedCompare5,
        relational_tags::TimedCompare6, relational_tags::TimedCompare7,
        relational_tags::TimedCompare8, relational_tags::TimedCompare9,
        relational_tags::TimedCompare10, relational_tags::TimedCompare11,
        relational_tags::TimedCompare12, relational_tags::TimedCompare13>
        t6{1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    global_time_mock = 0;
    CHECK_FALSE(t5 == t6);
    CHECK(global_time_mock == 1);

    global_time_mock = 0;
    CHECK(t5 != t6);
    CHECK(global_time_mock == 1);

    tuples::TaggedTuple<
        relational_tags::TimedCompare0, relational_tags::TimedCompare1,
        relational_tags::TimedCompare2, relational_tags::TimedCompare3,
        relational_tags::TimedCompare4, relational_tags::TimedCompare5,
        relational_tags::TimedCompare6, relational_tags::TimedCompare7,
        relational_tags::TimedCompare8, relational_tags::TimedCompare9,
        relational_tags::TimedCompare10, relational_tags::TimedCompare11,
        relational_tags::TimedCompare12, relational_tags::TimedCompare13>
        t7{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14};
    global_time_mock = 0;
    CHECK_FALSE(t5 == t7);
    CHECK(global_time_mock == 14);

    global_time_mock = 0;
    CHECK(t5 != t7);
    CHECK(global_time_mock == 14);
  }
  static_assert(not noexcept(NotNoExceptCompare{1} == NotNoExceptCompare{1}),
                "Failed testing Unit.Utilities.TaggedTuple.relational");
  static_assert(not noexcept(NotNoExceptCompare{1} != NotNoExceptCompare{0}),
                "Failed testing Unit.Utilities.TaggedTuple.relational");
  CHECK(NotNoExceptCompare{1} == NotNoExceptCompare{1});
  CHECK_FALSE(NotNoExceptCompare{1} == NotNoExceptCompare{0});
  CHECK(NotNoExceptCompare{1} != NotNoExceptCompare{0});
  CHECK_FALSE(NotNoExceptCompare{1} != NotNoExceptCompare{1});

#if __cplusplus >= 201402L
  {
    constexpr tuples::TaggedTuple<relational_tags::Int0, relational_tags::Int1,
                                  relational_tags::Int2>
        t0{1, 4, 5};
    constexpr tuples::TaggedTuple<relational_tags::Int0, relational_tags::Int1,
                                  relational_tags::Int2>
        t1{1, 4, 5};
    constexpr tuples::TaggedTuple<relational_tags::Int0, relational_tags::Int1,
                                  relational_tags::Int2>
        t2{1, 3, 5};
    constexpr tuples::TaggedTuple<relational_tags::Int0, relational_tags::Int1,
                                  relational_tags::Int2>
        t3{2, 4, 5};
    constexpr tuples::TaggedTuple<relational_tags::Int0, relational_tags::Int1,
                                  relational_tags::Int2>
        t4{1, 4, 6};
    static_assert(t0 == t1,
                  "Failed testing Unit.Utilities.TaggedTuple.relational");
    static_assert(not(t0 == t2),
                  "Failed testing Unit.Utilities.TaggedTuple.relational");
    static_assert(not(t0 == t3),
                  "Failed testing Unit.Utilities.TaggedTuple.relational");
    static_assert(not(t0 == t4),
                  "Failed testing Unit.Utilities.TaggedTuple.relational");
    static_assert(t0 != t2,
                  "Failed testing Unit.Utilities.TaggedTuple.relational");
    static_assert(t0 != t3,
                  "Failed testing Unit.Utilities.TaggedTuple.relational");
    static_assert(t0 != t4,
                  "Failed testing Unit.Utilities.TaggedTuple.relational");
    static_assert(noexcept(t0 != t4),
                  "Failed testing Unit.Utilities.TaggedTuple.relational");
    static_assert(noexcept(t0 == t2),
                  "Failed testing Unit.Utilities.TaggedTuple.relational");
  }
#endif
}

struct lex_time_compared {
  explicit lex_time_compared(char c) : c_(c) {}
  char c_;
};

bool operator<(lex_time_compared const& lhs,
               lex_time_compared const& rhs) noexcept {
  global_time_mock++;
  return lhs.c_ < rhs.c_;
}

namespace relational_tags {
struct Char0 {
  using type = char;
};
struct Char1 {
  using type = char;
};
struct Char2 {
  using type = char;
};

struct LexTimeComp0 {
  using type = lex_time_compared;
};
struct LexTimeComp1 {
  using type = lex_time_compared;
};
struct LexTimeComp2 {
  using type = lex_time_compared;
};
struct LexTimeComp3 {
  using type = lex_time_compared;
};
struct LexTimeComp4 {
  using type = lex_time_compared;
};
struct LexTimeComp5 {
  using type = lex_time_compared;
};
struct LexTimeComp6 {
  using type = lex_time_compared;
};
struct LexTimeComp7 {
  using type = lex_time_compared;
};
struct LexTimeComp8 {
  using type = lex_time_compared;
};
struct LexTimeComp9 {
  using type = lex_time_compared;
};
struct LexTimeComp10 {
  using type = lex_time_compared;
};
struct LexTimeComp11 {
  using type = lex_time_compared;
};
struct LexTimeComp12 {
  using type = lex_time_compared;
};
struct LexTimeComp13 {
  using type = lex_time_compared;
};
}  // namespace relational_tags

SPECTRE_TEST_CASE("Unit.Utilities.TaggedTuple.relational",
                  "[Unit][Utilities]") {
  {
    // Check lexicographical comparison
    tuples::TaggedTuple<relational_tags::Char0, relational_tags::Char1,
                        relational_tags::Char2>
        t0{'a', 'a', 'c'};
    tuples::TaggedTuple<relational_tags::Char0, relational_tags::Char1,
                        relational_tags::Char2>
        t1{'a', 'a', 'd'};
    tuples::TaggedTuple<relational_tags::Char0, relational_tags::Char1,
                        relational_tags::Char2>
        t2{'a', 'a', 'c'};
    tuples::TaggedTuple<relational_tags::Char0, relational_tags::Char1,
                        relational_tags::Char2>
        t3{'a', 'a', 'b'};
    CHECK(t0 < t1);
    CHECK(t0 <= t1);
    CHECK(t0 <= t2);

    CHECK(t0 > t3);
    CHECK(t0 >= t2);
    CHECK(t0 >= t3);
  }
  {
    // Check short circuiting works correctly
    tuples::TaggedTuple<
        relational_tags::LexTimeComp0, relational_tags::LexTimeComp1,
        relational_tags::LexTimeComp2, relational_tags::LexTimeComp3,
        relational_tags::LexTimeComp4, relational_tags::LexTimeComp5,
        relational_tags::LexTimeComp6, relational_tags::LexTimeComp7,
        relational_tags::LexTimeComp8, relational_tags::LexTimeComp9,
        relational_tags::LexTimeComp10, relational_tags::LexTimeComp11,
        relational_tags::LexTimeComp12, relational_tags::LexTimeComp13>
        t0{'a', 'a', 'a', 'a', 'a', 'a', 'a',
           'a', 'a', 'a', 'a', 'a', 'a', 'a'};
    tuples::TaggedTuple<
        relational_tags::LexTimeComp0, relational_tags::LexTimeComp1,
        relational_tags::LexTimeComp2, relational_tags::LexTimeComp3,
        relational_tags::LexTimeComp4, relational_tags::LexTimeComp5,
        relational_tags::LexTimeComp6, relational_tags::LexTimeComp7,
        relational_tags::LexTimeComp8, relational_tags::LexTimeComp9,
        relational_tags::LexTimeComp10, relational_tags::LexTimeComp11,
        relational_tags::LexTimeComp12, relational_tags::LexTimeComp13>
        t1{'b', 'a', 'a', 'a', 'a', 'a', 'a',
           'a', 'a', 'a', 'a', 'a', 'a', 'a'};
    global_time_mock = 0;
    CHECK(t0 < t1);
    CHECK(global_time_mock == 1);
    global_time_mock = 0;
    CHECK_FALSE(t0 > t1);
    CHECK(global_time_mock == 2);

    tuples::TaggedTuple<
        relational_tags::LexTimeComp0, relational_tags::LexTimeComp1,
        relational_tags::LexTimeComp2, relational_tags::LexTimeComp3,
        relational_tags::LexTimeComp4, relational_tags::LexTimeComp5,
        relational_tags::LexTimeComp6, relational_tags::LexTimeComp7,
        relational_tags::LexTimeComp8, relational_tags::LexTimeComp9,
        relational_tags::LexTimeComp10, relational_tags::LexTimeComp11,
        relational_tags::LexTimeComp12, relational_tags::LexTimeComp13>
        t2{'a', 'a', 'a', 'a', 'a', 'a', 'a',
           'a', 'a', 'a', 'a', 'a', 'a', 'a'};
    global_time_mock = 0;
    CHECK(t0 <= t2);
    CHECK(global_time_mock == 28);
  }
#if __cplusplus >= 201402L
  {
    // Check constexpr lexicographical comparison
    constexpr tuples::TaggedTuple<
        relational_tags::Char0, relational_tags::Char1, relational_tags::Char2>
        t0{'a', 'a', 'c'};
    constexpr tuples::TaggedTuple<
        relational_tags::Char0, relational_tags::Char1, relational_tags::Char2>
        t1{'a', 'a', 'd'};
    constexpr tuples::TaggedTuple<
        relational_tags::Char0, relational_tags::Char1, relational_tags::Char2>
        t2{'a', 'a', 'c'};
    constexpr tuples::TaggedTuple<
        relational_tags::Char0, relational_tags::Char1, relational_tags::Char2>
        t3{'a', 'a', 'b'};
    static_assert(t0 < t1, "Failed testing relational operators");
    static_assert(t0 <= t1, "Failed testing relational operators");
    static_assert(t0 <= t2, "Failed testing relational operators");

    static_assert(t0 > t3, "Failed testing relational operators");
    static_assert(t0 >= t2, "Failed testing relational operators");
    static_assert(t0 >= t3, "Failed testing relational operators");
  }
#endif
}

static_assert(
    tuples::tuple_size<
        tuples::TaggedTuple<tags::no_default, tags::empty_base>>::value == 2,
    "Failed check tuple_size");
static_assert(
    tuples::tuple_size<
        const tuples::TaggedTuple<tags::no_default, tags::empty_base>>::value ==
        2,
    "Failed check tuple_size");
static_assert(tuples::tuple_size<volatile tuples::TaggedTuple<
                      tags::no_default, tags::empty_base>>::value == 2,
              "Failed check tuple_size");
static_assert(tuples::tuple_size<const volatile tuples::TaggedTuple<
                      tags::no_default, tags::empty_base>>::value == 2,
              "Failed check tuple_size");

static_assert(tuples::tuple_size<tuples::TaggedTuple<>>::value == 0,
              "Failed check tuple_size");
static_assert(tuples::tuple_size<const tuples::TaggedTuple<>>::value == 0,
              "Failed check tuple_size");
static_assert(tuples::tuple_size<volatile tuples::TaggedTuple<>>::value == 0,
              "Failed check tuple_size");
static_assert(tuples::tuple_size<const volatile tuples::TaggedTuple<>>::value ==
                  0,
              "Failed check tuple_size");

// C++17 Draft 23.5.3.3 swap
struct not_swappable {
  explicit not_swappable(int v) : v_(v) {}
  int v_;
};

struct throws_swap {
  throws_swap() = default;
  explicit throws_swap(int v) : v_(v) {}

  int v_{0};
};

void swap(throws_swap& lhs, throws_swap& rhs) noexcept(false) {
  using std::swap;
  using tuples::swap;
  swap(lhs.v_, rhs.v_);
}

int global_swappable_value = 0;

struct empty_base_swappable {};

void swap(empty_base_swappable& /*lhs*/,
          empty_base_swappable& /*rhs*/) noexcept {
  global_swappable_value++;
}

struct empty_base_throws_swappable {};

void swap(empty_base_throws_swappable& /*lhs*/,
          empty_base_throws_swappable& /*rhs*/) noexcept(false) {
  global_swappable_value++;
}

static_assert(tuples::tuples_detail::is_swappable_with<double, double>::value,
              "Failed testing tuples::tuples_detail::is_swappable_with");
static_assert(tuples::tuples_detail::is_swappable_with<double&, double>::value,
              "Failed testing tuples::tuples_detail::is_swappable_with");
static_assert(tuples::tuples_detail::is_swappable_with<double&, double&>::value,
              "Failed testing tuples::tuples_detail::is_swappable_with");
static_assert(
    not tuples::tuples_detail::is_swappable_with<double const&, double>::value,
    "Failed testing tuples::tuples_detail::is_swappable_with");

static_assert(
    tuples::tuples_detail::is_nothrow_swappable_with<double, double>::value,
    "Failed testing tuples::tuples_detail::is_nothrow_swappable_with");
static_assert(
    tuples::tuples_detail::is_nothrow_swappable_with<double&, double>::value,
    "Failed testing tuples::tuples_detail::is_nothrow_swappable_with");
static_assert(
    tuples::tuples_detail::is_nothrow_swappable_with<double&, double&>::value,
    "Failed testing tuples::tuples_detail::is_nothrow_swappable_with");
static_assert(
    not tuples::tuples_detail::is_nothrow_swappable_with<double const&,
                                                         double>::value,
    "Failed testing tuples::tuples_detail::is_nothrow_swappable_with");

static_assert(
    not tuples::tuples_detail::is_swappable_with<double, not_swappable>::value,
    "Failed testing tuples::tuples_detail::is_swappable_with");
static_assert(
    not tuples::tuples_detail::is_swappable_with<double&, not_swappable>::value,
    "Failed testing tuples::tuples_detail::is_swappable_with");
static_assert(
    not tuples::tuples_detail::is_swappable_with<double, not_swappable&>::value,
    "Failed testing tuples::tuples_detail::is_swappable_with");
static_assert(not tuples::tuples_detail::is_swappable_with<
                  double&, not_swappable&>::value,
              "Failed testing tuples::tuples_detail::is_swappable_with");
static_assert(not tuples::tuples_detail::is_swappable_with<
                  double const&, not_swappable>::value,
              "Failed testing tuples::tuples_detail::is_swappable_with");

static_assert(
    not tuples::tuples_detail::is_nothrow_swappable_with<double,
                                                         not_swappable>::value,
    "Failed testing tuples::tuples_detail::is_nothrow_swappable_with");
static_assert(
    not tuples::tuples_detail::is_nothrow_swappable_with<double&,
                                                         not_swappable>::value,
    "Failed testing tuples::tuples_detail::is_nothrow_swappable_with");
static_assert(
    not tuples::tuples_detail::is_nothrow_swappable_with<double,
                                                         not_swappable&>::value,
    "Failed testing tuples::tuples_detail::is_nothrow_swappable_with");
static_assert(
    not tuples::tuples_detail::is_nothrow_swappable_with<double&,
                                                         not_swappable&>::value,
    "Failed testing tuples::tuples_detail::is_nothrow_swappable_with");
static_assert(
    not tuples::tuples_detail::is_nothrow_swappable_with<double const&,
                                                         not_swappable>::value,
    "Failed testing tuples::tuples_detail::is_nothrow_swappable_with");

static_assert(
    tuples::tuples_detail::is_swappable_with<throws_swap, throws_swap>::value,
    "Failed testing tuples::tuples_detail::is_swappable_with");
static_assert(
    not tuples::tuples_detail::is_nothrow_swappable_with<throws_swap,
                                                         throws_swap>::value,
    "Failed testing tuples::tuples_detail::is_nothrow_swappable_with");

namespace swap_tags {
struct Int0 {
  using type = int;
};
struct Int1 {
  using type = int;
};
struct NotSwappable {
  using type = not_swappable;
};

struct ThrowsSwap {
  using type = throws_swap;
};

struct EmptyBaseSwap0 {
  using type = empty_base_swappable;
};
struct EmptyBaseSwap1 {
  using type = empty_base_swappable;
};

struct EmptyBaseThrowsSwap0 {
  using type = empty_base_throws_swappable;
};
struct EmptyBaseThrowsSwap1 {
  using type = empty_base_throws_swappable;
};
}  // namespace swap_tags

SPECTRE_TEST_CASE("Unit.Utilities.TaggedTuple.swap", "[Unit][Utilities]") {
  {
    tuples::TaggedTuple<> t0{};
    tuples::TaggedTuple<> t1{};
    tuples::swap(t0, t1);
  }
  {
    tuples::TaggedTuple<swap_tags::Int0, swap_tags::Int1> t0{1, 3};
    tuples::TaggedTuple<swap_tags::Int0, swap_tags::Int1> t1{4, 5};
    CHECK(tuples::get<swap_tags::Int0>(t0) == 1);
    CHECK(tuples::get<swap_tags::Int1>(t0) == 3);
    CHECK(tuples::get<swap_tags::Int0>(t1) == 4);
    CHECK(tuples::get<swap_tags::Int1>(t1) == 5);
    tuples::swap(t0, t1);
    CHECK(tuples::get<swap_tags::Int0>(t1) == 1);
    CHECK(tuples::get<swap_tags::Int1>(t1) == 3);
    CHECK(tuples::get<swap_tags::Int0>(t0) == 4);
    CHECK(tuples::get<swap_tags::Int1>(t0) == 5);
    static_assert(noexcept(tuples::swap(t0, t1)),
                  "Failed testing Unit.Utilities.TaggedTuple.swap");
  }
  {
    tuples::TaggedTuple<swap_tags::ThrowsSwap> t0{1};
    tuples::TaggedTuple<swap_tags::ThrowsSwap> t1{2};
    CHECK(tuples::get<swap_tags::ThrowsSwap>(t0).v_ == 1);
    CHECK(tuples::get<swap_tags::ThrowsSwap>(t1).v_ == 2);
    tuples::swap(t0, t1);
    CHECK(tuples::get<swap_tags::ThrowsSwap>(t0).v_ == 2);
    CHECK(tuples::get<swap_tags::ThrowsSwap>(t1).v_ == 1);
    static_assert(not noexcept(tuples::swap(t0, t1)),
                  "Failed testing Unit.Utilities.TaggedTuple.swap");
  }
  {
    tuples::TaggedTuple<swap_tags::EmptyBaseSwap0, swap_tags::EmptyBaseSwap1>
        t0;
    tuples::TaggedTuple<swap_tags::EmptyBaseSwap0, swap_tags::EmptyBaseSwap1>
        t1;
    global_swappable_value = 0;
    tuples::swap(t0, t1);
    CHECK(global_swappable_value == 2);
    static_assert(noexcept(tuples::swap(t0, t1)),
                  "Failed testing Unit.Utilities.TaggedTuple.swap");
  }
  {
    tuples::TaggedTuple<swap_tags::EmptyBaseThrowsSwap0,
                        swap_tags::EmptyBaseThrowsSwap1>
        t0;
    tuples::TaggedTuple<swap_tags::EmptyBaseThrowsSwap0,
                        swap_tags::EmptyBaseThrowsSwap1>
        t1;
    global_swappable_value = 0;
    tuples::swap(t0, t1);
    CHECK(global_swappable_value == 2);
    static_assert(not noexcept(tuples::swap(t0, t1)),
                  "Failed testing Unit.Utilities.TaggedTuple.swap");
  }
}

namespace tags {
struct NonCopyable0 {
  using type = std::vector<std::unique_ptr<int>>;
};
struct NonCopyable1 {
  using type = std::vector<std::unique_ptr<int>>;
};
}  // namespace tags

SPECTRE_TEST_CASE("Unit.Utilities.TaggedTuple.NonCopyable",
                  "[Unit][Utilities]") {
  std::vector<std::unique_ptr<int>> a{};
  a.reserve(2);
  a.emplace_back(std::make_unique<int>(1));
  a.emplace_back(std::make_unique<int>(3));
  tuples::TaggedTuple<tags::NonCopyable0, tags::NonCopyable1> t(
      std::move(a), tags::NonCopyable1::type{});
  CHECK(*tuples::get<tags::NonCopyable0>(t)[0] == 1);
  CHECK(*tuples::get<tags::NonCopyable0>(t)[1] == 3);
}

namespace tags {
struct Copyable0 {
  using type = std::vector<double>;
};
struct Copyable1 {
  using type = std::vector<double>;
};
}  // namespace tags

SPECTRE_TEST_CASE("Unit.Utilities.TaggedTuple.SingleTagConstructor",
                  "[Unit][Utilities]") {
  {
    std::vector<std::unique_ptr<int>> a{};
    a.reserve(2);
    a.emplace_back(std::make_unique<int>(1));
    a.emplace_back(std::make_unique<int>(3));
    tuples::TaggedTuple<tags::NonCopyable0> t0(std::move(a));
    tuples::TaggedTuple<tags::NonCopyable1> t1(std::move(t0));
    CHECK(*tuples::get<tags::NonCopyable1>(t1)[0] == 1);
    CHECK(*tuples::get<tags::NonCopyable1>(t1)[1] == 3);
  }
  {
    std::vector<double> a{1.0, 3.0};
    tuples::TaggedTuple<tags::Copyable0> t0(a);
    tuples::TaggedTuple<tags::Copyable1> t1(t0);
    CHECK(tuples::get<tags::Copyable1>(t1) == a);
  }
  {
    std::vector<double> a{1.0, 3.0};
    tuples::TaggedTuple<tags::Copyable0> t0 = a;
    tuples::TaggedTuple<tags::Copyable1> t1(t0);
    CHECK(tuples::get<tags::Copyable1>(t1) == a);
  }
}
}  // namespace
