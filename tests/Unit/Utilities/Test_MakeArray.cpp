// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "Utilities/MakeArray.hpp"
#include "tests/Unit/TestHelpers.hpp"

static_assert(noexcept(make_array<2>(0)),
              "Failed Unit.Utilities.MakeArray testing noexcept calculation of "
              "make_array.");
static_assert(noexcept(make_array<2>(DoesNotThrow{})),
              "Failed Unit.Utilities.MakeArray testing noexcept calculation of "
              "make_array.");
static_assert(not noexcept(make_array<2>(DoesThrow{})),
              "Failed Unit.Utilities.MakeArray testing noexcept calculation of "
              "make_array.");

SPECTRE_TEST_CASE("Unit.Utilities.MakeArray", "[Unit][Utilities]") {
  constexpr auto same_array = make_array<4>(7.8);
  CHECK((same_array == std::array<double, 4>{{7.8, 7.8, 7.8, 7.8}}));
  constexpr auto varying_array = make_array(3.2, 4.3, 5.4, 6.5, 7.8);
  CHECK((varying_array == std::array<double, 5>{{3.2, 4.3, 5.4, 6.5, 7.8}}));

  auto array_n = make_array<6>(5);
  static_assert(
      std::is_same<typename std::decay<decltype(array_n)>::type::value_type,
                   int>::value,
      "Unit Test Failure: Incorrect type from make_array.");
  static_assert(array_n.size() == 6,
                "Unit Test Failure: Incorrect size from make_array.");
  for (const auto& p : array_n) {
    CHECK(5 == p);
  }

  auto array_non_copyable = make_array<1>(NonCopyable{});
  static_assert(array_non_copyable.size() == 1,
                "Unit Test Failure: Incorrect array size should be 1 for "
                "move-only types.");

  auto array_empty = make_array<0>(2);
  static_assert(array_empty.empty(),
                "Unit Test Failure: Incorrect array size for empty array.");

  auto array_non_copyable_empty = make_array<0>(NonCopyable{});
  static_assert(
      array_non_copyable_empty.empty(),
      "Unit Test Failure: Incorrect array size for empty array of move-only.");

  auto my_array = make_array(1, 3, 4, 8, 9);
  static_assert(my_array.size() == 5,
                "Unit Test Failure: Incorrect array size.");
  CHECK((my_array[0] == 1 and my_array[1] == 3 and my_array[2] == 4 and
         my_array[3] == 8 and my_array[4] == 9));

  CHECK((std::array<int, 3>{{2, 8, 6}}) ==
        (make_array<int, 3>(std::vector<int>{2, 8, 6})));
  CHECK((std::array<int, 3>{{2, 8, 6}}) ==
        (make_array<int, 3>(std::vector<int>{2, 8, 6, 9, 7})));
}

// [[OutputRegex, The sequence size must be at least as large as the array being
// created from it.]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Utilities.MakeArraySeqTooSmall",
                               "[Unit][Utilities]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  static_cast<void>(make_array<int, 3>(std::vector<int>{2, 8}));
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
