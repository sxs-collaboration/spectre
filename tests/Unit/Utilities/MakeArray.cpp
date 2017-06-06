// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "Utilities/MakeArray.hpp"

namespace {
struct NonCopyable {
  constexpr NonCopyable() = default;
  constexpr NonCopyable(const NonCopyable&) = delete;
  constexpr NonCopyable& operator=(const NonCopyable&) = delete;
  constexpr NonCopyable(NonCopyable&&) = default;   // NOLINT
  NonCopyable& operator=(NonCopyable&&) = default;  // NOLINT
  ~NonCopyable() = default;
};
}  // namespace

TEST_CASE("Unit.Utilities.MakeArray", "[Unit][Utilities]") {
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
}
