// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <initializer_list>
#include <string>

#include "Framework/TestHelpers.hpp"
#include "Utilities/Array.hpp"  // IWYU pragma: associated
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeArray.hpp"

static_assert(convert_to_cpp20_array(make_array<4>(7.8)) ==
                  cpp20::array<double, 4>{{7.8, 7.8, 7.8, 7.8}},
              "Failed testing cpp20::array conversion from std::array");
static_assert(convert_to_cpp20_array(make_array(3.2, 4.3, 5.4, 6.5, 7.8)) ==
                  cpp20::array<double, 5>{{3.2, 4.3, 5.4, 6.5, 7.8}},
              "Failed testing cpp20::array conversion from std::array");

namespace {
constexpr auto array_n = convert_to_cpp20_array(make_array<6>(5));
static_assert(
    std::is_same<typename std::decay<decltype(array_n)>::type::value_type,
                 int>::value,
    "Failed testing cpp20::array type from make_array");
static_assert(array_n.size() == 6,
              "Failed testing cpp20::array size from make_array");
constexpr bool test_constexpr_iter() {
  bool result = true;
  for (const auto& p : array_n) {
    result = result && 5 == p;
  }
  return result;
}
static_assert(test_constexpr_iter(), "Failed testing cpp20::array iterator");

constexpr auto array_empty = convert_to_cpp20_array(make_array<0>(2));
static_assert(array_empty.empty(),
              "Failed testing cpp20::array size of empty array");

constexpr auto array_non_copyable_empty =
    convert_to_cpp20_array(make_array<0>(NonCopyable{}));
static_assert(array_non_copyable_empty.empty(),
              "Failed testing cpp20::array size for empty array of move-only");

constexpr auto my_array = convert_to_cpp20_array(make_array(1, 3, 4, 8, 9));
static_assert(my_array.size() == 5, "Failed testing cpp20::array size");
static_assert(my_array.max_size() == 5, "Failed testing cpp20::array size");
static_assert(my_array[0] == 1 and my_array[1] == 3 and my_array[2] == 4 and
                  my_array[3] == 8 and my_array[4] == 9,
              "Failed testing cpp20::array values");

constexpr auto array_from_sequence = convert_to_cpp20_array(
    make_array<int, 3>(std::initializer_list<int>{2, 8, 6}));
static_assert(cpp20::array<int, 3>{{2, 8, 6}} == array_from_sequence,
              "Failed testing cpp20::array from sequence");
constexpr const auto array_from_truncated_sequence = convert_to_cpp20_array(
    make_array<int, 3>(std::initializer_list<int>{2, 8, 6, 9, 7}));
static_assert(cpp20::array<int, 3>{{2, 8, 6}} == array_from_truncated_sequence,
              "Failed testing cpp20::array from sequence");

constexpr cpp20::array<int, 3> create_an_array() {
  cpp20::array<int, 3> a{};
  a[0] = 8;
  a[1] = -9;
  a[2] = 10;
  return a;
}
static_assert(create_an_array() == cpp20::array<int, 3>{{8, -9, 10}},
              "Failed testing cpp20::array");
static_assert(create_an_array().back() == 10, "Failed testing cpp20::array");
static_assert(create_an_array().front() == 8, "Failed testing cpp20::array");

constexpr cpp20::array<int, 3> create_an_array_with_at() {
  cpp20::array<int, 3> a{};
  a.at(0) = 10;
  a.at(1) = -9;
  a.at(2) = -50;
  a.front() = 8;
  a.back() = 10;
  return a;
}
static_assert(create_an_array_with_at() == cpp20::array<int, 3>{{8, -9, 10}},
              "Failed testing cpp20::array");
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.Cpp20Array", "[Unit][Utilities]") {
  cpp20::array<int, 0> a0{};
  CHECK(get_output(a0) == "()");
  cpp20::array<int, 1> a1{{1}};
  CHECK(get_output(a1) == "(1)");
  cpp20::array<int, 5> a5{{1, 2, 3, 4, 5}};
  CHECK(get_output(a5) == "(1,2,3,4,5)");
}
