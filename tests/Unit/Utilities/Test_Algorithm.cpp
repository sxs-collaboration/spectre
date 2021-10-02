// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cctype>
#include <iterator>
#include <string>
#include <vector>

#include "Utilities/Algorithm.hpp"
#include "Utilities/Array.hpp"  // IWYU pragma: associated
#include "Utilities/Numeric.hpp"

namespace {
constexpr bool check_swap() {
  int x = 4;
  int y = 444;
  cpp20::swap(x, y);
  return x == 444 and y == 4;
}

constexpr bool check_iter_swap() {
  const size_t size = 6;
  cpp20::array<size_t, size> a{};
  cpp2b::iota(a.begin(), a.end(), size_t(1));
  cpp20::iter_swap(a.begin() + 1, a.begin() + 4);

  cpp20::array<size_t, size> expected{{1, 5, 3, 4, 2, 6}};
  return a == expected;
}

constexpr bool check_reverse() {
  const size_t size = 6;
  cpp20::array<size_t, size> a{};
  cpp2b::iota(a.begin(), a.end(), size_t(1));
  cpp20::detail::reverse(a.begin(), a.end(), std::bidirectional_iterator_tag());

  cpp20::array<size_t, size> expected{{6, 5, 4, 3, 2, 1}};
  return a == expected;
}

constexpr bool check_reverse_random_access() {
  const size_t size = 6;
  cpp20::array<size_t, size> a{};
  cpp2b::iota(a.begin(), a.end(), size_t(1));
  // a is a random_access_iterator, so the next line tests both the generic
  // cpp20::reverse() and the one for random access iterators
  cpp20::reverse(a.begin(), a.end());

  cpp20::array<size_t, size> expected{{6, 5, 4, 3, 2, 1}};
  return a == expected;
}

constexpr bool check_next_permutation() {
  const size_t size = 6;
  cpp20::array<size_t, size> a{{1, 2, 4, 6, 3, 5}};
  cpp20::next_permutation(a.begin(), a.end());

  cpp20::array<size_t, size> expected{{1, 2, 4, 6, 5, 3}};
  return a == expected;
}

void test_constexpr_swap_iter_swap_reverse_etc() {
  // Check the functions at runtime
  CHECK(check_swap());
  CHECK(check_iter_swap());
  CHECK(check_reverse());
  CHECK(check_reverse_random_access());
  CHECK(check_next_permutation());

  // Check the functions at compile time
  static_assert(check_swap(), "Failed test Unit.Utilities.Algorithm");
  static_assert(check_iter_swap(), "Failed test Unit.Utilities.Algorithm");
  static_assert(check_reverse(), "Failed test Unit.Utilities.Algorithm");
  static_assert(check_reverse_random_access(),
                "Failed test Unit.Utilities.Algorithm");
  static_assert(check_next_permutation(),
                "Failed test Unit.Utilities.Algorithm");
}

void test_all_of_and_any_of_and_none_of() {
  // Test wrappers around STL algorithms
  CHECK(alg::all_of(std::vector<int>{1, 1, 1, 1},
                    [](const int t) { return t == 1; }));
  CHECK_FALSE(alg::all_of(std::vector<int>{1, 2, 1, 1},
                          [](const int t) { return t == 1; }));
  CHECK_FALSE(alg::all_of(std::vector<int>{2, 1, 1, 1},
                          [](const int t) { return t == 1; }));
  CHECK_FALSE(alg::all_of(std::vector<int>{1, 1, 1, 2},
                          [](const int t) { return t == 1; }));
  CHECK(alg::any_of(std::vector<int>{4, 2, 1, 3},
                    [](const int t) { return t == 1; }));
  CHECK_FALSE(alg::any_of(std::vector<int>{4, 2, -1, 3},
                          [](const int t) { return t == 1; }));
  CHECK(alg::none_of(std::vector<int>{4, 2, -1, 3},
                     [](const int t) { return t == 1; }));
  CHECK_FALSE(alg::none_of(std::vector<int>{4, 2, 1, 3},
                           [](const int t) { return t == 1; }));
  CHECK_FALSE(alg::none_of(std::vector<int>{1, 2, -1, 3},
                           [](const int t) { return t == 1; }));
  CHECK_FALSE(alg::none_of(std::vector<int>{4, 2, -1, 1},
                           [](const int t) { return t == 1; }));
}

void test_count_related() {
  std::vector<int> a{1, 2, 3, 4, 3, 5};
  CHECK(alg::count(a, 0) == 0);
  CHECK(alg::count(a, -1) == 0);
  CHECK(alg::count(a, -2) == 0);
  for (int i = 1; i < 5; ++i) {
    CHECK(alg::count(a, i) == (i == 3 ? 2 : 1));
  }

  CHECK(alg::count_if(a, [](const int t) { return t == 0; }) == 0);
  CHECK(alg::count_if(a, [](const int t) { return t == -1; }) == 0);
  CHECK(alg::count_if(a, [](const int t) { return t == -2; }) == 0);
  CHECK(alg::count_if(a, [](const int t) { return t % 2 == 0; }) == 2);
  CHECK(alg::count_if(a, [](const int t) { return t % 3 == 0; }) == 2);
  CHECK(alg::count_if(a, [](const int t) { return t % 5 == 0; }) == 1);
  CHECK(alg::count_if(a, [](const int t) { return t % 7 == 0; }) == 0);
}

void test_equal() {
  // Test alg::equal
  CHECK(alg::equal(std::vector<int>{1, -7, 8, 9},
                   std::array<int, 4>{{1, -7, 8, 9}}));
  CHECK_FALSE(alg::equal(std::vector<int>{1, 7, 8, 9},
                         std::array<int, 4>{{1, -7, 8, 9}}));

  CHECK(alg::equal(std::vector<int>{1, -7, 8, 9},
                   std::array<int, 4>{{-1, 7, -8, -9}},
                   [](const int lhs, const int rhs) { return lhs == -rhs; }));
  CHECK_FALSE(alg::equal(
      std::vector<int>{1, -7, 8, 9}, std::array<int, 4>{{-1, -7, -8, -9}},
      [](const int lhs, const int rhs) { return lhs == -rhs; }));
}

void test_min_max_element() {
  std::vector<int> t{3, 5, 8, -1};
  CHECK(*alg::min_element(t) == -1);
  CHECK(*alg::max_element(t) == 8);
  CHECK(*alg::min_element(t, [](const int a, const int b) { return a > b; }) ==
        8);
  CHECK(*alg::max_element(t, [](const int a, const int b) { return a > b; }) ==
        -1);
}

void test_find_related() {
  const std::vector<int> a{1, 2, 3, 4};
  CHECK(alg::find(a, 3) == a.begin() + 2);
  CHECK(alg::find(a, 5) == a.end());
  CHECK(alg::found(a, 3));
  CHECK_FALSE(alg::found(a, 5));
  std::vector<int> a_copy = a;
  CHECK(alg::find(a_copy, 3) == a_copy.begin() + 2);
  *alg::find(a_copy, 3) = 4;
  CHECK(alg::find(a_copy, 3) == a_copy.end());

  const auto greater_than_three = [](const int t) { return t > 3; };
  CHECK(alg::find_if(a, greater_than_three) == a.end() - 1);
  CHECK(alg::find_if(a, [](const int t) { return t < 0; }) == a.end());
  CHECK(alg::found_if(a, greater_than_three));
  CHECK_FALSE(alg::found_if(a, [](const int t) { return t < 0; }));
  a_copy = a;
  CHECK(alg::find_if(a_copy, greater_than_three) == a_copy.begin() + 3);
  *alg::find_if(a_copy, greater_than_three) = 1;
  CHECK(alg::find_if(a_copy, greater_than_three) == a_copy.end());

  const auto less_than_three = [](const int t) { return t < 3; };
  CHECK(alg::find_if_not(a, less_than_three) == a.end() - 2);
  CHECK(alg::find_if_not(a, [](const int t) { return t < 8; }) == a.end());
  CHECK(alg::found_if_not(a, less_than_three));
  CHECK_FALSE(alg::found_if_not(a, [](const int t) { return t < 8; }));
  a_copy = a;
  CHECK(alg::find_if_not(a_copy, less_than_three) == a_copy.begin() + 2);
  *alg::find_if_not(a_copy, less_than_three) = 2;
  CHECK(alg::find_if_not(a_copy, less_than_three) == a_copy.begin() + 3);
}

void test_for_each() {
  int count = 0;
  const std::vector<size_t> a{1, 7, 234, 987, 32};
  alg::for_each(a, [&count](const size_t value) {
    if (value > 100) {
      count++;
    }
  });
  CHECK(count == 2);

  std::vector<size_t> b{1, 7, 234, 987, 32};
  alg::for_each(b, [](size_t& value) {
    if (value > 100) {
      value *= 2;
    }
  });
  CHECK(b == std::vector<size_t>{1, 7, 2 * 234, 2 * 987, 32});
}

void test_remove() {
  std::string str1 = "Test with some   spaces";
  str1.erase(alg::remove(str1, ' '), str1.end());
  CHECK(str1 == "Testwithsomespaces");

  std::string str2 = "Text\n with\tsome \t  whitespaces\n\n";
  str2.erase(
      alg::remove_if(str2, [](unsigned char x) { return std::isspace(x); }),
      str2.end());
  CHECK(str2 == "Textwithsomewhitespaces");
}

void test_sort() {
  std::vector<int> t_orig{3, 5, 8, -1};
  std::vector<int> t0 = t_orig;
  std::vector<int> t0_copy = t0;
  alg::sort(t0);
  std::sort(t0_copy.begin(), t0_copy.end());
  CHECK(t0 == t0_copy);

  auto t1 = t_orig;
  auto t1_copy = t1;
  alg::sort(t1, std::greater<>{});
  std::sort(t1_copy.begin(), t1_copy.end(), std::greater<>{});
  CHECK(t1 == t1_copy);
}

void test_transform() {
  std::vector<int> t0{3, 5, 8, -1};
  std::vector<int> t1{8, 2, 7, -5};
  std::vector<int> result(t0.size());
  alg::transform(t0, result.begin(), [](const size_t t) { return t + 5; });
  CHECK(result == std::vector<int>{8, 10, 13, 4});

  alg::transform(t0, t1, result.begin(),
                 [](const size_t a, const size_t b) { return a + b; });
  CHECK(result == std::vector<int>{11, 7, 15, -6});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.Algorithm", "[Unit][Utilities]") {
  test_constexpr_swap_iter_swap_reverse_etc();
  test_all_of_and_any_of_and_none_of();
  test_count_related();
  test_equal();
  test_find_related();
  test_for_each();
  test_min_max_element();
  test_remove();
  test_sort();
  test_transform();
}
