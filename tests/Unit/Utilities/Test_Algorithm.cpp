// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <deque>
#include <functional>
#include <iterator>
#include <list>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Array.hpp"
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

// A type in namespace `std` without its own `std::swap` overload, so
// argument-dependent lookup finds the generic `std::swap`
constexpr bool check_iter_swap_std_type() {
  using std::chrono::milliseconds;
  std::array<milliseconds, 3> a{
      {milliseconds{1}, milliseconds{2}, milliseconds{3}}};
  cpp20::iter_swap(a.begin(), a.begin() + 2);
  cpp20::reverse(a.begin(), a.end());
  return a == std::array<milliseconds, 3>{
                  {milliseconds{1}, milliseconds{2}, milliseconds{3}}};
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
  CHECK(check_iter_swap_std_type());
  CHECK(check_reverse());
  CHECK(check_reverse_random_access());
  CHECK(check_next_permutation());

  // Check the functions at compile time
  static_assert(check_swap(), "Failed test Unit.Utilities.Algorithm");
  static_assert(check_iter_swap(), "Failed test Unit.Utilities.Algorithm");
  static_assert(check_iter_swap_std_type(),
                "Failed test Unit.Utilities.Algorithm");
  static_assert(check_reverse(), "Failed test Unit.Utilities.Algorithm");
  static_assert(check_reverse_random_access(),
                "Failed test Unit.Utilities.Algorithm");
  static_assert(check_next_permutation(),
                "Failed test Unit.Utilities.Algorithm");
}

void test_all_of_and_any_of_and_none_of() {
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

  {
    const auto [min, max] = alg::minmax_element(t);
    CHECK(*min == -1);
    CHECK(*max == 8);
  }

  CHECK(*alg::min_element(t, [](const int a, const int b) { return a > b; }) ==
        8);
  CHECK(*alg::max_element(t, [](const int a, const int b) { return a > b; }) ==
        -1);

  {
    const auto [min, max] =
        alg::minmax_element(t, [](const int a, const int b) { return a > b; });
    CHECK(*min == 8);
    CHECK(*max == -1);
  }
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

struct CountCalls {
  size_t calls = 0;
  bool operator()(const size_t /*value*/) {
    ++calls;
    return true;
  }
};

template <typename Container>
void test_function_objects_are_copied(const Container& c) {
  // Like the standard library, the functions call copies of the function
  // object, so a `const` object with a non-`const` `operator()` works
  const CountCalls count_calls{};
  CHECK(alg::for_each(c, count_calls).calls == c.size());
  CHECK(alg::all_of(c, count_calls));
  CHECK(alg::count_if(c, count_calls) == 5);
}

void test_remove() {
  std::string str1 = "Test with some   spaces";
  str1.erase(alg::remove(str1, ' '), str1.end());
  CHECK(str1 == "Testwithsomespaces");

  std::string str2 = "Text\n with\tsome \t  whitespaces\n\n";
  str2.erase(
      alg::remove_if(str2,
                     [](const char x) {
                       return std::isspace(static_cast<unsigned char>(x)) != 0;
                     }),
      str2.end());
  CHECK(str2 == "Textwithsomewhitespaces");
}

void test_sample() {
  std::vector<int> v{-1, 3, 5, 9, -24, 8};
  std::mt19937 gen;
  for (size_t s = 0; s < 8; ++s) {
    std::vector<int> alg_s{};
    gen.seed(1);
    alg::sample(v, std::back_inserter(alg_s), s, gen);
    std::vector<int> std_s{};
    gen.seed(1);
    std::sample(std::begin(v), std::end(v), std::back_inserter(std_s), s, gen);
    CHECK(alg_s == std_s);
  }
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
  alg::transform(t0, result.begin(), [](const int t) { return t + 5; });
  CHECK(result == std::vector<int>{8, 10, 13, 4});

  alg::transform(t0, t1, result.begin(),
                 [](const int a, const int b) { return a + b; });
  CHECK(result == std::vector<int>{11, 7, 15, -6});
}

// `alg::minmax_element` returns the first smallest and the last largest
// element, like the standard library
template <typename Container>
void check_minmax_element(const Container& c) {
  const auto greater = std::greater<>{};
  const auto [min, max] = std::ranges::minmax_element(c);
  CHECK(alg::minmax_element(c) == std::pair{min, max});
  const auto [min_greater, max_greater] =
      std::ranges::minmax_element(c, greater);
  CHECK(alg::minmax_element(c, greater) == std::pair{min_greater, max_greater});
}

// The loops must match the standard library, also for empty containers and
// repeated values
void test_same_as_std_on_random_data() {
  MAKE_GENERATOR(generator);
  std::uniform_int_distribution<int> distribution(-3, 3);
  const auto less_than_one = [](const int x) { return x < 1; };
  const auto greater = std::greater<>{};
  for (size_t size = 0; size < 20; ++size) {
    for (size_t repeat = 0; repeat < 20; ++repeat) {
      std::vector<int> a(size);
      std::vector<int> b(size);
      for (size_t i = 0; i < size; ++i) {
        a[i] = distribution(generator);
        b[i] = distribution(generator) > 1 ? -a[i] : a[i];
      }
      CAPTURE(a);
      CAPTURE(b);
      CHECK(alg::all_of(a, less_than_one) ==
            std::ranges::all_of(a, less_than_one));
      CHECK(alg::any_of(a, less_than_one) ==
            std::ranges::any_of(a, less_than_one));
      CHECK(alg::none_of(a, less_than_one) ==
            std::ranges::none_of(a, less_than_one));
      CHECK(alg::count(a, 1) == std::ranges::count(a, 1));
      CHECK(alg::count_if(a, less_than_one) ==
            std::ranges::count_if(a, less_than_one));
      CHECK(alg::equal(a, b) == std::ranges::equal(a, b));
      CHECK(alg::equal(a, b, greater) == std::ranges::equal(a, b, greater));
      CHECK(alg::min_element(a) == std::ranges::min_element(a));
      CHECK(alg::max_element(a) == std::ranges::max_element(a));
      CHECK(alg::min_element(a, greater) ==
            std::ranges::min_element(a, greater));
      CHECK(alg::max_element(a, greater) ==
            std::ranges::max_element(a, greater));
      check_minmax_element(a);
      // The overloads for other containers are loops as well
      const std::list<int> a_list(a.begin(), a.end());
      CHECK(alg::min_element(a_list) == std::ranges::min_element(a_list));
      CHECK(alg::max_element(a_list) == std::ranges::max_element(a_list));
      CHECK(alg::min_element(a_list, greater) ==
            std::ranges::min_element(a_list, greater));
      CHECK(alg::max_element(a_list, greater) ==
            std::ranges::max_element(a_list, greater));
      check_minmax_element(a_list);

      std::vector<int> transformed(size);
      std::vector<int> expected_transformed(size);
      CHECK(alg::transform(a, b, transformed.begin(), std::plus<>{}) ==
            transformed.end());
      std::ranges::transform(a, b, expected_transformed.begin(), std::plus<>{});
      CHECK(transformed == expected_transformed);

      auto removed = a;
      auto expected_removed = a;
      const auto removed_end = alg::remove(removed, 1);
      const auto expected_removed_end =
          std::ranges::remove(expected_removed, 1).begin();
      CHECK(std::ranges::equal(removed.begin(), removed_end,
                               expected_removed.begin(), expected_removed_end));
      removed = a;
      expected_removed = a;
      const auto removed_if_end = alg::remove_if(removed, less_than_one);
      const auto expected_removed_if_end =
          std::ranges::remove_if(expected_removed, less_than_one).begin();
      CHECK(std::ranges::equal(removed.begin(), removed_if_end,
                               expected_removed.begin(),
                               expected_removed_if_end));

      auto sorted = a;
      std::ranges::sort(sorted);
      auto bubble_sorted = a;
      alg::bubble_sort(bubble_sorted);
      CHECK(bubble_sorted == sorted);
      std::ranges::sort(sorted, greater);
      bubble_sorted = a;
      alg::bubble_sort(bubble_sorted, greater);
      CHECK(bubble_sorted == sorted);
      // `alg::bubble_sort` is stable
      std::vector<std::pair<int, size_t>> pairs(size);
      for (size_t i = 0; i < size; ++i) {
        pairs[i] = {a[i], i};
      }
      const auto compare_first = [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first;
      };
      auto stable_sorted = pairs;
      std::ranges::stable_sort(stable_sorted, compare_first);
      alg::bubble_sort(pairs, compare_first);
      CHECK(pairs == stable_sorted);
      // A type for which argument-dependent lookup finds the generic
      // `std::swap`
      std::vector<std::chrono::milliseconds> durations(size);
      std::vector<std::chrono::milliseconds> expected_durations(size);
      for (size_t i = 0; i < size; ++i) {
        durations[i] = std::chrono::milliseconds{a[i]};
        expected_durations[i] = std::chrono::milliseconds{sorted[size - 1 - i]};
      }
      alg::bubble_sort(durations);
      CHECK(durations == expected_durations);
    }
  }
}

// The other tests use containers with contiguous iterators, for which the
// functions are the device-callable loops. Other containers use the host-only
// overloads.
static_assert(Algorithm_detail::device_iterable<std::array<int, 3>>);
static_assert(Algorithm_detail::device_iterable<std::vector<int>>);
static_assert(not Algorithm_detail::device_iterable<std::list<int>>);
static_assert(
    not Algorithm_detail::device_iterable<std::unordered_map<int, int>>);
static_assert(not Algorithm_detail::device_iterable<DataVector>);

void test_non_contiguous_containers() {
  const std::list<int> a{3, 1, 4, 1, 5, 9};
  const std::list<int> b{3, 1, 4, 1, 5, 8};
  const auto less_than_two = [](const int x) { return x < 2; };
  const auto greater = std::greater<>{};
  CHECK_FALSE(alg::all_of(a, less_than_two));
  CHECK(alg::any_of(a, less_than_two));
  CHECK_FALSE(alg::none_of(a, less_than_two));
  CHECK(alg::count(a, 1) == 2);
  CHECK(alg::count_if(a, less_than_two) == 2);
  CHECK(alg::equal(a, a));
  CHECK_FALSE(alg::equal(a, b));
  CHECK(alg::equal(a, a, std::equal_to<>{}));
  CHECK(*alg::find(a, 4) == 4);
  CHECK(alg::find(a, 2) == a.end());
  CHECK(*alg::find_if(a, [](const int x) { return x > 4; }) == 5);
  CHECK(*alg::find_if_not(a, [](const int x) { return x > 2; }) == 1);
  CHECK(alg::found(a, 9));
  CHECK_FALSE(alg::found(a, 2));
  CHECK(alg::found_if(a, less_than_two));
  CHECK_FALSE(alg::found_if_not(a, [](const int x) { return x > 0; }));
  int sum = 0;
  alg::for_each(a, [&sum](const int x) { sum += x; });
  CHECK(sum == 23);
  CHECK(*alg::max_element(a) == 9);
  CHECK(*alg::max_element(a, greater) == 1);
  CHECK(*alg::min_element(a) == 1);
  CHECK(*alg::min_element(a, greater) == 9);
  check_minmax_element(a);
  std::vector<int> transformed(a.size());
  alg::transform(a, b, transformed.begin(), std::plus<>{});
  CHECK(transformed == std::vector<int>{6, 2, 8, 2, 10, 17});
  alg::transform(a, transformed.begin(), [](const int x) { return 2 * x; });
  CHECK(transformed == std::vector<int>{6, 2, 8, 2, 10, 18});
  std::list<int> removed = a;
  removed.erase(alg::remove(removed, 1), removed.end());
  CHECK(removed == std::list<int>{3, 4, 5, 9});
  removed.erase(alg::remove_if(removed, [](const int x) { return x > 4; }),
                removed.end());
  CHECK(removed == std::list<int>{3, 4});
  std::deque<int> sorted{3, 1, 4, 1, 5, 9};
  alg::sort(sorted);
  CHECK(sorted == std::deque<int>{1, 1, 3, 4, 5, 9});
  alg::sort(sorted, greater);
  CHECK(sorted == std::deque<int>{9, 5, 4, 3, 1, 1});

  const std::unordered_map<int, int> map{{1, 2}, {3, 4}};
  CHECK(
      alg::all_of(map, [](const auto& p) { return p.second == p.first + 1; }));
  CHECK(alg::found_if(map, [](const auto& p) { return p.first == 3; }));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.Algorithm", "[Unit][Utilities]") {
  test_constexpr_swap_iter_swap_reverse_etc();
  test_all_of_and_any_of_and_none_of();
  test_count_related();
  test_equal();
  test_find_related();
  test_for_each();
  test_function_objects_are_copied(std::vector<size_t>{1, 7, 234, 987, 32});
  test_function_objects_are_copied(std::list<size_t>{1, 7, 234, 987, 32});
  test_min_max_element();
  test_remove();
  test_sample();
  test_sort();
  test_transform();
  test_same_as_std_on_random_data();
  test_non_contiguous_containers();
}
