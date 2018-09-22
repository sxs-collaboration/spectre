// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <iterator>
#include <vector>

#include "Utilities/Algorithm.hpp"
#include "Utilities/Array.hpp"  // IWYU pragma: associated
#include "Utilities/Numeric.hpp"

namespace {
constexpr bool check_swap() noexcept {
  int x = 4;
  int y = 444;
  cpp20::swap(x, y);
  return x == 444 and y == 4;
}

constexpr bool check_iter_swap() noexcept {
  const size_t size = 6;
  cpp17::array<size_t, size> a{};
  cpp2b::iota(a.begin(), a.end(), size_t(1));
  cpp20::iter_swap(a.begin() + 1, a.begin() + 4);

  cpp17::array<size_t, size> expected{{1, 5, 3, 4, 2, 6}};
  return a == expected;
}

constexpr bool check_reverse() noexcept {
  const size_t size = 6;
  cpp17::array<size_t, size> a{};
  cpp2b::iota(a.begin(), a.end(), size_t(1));
  cpp20::detail::reverse(a.begin(), a.end(), std::bidirectional_iterator_tag());

  cpp17::array<size_t, size> expected{{6, 5, 4, 3, 2, 1}};
  return a == expected;
}

constexpr bool check_reverse_random_access() noexcept {
  const size_t size = 6;
  cpp17::array<size_t, size> a{};
  cpp2b::iota(a.begin(), a.end(), size_t(1));
  // a is a random_access_iterator, so the next line tests both the generic
  // cpp20::reverse() and the one for random access iterators
  cpp20::reverse(a.begin(), a.end());

  cpp17::array<size_t, size> expected{{6, 5, 4, 3, 2, 1}};
  return a == expected;
}

constexpr bool check_next_permutation() noexcept {
  const size_t size = 6;
  cpp17::array<size_t, size> a{{1, 2, 4, 6, 3, 5}};
  cpp20::next_permutation(a.begin(), a.end());

  cpp17::array<size_t, size> expected{{1, 2, 4, 6, 5, 3}};
  return a == expected;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.Algorithm", "[Unit][Utilities]") {
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
  const std::vector<int> a{1, 2, 3, 4};
  CHECK(alg::find(a, 3) == a.begin() + 2);
  CHECK(alg::find(a, 5) == a.end());
  CHECK(alg::found(a, 3));
  CHECK_FALSE(alg::found(a, 5));

  CHECK(alg::find_if(a, [](const int t) { return t > 3; }) == a.end() - 1);
  CHECK(alg::find_if(a, [](const int t) { return t < 0; }) == a.end());
  CHECK(alg::found_if(a, [](const int t) { return t > 3; }));
  CHECK_FALSE(alg::found_if(a, [](const int t) { return t < 0; }));

  CHECK(alg::find_if_not(a, [](const int t) { return t < 3; }) == a.end() - 2);
  CHECK(alg::find_if_not(a, [](const int t) { return t < 8; }) == a.end());
  CHECK(alg::found_if_not(a, [](const int t) { return t < 3; }));
  CHECK_FALSE(alg::found_if_not(a, [](const int t) { return t < 8; }));

  int count = 0;
  alg::for_each(std::vector<size_t>{1, 7, 234, 987, 32},
                [&count](const size_t value) {
                  if (value > 100) {
                    count++;
                  }
                });
  CHECK(count == 2);

  // Test alg::equal
  CHECK(alg::equal(std::vector<int>{1, -7, 8, 9},
                   std::array<int, 4>{{1, -7, 8, 9}}));
  CHECK_FALSE(alg::equal(std::vector<int>{1, 7, 8, 9},
                         std::array<int, 4>{{1, -7, 8, 9}}));

  CHECK(alg::equal(
      std::vector<int>{1, -7, 8, 9}, std::array<int, 4>{{-1, 7, -8, -9}},
      [](const int lhs, const int rhs) noexcept { return lhs == -rhs; }));
  CHECK_FALSE(alg::equal(
      std::vector<int>{1, -7, 8, 9}, std::array<int, 4>{{-1, -7, -8, -9}},
      [](const int lhs, const int rhs) noexcept { return lhs == -rhs; }));
}
