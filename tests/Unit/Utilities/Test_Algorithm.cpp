// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <iterator>

#include "Utilities/Algorithm.hpp"
#include "Utilities/Array.hpp"  // IWYU pragma: associated
#include "Utilities/Numeric.hpp"

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
}
