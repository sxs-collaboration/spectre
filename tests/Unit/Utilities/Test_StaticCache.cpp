// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "ErrorHandling/Error.hpp"
#include "Utilities/StaticCache.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.StaticCache", "[Utilities][Unit]") {
  /// [static_cache]
  const static auto cache =
      make_static_cache<CacheRange<0, 3>, CacheRange<3, 5>>([](
          const size_t a, const size_t b) noexcept { return a + b; });
  CHECK(cache(0, 3) == 3);  // smallest entry
  CHECK(cache(2, 4) == 6);  // largest entry
  /// [static_cache]

  std::vector<std::pair<size_t, size_t>> calls;
  const auto cache2 =
      make_static_cache<CacheRange<0, 3>, CacheRange<3, 5>>([&calls](
          const size_t a, const size_t b) noexcept {
        calls.emplace_back(a, b);
        return a + b;
      });
  CHECK(calls.size() == 6);
  // Creation order is not specified.
  std::sort(calls.begin(), calls.end());
  const decltype(calls) expected_calls{{0, 3}, {0, 4}, {1, 3},
                                       {1, 4}, {2, 3}, {2, 4}};
  CHECK(calls == expected_calls);
  for (const auto& call : expected_calls) {
    CHECK(cache2(call.first, call.second) == call.first + call.second);
  }
  CHECK(calls == expected_calls);

  size_t small_calls = 0;
  const auto small_cache = make_static_cache([&small_calls]() noexcept {
    ++small_calls;
    return size_t{5};
  });
  CHECK(small_calls == 1);
  CHECK(small_cache() == 5);
  CHECK(small_calls == 1);
}

// [[OutputRegex, Index out of range: 3 <= 2 < 5]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Utilities.StaticCache.out_of_range.low",
                               "[Utilities][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const auto cache = make_static_cache<CacheRange<3, 5>>([](
      const size_t x) noexcept { return x; });
  cache(2);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Index out of range: 3 <= 5 < 5]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Utilities.StaticCache.out_of_range.high",
                               "[Utilities][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const auto cache = make_static_cache<CacheRange<3, 5>>([](
      const size_t x) noexcept { return x; });
  cache(5);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
