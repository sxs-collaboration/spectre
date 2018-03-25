// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <map>
#include <string>

#include "Utilities/CachedFunction.hpp"
#include "Utilities/TypeTraits.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.CachedFunction", "[Unit][Utilities]") {
  size_t call_count = 0;
  // Use types that cannot be implicitly converted
  const auto func = [&call_count](const std::string& s) noexcept {
    ++call_count;
    return s.size();
  };

  bool comparator_used = false;
  // Unused capture to silence clang-tidy warning in the example below.
  auto comparator = [&comparator_used, not_trivially_copyable=std::string{}](
      const std::string& a, const std::string& b) noexcept {
    comparator_used = true;
    return a < b;
  };

  /// [make_cached_function_example]
  auto cached = make_cached_function<const std::string&>(func);

  // Specifying all the arguments
  auto cached2 =
      make_cached_function<const std::string&, std::map, decltype(comparator)>(
          func, std::move(comparator));
  /// [make_cached_function_example]

  static_assert(cpp17::is_same_v<decltype(cached)::input, std::string>,
                "Wrong input type");
  static_assert(cpp17::is_same_v<decltype(cached)::output, size_t>,
                "Wrong output type");

  static_assert(cpp17::is_same_v<decltype(cached2)::input, std::string>,
                "Wrong input type");
  static_assert(cpp17::is_same_v<decltype(cached2)::output, size_t>,
                "Wrong output type");

  CHECK(call_count == 0);
  CHECK(cached("x") == 1);
  CHECK(call_count == 1);
  CHECK(cached("xx") == 2);
  CHECK(call_count == 2);
  CHECK(cached("x") == 1);
  CHECK(call_count == 2);
  cached.clear();
  CHECK(cached("x") == 1);
  CHECK(call_count == 3);

  CHECK(not comparator_used);
  CHECK(cached2("x") == 1);
  CHECK(cached2("xx") == 2);
  CHECK(comparator_used);
}
