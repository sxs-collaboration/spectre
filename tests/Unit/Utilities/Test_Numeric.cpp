// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "Utilities/Array.hpp"  // IWYU pragma: associated
#include "Utilities/Literals.hpp"
#include "Utilities/Numeric.hpp"

namespace {
constexpr bool check_iota() noexcept {
  const size_t size = 6;
  cpp17::array<size_t, size> a{};
  cpp2b::iota(a.begin(), a.end(), size_t(1));
  bool check{true};
  for (size_t i = 0; i < size; ++i) {
    check = check and a[i] == i + 1;
  }

  const cpp17::array<size_t, 4> inline_iota =
      alg::iota(cpp17::array<size_t, 4>{{}}, 7_st);
  for (size_t i = 0; i < inline_iota.size(); ++i) {
    check = check and inline_iota[i] == i + 7;
  }
  return check;
}

constexpr bool check_accumulate() noexcept {
  cpp17::array<size_t, 6> a{};
  cpp2b::iota(a.begin(), a.end(), size_t(1));
  size_t sum = cpp2b::accumulate(a.begin(), a.end(), size_t(23));
  return sum == 44;
}

SPECTRE_TEST_CASE("Unit.Utilities.Numeric", "[Unit][Utilities]") {
  // Check the functions at runtime
  CHECK(check_iota());
  CHECK(check_accumulate());

  // Check the functions at compile time
  static_assert(check_iota(), "Failed test Unit.Utilities.Numeric");
  static_assert(check_accumulate(), "Failed test Unit.Utilities.Numeric");

  // Check STL wrappers
  CHECK(alg::accumulate(std::array<int, 3>{{1, -3, 7}}, 4) == 9);
  CHECK(alg::accumulate(std::array<int, 3>{{1, -3, 7}}, 4,
                        std::multiplies<>{}) == -84);
  CHECK(alg::accumulate(std::array<int, 3>{{1, -3, 7}},
                        4, [](const int state, const int element) noexcept {
                          return state * element;
                        }) == -84);
}
}  // namespace
