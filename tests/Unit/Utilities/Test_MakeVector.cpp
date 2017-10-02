// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "Utilities/MakeVector.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
struct NonCopyable {
  constexpr NonCopyable() = default;
  constexpr NonCopyable(const NonCopyable&) = delete;
  constexpr NonCopyable& operator=(const NonCopyable&) = delete;
  constexpr NonCopyable(NonCopyable&&) = default;
  NonCopyable& operator=(NonCopyable&&) = default;
  ~NonCopyable() = default;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.MakeVector", "[Unit][Utilities]") {
  auto varying_vector = make_vector(3.2, 4.3, 5.4, 6.5, 7.8);
  CHECK((varying_vector == std::vector<double>{{3.2, 4.3, 5.4, 6.5, 7.8}}));

  auto vector_n = make_vector(5, 5, 5, 5, 5, 5);
  static_assert(
      std::is_same<typename std::decay<decltype(vector_n)>::type::value_type,
                   int>::value,
      "Unit Test Failure: Incorrect type from make_vector.");

  CHECK(vector_n.size() == 6);
  for (const auto& p : vector_n) {
    CHECK(5 == p);
  }

  /// [make_vector_example]
  auto vector_non_copyable = make_vector(NonCopyable{}, NonCopyable{});
  CHECK(vector_non_copyable.size() == 2);

  auto vector_empty = make_vector<int>();
  CHECK(vector_empty.empty());

  auto my_vector = make_vector(1, 3, 4, 8, 9);
  CHECK(my_vector.size() == 5);

  auto vector_explicit_type = make_vector<double>(1, 2, 3);
  CHECK(vector_explicit_type == (std::vector<double>{1.,2.,3.}));
  /// [make_vector_example]
}
