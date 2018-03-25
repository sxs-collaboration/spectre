// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <type_traits>
#include <vector>

#include "Utilities/MakeVector.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.MakeVector", "[Unit][Utilities]") {
  static_assert(
      std::is_same<
          typename std::decay<decltype(make_vector(0., 0.))>::type::value_type,
          double>::value,
      "Unit Test Failure: Incorrect type from make_vector.");

  /// [make_vector_example]
  auto vector = make_vector(3.2, 4.3, 5.4, 6.5, 7.8);
  CHECK(vector == (std::vector<double>{{3.2, 4.3, 5.4, 6.5, 7.8}}));

  auto vector_non_copyable = make_vector(NonCopyable{}, NonCopyable{});
  CHECK(vector_non_copyable.size() == 2);

  auto vector_empty = make_vector<int>();
  CHECK(vector_empty.empty());

  auto vector_size_one = make_vector(3);
  CHECK(vector_size_one == std::vector<int>{3});

  auto vector_explicit_type = make_vector<double>(1, 2, 3);
  CHECK(vector_explicit_type == (std::vector<double>{1.,2.,3.}));
  /// [make_vector_example]
}
