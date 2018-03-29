// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "DataStructures/GeneralIndexIterator.hpp"
#include "Utilities/Gsl.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.GeneralIndexIterator",
                  "[Unit][DataStructures]") {
  // Test constructing from const value
  const std::array<std::pair<int, int>, 3> ranges{{{0, 2}, {1, 2}, {1, 3}}};
  auto coordinates = make_general_index_iterator(ranges);

  auto check_values = [&](std::array<int, 3> expected) {
    CHECK(coordinates);
    auto it = coordinates.begin();
    for(size_t i = 0; i < expected.size(); ++i, ++it) {
      CHECK(coordinates[i] == gsl::at(expected, i));
      CHECK(*it == gsl::at(expected, i));
    }
    CHECK(it == coordinates.end());
  };
  check_values({{0, 1, 1}});
  ++coordinates;
  check_values({{1, 1, 1}});
  ++coordinates;
  check_values({{0, 1, 2}});
  ++coordinates;
  check_values({{1, 1, 2}});
  ++coordinates;
  CHECK_FALSE(coordinates);

  // Test constructing from temporary
  auto zero_dimensional_coordinates = make_general_index_iterator(
      std::array<std::pair<int, int>, 0>{});
  CHECK(zero_dimensional_coordinates);
  CHECK(zero_dimensional_coordinates.begin() ==
        zero_dimensional_coordinates.end());
  ++zero_dimensional_coordinates;
  CHECK_FALSE(zero_dimensional_coordinates);

  auto degenerate_coordinates = make_general_index_iterator(
      std::array<std::pair<int, int>, 1>{{{2, 2}}});
  CHECK_FALSE(degenerate_coordinates);
}
