// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <utility>

#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Structure/TrimMap.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <size_t Dim>
void test_remove_nonexistent_neighbors() {
  FixedHashMap<maximum_number_of_neighbors(Dim),
               std::pair<Direction<Dim>, ElementId<Dim>>, int,
               boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>
      map_to_trim{};

  DirectionMap<Dim, Neighbors<Dim>> neighbors{};
  for (size_t i = 0; i < 2 * Dim; ++i) {
    ElementId<Dim> id{i + 1, {}};
    neighbors[gsl::at(Direction<Dim>::all_directions(), i)] =
        Neighbors<Dim>{{id}, {}};
    map_to_trim[std::pair{gsl::at(Direction<Dim>::all_directions(), i), id}] =
        i * i + 10;  // Assign some number
  }
  const Element<Dim> element{ElementId<Dim>{0, {}}, neighbors};
  // Add extra neighbors that will be removed
  for (size_t i = 0; i < 2 * Dim and Dim > 1; ++i) {
    ElementId<Dim> id{i + 100, {}};
    map_to_trim[std::pair{gsl::at(Direction<Dim>::all_directions(), i), id}] =
        i * i + 1000;  // Assign some number
  }
  REQUIRE(map_to_trim.size() == (Dim == 1 ? 2 : 4) * Dim);
  domain::remove_nonexistent_neighbors(make_not_null(&map_to_trim), element);

  // Check that all expected neighbors are there
  for (const auto& [direction, neighbors_in_direction] : element.neighbors()) {
    for (const auto& neighbor : neighbors_in_direction) {
      CHECK(map_to_trim.contains(std::pair{direction, neighbor}));
    }
  }
  CHECK(map_to_trim.size() == 2 * Dim);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Structure.TrimMap", "[Domain][Unit]") {
  test_remove_nonexistent_neighbors<1>();
  test_remove_nonexistent_neighbors<2>();
  test_remove_nonexistent_neighbors<3>();
}
