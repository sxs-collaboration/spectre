// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <unordered_set>

#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/Neighbors.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/Tags.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep
#include "tests/Unit/TestHelpers.hpp"

namespace {
template <size_t VolumeDim>
void check_element() {
  const domain::ElementId<VolumeDim> id{5};

  const domain::Neighbors<VolumeDim> neighbors(
      std::unordered_set<domain::ElementId<VolumeDim>>{
          domain::ElementId<VolumeDim>(7), domain::ElementId<VolumeDim>(4)},
      domain::OrientationMap<VolumeDim>{});
  const typename domain::Element<VolumeDim>::Neighbors_t two_neighbors{
      {domain::Direction<VolumeDim>::lower_xi(), neighbors},
      {domain::Direction<VolumeDim>::upper_xi(), neighbors}};

  const domain::Element<VolumeDim> element(id, two_neighbors);

  CHECK(element.id() == id);
  CHECK(element.neighbors() == two_neighbors);
  CHECK(element.number_of_neighbors() == 4);
  for (const auto& direction : domain::Direction<VolumeDim>::all_directions()) {
    // Either a xi direction or an external boundary, but not both.
    CHECK((direction.axis() == domain::Direction<VolumeDim>::Axis::Xi) !=
          (element.external_boundaries().count(direction) == 1));
  }

  CHECK(element == element);
  CHECK_FALSE(element != element);
  const domain::Element<VolumeDim> element_diff_id(
      domain::ElementId<VolumeDim>(3), two_neighbors);
  CHECK(element != element_diff_id);
  CHECK_FALSE(element == element_diff_id);
  const domain::Element<VolumeDim> element_diff_neighbors(
      id, typename domain::Element<VolumeDim>::Neighbors_t{
              {domain::Direction<VolumeDim>::lower_xi(), neighbors}});
  CHECK(element != element_diff_neighbors);
  CHECK_FALSE(element == element_diff_neighbors);

  CHECK(get_output(element) ==
        "Element " + get_output(element.id()) + ":\n"
        "  Neighbors: " + get_output(element.neighbors()) + "\n"
        "  External boundaries: " + get_output(element.external_boundaries()) +
        "\n");

  test_serialization(element);

  CHECK(domain::Tags::Element<VolumeDim>::name() == "Element");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Element", "[Domain][Unit]") {
  check_element<1>();
  check_element<2>();
  check_element<3>();
}
