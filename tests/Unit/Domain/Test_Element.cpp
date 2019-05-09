// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <functional>
#include <unordered_set>

#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/Neighbors.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/Side.hpp"
#include "Domain/Tags.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep
#include "tests/Unit/TestHelpers.hpp"

namespace {
template <size_t VolumeDim>
void check_element_work(const typename Element<VolumeDim>::Neighbors_t&
                            neighbors_in_largest_dimension,
                        const size_t expected_number_of_neighbors) {
  const ElementId<VolumeDim> id{5};
  const Element<VolumeDim> element(id, neighbors_in_largest_dimension);

  CHECK(element.id() == id);
  CHECK(element.neighbors() == neighbors_in_largest_dimension);
  CHECK(element.number_of_neighbors() == expected_number_of_neighbors);
  for (const auto& direction : Direction<VolumeDim>::all_directions()) {
    // The highest spatial dimension has neighbors; else, external boundary.
    CHECK((direction.dimension() == VolumeDim - 1) !=
          (element.external_boundaries().count(direction) == 1));
  }
  CHECK(element == element);
  CHECK_FALSE(element != element);

  const Element<VolumeDim> element_diff_id(ElementId<VolumeDim>(3),
                                           neighbors_in_largest_dimension);
  CHECK(element != element_diff_id);
  CHECK_FALSE(element == element_diff_id);

  const Element<VolumeDim> element_diff_neighbors(
      id, typename Element<VolumeDim>::Neighbors_t{
              {Direction<VolumeDim>::lower_xi(),
               neighbors_in_largest_dimension.at(
                   Direction<VolumeDim>(VolumeDim - 1, Side::Upper))}});
  CHECK(element != element_diff_neighbors);
  CHECK_FALSE(element == element_diff_neighbors);

  CHECK(get_output(element) ==
        "Element " + get_output(element.id()) + ":\n"
        "  Neighbors: " + get_output(element.neighbors()) + "\n"
        "  External boundaries: " + get_output(element.external_boundaries()) +
        "\n");

  test_serialization(element);

  CHECK(Tags::Element<VolumeDim>::name() == "Element");
}

void check_element_1d() {
  const Neighbors<1> neighbors_lower_xi(
      std::unordered_set<ElementId<1>>{ElementId<1>(7)}, OrientationMap<1>{});
  const Neighbors<1> neighbors_upper_xi(
      std::unordered_set<ElementId<1>>{ElementId<1>(7)}, OrientationMap<1>{});
  const typename Element<1>::Neighbors_t xi_neighbors{
      {Direction<1>::lower_xi(), neighbors_lower_xi},
      {Direction<1>::upper_xi(), neighbors_upper_xi}};
  check_element_work<1>(xi_neighbors, 2);
}

void check_element_2d() {
  const Neighbors<2> neighbors_lower_eta(
      std::unordered_set<ElementId<2>>{ElementId<2>(7), ElementId<2>(4)},
      OrientationMap<2>{});
  const Neighbors<2> neighbors_upper_eta(
      std::unordered_set<ElementId<2>>{ElementId<2>(7), ElementId<2>(4)},
      OrientationMap<2>{});
  const typename Element<2>::Neighbors_t eta_neighbors{
      {Direction<2>::lower_eta(), neighbors_lower_eta},
      {Direction<2>::upper_eta(), neighbors_upper_eta}};
  check_element_work<2>(eta_neighbors, 4);
}

void check_element_3d() {
  const Neighbors<3> neighbors_lower_zeta(
      std::unordered_set<ElementId<3>>{ElementId<3>(7), ElementId<3>(4),
                                       ElementId<3>(9), ElementId<3>(2)},
      OrientationMap<3>{});
  const Neighbors<3> neighbors_upper_zeta(
      std::unordered_set<ElementId<3>>{ElementId<3>(7), ElementId<3>(4),
                                       ElementId<3>(9), ElementId<3>(2)},
      OrientationMap<3>{});
  const typename Element<3>::Neighbors_t zeta_neighbors{
      {Direction<3>::lower_zeta(), neighbors_lower_zeta},
      {Direction<3>::upper_zeta(), neighbors_upper_zeta}};
  check_element_work<3>(zeta_neighbors, 8);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Element", "[Domain][Unit]") {
  check_element_1d();
  check_element_2d();
  check_element_3d();
}
