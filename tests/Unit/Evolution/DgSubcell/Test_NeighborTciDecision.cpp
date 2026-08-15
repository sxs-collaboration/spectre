// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/NeighborTciDecision.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell {
namespace {
// Builds an element with:
// - lower_xi: one conforming neighbor (block 0) -> ConformingAligned
// - upper_xi: two non-conforming neighbors (blocks 2, 3) ->
//             MultipleNonconforming
template <size_t Dim>
Element<Dim> make_test_element() {
  Neighbors<Dim> conforming_nbrs{{ElementId<Dim>{0}},
                                 OrientationMap<Dim>::create_aligned()};
  const std::unordered_map<size_t, OrientationMap<Dim>> nc_orientations{
      {2, OrientationMap<Dim>::create_aligned()},
      {3, OrientationMap<Dim>::create_aligned()}};
  Neighbors<Dim> nc_nbrs{
      {ElementId<Dim>{2}, ElementId<Dim>{3}}, nc_orientations, false};
  return Element<Dim>{ElementId<Dim>{1},
                      {{Direction<Dim>::lower_xi(), conforming_nbrs},
                       {Direction<Dim>::upper_xi(), nc_nbrs}}};
}

template <size_t Dim>
void test() {
  using tag = subcell::Tags::NeighborTciDecisions<Dim>;
  using Type = typename tag::type;

  const Element<Dim> element = make_test_element<Dim>();
  const DirectionalId<Dim> conforming_did{Direction<Dim>::lower_xi(),
                                          ElementId<Dim>{0}};
  const DirectionalId<Dim> nc_did{Direction<Dim>::upper_xi(),
                                  ElementId<Dim>{2}};

  evolution::dg::BoundaryData<Dim> neighbor_data{};
  neighbor_data.tci_status = 10;

  {
    INFO("Empty map: function exits early without update or error");
    auto box = db::create<db::AddSimpleTags<tag, domain::Tags::Element<Dim>>>(
        Type{}, element);
    neighbor_tci_decision(make_not_null(&box), conforming_did, neighbor_data);
    CHECK(db::get<tag>(box).empty());
  }

  {
    INFO("MultipleNonconforming direction: no-op even with a non-empty map");
    Type decisions;
    decisions.insert({conforming_did, 0});
    auto box = db::create<db::AddSimpleTags<tag, domain::Tags::Element<Dim>>>(
        std::move(decisions), element);
    neighbor_tci_decision(make_not_null(&box), nc_did, neighbor_data);
    CHECK(db::get<tag>(box).at(conforming_did) == 0);  // unchanged
  }

  {
    INFO("Normal conforming case: status is updated");
    Type decisions;
    decisions.insert({conforming_did, 0});
    auto box = db::create<db::AddSimpleTags<tag, domain::Tags::Element<Dim>>>(
        std::move(decisions), element);
    neighbor_tci_decision(make_not_null(&box), conforming_did, neighbor_data);
    CHECK(db::get<tag>(box).at(conforming_did) == 10);
  }

#ifdef SPECTRE_DEBUG
  {
    INFO("Non-empty map, conforming direction, neighbor not present -> ASSERT");
    const DirectionalId<Dim> absent_did{Direction<Dim>::lower_xi(),
                                        ElementId<Dim>{99}};
    Type decisions;
    decisions.insert({conforming_did, 0});
    auto box = db::create<db::AddSimpleTags<tag, domain::Tags::Element<Dim>>>(
        std::move(decisions), element);
    CHECK_THROWS_WITH(
        neighbor_tci_decision(make_not_null(&box), absent_did, neighbor_data),
        Catch::Matchers::ContainsSubstring(
            "NeighborTciDecisions does not contain the neighbor"));
  }
#endif
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.NeighborTciDecision",
                  "[Evolution][Unit]") {
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace
}  // namespace evolution::dg::subcell
