// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/NeighborTciDecision.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell {
namespace {
template <size_t Dim>
void test() {
  using tag = subcell::Tags::NeighborTciDecisions<Dim>;
  using Type = typename tag::type;
  auto box = db::create<db::AddSimpleTags<tag>>(Type{});
  using StorageType = evolution::dg::BoundaryData<Dim>;
  StorageType neighbor_data{};
  const DirectionalId<Dim> id_xi{Direction<Dim>::lower_xi(), ElementId<Dim>{0}};
  neighbor_data.tci_status = 10;
#ifdef SPECTRE_DEBUG
  // check ASSERT for neighbors works
  CHECK_THROWS_WITH(
      neighbor_tci_decision(make_not_null(&box), id_xi, neighbor_data),
      Catch::Matchers::ContainsSubstring(
          "The NeighborTciDecisions tag does not contain the neighbor"));
#endif
  db::mutate<tag>(
      [&id_xi](const auto neighbor_decisions_ptr) {
        neighbor_decisions_ptr->insert(std::pair{id_xi, 0});
      },
      make_not_null(&box));
  neighbor_tci_decision(make_not_null(&box), id_xi, neighbor_data);
  CHECK(db::get<tag>(box).at(id_xi) == 10);
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.NeighborTciDecision",
                  "[Evolution][Unit]") {
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace
}  // namespace evolution::dg::subcell
