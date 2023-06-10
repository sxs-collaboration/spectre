// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <optional>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Evolution/DgSubcell/NeighborTciDecision.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell {
template <size_t Dim>
void test() {
  using tag = subcell::Tags::NeighborTciDecisions<Dim>;
  using Type = typename tag::type;
  auto box = db::create<db::AddSimpleTags<tag>>(Type{});
  using StorageType =
      std::tuple<Mesh<Dim>, Mesh<Dim - 1>, std::optional<DataVector>,
                 std::optional<DataVector>, ::TimeStepId, int>;
  std::pair<
      const TimeStepId,
      FixedHashMap<maximum_number_of_neighbors(Dim),
                   std::pair<Direction<Dim>, ElementId<Dim>>, StorageType,
                   boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>>
      neighbor_data{};
  const std::pair id_xi{Direction<Dim>::lower_xi(), ElementId<Dim>{0}};
  neighbor_data.second.insert(std::pair{id_xi, StorageType{}});
  std::get<5>(neighbor_data.second.at(id_xi)) = 10;
  std::pair<Direction<Dim>, ElementId<Dim>> id_eta;
  std::pair<Direction<Dim>, ElementId<Dim>> id_zeta;
  if constexpr (Dim > 1) {
    id_eta = std::pair{Direction<Dim>::lower_eta(), ElementId<Dim>{2}};
    neighbor_data.second.insert(std::pair{id_eta, StorageType{}});
    std::get<5>(neighbor_data.second.at(id_eta)) = 12;
  }
  if constexpr (Dim > 2) {
    id_zeta = std::pair{Direction<Dim>::lower_zeta(), ElementId<Dim>{5}};
    neighbor_data.second.insert(std::pair{id_zeta, StorageType{}});
    std::get<5>(neighbor_data.second.at(id_zeta)) = 15;
  }
#ifdef SPECTRE_DEBUG
  // check ASSERT for neighbors works
  CHECK_THROWS_WITH(
      neighbor_tci_decision(make_not_null(&box), neighbor_data),
      Catch::Matchers::Contains(
          "The NeighborTciDecisions tag does not contain the neighbor"));
#endif
  db::mutate<tag>(
      [&id_xi, &id_eta, &id_zeta](const auto neighbor_decisions_ptr) {
        (void)id_eta, (void)id_zeta;
        neighbor_decisions_ptr->insert(std::pair{id_xi, 0});
        if constexpr (Dim > 1) {
          neighbor_decisions_ptr->insert(std::pair{id_eta, 0});
        }
        if constexpr (Dim > 2) {
          neighbor_decisions_ptr->insert(std::pair{id_zeta, 0});
        }
      },
      make_not_null(&box));
  neighbor_tci_decision(make_not_null(&box), neighbor_data);
  CHECK(db::get<tag>(box).at(id_xi) == 10);
  if constexpr (Dim > 1) {
    CHECK(db::get<tag>(box).at(id_eta) == 12);
  }
  if constexpr (Dim > 2) {
    CHECK(db::get<tag>(box).at(id_zeta) == 15);
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.NeighborTciDecision",
                  "[Evolution][Unit]") {
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace evolution::dg::subcell
