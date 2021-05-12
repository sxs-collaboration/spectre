// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <deque>
#include <iterator>
#include <memory>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Actions/ReconstructionCommunication.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/NeighborData.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/NeighborData.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/TciGridHistory.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforthN.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct System {
  static constexpr size_t volume_dim = Dim;
  using variables_tag = Tags::Variables<tmpl::list<Var1>>;
};

template <size_t Dim, typename Metavariables>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Dim>;

  using initial_tags = tmpl::list<
      Tags::TimeStepId, Tags::Next<Tags::TimeStepId>, domain::Tags::Mesh<Dim>,
      evolution::dg::subcell::Tags::Mesh<Dim>,
      evolution::dg::subcell::Tags::ActiveGrid, domain::Tags::Element<Dim>,
      evolution::dg::subcell::Tags::NeighborDataForReconstructionAndRdmpTci<
          Dim>,
      Tags::Variables<tmpl::list<Var1>>, evolution::dg::Tags::MortarData<Dim>,
      evolution::dg::Tags::MortarNextTemporalId<Dim>>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<
          ActionTesting::InitializeDataBox<initial_tags>,
          evolution::dg::subcell::Actions::SendDataForReconstruction<
              Dim, typename Metavariables::GhostDataMutator>,
          evolution::dg::subcell::Actions::ReceiveDataForReconstruction<Dim>>>>;
};

template <size_t Dim>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  static constexpr bool local_time_stepping = false;
  using component_list = tmpl::list<component<Dim, Metavariables>>;
  using system = System<Dim>;
  using const_global_cache_tags = tmpl::list<>;
  enum class Phase { Initialization, Exit };

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static bool ghost_zone_size_invoked;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static bool ghost_data_mutator_invoked;

  struct SubcellOptions {
    template <typename DbTagsList>
    static size_t ghost_zone_size(const db::DataBox<DbTagsList>& box) noexcept {
      CHECK(db::get<domain::Tags::Mesh<Dim>>(box) ==
            Mesh<Dim>(5, Spectral::Basis::Legendre,
                      Spectral::Quadrature::GaussLobatto));
      ghost_zone_size_invoked = true;
      return 2;
    }
  };

  struct GhostDataMutator {
    using return_tags = tmpl::list<>;
    using argument_tags = tmpl::list<Tags::Variables<tmpl::list<Var1>>>;
    static Variables<tmpl::list<Var1>> apply(
        const Variables<tmpl::list<Var1>>& vars) noexcept {
      ghost_data_mutator_invoked = true;
      // make some trivial but testable modification
      auto result = vars;
      get(get<Var1>(result)) *= 2.0;
      return result;
    }
  };
};

template <size_t Dim>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
bool Metavariables<Dim>::ghost_zone_size_invoked = false;
template <size_t Dim>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
bool Metavariables<Dim>::ghost_data_mutator_invoked = false;

template <size_t Dim>
void test() {
  CAPTURE(Dim);

  using metavars = Metavariables<Dim>;
  metavars::ghost_zone_size_invoked = false;
  metavars::ghost_data_mutator_invoked = false;
  using comp = component<Dim, metavars>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  MockRuntimeSystem runner{{}};

  const TimeStepId time_step_id{true, 1, Time{Slab{1.0, 2.0}, {0, 10}}};
  const TimeStepId next_time_step_id{true, 1, Time{Slab{1.0, 2.0}, {1, 10}}};
  const Mesh<Dim> dg_mesh{5, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const evolution::dg::subcell::ActiveGrid active_grid =
      evolution::dg::subcell::ActiveGrid::Subcell;

  // ^ eta
  // +-+-+> xi
  // |X| |
  // +-+-+
  // | | |
  // +-+-+
  //
  // The "self_id" for the element that we are considering is marked by an X in
  // the diagram. We consider a configuration with one neighbor in the +xi
  // direction (east_id), and (in 2d and 3d) one in the -eta (south_id)
  // direction.
  //
  // In 1d there aren't any projections to test, and in 3d we only have 1
  // element in the z-direction.
  DirectionMap<Dim, Neighbors<Dim>> neighbors{};
  ElementId<Dim> self_id{};
  ElementId<Dim> east_id{};
  ElementId<Dim> south_id{};  // not used in 1d
  OrientationMap<Dim> orientation{};
  // Note: in 2d and 3d it is the neighbor in the lower eta direction that has a
  // non-trivial orientation.

  if constexpr (Dim == 1) {
    self_id = ElementId<Dim>{0, {{{1, 0}}}};
    east_id = ElementId<Dim>{0, {{{1, 1}}}};
    neighbors[Direction<Dim>::upper_xi()] = Neighbors<Dim>{{east_id}, {}};
  } else if constexpr (Dim == 2) {
    orientation = OrientationMap<Dim>{
        std::array{Direction<Dim>::lower_xi(), Direction<Dim>::lower_eta()}};
    self_id = ElementId<Dim>{0, {{{1, 0}, {0, 0}}}};
    east_id = ElementId<Dim>{0, {{{1, 1}, {0, 0}}}};
    south_id = ElementId<Dim>{1, {{{0, 0}, {0, 0}}}};
    neighbors[Direction<Dim>::upper_xi()] = Neighbors<Dim>{{east_id}, {}};
    neighbors[Direction<Dim>::lower_eta()] =
        Neighbors<Dim>{{south_id}, orientation};
  } else {
    static_assert(Dim == 3, "Only implemented tests in 1, 2, and 3d");
    orientation = OrientationMap<Dim>{std::array{Direction<Dim>::lower_xi(),
                                                 Direction<Dim>::lower_eta(),
                                                 Direction<Dim>::upper_zeta()}};
    self_id = ElementId<Dim>{0, {{{1, 0}, {0, 0}, {0, 0}}}};
    east_id = ElementId<Dim>{0, {{{1, 1}, {0, 0}, {0, 0}}}};
    south_id = ElementId<Dim>{1, {{{0, 0}, {0, 0}, {0, 0}}}};
    neighbors[Direction<Dim>::upper_xi()] = Neighbors<Dim>{{east_id}, {}};
    neighbors[Direction<Dim>::lower_eta()] =
        Neighbors<Dim>{{south_id}, orientation};
  }
  const Element<Dim> element{self_id, neighbors};

  using NeighborDataMap =
      FixedHashMap<maximum_number_of_neighbors(Dim) + 1,
                   std::pair<Direction<Dim>, ElementId<Dim>>,
                   evolution::dg::subcell::NeighborData,
                   boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>;
  NeighborDataMap neighbor_data{};
  const std::pair self_neighbor_id{Direction<Dim>::lower_xi(),
                                   ElementId<Dim>::external_boundary_id()};
  const std::pair east_neighbor_id{Direction<Dim>::upper_xi(), east_id};
  // insert data from one of the neighbors to make sure the send actions clears
  // it.
  neighbor_data[east_neighbor_id] = {};

  using evolved_vars_tags = tmpl::list<Var1>;
  Variables<evolved_vars_tags> evolved_vars{
      subcell_mesh.number_of_grid_points()};
  // Set Var1 to the logical coords, just need some data
  get(get<Var1>(evolved_vars)) = get<0>(logical_coordinates(subcell_mesh));

  using MortarData = typename evolution::dg::Tags::MortarData<Dim>::type;
  using MortarNextId =
      typename evolution::dg::Tags::MortarNextTemporalId<Dim>::type;
  MortarData mortar_data{};
  MortarNextId mortar_next_id{};
  mortar_data[east_neighbor_id] = {};
  mortar_next_id[east_neighbor_id] = {};
  if constexpr (Dim > 1) {
    const std::pair south_neighbor_id{Direction<Dim>::lower_eta(), south_id};
    mortar_data[south_neighbor_id] = {};
    mortar_next_id[south_neighbor_id] = {};
  }

  ActionTesting::emplace_array_component_and_initialize<comp>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, self_id,
      {time_step_id, next_time_step_id, dg_mesh, subcell_mesh, active_grid,
       element, neighbor_data, evolved_vars, mortar_data, mortar_next_id});
  for (const auto& [direction, neighbor_ids] : neighbors) {
    (void)direction;
    for (const auto& neighbor_id : neighbor_ids) {
      // Initialize neighbors with garbage data. We won't ever run any actions
      // on them, we just need to insert them to make sure things are sent to
      // the right places. We'll check their inboxes directly.
      ActionTesting::emplace_array_component_and_initialize<comp>(
          &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0},
          neighbor_id,
          {time_step_id, next_time_step_id, Mesh<Dim>{}, Mesh<Dim>{},
           active_grid, Element<Dim>{}, NeighborDataMap{},
           Variables<evolved_vars_tags>{}, MortarData{}, MortarNextId{}});
    }
  }

  using neighbor_data_tag =
      evolution::dg::subcell::Tags::NeighborDataForReconstructionAndRdmpTci<
          Dim>;
  using ActionTesting::get_databox_tag;
  CHECK(get_databox_tag<comp, neighbor_data_tag>(runner, self_id).size() == 1);
  CHECK(get_databox_tag<comp, neighbor_data_tag>(runner, self_id)
            .count(east_neighbor_id) == 1);

  // Run the SendDataForReconstruction action on self_id
  ActionTesting::next_action<comp>(make_not_null(&runner), self_id);

  CHECK(get_databox_tag<comp, neighbor_data_tag>(runner, self_id).size() == 1);
  CHECK(get_databox_tag<comp, neighbor_data_tag>(runner, self_id)
            .count(self_neighbor_id) == 1);

  // Check local RDMP data
  const evolution::dg::subcell::NeighborData& local_neighbor_data =
      get_databox_tag<comp, neighbor_data_tag>(runner, self_id)
          .find(self_neighbor_id)
          ->second;
  CHECK(local_neighbor_data.max_variables_values.size() == 1);
  CHECK(local_neighbor_data.max_variables_values[0] ==
        max(get(get<Var1>(evolved_vars))));
  CHECK(local_neighbor_data.min_variables_values.size() == 1);
  CHECK(local_neighbor_data.min_variables_values[0] ==
        min(get(get<Var1>(evolved_vars))));
  // Check data sent to neighbors
  DirectionMap<Dim, bool> directions_to_slice{};
  for (const auto& direction_neighbors : element.neighbors()) {
    if (direction_neighbors.second.size() == 0) {
      directions_to_slice[direction_neighbors.first] = false;
    } else {
      directions_to_slice[direction_neighbors.first] = true;
    }
  }
  const size_t ghost_zone_size = 2;
  const DirectionMap<Dim, std::vector<double>> all_sliced_data =
      evolution::dg::subcell::slice_data(evolved_vars, subcell_mesh.extents(),
                                         ghost_zone_size, directions_to_slice);
  {
    std::vector<double> expected_east_data =
        all_sliced_data.at(east_neighbor_id.first);
    for (double& value : expected_east_data) {
      value *= 2.0;
    }
    expected_east_data.insert(expected_east_data.end(),
                              local_neighbor_data.max_variables_values.cbegin(),
                              local_neighbor_data.max_variables_values.cend());
    expected_east_data.insert(expected_east_data.end(),
                              local_neighbor_data.min_variables_values.cbegin(),
                              local_neighbor_data.min_variables_values.cend());
    const auto& east_data = ActionTesting::get_inbox_tag<
        comp, evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<Dim>>(
        runner, east_id);
    CHECK_ITERABLE_APPROX(
        expected_east_data,
        *std::get<1>(east_data.at(time_step_id)
                         .at(std::pair{Direction<Dim>::lower_xi(), self_id})));
  }
  if constexpr (Dim > 1) {
    const auto direction = Direction<Dim>::lower_eta();

    std::array<size_t, Dim> slice_extents{};
    for (size_t d = 0; d < Dim; ++d) {
      gsl::at(slice_extents, d) = subcell_mesh.extents(d);
    }
    gsl::at(slice_extents, direction.dimension()) = ghost_zone_size;

    std::vector<double> expected_south_data = orient_variables(
        all_sliced_data.at(direction), Index<Dim>{slice_extents}, orientation);
    for (double& value : expected_south_data) {
      value *= 2.0;
    }
    expected_south_data.insert(
        expected_south_data.end(),
        local_neighbor_data.max_variables_values.cbegin(),
        local_neighbor_data.max_variables_values.cend());
    expected_south_data.insert(
        expected_south_data.end(),
        local_neighbor_data.min_variables_values.cbegin(),
        local_neighbor_data.min_variables_values.cend());

    const auto& south_data = ActionTesting::get_inbox_tag<
        comp, evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<Dim>>(
        runner, south_id);
    CHECK(expected_south_data ==
          *std::get<1>(
              south_data.at(time_step_id)
                  .at(std::pair{orientation(direction.opposite()), self_id})));
  }

  // Set the inbox data on self_id and then check that it gets processed
  // correctly. We need to check both if a neighbor is doing DG or if a neighbor
  // is doing subcell.
  auto& self_inbox = ActionTesting::get_inbox_tag<
      comp, evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<Dim>>(
      make_not_null(&runner), self_id);
  REQUIRE_FALSE(ActionTesting::next_action_if_ready<comp>(
      make_not_null(&runner), self_id));

  const size_t rdmp_max_min_total_number = 2;
  // Send data from east neighbor
  std::vector<double> east_ghost_cells_and_rdmp{};
  {
    const auto face_mesh = dg_mesh.slice_away(0);
    east_ghost_cells_and_rdmp.resize(
        subcell_mesh.slice_away(0).number_of_grid_points() * ghost_zone_size +
        rdmp_max_min_total_number);
    alg::iota(east_ghost_cells_and_rdmp, 0.0);
    std::vector<double> boundary_data(face_mesh.number_of_grid_points() *
                                      (2 + Dim));
    alg::iota(boundary_data, 1000.0);
    evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<Dim>::
        insert_into_inbox(
            make_not_null(&self_inbox), time_step_id,
            std::pair{std::pair{Direction<Dim>::upper_xi(), east_id},
                      std::tuple{face_mesh, east_ghost_cells_and_rdmp,
                                 boundary_data, next_time_step_id}});
  }
  [[maybe_unused]] std::vector<double> south_ghost_cells_and_rdmp{};
  if constexpr (Dim > 1) {
    REQUIRE_FALSE(ActionTesting::next_action_if_ready<comp>(
        make_not_null(&runner), self_id));

    const auto face_mesh = subcell_mesh.slice_away(1);
    south_ghost_cells_and_rdmp.resize(
        subcell_mesh.slice_away(0).number_of_grid_points() * ghost_zone_size +
        rdmp_max_min_total_number);
    alg::iota(south_ghost_cells_and_rdmp, 10000.0);
    *std::prev(south_ghost_cells_and_rdmp.end()) = -10.0;
    evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<Dim>::
        insert_into_inbox(
            make_not_null(&self_inbox), time_step_id,
            std::pair{std::pair{Direction<Dim>::lower_eta(), south_id},
                      std::tuple{face_mesh, south_ghost_cells_and_rdmp,
                                 std::nullopt, next_time_step_id}});
  }

  // Run the ReceiveDataForReconstruction action on self_id
  ActionTesting::next_action<comp>(make_not_null(&runner), self_id);

  // Check the received data was stored correctly
  const auto& neighbor_data_from_box =
      get_databox_tag<comp, neighbor_data_tag>(runner, self_id);
  const evolution::dg::subcell::NeighborData& self_neighbor_data =
      neighbor_data_from_box.find(self_neighbor_id)->second;
  CHECK(self_neighbor_data.max_variables_values.size() == 1);
  CHECK(approx(self_neighbor_data.max_variables_values[0]) ==
        (Dim > 1 ? static_cast<double>(
                       ghost_zone_size *
                       subcell_mesh.slice_away(1).number_of_grid_points()) +
                       10000.0
                 : static_cast<double>(ghost_zone_size)));
  CHECK(self_neighbor_data.min_variables_values.size() == 1);
  CHECK(approx(self_neighbor_data.min_variables_values[0]) ==
        (Dim > 1 ? -10.0 : min(get<0>(logical_coordinates(subcell_mesh)))));

  REQUIRE(neighbor_data_from_box.find(east_neighbor_id) !=
          neighbor_data_from_box.end());
  east_ghost_cells_and_rdmp.pop_back();
  east_ghost_cells_and_rdmp.pop_back();
  CHECK(neighbor_data_from_box.find(east_neighbor_id)
            ->second.data_for_reconstruction == east_ghost_cells_and_rdmp);
  if constexpr (Dim > 1) {
    const std::pair south_neighbor_id{Direction<Dim>::lower_eta(), south_id};
    REQUIRE(neighbor_data_from_box.find(south_neighbor_id) !=
            neighbor_data_from_box.end());
    south_ghost_cells_and_rdmp.pop_back();
    south_ghost_cells_and_rdmp.pop_back();
    CHECK(neighbor_data_from_box.find(south_neighbor_id)
              ->second.data_for_reconstruction == south_ghost_cells_and_rdmp);
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.Actions.ReconstructionCommunication",
                  "[Evolution][Unit]") {
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace
