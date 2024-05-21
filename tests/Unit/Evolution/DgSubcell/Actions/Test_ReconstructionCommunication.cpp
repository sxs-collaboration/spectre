// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <deque>
#include <iterator>
#include <memory>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/NeighborMesh.hpp"
#include "Evolution/DgSubcell/Actions/ReconstructionCommunication.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/CellCenteredFlux.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Interpolators.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/Reconstructor.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/TciGridHistory.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/Phase.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
class DummyReconstructor {
 public:
  static size_t ghost_zone_size() { return 2; }
  void pup(PUP::er& /*p*/) {}
};

namespace Tags {
struct Reconstructor : db::SimpleTag,
                       evolution::dg::subcell::Tags::Reconstructor {
  using type = std::unique_ptr<DummyReconstructor>;
};
}  // namespace Tags

struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct System {
  static constexpr size_t volume_dim = Dim;
  using variables_tag = ::Tags::Variables<tmpl::list<Var1>>;
  using flux_variables = tmpl::list<Var1>;
};

template <size_t Dim, typename Metavariables>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Dim>;

  using initial_tags = tmpl::list<
      Tags::Reconstructor, ::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>,
      domain::Tags::Mesh<Dim>, evolution::dg::subcell::Tags::Mesh<Dim>,
      evolution::dg::subcell::Tags::ActiveGrid, domain::Tags::Element<Dim>,
      evolution::dg::subcell::Tags::GhostDataForReconstruction<Dim>,
      evolution::dg::subcell::Tags::DataForRdmpTci,
      evolution::dg::subcell::Tags::TciDecision,
      evolution::dg::subcell::Tags::NeighborTciDecisions<Dim>,
      ::Tags::Variables<tmpl::list<Var1>>, evolution::dg::Tags::MortarData<Dim>,
      evolution::dg::Tags::MortarNextTemporalId<Dim>,
      domain::Tags::NeighborMesh<Dim>,
      evolution::dg::subcell::Tags::CellCenteredFlux<tmpl::list<Var1>, Dim>,
      evolution::dg::subcell::Tags::InterpolatorsFromFdToNeighborFd<Dim>,
      evolution::dg::subcell::Tags::InterpolatorsFromNeighborDgToFd<Dim>>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<
          ActionTesting::InitializeDataBox<initial_tags>,
          evolution::dg::subcell::Actions::SendDataForReconstruction<
              Dim, typename Metavariables::GhostDataMutator,
              // No local time stepping
              false>,
          evolution::dg::subcell::Actions::ReceiveDataForReconstruction<Dim>>>>;
};

template <size_t Dim>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  using component_list = tmpl::list<component<Dim, Metavariables>>;
  using system = System<Dim>;
  using const_global_cache_tags = tmpl::list<>;

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static bool ghost_zone_size_invoked;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static bool ghost_data_mutator_invoked;

  struct GhostDataMutator {
    using return_tags = tmpl::list<>;
    using argument_tags = tmpl::list<::Tags::Variables<tmpl::list<Var1>>>;
    static DataVector apply(const Variables<tmpl::list<Var1>>& vars,
                            const size_t rdmp_size) {
      CAPTURE(rdmp_size);
      CAPTURE(Dim);
      CAPTURE(vars.number_of_grid_points());
      CHECK(
          (rdmp_size == 0 or rdmp_size == Dim * vars.number_of_grid_points()));
      DataVector buffer{vars.size() + rdmp_size};
      ghost_data_mutator_invoked = true;
      // make some trivial but testable modification
      DataVector view{buffer.data(), vars.size()};
      view = 2.0 * get(get<Var1>(vars));
      return buffer;
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
void test(const bool use_cell_centered_flux) {
  CAPTURE(Dim);
  CAPTURE(use_cell_centered_flux);

  using Interps = DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>;

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
      DirectionalIdMap<Dim, evolution::dg::subcell::GhostData>;
  NeighborDataMap neighbor_data{};
  const DirectionalId<Dim> east_neighbor_id{Direction<Dim>::upper_xi(),
                                            east_id};
  // insert data from one of the neighbors to make sure the send actions clears
  // it.
  neighbor_data[east_neighbor_id] = {};

  using evolved_vars_tags = tmpl::list<Var1>;
  Variables<evolved_vars_tags> evolved_vars{
      subcell_mesh.number_of_grid_points()};
  // Set Var1 to the logical coords, just need some data
  get(get<Var1>(evolved_vars)) = get<0>(logical_coordinates(subcell_mesh));

  using CellCenteredFluxTag =
      evolution::dg::subcell::Tags::CellCenteredFlux<tmpl::list<Var1>, Dim>;
  typename CellCenteredFluxTag::type cell_centered_flux{};
  if (use_cell_centered_flux) {
    cell_centered_flux = typename CellCenteredFluxTag::type{
        subcell_mesh.number_of_grid_points()};
    const auto logical_coords = logical_coordinates(subcell_mesh);
    for (size_t i = 0; i < Dim; ++i) {
      get<::Tags::Flux<Var1, tmpl::size_t<Dim>, Frame::Inertial>>(
          cell_centered_flux.value())
          .get(i) = logical_coords.get(i) + (static_cast<double>(i) + 1.0);
    }
  }

  using MortarData = typename evolution::dg::Tags::MortarData<Dim>::type;
  using MortarNextId =
      typename evolution::dg::Tags::MortarNextTemporalId<Dim>::type;
  MortarData mortar_data{};
  MortarNextId mortar_next_id{};
  mortar_data[east_neighbor_id] = {};
  mortar_next_id[east_neighbor_id] = {};
  if constexpr (Dim > 1) {
    const DirectionalId<Dim> south_neighbor_id{Direction<Dim>::lower_eta(),
                                               south_id};
    mortar_data[south_neighbor_id] = {};
    mortar_next_id[south_neighbor_id] = {};
  }

  size_t neighbor_tci_decision = 0;
  typename evolution::dg::subcell::Tags::NeighborTciDecisions<Dim>::type
      neighbor_decision{};
  for (const auto& [direction, neighbor_ids] : neighbors) {
    (void)direction;
    for (const auto& neighbor_id : neighbor_ids) {
      neighbor_decision.insert(
          std::pair{DirectionalId<Dim>{direction, neighbor_id}, 0});
      // Initialize neighbors with garbage data. We won't ever run any actions
      // on them, we just need to insert them to make sure things are sent to
      // the right places. We'll check their inboxes directly.
      ActionTesting::emplace_array_component_and_initialize<comp>(
          &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0},
          neighbor_id,
          {std::make_unique<DummyReconstructor>(), time_step_id,
           next_time_step_id, Mesh<Dim>{}, Mesh<Dim>{}, active_grid,
           Element<Dim>{}, NeighborDataMap{},
           evolution::dg::subcell::RdmpTciData{}, neighbor_tci_decision,
           typename evolution::dg::subcell::Tags::NeighborTciDecisions<
               Dim>::type{},
           Variables<evolved_vars_tags>{}, MortarData{}, MortarNextId{},
           typename domain::Tags::NeighborMesh<Dim>::type{}, cell_centered_flux,
           Interps{}, Interps{}});
      ++neighbor_tci_decision;
    }
  }
  Interps fd_to_neighbor_fd_interpolants{};
  Interps neighbor_dg_to_fd_interpolants{};
  const int self_tci_decision = 100;
  ActionTesting::emplace_array_component_and_initialize<comp>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, self_id,
      {std::make_unique<DummyReconstructor>(), time_step_id, next_time_step_id,
       dg_mesh, subcell_mesh, active_grid, element, neighbor_data,
       // Explicitly set RDMP data since this would be set previously by the TCI
       evolution::dg::subcell::RdmpTciData{{max(get(get<Var1>(evolved_vars)))},
                                           {min(get(get<Var1>(evolved_vars)))}},
       self_tci_decision, neighbor_decision, evolved_vars, mortar_data,
       mortar_next_id, typename domain::Tags::NeighborMesh<Dim>::type{},
       cell_centered_flux, fd_to_neighbor_fd_interpolants,
       neighbor_dg_to_fd_interpolants});

  using ghost_data_tag =
      evolution::dg::subcell::Tags::GhostDataForReconstruction<Dim>;
  using rdmp_tci_data_tag = evolution::dg::subcell::Tags::DataForRdmpTci;
  using ActionTesting::get_databox_tag;
  CHECK(get_databox_tag<comp, ghost_data_tag>(runner, self_id).size() == 1);
  CHECK(get_databox_tag<comp, ghost_data_tag>(runner, self_id)
            .count(east_neighbor_id) == 1);

  // Run the SendDataForReconstruction action on self_id
  ActionTesting::next_action<comp>(make_not_null(&runner), self_id);

  CHECK(get_databox_tag<comp, ghost_data_tag>(runner, self_id).empty());

  // Check local RDMP data
  const evolution::dg::subcell::RdmpTciData& rdmp_tci_data =
      get_databox_tag<comp, rdmp_tci_data_tag>(runner, self_id);
  REQUIRE(rdmp_tci_data.max_variables_values.size() == 1);
  CHECK(rdmp_tci_data.max_variables_values[0] ==
        max(get(get<Var1>(evolved_vars))));
  REQUIRE(rdmp_tci_data.min_variables_values.size() == 1);
  CHECK(rdmp_tci_data.min_variables_values[0] ==
        min(get(get<Var1>(evolved_vars))));
  // Check data sent to neighbors
  const auto& directions_to_slice = element.internal_boundaries();
  const size_t ghost_zone_size = 2;
  const size_t rdmp_size = rdmp_tci_data.max_variables_values.size() +
                           rdmp_tci_data.min_variables_values.size();
  const DirectionMap<Dim, DataVector> all_sliced_data =
      [&evolved_vars, &subcell_mesh, ghost_zone_size, &directions_to_slice,
       &cell_centered_flux, &fd_to_neighbor_fd_interpolants]() {
        (void)ghost_zone_size;
        if (cell_centered_flux.has_value()) {
          DataVector buffer{evolved_vars.size() +
                            cell_centered_flux.value().size()};
          std::copy(evolved_vars.data(),
                    std::next(evolved_vars.data(),
                              static_cast<std::ptrdiff_t>(evolved_vars.size())),
                    buffer.data());
          std::copy(cell_centered_flux.value().data(),
                    std::next(cell_centered_flux.value().data(),
                              static_cast<std::ptrdiff_t>(
                                  cell_centered_flux.value().size())),
                    std::next(buffer.data(), static_cast<std::ptrdiff_t>(
                                                 evolved_vars.size())));
          return evolution::dg::subcell::slice_data(
              buffer, subcell_mesh.extents(), ghost_zone_size,
              directions_to_slice, 0, fd_to_neighbor_fd_interpolants);
        } else {
          return evolution::dg::subcell::slice_data(
              evolved_vars, subcell_mesh.extents(), ghost_zone_size,
              directions_to_slice, 0, fd_to_neighbor_fd_interpolants);
        }
      }();
  {
    const auto& east_sliced_neighbor_data =
        all_sliced_data.at(east_neighbor_id.direction());
    DataVector expected_east_data{east_sliced_neighbor_data.size() + rdmp_size};
    std::copy(east_sliced_neighbor_data.begin(),
              east_sliced_neighbor_data.end(), expected_east_data.begin());
    // We only multiple Var1 by 2, not the fluxes.
    const size_t bound_for_vars_to_multiply =
        east_sliced_neighbor_data.size() /
        (cell_centered_flux.has_value() ? (1 + Dim) : 1_st);
    for (size_t i = 0; i < bound_for_vars_to_multiply; i++) {
      expected_east_data[i] *= 2.0;
    }
    std::copy(rdmp_tci_data.max_variables_values.cbegin(),
              rdmp_tci_data.max_variables_values.cend(),
              std::prev(expected_east_data.end(), static_cast<int>(rdmp_size)));
    std::copy(
        rdmp_tci_data.min_variables_values.cbegin(),
        rdmp_tci_data.min_variables_values.cend(),
        std::prev(expected_east_data.end(),
                  static_cast<int>(rdmp_tci_data.min_variables_values.size())));

    const auto& east_data = ActionTesting::get_inbox_tag<
        comp, evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<Dim>>(
        runner, east_id);
    CHECK_ITERABLE_APPROX(
        expected_east_data,
        east_data.at(time_step_id)
            .at(DirectionalId<Dim>{Direction<Dim>::lower_xi(), self_id})
            .ghost_cell_data.value());
    CHECK(east_data.at(time_step_id)
              .at(DirectionalId<Dim>{Direction<Dim>::lower_xi(), self_id})
              .tci_status == self_tci_decision);
  }
  if constexpr (Dim > 1) {
    const auto direction = Direction<Dim>::lower_eta();

    std::array<size_t, Dim> slice_extents{};
    for (size_t d = 0; d < Dim; ++d) {
      gsl::at(slice_extents, d) = subcell_mesh.extents(d);
    }
    gsl::at(slice_extents, direction.dimension()) = ghost_zone_size;

    const auto& south_sliced_neighbor_data = all_sliced_data.at(direction);

    DataVector expected_south_data{south_sliced_neighbor_data.size() +
                                   rdmp_size};
    DataVector expected_south_data_view{expected_south_data.data(),
                                        south_sliced_neighbor_data.size()};
    // Note: We do not orient the variables because that's currently handled
    // by the interpolation code.
    //
    // orient_variables(make_not_null(&expected_south_data_view),
    //                  all_sliced_data.at(direction),
    //                  Index<Dim>{slice_extents},
    //                  orientation);
    expected_south_data_view = all_sliced_data.at(direction);
    const size_t bound_for_vars_to_multiply =
        expected_south_data_view.size() /
        (cell_centered_flux.has_value() ? (1 + Dim) : 1_st);
    for (size_t i = 0; i < bound_for_vars_to_multiply; i++) {
      expected_south_data_view[i] *= 2.0;
    }
    std::copy(
        rdmp_tci_data.max_variables_values.cbegin(),
        rdmp_tci_data.max_variables_values.cend(),
        std::prev(expected_south_data.end(), static_cast<int>(rdmp_size)));
    std::copy(
        rdmp_tci_data.min_variables_values.cbegin(),
        rdmp_tci_data.min_variables_values.cend(),
        std::prev(expected_south_data.end(),
                  static_cast<int>(rdmp_tci_data.min_variables_values.size())));

    const auto& south_data = ActionTesting::get_inbox_tag<
        comp, evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<Dim>>(
        runner, south_id);
    CHECK(
        expected_south_data ==
        south_data.at(time_step_id)
            .at(DirectionalId<Dim>{orientation(direction.opposite()), self_id})
            .ghost_cell_data.value());
    CHECK(
        south_data.at(time_step_id)
            .at(DirectionalId<Dim>{orientation(direction.opposite()), self_id})
            .tci_status == self_tci_decision);
  }

  // Set the inbox data on self_id and then check that it gets processed
  // correctly. We need to check both if a neighbor is doing DG or if a neighbor
  // is doing subcell.
  auto& self_inbox = ActionTesting::get_inbox_tag<
      comp, evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<Dim>>(
      make_not_null(&runner), self_id);
  REQUIRE_FALSE(ActionTesting::next_action_if_ready<comp>(
      make_not_null(&runner), self_id));

  // Send data from east neighbor
  DataVector east_ghost_cells_and_rdmp{};
  {
    const auto face_mesh = dg_mesh.slice_away(0);
    east_ghost_cells_and_rdmp = DataVector{
        subcell_mesh.slice_away(0).number_of_grid_points() * ghost_zone_size +
        rdmp_size};
    alg::iota(east_ghost_cells_and_rdmp, 0.0);
    DataVector boundary_data{face_mesh.number_of_grid_points() * (2 + Dim)};
    alg::iota(boundary_data, 1000.0);
    evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<Dim>::
        insert_into_inbox(
            make_not_null(&self_inbox), time_step_id,
            std::pair{DirectionalId<Dim>{Direction<Dim>::upper_xi(), east_id},
                      evolution::dg::BoundaryData<Dim>{
                          // subcell_mesh because we are sending the projected
                          // data right now.
                          subcell_mesh, face_mesh, east_ghost_cells_and_rdmp,
                          boundary_data, next_time_step_id, -10}});
  }
  [[maybe_unused]] DataVector south_ghost_cells_and_rdmp{};
  if constexpr (Dim > 1) {
    REQUIRE_FALSE(ActionTesting::next_action_if_ready<comp>(
        make_not_null(&runner), self_id));

    const auto face_mesh = subcell_mesh.slice_away(1);
    south_ghost_cells_and_rdmp = DataVector{
        subcell_mesh.slice_away(0).number_of_grid_points() * ghost_zone_size +
        rdmp_size};
    alg::iota(south_ghost_cells_and_rdmp, 10000.0);
    *std::prev(south_ghost_cells_and_rdmp.end()) = -10.0;
    evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<Dim>::
        insert_into_inbox(
            make_not_null(&self_inbox), time_step_id,
            std::pair{DirectionalId<Dim>{Direction<Dim>::lower_eta(), south_id},
                      evolution::dg::BoundaryData<Dim>{
                          // subcell_mesh because we are sending the projected
                          // data right now.
                          subcell_mesh, face_mesh, south_ghost_cells_and_rdmp,
                          std::nullopt, next_time_step_id, -15}});
  }

  // Run the ReceiveDataForReconstruction action on self_id
  ActionTesting::next_action<comp>(make_not_null(&runner), self_id);

  // Check the received data was stored correctly
  const auto& ghost_data_from_box =
      get_databox_tag<comp, ghost_data_tag>(runner, self_id);
  CHECK(rdmp_tci_data.max_variables_values.size() == 1);
  CHECK(approx(rdmp_tci_data.max_variables_values[0]) ==
        (Dim > 1 ? static_cast<double>(
                       ghost_zone_size *
                       subcell_mesh.slice_away(1).number_of_grid_points()) +
                       10000.0
                 : static_cast<double>(ghost_zone_size)));
  CHECK(rdmp_tci_data.min_variables_values.size() == 1);
  CHECK(approx(rdmp_tci_data.min_variables_values[0]) ==
        (Dim > 1 ? -10.0 : min(get<0>(logical_coordinates(subcell_mesh)))));

  REQUIRE(ghost_data_from_box.find(east_neighbor_id) !=
          ghost_data_from_box.end());
  const DataVector east_ghost_cells_and_rdmp_view{
      east_ghost_cells_and_rdmp.data(),
      east_ghost_cells_and_rdmp.size() - rdmp_size};
  CHECK(ghost_data_from_box.find(east_neighbor_id)
            ->second.neighbor_ghost_data_for_reconstruction() ==
        east_ghost_cells_and_rdmp_view);
  CHECK(
      get_databox_tag<comp,
                      evolution::dg::subcell::Tags::NeighborTciDecisions<Dim>>(
          runner, self_id)
          .at(east_neighbor_id) == -10);
  if constexpr (Dim > 1) {
    const DirectionalId<Dim> south_neighbor_id{Direction<Dim>::lower_eta(),
                                               south_id};
    REQUIRE(ghost_data_from_box.find(south_neighbor_id) !=
            ghost_data_from_box.end());
    const DataVector south_ghost_cells_and_rdmp_view{
        south_ghost_cells_and_rdmp.data(),
        south_ghost_cells_and_rdmp.size() - rdmp_size};
    CHECK(ghost_data_from_box.find(south_neighbor_id)
              ->second.neighbor_ghost_data_for_reconstruction() ==
          south_ghost_cells_and_rdmp_view);
    CHECK(get_databox_tag<
              comp, evolution::dg::subcell::Tags::NeighborTciDecisions<Dim>>(
              runner, self_id)
              .at(south_neighbor_id) == -15);
  }

  // Check that we got a neighbor mesh from all neighbors.
  size_t total_neighbors = 0;
  const auto& neighbor_meshes =
      get_databox_tag<comp, ::domain::Tags::NeighborMesh<Dim>>(runner, self_id);
  for (const auto& [direction, neighbors_in_direction] : element.neighbors()) {
    for (const auto& neighbor : neighbors_in_direction) {
      const auto it =
          neighbor_meshes.find(DirectionalId<Dim>{direction, neighbor});
      REQUIRE(it != neighbor_meshes.end());
      // Currently all neighbors are doing subcell. We probably want to
      // generalize this in the future.
      CHECK(it->second == subcell_mesh);
      ++total_neighbors;
    }
  }
  CHECK(neighbor_meshes.size() == total_neighbors);
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.Actions.ReconstructionCommunication",
                  "[Evolution][Unit]") {
  for (const bool use_cell_centered_flux : {false, true}) {
    test<1>(use_cell_centered_flux);
    test<2>(use_cell_centered_flux);
    test<3>(use_cell_centered_flux);
  }
}
}  // namespace
