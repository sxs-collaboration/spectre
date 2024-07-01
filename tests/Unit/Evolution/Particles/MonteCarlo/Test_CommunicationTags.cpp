// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
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
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Interpolators.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/Particles/MonteCarlo/GhostZoneCommunication.hpp"
#include "Evolution/Particles/MonteCarlo/GhostZoneCommunicationStep.hpp"
#include "Evolution/Particles/MonteCarlo/GhostZoneCommunicationTags.hpp"
#include "Evolution/Particles/MonteCarlo/MortarData.hpp"
#include "Evolution/Particles/MonteCarlo/Packet.hpp"
#include "Evolution/Particles/MonteCarlo/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/Phase.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
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
      ::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>,
      domain::Tags::Mesh<Dim>, evolution::dg::subcell::Tags::Mesh<Dim>,
      evolution::dg::subcell::Tags::ActiveGrid, domain::Tags::Element<Dim>,
      Particles::MonteCarlo::Tags::McGhostZoneDataTag<Dim>,
      Particles::MonteCarlo::Tags::PacketsOnElement,
      ::Tags::Variables<tmpl::list<Var1>>,
      Particles::MonteCarlo::Tags::MortarDataTag<Dim>,
      hydro::Tags::RestMassDensity<DataVector>,
      hydro::Tags::ElectronFraction<DataVector>,
      hydro::Tags::Temperature<DataVector>,
      Particles::MonteCarlo::Tags::CellLightCrossingTime<DataVector>,
      domain::Tags::NeighborMesh<Dim>,
      evolution::dg::subcell::Tags::InterpolatorsFromFdToNeighborFd<Dim>>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<
          ActionTesting::InitializeDataBox<initial_tags>,
          Particles::MonteCarlo::Actions::SendDataForMcCommunication<
              Dim,
              // No local time stepping
              false, Particles::MonteCarlo::CommunicationStep::PreStep>,
          Particles::MonteCarlo::Actions::ReceiveDataForMcCommunication<
              Dim, Particles::MonteCarlo::CommunicationStep::PreStep>,
          Particles::MonteCarlo::Actions::SendDataForMcCommunication<
              Dim,
              // No local time stepping
              false, Particles::MonteCarlo::CommunicationStep::PostStep>,
          Particles::MonteCarlo::Actions::ReceiveDataForMcCommunication<
              Dim, Particles::MonteCarlo::CommunicationStep::PostStep>>>>;
};

template <size_t Dim>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  using component_list = tmpl::list<component<Dim, Metavariables>>;
  using system = System<Dim>;
  using const_global_cache_tags = tmpl::list<>;
};

template <size_t Dim>
void test_send_receive_actions() {
  CAPTURE(Dim);

  using Interps = DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>;
  using metavars = Metavariables<Dim>;
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
  // NOLINTNEXTLINE(misc-const-correctness)
  ElementId<Dim> south_id{};  // not used in 1d
  // NOLINTNEXTLINE(misc-const-correctness)
  OrientationMap<Dim> orientation{};
  // Note: in 2d and 3d it is the neighbor in the lower eta direction that has a
  // non-trivial orientation.

  if constexpr (Dim == 1) {
    self_id = ElementId<Dim>{0, {{{1, 0}}}};
    east_id = ElementId<Dim>{0, {{{1, 1}}}};
    neighbors[Direction<Dim>::upper_xi()] = Neighbors<Dim>{{east_id},
      OrientationMap<Dim>::create_aligned()};
  } else if constexpr (Dim == 2) {
    orientation = OrientationMap<Dim>{
        std::array{Direction<Dim>::lower_xi(), Direction<Dim>::lower_eta()}};
    self_id = ElementId<Dim>{0, {{{1, 0}, {0, 0}}}};
    east_id = ElementId<Dim>{0, {{{1, 1}, {0, 0}}}};
    south_id = ElementId<Dim>{1, {{{0, 0}, {0, 0}}}};
    neighbors[Direction<Dim>::upper_xi()] = Neighbors<Dim>{{east_id},
      OrientationMap<Dim>::create_aligned()};
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
    neighbors[Direction<Dim>::upper_xi()] = Neighbors<Dim>{{east_id},
      OrientationMap<Dim>::create_aligned()};
    neighbors[Direction<Dim>::lower_eta()] =
        Neighbors<Dim>{{south_id}, orientation};
  }
  const Element<Dim> element{self_id, neighbors};

  using NeighborDataMap =
      DirectionalIdMap<Dim, Particles::MonteCarlo::McGhostZoneData<Dim>>;
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

  Scalar<DataVector> rest_mass_density(
      get<0>(logical_coordinates(subcell_mesh)));
  Scalar<DataVector> electron_fraction(
      get<0>(logical_coordinates(subcell_mesh)) / 2.0);
  Scalar<DataVector> temperature(get<0>(logical_coordinates(subcell_mesh)) *
                                 3.5);
  Scalar<DataVector> cell_light_crossing_time(
      get<0>(logical_coordinates(subcell_mesh)) * 2.0);
  std::vector<Particles::MonteCarlo::Packet> packets_on_element{};

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
           active_grid, Element<Dim>{}, NeighborDataMap{}, packets_on_element,
           Variables<evolved_vars_tags>{},
           typename Particles::MonteCarlo::Tags::MortarDataTag<Dim>::type{},
           Scalar<DataVector>{}, Scalar<DataVector>{}, Scalar<DataVector>{},
           Scalar<DataVector>{},
           typename domain::Tags::NeighborMesh<Dim>::type{}, Interps{}});
    }
  }

  // Initialize MortarData as needed
  Particles::MonteCarlo::MortarData<Dim> mortar_data{};
  {
    const size_t number_of_points_in_ghost_zone =
      Dim > 1 ?
      subcell_mesh.slice_away(0).number_of_grid_points() :
      1;
    mortar_data.rest_mass_density[east_neighbor_id] =
        DataVector(number_of_points_in_ghost_zone, 0.1);
    mortar_data.electron_fraction[east_neighbor_id] =
        DataVector(number_of_points_in_ghost_zone, 0.1);
    mortar_data.temperature[east_neighbor_id] =
        DataVector(number_of_points_in_ghost_zone, 0.1);
    mortar_data.cell_light_crossing_time[east_neighbor_id] =
        DataVector(number_of_points_in_ghost_zone, 0.1);
  }
  if constexpr (Dim > 1) {
    const DirectionalId<Dim> south_neighbor_id{Direction<Dim>::lower_eta(),
                                               south_id};
    const Mesh<Dim - 1> ghost_mesh = subcell_mesh.slice_away(1);
    mortar_data.rest_mass_density[south_neighbor_id] =
        DataVector(ghost_mesh.number_of_grid_points(), 0.1);
    mortar_data.electron_fraction[south_neighbor_id] =
        DataVector(ghost_mesh.number_of_grid_points(), 0.1);
    mortar_data.temperature[south_neighbor_id] =
        DataVector(ghost_mesh.number_of_grid_points(), 0.1);
    mortar_data.cell_light_crossing_time[south_neighbor_id] =
        DataVector(ghost_mesh.number_of_grid_points(), 0.1);
  }

  const size_t species = 1;
  const double number_of_neutrinos = 2.0;
  const size_t index_of_closest_grid_point = 0;
  const double t0 = 1.2;
  const double x0 = 0.3;
  const double y0 = 0.5;
  const double z0 = -0.7;
  const double p_upper_t0 = 1.1;
  const double p_x0 = 0.9;
  const double p_y0 = 0.7;
  const double p_z0 = 0.1;
  // Packet to be sent to east element
  Particles::MonteCarlo::Packet packet_east(
      species, number_of_neutrinos, index_of_closest_grid_point, t0, 1.3, y0,
      z0, p_upper_t0, p_x0, p_y0, p_z0);
  // Packet to be kept by current element
  Particles::MonteCarlo::Packet packet_keep(
      species, number_of_neutrinos, index_of_closest_grid_point, t0, x0, y0, z0,
      p_upper_t0, p_x0, p_y0, p_z0);
  // Packet to be deleted for being out of bounds
  const Particles::MonteCarlo::Packet packet_delete(
      species, number_of_neutrinos, index_of_closest_grid_point, t0, -1.3, y0,
      z0, p_upper_t0, p_x0, p_y0, p_z0);
  // Packet to be sent to south element
  // Note that the packet extends outside of the current domain in both the y
  // and z direction, but should be moved in the direction where it is the
  // farthest from the boundary (y, here).
  Particles::MonteCarlo::Packet packet_south(
      species, number_of_neutrinos, index_of_closest_grid_point, t0, x0, -1.2,
      -1.1, p_upper_t0, p_x0, p_y0, p_z0);
  packets_on_element.push_back(packet_east);
  packets_on_element.push_back(packet_keep);
  packets_on_element.push_back(packet_delete);
  if constexpr (Dim > 1) {
    packets_on_element.push_back(packet_south);
  }

  Interps fd_to_neighbor_fd_interpolants{};
  ActionTesting::emplace_array_component_and_initialize<comp>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, self_id,
      {time_step_id, next_time_step_id, dg_mesh, subcell_mesh, active_grid,
       element, neighbor_data, packets_on_element, evolved_vars, mortar_data,
       rest_mass_density, electron_fraction, temperature,
       cell_light_crossing_time,
       typename domain::Tags::NeighborMesh<Dim>::type{},
       fd_to_neighbor_fd_interpolants});

  using ghost_data_tag = Particles::MonteCarlo::Tags::McGhostZoneDataTag<Dim>;
  using ActionTesting::get_databox_tag;
  CHECK(get_databox_tag<comp, ghost_data_tag>(runner, self_id).size() == 1);
  CHECK(get_databox_tag<comp, ghost_data_tag>(runner, self_id)
            .count(east_neighbor_id) == 1);

  // Run the SendDataForReconstruction action on self_id (PreStep)
  ActionTesting::next_action<comp>(make_not_null(&runner), self_id);
  CHECK(get_databox_tag<comp, ghost_data_tag>(runner, self_id).empty());

  // Check data sent to neighbors
  const size_t ghost_zone_size = 1;
  const auto& directions_to_slice = element.internal_boundaries();
  const DirectionMap<Dim, DataVector> all_sliced_data =
      [&rest_mass_density, &electron_fraction, &temperature,
       &cell_light_crossing_time, &subcell_mesh, ghost_zone_size,
       &directions_to_slice, &fd_to_neighbor_fd_interpolants]() {
        (void)ghost_zone_size;
        const size_t dv_size = subcell_mesh.number_of_grid_points();
        const size_t number_of_vars = 4;
        DataVector buffer{dv_size * number_of_vars};
        std::copy(
            get(rest_mass_density).data(),
            std::next(get(rest_mass_density).data(), static_cast<int>(dv_size)),
            buffer.data());
        std::copy(
            get(electron_fraction).data(),
            std::next(get(electron_fraction).data(), static_cast<int>(dv_size)),
            std::next(buffer.data(), static_cast<int>(dv_size)));
        std::copy(get(temperature).data(),
                  std::next(get(temperature).data(), static_cast<int>(dv_size)),
                  std::next(buffer.data(), static_cast<int>(dv_size * 2)));
        std::copy(get(cell_light_crossing_time).data(),
                  std::next(get(cell_light_crossing_time).data(),
                            static_cast<int>(dv_size)),
                  std::next(buffer.data(), static_cast<int>(dv_size * 3)));

        return evolution::dg::subcell::slice_data(
            buffer, subcell_mesh.extents(), ghost_zone_size,
            directions_to_slice, 0, fd_to_neighbor_fd_interpolants);
      }();

  {
    const auto& expected_east_data =
        all_sliced_data.at(east_neighbor_id.direction());
    const auto& east_data = ActionTesting::get_inbox_tag<
        comp, Particles::MonteCarlo::McGhostZoneDataInboxTag<
                  Dim, Particles::MonteCarlo::CommunicationStep::PreStep>>(
        runner, east_id);
    CHECK(east_data.at(time_step_id)
              .at(DirectionalId<Dim>{Direction<Dim>::lower_xi(), self_id})
              .ghost_zone_hydro_variables == expected_east_data);
    CHECK(east_data.at(time_step_id)
              .at(DirectionalId<Dim>{Direction<Dim>::lower_xi(), self_id})
              .packets_entering_this_element == std::nullopt);
  }

  if constexpr (Dim > 1) {
    const auto direction = Direction<Dim>::lower_eta();
    const auto& expected_south_data = all_sliced_data.at(direction);
    const auto& south_data = ActionTesting::get_inbox_tag<
        comp, Particles::MonteCarlo::McGhostZoneDataInboxTag<
                  Dim, Particles::MonteCarlo::CommunicationStep::PreStep>>(
        runner, south_id);
    CHECK(
        south_data.at(time_step_id)
            .at(DirectionalId<Dim>{orientation(direction.opposite()), self_id})
            .ghost_zone_hydro_variables == expected_south_data);
    CHECK(south_data.at(time_step_id)
              .at(DirectionalId<Dim>{direction, self_id})
              .packets_entering_this_element == std::nullopt);
  }

  // Set the inbox data on self_id and then check that it gets processed
  // correctly.
  auto& self_inbox_pre = ActionTesting::get_inbox_tag<
      comp, Particles::MonteCarlo::McGhostZoneDataInboxTag<
                Dim, Particles::MonteCarlo::CommunicationStep::PreStep>>(
      make_not_null(&runner), self_id);
  REQUIRE_FALSE(ActionTesting::next_action_if_ready<comp>(
      make_not_null(&runner), self_id));

  // Send data from east neighbor
  DataVector east_ghost_cells{};
  {
    const size_t number_of_vars = 4;
    east_ghost_cells =
        DataVector{subcell_mesh.slice_away(0).number_of_grid_points() *
                   ghost_zone_size * number_of_vars};
    alg::iota(east_ghost_cells, 2.0);
    const size_t items_in_inbox =
        Particles::MonteCarlo::McGhostZoneDataInboxTag<
            Dim, Particles::MonteCarlo::CommunicationStep::PreStep>::
            insert_into_inbox(
                make_not_null(&self_inbox_pre), time_step_id,
                std::pair{
                    DirectionalId<Dim>{Direction<Dim>::upper_xi(), east_id},
                    Particles::MonteCarlo::McGhostZoneData<Dim>{
                        east_ghost_cells, std::nullopt}});
    CHECK(items_in_inbox == 1);
  }
  // NOLINTNEXTLINE(misc-const-correctness)
  [[maybe_unused]] DataVector south_ghost_cells{};

  if constexpr (Dim > 1) {
    REQUIRE_FALSE(ActionTesting::next_action_if_ready<comp>(
        make_not_null(&runner), self_id));

    const size_t number_of_vars = 4;
    south_ghost_cells =
        DataVector{subcell_mesh.slice_away(1).number_of_grid_points() *
                   ghost_zone_size * number_of_vars};
    alg::iota(south_ghost_cells, 10000.0);
    *std::prev(south_ghost_cells.end()) = -10.0;
    const size_t items_in_inbox =
        Particles::MonteCarlo::McGhostZoneDataInboxTag<
            Dim, Particles::MonteCarlo::CommunicationStep::PreStep>::
            insert_into_inbox(
                make_not_null(&self_inbox_pre), time_step_id,
                std::pair{
                    DirectionalId<Dim>{Direction<Dim>::lower_eta(), south_id},
                    Particles::MonteCarlo::McGhostZoneData<Dim>{
                        south_ghost_cells, std::nullopt}});
    CHECK(items_in_inbox == 2);
  }

  // Run the ReceiveDataForReconstruction action on self_id (PreStep)
  ActionTesting::next_action<comp>(make_not_null(&runner), self_id);

  // Check the received data was stored correctly
  using mortar_data_tag = Particles::MonteCarlo::Tags::MortarDataTag<Dim>;
  const auto& mortar_data_from_box =
      get_databox_tag<comp, mortar_data_tag>(runner, self_id);

  {
    REQUIRE(mortar_data_from_box.rest_mass_density.find(east_neighbor_id) !=
            mortar_data_from_box.rest_mass_density.end());
    REQUIRE(mortar_data_from_box.electron_fraction.find(east_neighbor_id) !=
            mortar_data_from_box.electron_fraction.end());
    REQUIRE(mortar_data_from_box.temperature.find(east_neighbor_id) !=
            mortar_data_from_box.temperature.end());
    REQUIRE(
        mortar_data_from_box.cell_light_crossing_time.find(east_neighbor_id) !=
        mortar_data_from_box.cell_light_crossing_time.end());

    const size_t number_of_east_points = east_ghost_cells.size() / 4;
    const DataVector rest_mass_density_view{east_ghost_cells.data(),
                                            number_of_east_points};
    const DataVector electron_fraction_view{
        east_ghost_cells.data() + number_of_east_points, number_of_east_points};
    const DataVector temperature_view{
        east_ghost_cells.data() + number_of_east_points * 2,
        number_of_east_points};
    const DataVector cell_light_crossing_time_view{
        east_ghost_cells.data() + number_of_east_points * 3,
        east_ghost_cells.size() - number_of_east_points * 3};

    CHECK(
        mortar_data_from_box.rest_mass_density.find(east_neighbor_id)->second ==
        rest_mass_density_view);
    CHECK(
        mortar_data_from_box.electron_fraction.find(east_neighbor_id)->second ==
        electron_fraction_view);
    CHECK(mortar_data_from_box.temperature.find(east_neighbor_id)->second ==
          temperature_view);
    CHECK(mortar_data_from_box.cell_light_crossing_time.find(east_neighbor_id)
              ->second == cell_light_crossing_time_view);
  }
  if constexpr (Dim > 1) {
    const DirectionalId<Dim> south_neighbor_id{Direction<Dim>::lower_eta(),
                                               south_id};
    REQUIRE(mortar_data_from_box.rest_mass_density.find(south_neighbor_id) !=
            mortar_data_from_box.rest_mass_density.end());
    REQUIRE(mortar_data_from_box.electron_fraction.find(south_neighbor_id) !=
            mortar_data_from_box.electron_fraction.end());
    REQUIRE(mortar_data_from_box.temperature.find(south_neighbor_id) !=
            mortar_data_from_box.temperature.end());
    REQUIRE(
        mortar_data_from_box.cell_light_crossing_time.find(south_neighbor_id) !=
        mortar_data_from_box.cell_light_crossing_time.end());

    const size_t number_of_south_points = south_ghost_cells.size() / 4;
    const DataVector rest_mass_density_view{south_ghost_cells.data(),
                                            number_of_south_points};
    const DataVector electron_fraction_view{
        south_ghost_cells.data() + number_of_south_points,
        number_of_south_points};
    const DataVector temperature_view{
        south_ghost_cells.data() + number_of_south_points * 2,
        number_of_south_points};
    const DataVector cell_light_crossing_time_view{
        south_ghost_cells.data() + number_of_south_points * 3,
        south_ghost_cells.size() - number_of_south_points * 3};

    CHECK(mortar_data_from_box.rest_mass_density.find(south_neighbor_id)
              ->second == rest_mass_density_view);
    CHECK(mortar_data_from_box.electron_fraction.find(south_neighbor_id)
              ->second == electron_fraction_view);
    CHECK(mortar_data_from_box.temperature.find(south_neighbor_id)->second ==
          temperature_view);
    CHECK(mortar_data_from_box.cell_light_crossing_time.find(south_neighbor_id)
              ->second == cell_light_crossing_time_view);
  }
  {
    const auto& packets_from_box =
        get_databox_tag<comp, Particles::MonteCarlo::Tags::PacketsOnElement>(
            runner, self_id);
    CHECK(packets_from_box.size() == packets_on_element.size());
  }

  // Run the SendDataForReconstruction action on self_id (PostStep)
  ActionTesting::next_action<comp>(make_not_null(&runner), self_id);
  CHECK(get_databox_tag<comp, ghost_data_tag>(runner, self_id).empty());

  {
    // Verify that the packets out of the element have been removed.
    const auto& packets_from_box =
        get_databox_tag<comp, Particles::MonteCarlo::Tags::PacketsOnElement>(
            runner, self_id);
    CHECK(packets_from_box.size() == 1);
    CHECK(packets_from_box[0] == packet_keep);
  }

  // Correct coordinates of packets sent east/south to get in the
  // topological coordinate of their new element.
  packet_east.coordinates[0] -= 2.0;
  packet_south.coordinates[1] += 2.0;
  {
    const auto& east_data = ActionTesting::get_inbox_tag<
        comp, Particles::MonteCarlo::McGhostZoneDataInboxTag<
                  Dim, Particles::MonteCarlo::CommunicationStep::PostStep>>(
        runner, east_id);
    CHECK(east_data.at(time_step_id)
              .at(DirectionalId<Dim>{Direction<Dim>::lower_xi(), self_id})
              .packets_entering_this_element != std::nullopt);
    CHECK(east_data.at(time_step_id)
              .at(DirectionalId<Dim>{Direction<Dim>::lower_xi(), self_id})
              .packets_entering_this_element.value()
              .size() == 1);
    const Particles::MonteCarlo::Packet received_packet =
        east_data.at(time_step_id)
            .at(DirectionalId<Dim>{Direction<Dim>::lower_xi(), self_id})
            .packets_entering_this_element.value()[0];
    CHECK(packet_east == received_packet);
  }
  if constexpr (Dim > 1) {
    const auto direction = Direction<Dim>::lower_eta();
    const auto& south_data = ActionTesting::get_inbox_tag<
        comp, Particles::MonteCarlo::McGhostZoneDataInboxTag<
                  Dim, Particles::MonteCarlo::CommunicationStep::PostStep>>(
        runner, south_id);
    CHECK(
        south_data.at(time_step_id)
            .at(DirectionalId<Dim>{orientation(direction.opposite()), self_id})
            .packets_entering_this_element != std::nullopt);
    CHECK(
        south_data.at(time_step_id)
            .at(DirectionalId<Dim>{orientation(direction.opposite()), self_id})
            .packets_entering_this_element.value()
            .size() == 1);
    const Particles::MonteCarlo::Packet received_packet =
        south_data.at(time_step_id)
            .at(DirectionalId<Dim>{orientation(direction.opposite()), self_id})
            .packets_entering_this_element.value()[0];
    CHECK(packet_south == received_packet);
  }

  // Set the inbox data on self_id and then check that it gets processed
  // correctly.
  auto& self_inbox_post = ActionTesting::get_inbox_tag<
      comp, Particles::MonteCarlo::McGhostZoneDataInboxTag<
                Dim, Particles::MonteCarlo::CommunicationStep::PostStep>>(
      make_not_null(&runner), self_id);
  // Check that we are not ready to receive yet (Inboxes unfilled)
  REQUIRE_FALSE(ActionTesting::next_action_if_ready<comp>(
      make_not_null(&runner), self_id));

  // Set up fake data coming from east neighbor
  {
    const std::optional<std::vector<Particles::MonteCarlo::Packet>>
        packets_from_east =
            std::vector<Particles::MonteCarlo::Packet>{packet_east};
    const DataVector east_ghost_cells_post{};
    const size_t items_in_inbox =
        Particles::MonteCarlo::McGhostZoneDataInboxTag<
            Dim, Particles::MonteCarlo::CommunicationStep::PostStep>::
            insert_into_inbox(
                make_not_null(&self_inbox_post), time_step_id,
                std::pair{
                    DirectionalId<Dim>{Direction<Dim>::upper_xi(), east_id},
                    Particles::MonteCarlo::McGhostZoneData<Dim>{
                        east_ghost_cells_post, packets_from_east}});
    CHECK(items_in_inbox == 1);
  }
  // Set up fake data coming from south neighbor
  if constexpr (Dim > 1) {
    REQUIRE_FALSE(ActionTesting::next_action_if_ready<comp>(
        make_not_null(&runner), self_id));

    const std::optional<std::vector<Particles::MonteCarlo::Packet>>
        packets_from_south =
            std::vector<Particles::MonteCarlo::Packet>{packet_south};
    const DataVector south_ghost_cells_post{};
    const size_t items_in_inbox =
        Particles::MonteCarlo::McGhostZoneDataInboxTag<
            Dim, Particles::MonteCarlo::CommunicationStep::PostStep>::
            insert_into_inbox(
                make_not_null(&self_inbox_post), time_step_id,
                std::pair{
                    DirectionalId<Dim>{Direction<Dim>::lower_eta(), south_id},
                    Particles::MonteCarlo::McGhostZoneData<Dim>{
                        south_ghost_cells_post, packets_from_south}});
    CHECK(items_in_inbox == 2);
  }
  // Run the ReceiveDataForReconstruction action on self_id (PostStep)
  ActionTesting::next_action<comp>(make_not_null(&runner), self_id);

  // We should now have 3 packets (2 for Dim=1)
  {
    const auto& packets_from_box =
        get_databox_tag<comp, Particles::MonteCarlo::Tags::PacketsOnElement>(
            runner, self_id);
    CHECK(packets_from_box.size() == (Dim > 1 ? 3 : 2));
    CHECK(packets_from_box[0] == packet_keep);
    CHECK(packets_from_box[1] == packet_east);
    if constexpr (Dim > 1) {
      CHECK(packets_from_box[2] == packet_south);
    }
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Particles.MonteCarlo.CommunicationTags",
                  "[Evolution][Unit]") {
  test_send_receive_actions<1>();
  test_send_receive_actions<2>();
  test_send_receive_actions<3>();
}
