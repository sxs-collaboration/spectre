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
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/Particles/MonteCarlo/Actions/InitializeMonteCarlo.hpp"
#include "Evolution/Particles/MonteCarlo/MonteCarloOptions.hpp"
#include "Evolution/Particles/MonteCarlo/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RadiationTransport/MonteCarlo/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RadiationTransport/MonteCarlo/HomogeneousSphere.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Time/Tags/Time.hpp"
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

// inertial coordinates set to logical coords here
tnsr::I<DataVector, 3, Frame::Inertial> spatial_coords_inertial(
    tnsr::I<DataVector, 3, Frame::ElementLogical> logical_coords) {
  auto x = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      logical_coords.get(0), 0.0);
  x.get(0) = logical_coords.get(0);
  x.get(1) = logical_coords.get(1);
  x.get(2) = logical_coords.get(2);
  return x;
}

template <size_t Dim>
struct System {
  static constexpr size_t volume_dim = Dim;
  using variables_tag = ::Tags::Variables<tmpl::list<Var1>>;
  using hydro_variables_tag = ::Tags::Variables<hydro::grmhd_tags<DataVector>>;
  using flux_variables = tmpl::list<Var1>;
};

template <size_t Dim, typename Metavariables, size_t EnergyBins,
          size_t NeutrinoSpecies>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Dim>;

  using initial_tags = tmpl::list<
      domain::Tags::Mesh<Dim>, evolution::dg::subcell::Tags::Mesh<Dim>,
      evolution::dg::subcell::Tags::ActiveGrid, domain::Tags::Element<Dim>,
      ::Tags::Variables<tmpl::list<Var1>>, ::Tags::Time,
      evolution::dg::subcell::Tags::Coordinates<Dim, Frame::ElementLogical>,
      evolution::dg::subcell::Tags::Coordinates<Dim, Frame::Inertial>>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<initial_tags>,
                 Initialization::Actions::InitializeMCTags<
                     System<Dim>, EnergyBins, NeutrinoSpecies>>>>;
};

template <size_t Dim, size_t EnergyBins, size_t NeutrinoSpecies>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  using component_list =
      tmpl::list<component<Dim, Metavariables, EnergyBins, NeutrinoSpecies>>;
  using system = System<Dim>;

  using initial_data_list =
      RadiationTransport::MonteCarlo::Solutions::all_solutions;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<evolution::initial_data::InitialData, initial_data_list>>;
  };

  using const_global_cache_tags = tmpl::list<
      hydro::Tags::GrmhdEquationOfState,
      Particles::MonteCarlo::Tags::MonteCarloOptions<NeutrinoSpecies>,
      evolution::initial_data::Tags::InitialData>;
};

void test_initialize_monte_carlo() {
  const size_t Dim = 3;
  const size_t energy_bins = 4;
  const size_t neutrino_species = 3;

  register_classes_with_charm<EquationsOfState::Tabulated3D<true>>();
  register_classes_with_charm<
      Particles::MonteCarlo::MonteCarloOptions<neutrino_species>>();
  register_classes_with_charm<
      RadiationTransport::MonteCarlo::Solutions::HomogeneousSphere>();

  // Fake EoS and NuLib tables
  const std::string h5_file_name_compose{
      unit_test_src_path() +
      "PointwiseFunctions/Hydro/EquationsOfState/dd2_unit_test.h5"};
  std::unique_ptr<EquationsOfState::EquationOfState<true, 3>>
      equation_of_state_ptr =
          std::make_unique<EquationsOfState::Tabulated3D<true>>(
              h5_file_name_compose, "/dd2");

  using metavars = Metavariables<Dim, energy_bins, neutrino_species>;
  using comp = component<Dim, metavars, energy_bins, neutrino_species>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;

  const double time = 0.0;
  const Mesh<Dim> dg_mesh{5, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const evolution::dg::subcell::ActiveGrid active_grid =
      evolution::dg::subcell::ActiveGrid::Subcell;
  const size_t n_pts = subcell_mesh.number_of_grid_points();

  DirectionMap<Dim, Neighbors<Dim>> neighbors{};
  const ElementId<Dim> self_id = ElementId<Dim>{0, {{{1, 0}, {0, 0}, {0, 0}}}};
  const ElementId<Dim> east_id = ElementId<Dim>{0, {{{1, 1}, {0, 0}, {0, 0}}}};
  const ElementId<Dim> south_id = ElementId<Dim>{1, {{{0, 0}, {0, 0}, {0, 0}}}};
  neighbors[Direction<Dim>::upper_xi()] =
      Neighbors<Dim>{{east_id}, OrientationMap<Dim>::create_aligned()};
  const OrientationMap<Dim> orientation = OrientationMap<Dim>{
      std::array{Direction<Dim>::lower_xi(), Direction<Dim>::lower_eta(),
                 Direction<Dim>::upper_zeta()}};
  neighbors[Direction<Dim>::lower_eta()] =
      Neighbors<Dim>{{south_id}, orientation};
  const Element<Dim> element{self_id, neighbors};

  using evolved_vars_tags = tmpl::list<Var1>;
  Variables<evolved_vars_tags> evolved_vars{n_pts};
  // Set Var1 to the logical coords, just need some data
  get(get<Var1>(evolved_vars)) = get<0>(logical_coordinates(subcell_mesh));

  const DataVector zero_dv(n_pts, 0.0);

  // Mesh with ghost zones, for size checking
  const auto& subcell_extents = subcell_mesh.extents();
  const size_t num_ghost_zones = 1;
  size_t mesh_size_with_ghost = 1;
  size_t mortar_data_size = 1;
  size_t coupling_data_size = 1;
  for (size_t d = 0; d < Dim; d++) {
    mesh_size_with_ghost *= subcell_extents[d] + 2 * num_ghost_zones;
    if (d < Dim - 1) {
      mortar_data_size *= subcell_extents[d];
      coupling_data_size *= subcell_extents[d] + 2 * num_ghost_zones;
    }
  }

  // Grid coordinates on subcell mesh
  const size_t mesh_size = subcell_extents[0];
  CHECK(subcell_mesh.extents()[1] == mesh_size);
  CHECK(subcell_mesh.extents()[2] == mesh_size);
  tnsr::I<DataVector, 3, Frame::ElementLogical> mesh_coordinates =
      logical_coordinates(subcell_mesh);
  tnsr::I<DataVector, 3, Frame::Inertial> inertial_coordinates =
      spatial_coords_inertial(mesh_coordinates);

  // Set up Monte Carlo options
  std::array<double, neutrino_species> initial_packet_energy = {1.e-12, 1.e-12,
                                                                1.e-12};
  const size_t desired_packets_per_species = 1e6;
  std::unique_ptr<Particles::MonteCarlo::MonteCarloOptions<neutrino_species>>
      monte_carlo_options_ptr = std::make_unique<
          Particles::MonteCarlo::MonteCarloOptions<neutrino_species>>(
          initial_packet_energy, desired_packets_per_species);

  // Set up mock initial data
  std::unique_ptr<RadiationTransport::MonteCarlo::Solutions::HomogeneousSphere>
      initial_data_ptr = std::make_unique<
          RadiationTransport::MonteCarlo::Solutions::HomogeneousSphere>(
          1.0, std::array<double, 2>{1.0, 0.1}, std::array<double, 2>{1.0, 0.1},
          std::array<double, 2>{0.1, 0.05}, std::move(equation_of_state_ptr));

  MockRuntimeSystem runner{{std::move(equation_of_state_ptr),
                            std::move(monte_carlo_options_ptr),
                            std::move(initial_data_ptr)}};

  ActionTesting::emplace_array_component_and_initialize<comp>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, self_id,
      {dg_mesh, subcell_mesh, active_grid, element, evolved_vars, time,
       mesh_coordinates, inertial_coordinates});

  // Run initialize action
  ActionTesting::next_action<comp>(make_not_null(&runner), self_id);

  // Check that expected variables have been created
  const std::vector<Particles::MonteCarlo::Packet>& packets_from_box =
      ActionTesting::get_databox_tag<
          comp, Particles::MonteCarlo::Tags::PacketsOnElement>(runner, self_id);
  CHECK(packets_from_box.empty());
  const std::mt19937& generator = ActionTesting::get_databox_tag<
      comp, Particles::MonteCarlo::Tags::RandomNumberGenerator>(runner,
                                                                self_id);
  // Min value is always zero per the mersenner twister engine documentation;
  // called as a simple use of the generator.
  CHECK(generator.min() == 0);
  const std::array<double, neutrino_species>& minimum_energy_at_emission =
      ActionTesting::get_databox_tag<
          comp, Particles::MonteCarlo::Tags::MinimumPacketEnergyAtEmission<
                    neutrino_species>>(runner, self_id);
  for (size_t s = 0; s < neutrino_species; s++) {
    CHECK(gsl::at(minimum_energy_at_emission, s) ==
          gsl::at(initial_packet_energy, s));
  }
  // Check size of the fluid variables
  const Scalar<DataVector>& lorentz_factor =
      ActionTesting::get_databox_tag<comp,
                                     hydro::Tags::LorentzFactor<DataVector>>(
          runner, self_id);
  CHECK(get(lorentz_factor).size() == n_pts);
  // Check size of coupling terms
  const Scalar<DataVector>& coupling_tilde_tau = ActionTesting::get_databox_tag<
      comp, Particles::MonteCarlo::Tags::CouplingTildeTau<DataVector>>(runner,
                                                                       self_id);
  CHECK(get(coupling_tilde_tau).size() == mesh_size_with_ghost);

  using MortarData =
      typename Particles::MonteCarlo::Tags::MortarDataTag<Dim>::type;
  const MortarData& mortar_data = ActionTesting::get_databox_tag<
      comp, Particles::MonteCarlo::Tags::MortarDataTag<Dim>>(runner, self_id);
  using CouplingData =
      typename Particles::MonteCarlo::Tags::GhostZoneCouplingDataTag<Dim>::type;
  const CouplingData& coupling_data = ActionTesting::get_databox_tag<
      comp, Particles::MonteCarlo::Tags::GhostZoneCouplingDataTag<Dim>>(
      runner, self_id);
  const DirectionalId<Dim> east_neighbor_id{Direction<Dim>::upper_xi(),
                                            east_id};
  const DirectionalId<Dim> south_neighbor_id{Direction<Dim>::lower_eta(),
                                             south_id};
  CHECK(
      (mortar_data.temperature.find(east_neighbor_id)->second).value().size() ==
      mortar_data_size);
  CHECK((mortar_data.cell_light_crossing_time.find(south_neighbor_id)->second)
            .value()
            .size() == mortar_data_size);
  CHECK((coupling_data.coupling_tilde_tau.find(east_neighbor_id)->second)
            .value()
            .size() == coupling_data_size);
  CHECK((coupling_data.coupling_tilde_s.find(south_neighbor_id)->second)
            .value()
            .get(Dim - 1)
            .size() == coupling_data_size);
  using GhostZoneData =
      typename Particles::MonteCarlo::Tags::McGhostZoneDataTag<Dim>::type;
  const GhostZoneData& ghost_data_in_box = ActionTesting::get_databox_tag<
      comp, Particles::MonteCarlo::Tags::McGhostZoneDataTag<Dim>>(runner,
                                                                  self_id);
  CHECK(ghost_data_in_box.empty());
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Particles.MonteCarloInitializeAction",
                  "[Unit][Evolution]") {
  test_initialize_monte_carlo();
}
