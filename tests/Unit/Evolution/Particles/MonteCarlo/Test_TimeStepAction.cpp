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
#include "Evolution/Particles/MonteCarlo/Actions/TimeStepActions.hpp"
#include "Evolution/Particles/MonteCarlo/GhostZoneCommunication.hpp"
#include "Evolution/Particles/MonteCarlo/GhostZoneCommunicationTags.hpp"
#include "Evolution/Particles/MonteCarlo/MortarData.hpp"
#include "Evolution/Particles/MonteCarlo/Packet.hpp"
#include "Evolution/Particles/MonteCarlo/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Informer/InfoFromBuild.hpp"
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
      Particles::MonteCarlo::Tags::RandomNumberGenerator, ::Tags::TimeStepId,
      ::Tags::Next<::Tags::TimeStepId>, domain::Tags::Mesh<Dim>,
      evolution::dg::subcell::Tags::Mesh<Dim>,
      evolution::dg::subcell::Tags::ActiveGrid, domain::Tags::Element<Dim>,
      Particles::MonteCarlo::Tags::McGhostZoneDataTag<Dim>,
      Particles::MonteCarlo::Tags::PacketsOnElement,
      ::Tags::Variables<tmpl::list<Var1>>,
      Particles::MonteCarlo::Tags::MortarDataTag<Dim>,
      hydro::Tags::LorentzFactor<DataVector>,
      hydro::Tags::RestMassDensity<DataVector>,
      hydro::Tags::ElectronFraction<DataVector>,
      hydro::Tags::Temperature<DataVector>,
      Particles::MonteCarlo::Tags::CellLightCrossingTime<DataVector>,
      domain::Tags::NeighborMesh<Dim>,
      Particles::MonteCarlo::Tags::DesiredPacketEnergyAtEmission<3>,
      hydro::Tags::LowerSpatialFourVelocity<DataVector, Dim, Frame::Inertial>,
      gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<DataVector, Dim, Frame::Inertial>,
      Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<Dim>,
                  Frame::Inertial>,
      Tags::deriv<gr::Tags::Shift<DataVector, Dim>, tmpl::size_t<Dim>,
                  Frame::Inertial>,
      Tags::deriv<gr::Tags::InverseSpatialMetric<DataVector, Dim>,
                  tmpl::size_t<Dim>, Frame::Inertial>,
      gr::Tags::SpatialMetric<DataVector, Dim, Frame::Inertial>,
      gr::Tags::InverseSpatialMetric<DataVector, Dim, Frame::Inertial>,
      gr::Tags::DetSpatialMetric<DataVector>,
      evolution::dg::subcell::Tags::Coordinates<Dim, Frame::ElementLogical>,
      domain::Tags::MeshVelocity<Dim>,
      evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToInertial<Dim>,
      evolution::dg::subcell::fd::Tags::DetInverseJacobianLogicalToInertial,
      domain::Tags::InverseJacobian<Dim + 1, Frame::Inertial, Frame::Fluid>,
      domain::Tags::DetInvJacobian<Frame::Inertial, Frame::Fluid>,
      domain::Tags::Jacobian<Dim + 1, Frame::Inertial, Frame::Fluid>,
      evolution::dg::subcell::Tags::InterpolatorsFromFdToNeighborFd<Dim>>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<initial_tags>,
                 Particles::MonteCarlo::Actions::TakeTimeStep<4, 3>>>>;
};

template <size_t Dim>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  using component_list = tmpl::list<component<Dim, Metavariables>>;
  using system = System<Dim>;

  using const_global_cache_tags =
      tmpl::list<hydro::Tags::GrmhdEquationOfState,
                 Particles::MonteCarlo::Tags::InteractionRatesTable<4, 3>>;
};

void test_advance_packets() {
  MAKE_GENERATOR(generator);
  const size_t Dim = 3;

  register_classes_with_charm<EquationsOfState::Tabulated3D<true>>();
  register_classes_with_charm<
      Particles::MonteCarlo::NeutrinoInteractionTable<4, 3>>();

  // Fake EoS and NuLib tables
  const std::string h5_file_name_compose{
      unit_test_src_path() +
      "PointwiseFunctions/Hydro/EquationsOfState/dd2_unit_test.h5"};
  std::unique_ptr<EquationsOfState::EquationOfState<true, 3>>
      equation_of_state_ptr =
          std::make_unique<EquationsOfState::Tabulated3D<true>>(
              h5_file_name_compose, "/dd2");
  const std::string h5_file_name_nulib{
      unit_test_src_path() +
      "Evolution/Particles/MonteCarlo/NuLib_TestTable.h5"};
  std::unique_ptr<Particles::MonteCarlo::NeutrinoInteractionTable<4, 3>>
      interaction_table_ptr = std::make_unique<
          Particles::MonteCarlo::NeutrinoInteractionTable<4, 3>>(
          h5_file_name_nulib);

  using Interps = DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>;
  using metavars = Metavariables<Dim>;
  using comp = component<Dim, metavars>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;

  const TimeStepId time_step_id{true, 1, Time{Slab{1.0, 2.0}, {0, 10}}};
  const TimeStepId next_time_step_id{true, 1, Time{Slab{1.0, 2.0}, {1, 10}}};
  const Mesh<Dim> dg_mesh{5, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const evolution::dg::subcell::ActiveGrid active_grid =
      evolution::dg::subcell::ActiveGrid::Subcell;
  const size_t n_pts = subcell_mesh.number_of_grid_points();

  const DirectionMap<Dim, Neighbors<Dim>> neighbors{};
  const ElementId<Dim> self_id = ElementId<Dim>{0, {{{1, 0}, {0, 0}, {0, 0}}}};
  const Element<Dim> element{self_id, neighbors};

  using NeighborDataMap =
      DirectionalIdMap<Dim, Particles::MonteCarlo::McGhostZoneData<Dim>>;
  NeighborDataMap neighbor_data{};

  using evolved_vars_tags = tmpl::list<Var1>;
  Variables<evolved_vars_tags> evolved_vars{n_pts};
  // Set Var1 to the logical coords, just need some data
  get(get<Var1>(evolved_vars)) = get<0>(logical_coordinates(subcell_mesh));

  const DataVector zero_dv(n_pts, 0.0);

  // Minkowski metric
  Scalar<DataVector> lapse{DataVector(n_pts, 1.0)};
  tnsr::II<DataVector, 3, Frame::Inertial> inv_spatial_metric =
      make_with_value<tnsr::II<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  inv_spatial_metric.get(0, 0) = 1.0;
  inv_spatial_metric.get(1, 1) = 1.0;
  inv_spatial_metric.get(2, 2) = 1.0;
  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  spatial_metric.get(0, 0) = 1.0;
  spatial_metric.get(1, 1) = 1.0;
  spatial_metric.get(2, 2) = 1.0;
  Scalar<DataVector> determinant_spatial_metric(n_pts, 1.0);
  tnsr::I<DataVector, 3, Frame::Inertial> shift =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  tnsr::i<DataVector, 3, Frame::Inertial> d_lapse =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  tnsr::iJ<DataVector, 3, Frame::Inertial> d_shift =
      make_with_value<tnsr::iJ<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  tnsr::iJJ<DataVector, 3, Frame::Inertial> d_inv_spatial_metric =
      make_with_value<tnsr::iJJ<DataVector, 3, Frame::Inertial>>(lapse, 0.0);

  // Fluid variables
  Scalar<DataVector> rest_mass_density(zero_dv);
  Scalar<DataVector> lorentz_factor(zero_dv);
  Scalar<DataVector> electron_fraction(zero_dv);
  Scalar<DataVector> temperature(zero_dv);
  tnsr::i<DataVector, 3, Frame::Inertial> lower_spatial_four_velocity =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  Scalar<DataVector> cell_light_crossing_time(zero_dv);
  std::array<DataVector, 3> single_packet_energy = {zero_dv, zero_dv, zero_dv};
  gsl::at(single_packet_energy, 0) = 1.0;
  gsl::at(single_packet_energy, 1) = 1.0;
  gsl::at(single_packet_energy, 2) = 1.0;
  get(cell_light_crossing_time) = 1.0;

  // Grid coordinates on subcell mesh
  const size_t mesh_size = subcell_mesh.extents()[0];
  CHECK(subcell_mesh.extents()[1] == mesh_size);
  CHECK(subcell_mesh.extents()[2] == mesh_size);
  tnsr::I<DataVector, 3, Frame::ElementLogical> mesh_coordinates =
      make_with_value<tnsr::I<DataVector, 3, Frame::ElementLogical>>(lapse,
                                                                     0.0);
  for (size_t iz = 0; iz < mesh_size; iz++) {
    const double z_coord = -1.0 + (0.5 + static_cast<double>(iz)) /
                                      static_cast<double>(mesh_size) * 2.0;
    for (size_t iy = 0; iy < mesh_size; iy++) {
      const double y_coord = -1.0 + (0.5 + static_cast<double>(iy)) /
                                        static_cast<double>(mesh_size) * 2.0;
      for (size_t ix = 0; ix < mesh_size; ix++) {
        const double x_coord = -1.0 + (0.5 + static_cast<double>(ix)) /
                                          static_cast<double>(mesh_size) * 2.0;
        const size_t idx = ix + iy * mesh_size + iz * mesh_size * mesh_size;
        mesh_coordinates.get(0)[idx] = x_coord;
        mesh_coordinates.get(1)[idx] = y_coord;
        mesh_coordinates.get(2)[idx] = z_coord;
      }
    }
  }

  const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>> mesh_velocity =
      std::nullopt;

  // Jacobian set to identity for now
  Scalar<DataVector> det_inverse_jacobian_inertial_to_fluid(n_pts, 1.0);
  InverseJacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>
      inverse_jacobian_inertial_to_fluid = make_with_value<
          InverseJacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>>(lapse,
                                                                         0.0);
  inverse_jacobian_inertial_to_fluid.get(0, 0) = 1.0;
  inverse_jacobian_inertial_to_fluid.get(1, 1) = 1.0;
  inverse_jacobian_inertial_to_fluid.get(2, 2) = 1.0;
  inverse_jacobian_inertial_to_fluid.get(3, 3) = 1.0;
  Jacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>
      jacobian_inertial_to_fluid = make_with_value<
          Jacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>>(lapse, 0.0);
  jacobian_inertial_to_fluid.get(0, 0) = 1.0;
  jacobian_inertial_to_fluid.get(1, 1) = 1.0;
  jacobian_inertial_to_fluid.get(2, 2) = 1.0;
  jacobian_inertial_to_fluid.get(3, 3) = 1.0;
  // Logical to inertial inverse jacobian, also identity for now
  InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
      inverse_jacobian_logical_to_inertial =
          make_with_value<InverseJacobian<DataVector, 3, Frame::ElementLogical,
                                          Frame::Inertial>>(lapse, 0.0);
  inverse_jacobian_logical_to_inertial.get(0, 0) = 1.0;
  inverse_jacobian_logical_to_inertial.get(1, 1) = 1.0;
  inverse_jacobian_logical_to_inertial.get(2, 2) = 1.0;
  Scalar<DataVector> det_inverse_jacobian_logical_to_inertial(n_pts, 1.0);

  // Initialize MortarData
  Particles::MonteCarlo::MortarData<Dim> mortar_data{};

  std::vector<Particles::MonteCarlo::Packet> packets_on_element{};
  const size_t species = 1;
  const double number_of_neutrinos = 2.0;
  const size_t index_of_closest_grid_point = 0;
  const double t0 = time_step_id.step_time().value();
  const double x0 = 0.3;
  const double y0 = 0.5;
  const double z0 = -0.7;
  const double p_upper_t0 = 1.1;
  const double p_x0 = 0.9;
  const double p_y0 = 0.7;
  const double p_z0 = 0.1;
  const Particles::MonteCarlo::Packet packet(species, number_of_neutrinos,
                                       index_of_closest_grid_point, t0, x0, y0,
                                       z0, p_upper_t0, p_x0, p_y0, p_z0);
  packets_on_element.push_back(packet);

  MockRuntimeSystem runner{
      {std::move(equation_of_state_ptr), std::move(interaction_table_ptr)}};

  Interps fd_to_neighbor_fd_interpolants{};
  ActionTesting::emplace_array_component_and_initialize<comp>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, self_id,
      {generator,
       time_step_id,
       next_time_step_id,
       dg_mesh,
       subcell_mesh,
       active_grid,
       element,
       neighbor_data,
       packets_on_element,
       evolved_vars,
       mortar_data,
       lorentz_factor,
       rest_mass_density,
       electron_fraction,
       temperature,
       cell_light_crossing_time,
       typename domain::Tags::NeighborMesh<Dim>::type{},
       single_packet_energy,
       lower_spatial_four_velocity,
       lapse,
       shift,
       d_lapse,
       d_shift,
       d_inv_spatial_metric,
       spatial_metric,
       inv_spatial_metric,
       determinant_spatial_metric,
       mesh_coordinates,
       mesh_velocity,
       inverse_jacobian_logical_to_inertial,
       det_inverse_jacobian_logical_to_inertial,
       inverse_jacobian_inertial_to_fluid,
       det_inverse_jacobian_inertial_to_fluid,
       jacobian_inertial_to_fluid,
       fd_to_neighbor_fd_interpolants});

  // Run singe time step
  ActionTesting::next_action<comp>(make_not_null(&runner), self_id);

  const auto& packets_from_box =
      get_databox_tag<comp, Particles::MonteCarlo::Tags::PacketsOnElement>(
          runner, self_id);
  CHECK(packets_from_box[0].time == next_time_step_id.step_time().value());
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Particles.MonteCarloTimeStepAction",
                  "[Unit][Evolution]") {
  test_advance_packets();
}
