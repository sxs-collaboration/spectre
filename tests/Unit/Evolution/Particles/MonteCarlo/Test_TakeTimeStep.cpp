// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Evolution/Particles/MonteCarlo/EvolvePackets.hpp"
#include "Evolution/Particles/MonteCarlo/NeutrinoInteractionTable.hpp"
#include "Evolution/Particles/MonteCarlo/Packet.hpp"
#include "Evolution/Particles/MonteCarlo/TemplatedLocalFunctions.hpp"
#include "Framework/TestHelpers.hpp"
#include "Framework/TestingFramework.hpp"
#include "Helpers/PointwiseFunctions/Hydro/EquationsOfState/TestHelpers.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Tabulated3d.hpp"

namespace{

void test_flat_space_time_step() {
  const Mesh<3> mesh(2, Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered);

  MAKE_GENERATOR(generator);

  const size_t NeutrinoSpecies = 2;
  const size_t NeutrinoEnergies = 2;

  const size_t size_1d = 2;
  const size_t num_ghost_zones = 1;

  const size_t dv_size = cube(size_1d);
  DataVector zero_dv(dv_size, 0.0);
  const size_t dv_size_with_ghost = cube(size_1d + 2 * num_ghost_zones);
  DataVector zero_dv_with_ghost(dv_size_with_ghost, 0.0);
  const size_t dv_size_in_ghost = square(size_1d) * num_ghost_zones;
  DataVector zero_dv_in_ghost(dv_size_in_ghost, 0.0);
  DataVector one_dv_in_ghost(dv_size_in_ghost, 1.0);

  // Minkowski metric
  Scalar<DataVector> lapse{DataVector(dv_size, 1.0)};
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
  Scalar<DataVector> determinant_spatial_metric(dv_size, 1.0);
  tnsr::I<DataVector, 3, Frame::Inertial> shift =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  tnsr::i<DataVector, 3, Frame::Inertial> d_lapse =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  tnsr::iJ<DataVector, 3, Frame::Inertial> d_shift =
      make_with_value<tnsr::iJ<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  tnsr::iJJ<DataVector, 3, Frame::Inertial> d_inv_spatial_metric =
      make_with_value<tnsr::iJJ<DataVector, 3, Frame::Inertial>>(lapse, 0.0);

  // Mesh velocity set to std::null for now
  const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>> mesh_velocity =
      std::nullopt;

  // Jacobian set to identity for now
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
  Scalar<DataVector> det_jacobian_logical_to_inertial(dv_size, 1.0);

  // Coordinates
  tnsr::I<DataVector, 3, Frame::ElementLogical> mesh_coordinates =
      make_with_value<tnsr::I<DataVector, 3, Frame::ElementLogical>>(lapse,
                                                                     0.0);
  mesh_coordinates.get(0) =
      DataVector{-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5};
  mesh_coordinates.get(1) =
      DataVector{-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5};
  mesh_coordinates.get(2) =
      DataVector{-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5};

  const size_t species = 1;
  const double number_of_neutrinos = 1.0;
  const size_t index_of_closest_grid_point = 0;
  const double t0 = 0.0;
  const double x0 = -0.5;
  const double y0 = -0.5;
  const double z0 = -0.5;
  const double p_upper_t0 = 1.0;
  const double p_x0 = 1.0;
  const double p_y0 = 0.0;
  const double p_z0 = 0.0;
  Particles::MonteCarlo::Packet packet(species, number_of_neutrinos,
                                       index_of_closest_grid_point, t0, x0, y0,
                                       z0, p_upper_t0, p_x0, p_y0, p_z0);
  packet.renormalize_momentum(inv_spatial_metric, lapse);
  CHECK(packet.momentum_upper_t == 1.0);

  Scalar<DataVector> coupling_tilde_tau =
      make_with_value<Scalar<DataVector>>(zero_dv, 0.0);
  Scalar<DataVector> coupling_rho_ye =
      make_with_value<Scalar<DataVector>>(zero_dv, 0.0);
  tnsr::i<DataVector, 3, Frame::Inertial> coupling_tilde_s =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(zero_dv, 0.0);

  std::vector<Particles::MonteCarlo::Packet> packets{packet};
  Particles::MonteCarlo::TemplatedLocalFunctions<NeutrinoEnergies,
                                                 NeutrinoSpecies>
      MonteCarloStruct;

  const double start_time = 0.0;
  const double final_time = 1.6;
  const double time_step = 0.4;
  std::array<DataVector, NeutrinoSpecies> single_packet_energy = {zero_dv,
                                                                  zero_dv};
  for (size_t s = 0; s < NeutrinoSpecies; s++) {
    gsl::at(single_packet_energy,s) = 1.0;
  }

  // Neutrino-matter interactions
  const std::vector<double> table_log_density{-10.0, 0.0};
  const std::vector<double> table_log_temperature{-2.0, 0.0};
  const std::vector<double> table_electron_fraction{0.05, 0.06};
  const size_t n_points_table = 8;
  const size_t number_of_vars = 3 * NeutrinoEnergies * NeutrinoSpecies;
  const std::array<double, NeutrinoEnergies> table_neutrino_energies = {1.0,
                                                                        3.0};
  std::vector<double> table_data{};
  table_data.resize(number_of_vars * n_points_table);
  // Fill in fake data for interaction rates. Fastest moving index is density,
  // then temperature, then electron fraction.
  // Here, we set interaction rates that linearly grow in log(density) so
  // we fill all 'high density' points of the table.
  for (size_t p = 1; p < n_points_table; p += 2) {
    // Emissivity
    // nu_e
    table_data[p * number_of_vars] = 1.0;
    table_data[p * number_of_vars + 1] = 9.0;
    // nu_a
    table_data[p * number_of_vars + NeutrinoEnergies] = 0.5;
    table_data[p * number_of_vars + NeutrinoEnergies + 1] = 4.5;
    // Absorption
    size_t idx_shift = NeutrinoEnergies * NeutrinoSpecies;
    // nu_e
    table_data[p * number_of_vars + idx_shift] = 1.0;
    table_data[p * number_of_vars + 1 + idx_shift] = 9.0;
    // nu_a
    table_data[p * number_of_vars + NeutrinoEnergies + idx_shift] = 0.5;
    table_data[p * number_of_vars + NeutrinoEnergies + 1 + idx_shift] = 4.5;
    // Scattering
    idx_shift = 2 * NeutrinoEnergies * NeutrinoSpecies;
    // nu_e
    table_data[p * number_of_vars + idx_shift] = 1.0;
    table_data[p * number_of_vars + 1 + idx_shift] = 9.0;
    // nu_a
    table_data[p * number_of_vars + NeutrinoEnergies + idx_shift] = 0.5;
    table_data[p * number_of_vars + NeutrinoEnergies + 1 + idx_shift] = 4.5;
  }
  const Particles::MonteCarlo::NeutrinoInteractionTable<2, 2> interaction_table(
      table_data, table_neutrino_energies, table_log_density,
      table_log_temperature, table_electron_fraction);

  std::string h5_file_name_compose{
    unit_test_src_path() +
    "PointwiseFunctions/Hydro/EquationsOfState/dd2_unit_test.h5"};
  EquationsOfState::Tabulated3D<true> equation_of_state(h5_file_name_compose,
                                                        "/dd2");

  // Currently we choose values leading to no interaction, for
  // predictability...
  Scalar<DataVector> baryon_density(dv_size, 1.e-10);
  Scalar<DataVector> temperature(dv_size, 0.01);
  Scalar<DataVector> electron_fraction(dv_size, 0.05);
  Scalar<DataVector> lorentz_factor(dv_size, 1.0);
  tnsr::i<DataVector, 3, Frame::Inertial> lower_spatial_four_velocity =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  Scalar<DataVector> cell_light_crossing_time(dv_size, 0.6);

  // Ghost zone data (currently zero for all fluid variables on lower end of
  // element and nullopt on upper end of element)
  DirectionalIdMap<3, std::optional<DataVector>> baryon_density_ghost_zones{};
  DirectionalIdMap<3, std::optional<DataVector>> temperature_ghost_zones{};
  DirectionalIdMap<3, std::optional<DataVector>>
      electron_fraction_ghost_zones{};
  DirectionalIdMap<3, std::optional<DataVector>>
      cell_light_crossing_time_ghost_zones{};
  for (size_t d = 0; d < 3; d++) {
    Direction<3> up(d, Side::Upper);
    Direction<3> down(d, Side::Lower);
    const ElementId<3> dummy_neighbor_index(0, 0);
    baryon_density_ghost_zones.insert(
        std::pair{DirectionalId<3>{up, dummy_neighbor_index}, std::nullopt});
    baryon_density_ghost_zones.insert(std::pair{
        DirectionalId<3>{down, dummy_neighbor_index}, zero_dv_in_ghost});
    temperature_ghost_zones.insert(
        std::pair{DirectionalId<3>{up, dummy_neighbor_index}, std::nullopt});
    temperature_ghost_zones.insert(std::pair{
        DirectionalId<3>{down, dummy_neighbor_index}, zero_dv_in_ghost});
    electron_fraction_ghost_zones.insert(
        std::pair{DirectionalId<3>{up, dummy_neighbor_index}, std::nullopt});
    electron_fraction_ghost_zones.insert(std::pair{
        DirectionalId<3>{down, dummy_neighbor_index}, zero_dv_in_ghost});
    cell_light_crossing_time_ghost_zones.insert(
        std::pair{DirectionalId<3>{up, dummy_neighbor_index}, std::nullopt});
    cell_light_crossing_time_ghost_zones.insert(std::pair{
        DirectionalId<3>{down, dummy_neighbor_index}, one_dv_in_ghost});
  }

  double current_time = start_time;
  while (current_time < final_time) {
    MonteCarloStruct.take_time_step_on_element(
        &packets, &generator, &single_packet_energy, current_time,
        current_time + time_step, equation_of_state, interaction_table,
        electron_fraction, baryon_density, temperature, lorentz_factor,
        lower_spatial_four_velocity, lapse, shift, d_lapse, d_shift,
        d_inv_spatial_metric, spatial_metric, inv_spatial_metric,
        determinant_spatial_metric, cell_light_crossing_time, mesh,
        mesh_coordinates, num_ghost_zones, mesh_velocity,
        inverse_jacobian_logical_to_inertial, det_jacobian_logical_to_inertial,
        jacobian_inertial_to_fluid, inverse_jacobian_inertial_to_fluid,
        electron_fraction_ghost_zones, baryon_density_ghost_zones,
        temperature_ghost_zones, cell_light_crossing_time_ghost_zones);
    current_time += time_step;
    const double expected_x0 = -0.5 + current_time;
    // Note that the index dv_size is used to represent all GZs
    const size_t expected_idx =
        (expected_x0 < 0.0) ? 0 : (expected_x0 < 1.0 ? 1 : dv_size);

    // In the current setup, we just propagate a single packet to the
    // final time.
    CHECK(packets[0].coordinates.get(0) == expected_x0);
    CHECK(packets[0].index_of_closest_grid_point == expected_idx);
  }
  size_t n_packets = packets.size();
  CHECK(n_packets==1);
  CHECK(packets[0].time==final_time);
}

} // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Particles.MonteCarloTakeTimeStep",
                  "[Unit][Evolution]") {
  test_flat_space_time_step();
}
