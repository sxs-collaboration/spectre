
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
#include "Informer/InfoFromBuild.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Tabulated3d.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"

namespace {

void test_optically_thick_sphere() {
  const size_t mesh_size = 20;
  const Mesh<3> mesh(mesh_size, Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered);
  const size_t num_ghost_zones = 1;

  MAKE_GENERATOR(generator);

  const size_t dv_size = mesh_size * mesh_size * mesh_size;
  DataVector zero_dv(dv_size, 0.0);
  const size_t dv_size_with_ghost = cube(mesh_size + 2 * num_ghost_zones);
  DataVector zero_dv_with_ghost(dv_size_with_ghost, 0.0);
  const size_t dv_size_in_ghost = square(mesh_size) * num_ghost_zones;
  DataVector zero_dv_in_ghost(dv_size_in_ghost, 0.0);
  DataVector one_dv_in_ghost(dv_size_in_ghost, 1.0);

  // Logical coordinates
  tnsr::I<DataVector, 3, Frame::ElementLogical> mesh_coordinates =
      make_with_value<tnsr::I<DataVector, 3, Frame::ElementLogical>>(zero_dv,
                                                                     0.0);
  tnsr::I<DataVector, 3, Frame::Inertial> inertial_coordinates =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(zero_dv, 0.0);
  Scalar<DataVector> radius = make_with_value<Scalar<DataVector>>(zero_dv, 0.0);
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
        inertial_coordinates.get(0)[idx] = x_coord * 2.0;
        inertial_coordinates.get(1)[idx] = y_coord * 2.0;
        inertial_coordinates.get(2)[idx] = z_coord * 2.0;
        get(radius)[idx] =
            sqrt(x_coord * x_coord + y_coord * y_coord + z_coord * z_coord);
      }
    }
  }

  // Fluid variables; set to create a high-density sphere of radius 1
  // Note that there is a factor of 2 between inertial and logical
  // coordinates, per the jacobian chosen below
  Scalar<DataVector> lorentz_factor =
      make_with_value<Scalar<DataVector>>(zero_dv, 1.0);
  tnsr::i<DataVector, 3, Frame::Inertial> lower_spatial_four_velocity =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(zero_dv, 0.0);
  // Default to values giving no interactions
  Scalar<DataVector> baryon_density(dv_size, exp(-10.0));
  Scalar<DataVector> temperature(dv_size, 0.5);
  Scalar<DataVector> electron_fraction(dv_size, 0.055);
  Scalar<DataVector> cell_light_crossing_time(
      dv_size, 1.0 / static_cast<double>(mesh_size));
  // Set density higher inside sphere of radius 0.5 (in logical coords)
  // where emission will be higher
  for (size_t i = 0; i < dv_size; i++) {
    if (get(radius)[i] < 0.5) {
      get(baryon_density)[i] = 1.0;
    }
  }

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
  Scalar<DataVector> determinant_spatial_metric =
      make_with_value<Scalar<DataVector>>(lapse, 1.0);
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

  // Inertial-to-fluid jacobian set to identity for now
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

  // Logical to inertial inverse jacobian, rescale by factor of 2 to have
  // sphere of radius 0.5 mapped to unit sphere
  InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
      inverse_jacobian_logical_to_inertial =
          make_with_value<InverseJacobian<DataVector, 3, Frame::ElementLogical,
                                          Frame::Inertial>>(lapse, 0.0);
  inverse_jacobian_logical_to_inertial.get(0, 0) = 0.5;
  inverse_jacobian_logical_to_inertial.get(1, 1) = 0.5;
  inverse_jacobian_logical_to_inertial.get(2, 2) = 0.5;
  Scalar<DataVector> det_jacobian_logical_to_inertial =
      make_with_value<Scalar<DataVector>>(lapse, 8.0);

  // Create neutrino interaction tables from explicit data
  const std::vector<double> table_log_density{-10.0, 0.0};
  const std::vector<double> table_log_temperature{-2.0, 0.0};
  const std::vector<double> table_electron_fraction{0.05, 0.06};
  const size_t n_points_table = 8;
  const size_t EnergyBins = 2;
  const size_t NeutrinoSpecies = 2;
  const std::array<double, EnergyBins> table_neutrino_energies = {1.0, 3.0};
  const double eps_opacity = 1.e-70;

  const size_t number_of_vars = 3 * EnergyBins * NeutrinoSpecies;
  std::vector<double> table_data{};
  table_data.resize(number_of_vars * n_points_table);
  // Fill in fake data for interaction rates. Fastest moving index is density,
  // then temperature, then electron fraction.
  // Here, we set interaction rates that linearly grow in log(density) so
  // we fill all 'high density' points of the table.
  const double ka = 10.0;
  const double eta = 9.0;
  const size_t shift_table = EnergyBins * NeutrinoSpecies;
  for (size_t p = 0; p < n_points_table; p += 1) {
    // Fill in all scattering / absorption values with eps_opacity
    for (size_t q = 0; q < 2 * shift_table; q++){
      table_data[p * number_of_vars + shift_table + q] = eps_opacity;
    }
  }
  for (size_t p = 1; p < n_points_table; p += 2) {
    // Emissivity
    // We only emit in the lowest energy bin and for nu_e
    table_data[p * number_of_vars] = eta;
    // Absorption
    // Only set nu_e, but in all energy bins
    table_data[p * number_of_vars + shift_table] = ka;
    table_data[p * number_of_vars + 1 + shift_table] = ka;
  }
  const Particles::MonteCarlo::NeutrinoInteractionTable<2, 2> interaction_table(
      table_data,
      table_neutrino_energies, table_log_density, table_log_temperature,
      table_electron_fraction);

  std::string h5_file_name_compose{
      unit_test_src_path() +
      "PointwiseFunctions/Hydro/EquationsOfState/dd2_unit_test.h5"};
  EquationsOfState::Tabulated3D<true> equation_of_state(h5_file_name_compose,
                                                        "/dd2");

  // Coupling terms (not used at this point)
  Scalar<DataVector> coupling_tilde_tau =
      make_with_value<Scalar<DataVector>>(zero_dv, 0.0);
  Scalar<DataVector> coupling_rho_ye =
      make_with_value<Scalar<DataVector>>(zero_dv, 0.0);
  tnsr::i<DataVector, 3, Frame::Inertial> coupling_tilde_s =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(zero_dv, 0.0);

  // Create list of packets
  std::vector<Particles::MonteCarlo::Packet> packets{};
  Particles::MonteCarlo::TemplatedLocalFunctions<2, 2> MonteCarloStruct;

  // Set energy of packets to be created
  std::array<DataVector, 2> single_packet_energy = {zero_dv, zero_dv};
  for (size_t s = 0; s < NeutrinoSpecies; s++) {
    gsl::at(single_packet_energy, s) = 0.001;
  }

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

  // Time stepping information
  const double final_time = 20.0;
  const double dt = 1.0 / static_cast<double>(mesh_size);
  const double dt_obs = 1.0;
  double current_time = 0.0;
  double next_obs_time = dt_obs;
  double last_obs_time = 0.0;

  // Actual evolution of the system
  size_t packets_escaping = 0;
  while (current_time < final_time) {
    MonteCarloStruct.take_time_step_on_element(
        &packets, &generator, &single_packet_energy, current_time,
        current_time + dt, equation_of_state, interaction_table,
        electron_fraction, baryon_density, temperature, lorentz_factor,
        lower_spatial_four_velocity, lapse, shift, d_lapse, d_shift,
        d_inv_spatial_metric, spatial_metric, inv_spatial_metric,
        determinant_spatial_metric, cell_light_crossing_time, mesh,
        mesh_coordinates, num_ghost_zones, mesh_velocity,
        inverse_jacobian_logical_to_inertial, det_jacobian_logical_to_inertial,
        jacobian_inertial_to_fluid, inverse_jacobian_inertial_to_fluid,
        electron_fraction_ghost_zones, baryon_density_ghost_zones,
        temperature_ghost_zones, cell_light_crossing_time_ghost_zones);
    current_time += dt;

    size_t n_packets = packets.size();
    // Remove packets with logical coordinate radius > 0.9
    // (assume these are escaping)
    for (size_t p = 0; p < n_packets; p++) {
      Particles::MonteCarlo::Packet& packet = packets[p];
      double rad = 0.0;
      for (size_t d = 0; d < 3; d++) {
        rad += packet.coordinates.get(d) * packet.coordinates.get(d);
      }
      rad = sqrt(rad);
      if (rad > 0.9) {
        std::swap(packets[p], packets[n_packets - 1]);
        packets.pop_back();
        p--;
        n_packets--;
        packets_escaping++;
      }
    }
    if (current_time >= next_obs_time) {
      Parallel::printf(
          "Time: %.5f; Luminosity: %.5f\n", current_time,
          static_cast<double>(packets_escaping) * 0.001 /
          (current_time - last_obs_time));
      last_obs_time = current_time;
      next_obs_time = current_time + dt_obs;
      packets_escaping = 0;
    }
  }
}

void test_boosted_sphere() {

  const size_t mesh_size = 20;
  const Mesh<3> mesh(mesh_size, Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered);
  const size_t num_ghost_zones = 1;

  MAKE_GENERATOR(generator);

  const size_t dv_size = mesh_size * mesh_size * mesh_size;
  DataVector zero_dv(dv_size, 0.0);
  const size_t dv_size_with_ghost = cube(mesh_size + 2 * num_ghost_zones);
  DataVector zero_dv_with_ghost(dv_size_with_ghost, 0.0);
  const size_t dv_size_in_ghost = square(mesh_size) * num_ghost_zones;
  DataVector zero_dv_in_ghost(dv_size_in_ghost, 0.0);
  DataVector one_dv_in_ghost(dv_size_in_ghost, 1.0);

  // Logical coordinates
  tnsr::I<DataVector, 3, Frame::ElementLogical> mesh_coordinates =
      make_with_value<tnsr::I<DataVector, 3, Frame::ElementLogical>>(zero_dv,
                                                                     0.0);
  tnsr::I<DataVector, 3, Frame::Inertial> inertial_coordinates =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(zero_dv, 0.0);
  Scalar<DataVector> radius = make_with_value<Scalar<DataVector>>(zero_dv, 0.0);
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
        inertial_coordinates.get(0)[idx] = x_coord * 2.0;
        inertial_coordinates.get(1)[idx] = y_coord * 2.0;
        inertial_coordinates.get(2)[idx] = z_coord * 2.0;
        get(radius)[idx] =
            sqrt(x_coord * x_coord + y_coord * y_coord + z_coord * z_coord);
      }
    }
  }

  // Fluid variables; set to create a high-density sphere of radius 1
  // Note that there is a factor of 2 between inertial and logical
  // coordinates, per the jacobian chosen below
  Scalar<DataVector> lorentz_factor =
    make_with_value<Scalar<DataVector>>(zero_dv, sqrt(3)/2.0);
  tnsr::i<DataVector, 3, Frame::Inertial> lower_spatial_four_velocity =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(zero_dv, 0.0);
  lower_spatial_four_velocity.get(0) = 0.5;
  // Default to values giving no interactions
  Scalar<DataVector> baryon_density(dv_size, exp(-10.0));
  Scalar<DataVector> temperature(dv_size, 0.5);
  Scalar<DataVector> electron_fraction(dv_size, 0.055);
  Scalar<DataVector> cell_light_crossing_time(
      dv_size, 1.0 / static_cast<double>(mesh_size));
  // Set density higher inside sphere of radius 0.5 (in logical coords)
  // where emission will be higher
  for (size_t i = 0; i < dv_size; i++) {
    if (get(radius)[i] < 0.5) {
      get(baryon_density)[i] = 1.0;
    }
  }

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
  Scalar<DataVector> determinant_spatial_metric =
      make_with_value<Scalar<DataVector>>(lapse, 1.0);
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

  // Inertial-to-fluid jacobian set to identity for now
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

  // Logical to inertial inverse jacobian, rescale by factor of 2 to have
  // sphere of radius 0.5 mapped to unit sphere
  InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
      inverse_jacobian_logical_to_inertial =
          make_with_value<InverseJacobian<DataVector, 3, Frame::ElementLogical,
                                          Frame::Inertial>>(lapse, 0.0);
  inverse_jacobian_logical_to_inertial.get(0, 0) = 0.5;
  inverse_jacobian_logical_to_inertial.get(1, 1) = 0.5;
  inverse_jacobian_logical_to_inertial.get(2, 2) = 0.5;
  Scalar<DataVector> det_jacobian_logical_to_inertial =
      make_with_value<Scalar<DataVector>>(lapse, 8.0);

  // Create neutrino interaction tables from explicit data
  const std::vector<double> table_log_density{-10.0, 0.0};
  const std::vector<double> table_log_temperature{-2.0, 0.0};
  const std::vector<double> table_electron_fraction{0.05, 0.06};
  const size_t n_points_table = 8;
  const size_t EnergyBins = 2;
  const size_t NeutrinoSpecies = 2;
  const std::array<double, EnergyBins> table_neutrino_energies = {1.0, 3.0};
  const double eps_opacity = 1.e-70;

  const size_t number_of_vars = 3 * EnergyBins * NeutrinoSpecies;
  std::vector<double> table_data{};
  table_data.resize(number_of_vars * n_points_table);
  // Fill in fake data for interaction rates. Fastest moving index is density,
  // then temperature, then electron fraction.
  // Here, we set interaction rates that linearly grow in log(density) so
  // we fill all 'high density' points of the table.
  const double ka = 10.0;
  const double eta = 9.0;
  const size_t shift_table = EnergyBins * NeutrinoSpecies;
  for (size_t p = 0; p < n_points_table; p += 1) {
    // Fill in all scattering / absorption values with eps_opacity
    for (size_t q = 0; q < 2 * shift_table; q++){
      table_data[p * number_of_vars + shift_table + q] = eps_opacity;
    }
  }
  for (size_t p = 1; p < n_points_table; p += 2) {
    // Emissivity
    // We only emit in the lowest energy bin and for nu_e
    table_data[p * number_of_vars] = eta;
    // Absorption
    // Only set nu_e, but in all energy bins
    table_data[p * number_of_vars + shift_table] = ka;
    table_data[p * number_of_vars + 1 + shift_table] = ka;
  }
  const Particles::MonteCarlo::NeutrinoInteractionTable<2, 2> interaction_table(
      table_data,
      table_neutrino_energies, table_log_density, table_log_temperature,
      table_electron_fraction);

  std::string h5_file_name_compose{
      unit_test_src_path() +
      "PointwiseFunctions/Hydro/EquationsOfState/dd2_unit_test.h5"};
  EquationsOfState::Tabulated3D<true> equation_of_state(h5_file_name_compose,
                                                        "/dd2");

  // Coupling terms (not used at this point)
  Scalar<DataVector> coupling_tilde_tau =
      make_with_value<Scalar<DataVector>>(zero_dv, 0.0);
  Scalar<DataVector> coupling_rho_ye =
      make_with_value<Scalar<DataVector>>(zero_dv, 0.0);
  tnsr::i<DataVector, 3, Frame::Inertial> coupling_tilde_s =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(zero_dv, 0.0);

  // Create list of packets
  std::vector<Particles::MonteCarlo::Packet> packets{};
  Particles::MonteCarlo::TemplatedLocalFunctions<2, 2> MonteCarloStruct;

  // Set energy of packets to be created
  std::array<DataVector, 2> single_packet_energy = {zero_dv, zero_dv};
  for (size_t s = 0; s < NeutrinoSpecies; s++) {
    gsl::at(single_packet_energy, s) = 0.001;
  }

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

  // Time stepping information
  const double final_time = 5.0;
  const double dt = 0.05;
  const double dt_obs = 1.0;
  double current_time = 0.0;
  double next_obs_time = dt_obs;
  double last_obs_time = 0.0;

  // Actual evolution of the system
  size_t packets_escaping = 0;
  while (current_time < final_time) {
    MonteCarloStruct.take_time_step_on_element(
        &packets, &generator, &single_packet_energy, current_time,
        current_time + dt, equation_of_state, interaction_table,
        electron_fraction, baryon_density, temperature, lorentz_factor,
        lower_spatial_four_velocity, lapse, shift, d_lapse, d_shift,
        d_inv_spatial_metric, spatial_metric, inv_spatial_metric,
        determinant_spatial_metric, cell_light_crossing_time, mesh,
        mesh_coordinates, num_ghost_zones, mesh_velocity,
        inverse_jacobian_logical_to_inertial, det_jacobian_logical_to_inertial,
        jacobian_inertial_to_fluid, inverse_jacobian_inertial_to_fluid,
        electron_fraction_ghost_zones, baryon_density_ghost_zones,
        temperature_ghost_zones, cell_light_crossing_time_ghost_zones);
    current_time += dt;

    size_t n_packets = packets.size();
    // Remove packets with logical coordinate radius > 0.9
    // (assume these are escaping)
    for (size_t p = 0; p < n_packets; p++) {
      Particles::MonteCarlo::Packet& packet = packets[p];
      double rad = 0.0;
      for (size_t d = 0; d < 3; d++) {
        rad += packet.coordinates.get(d) * packet.coordinates.get(d);
      }
      rad = sqrt(rad);
      if (rad > 0.9) {
        std::swap(packets[p], packets[n_packets - 1]);
        packets.pop_back();
        p--;
        n_packets--;
        packets_escaping++;
      }
    }
    if (current_time >= next_obs_time) {
      Parallel::printf(
          "Time: %.5f; Luminosity: %.5f\n", current_time,
          static_cast<double>(packets_escaping) * 0.001 /
         (current_time - last_obs_time));
      last_obs_time = current_time;
      next_obs_time = current_time + dt_obs;
      packets_escaping = 0;
    }
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Particles.MonteCarloOpticallyThickSphere",
                  "[Unit][Evolution]") {
  // Not turned on by defaults... too long for automated tests,
  // but useful framework to test diffusion regime.
  // NOLINTNEXTLINE(readability-simplify-boolean-expr)
  if ((false)) {
    test_optically_thick_sphere();
    test_boosted_sphere();
  }
}
