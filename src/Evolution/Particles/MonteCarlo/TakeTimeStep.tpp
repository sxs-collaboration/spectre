// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Particles/MonteCarlo/TemplatedLocalFunctions.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Evolution/Particles/MonteCarlo/CellVolume.hpp"
#include "Evolution/Particles/MonteCarlo/NeutrinoInteractionTable.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

namespace Particles::MonteCarlo {

namespace {

void combine_ghost_data(
    gsl::not_null<DataVector*> with_ghost_data, const Mesh<3> local_mesh,
    const size_t num_ghost_zones, const DataVector& local_data,
    const DirectionalIdMap<3, std::optional<DataVector>>& ghost_data) {
  const Index<3> local_extents = local_mesh.extents();
  Index<3> ghost_extents = local_extents;
  for (size_t d = 0; d < 3; d++) {
    ghost_extents[d] += 2 * num_ghost_zones;
  }
  for (size_t i = 0; i < local_mesh.extents(0); ++i) {
    for (size_t j = 0; j < local_mesh.extents(1); ++j) {
      for (size_t k = 0; k < local_mesh.extents(2); ++k) {
        (*with_ghost_data)[collapsed_index(
            Index<3>{i + num_ghost_zones, j + num_ghost_zones,
                     k + num_ghost_zones},
            ghost_extents)] =
            local_data[collapsed_index(Index<3>{i, j, k}, local_extents)];
      }
    }
  }
  // Loop over each direction. We assume at most one neighber in each
  // direction.
  for (auto& [direction_id, ghost_data_dir] : ghost_data) {
    if (ghost_data_dir) {
      const size_t dimension = direction_id.direction().dimension();
      const Side side = direction_id.direction().side();
      Index<3> ghost_zone_extents = local_extents;
      ghost_zone_extents[dimension] = num_ghost_zones;
      for (size_t i = 0; i < ghost_zone_extents[0]; ++i) {
        for (size_t j = 0; j < ghost_zone_extents[1]; ++j) {
          for (size_t k = 0; k < ghost_zone_extents[2]; ++k) {
            const Index<3> ghost_index_3d{i, j, k};
            const size_t ghost_index =
                collapsed_index(ghost_index_3d, ghost_zone_extents);
            Index<3> extended_index_3d{i + num_ghost_zones, j + num_ghost_zones,
                                       k + num_ghost_zones};
            extended_index_3d[dimension] = (side == Side::Lower)
                                               ? ghost_index_3d[dimension]
                                               : local_extents[dimension] +
                                                     num_ghost_zones +
                                                     ghost_index_3d[dimension];
            const size_t extended_index =
                collapsed_index(extended_index_3d, ghost_extents);
            (*with_ghost_data)[extended_index] =
                ghost_data_dir.value()[ghost_index];
          }
        }
      }
    }
  }
}

}  // namespace

template <size_t EnergyBins, size_t NeutrinoSpecies>
void TemplatedLocalFunctions<EnergyBins, NeutrinoSpecies>::
    take_time_step_on_element(
        const gsl::not_null<std::vector<Packet>*> packets,
        const gsl::not_null<std::mt19937*> random_number_generator,
        const gsl::not_null<std::array<DataVector, NeutrinoSpecies>*>
            single_packet_energy,

        const double start_time, const double target_end_time,
        const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
        const NeutrinoInteractionTable<EnergyBins, NeutrinoSpecies>&
            interaction_table,

        const Scalar<DataVector>& electron_fraction,
        const Scalar<DataVector>& rest_mass_density,
        const Scalar<DataVector>& temperature,
        const Scalar<DataVector>& lorentz_factor,
        const tnsr::i<DataVector, 3, Frame::Inertial>&
            lower_spatial_four_velocity,

        const Scalar<DataVector>& lapse,
        const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
        const tnsr::i<DataVector, 3, Frame::Inertial>& d_lapse,
        const tnsr::iJ<DataVector, 3, Frame::Inertial>& d_shift,
        const tnsr::iJJ<DataVector, 3, Frame::Inertial>& d_inv_spatial_metric,
        const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
        const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
        const Scalar<DataVector>& determinant_spatial_metric,
        const Scalar<DataVector>& cell_light_crossing_time,

        const Mesh<3>& mesh,
        const tnsr::I<DataVector, 3, Frame::ElementLogical>& mesh_coordinates,
        const size_t num_ghost_zones,
        const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
            mesh_velocity,
        const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                              Frame::Inertial>&
            inverse_jacobian_logical_to_inertial,
        const Scalar<DataVector>& det_jacobian_logical_to_inertial,
        const Jacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>&
            inertial_to_fluid_jacobian,
        const InverseJacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>&
            inertial_to_fluid_inverse_jacobian,

        const DirectionalIdMap<3, std::optional<DataVector>>&
            electron_fraction_ghost,
        const DirectionalIdMap<3, std::optional<DataVector>>&
            rest_mass_density_ghost,
        const DirectionalIdMap<3, std::optional<DataVector>>&
            temperature_ghost,
        const DirectionalIdMap<3, std::optional<DataVector>>&
            cell_light_crossing_time_ghost) {
  // Minimum  temperature for use of the NuLib table
  // (Could be an option).
  const double minimum_temperature_table = 0.5;
  const double time_step = target_end_time - start_time;

  // Calculate volume elements. Proper volume is needed
  // to calculate the total emission in a cell over the
  // given time step. The 3-volume will be needed to
  // calculate the coupling to the fluid.
  Scalar<DataVector> cell_proper_four_volume =
      make_with_value<Scalar<DataVector>>(lapse, 0.0);
  Scalar<DataVector> cell_inertial_three_volume =
      make_with_value<Scalar<DataVector>>(lapse, 0.0);
  cell_proper_four_volume_finite_difference(
      &cell_proper_four_volume, lapse, determinant_spatial_metric, time_step,
      mesh, det_jacobian_logical_to_inertial);
  cell_inertial_coordinate_three_volume_finite_difference(
      &cell_inertial_three_volume, mesh, det_jacobian_logical_to_inertial);

  Index<3> extents_with_ghost_zone = mesh.extents();
  size_t mesh_size_with_ghost_zones = 1;
  for (size_t d = 0; d < 3; d++) {
    extents_with_ghost_zone[d] += 2 * num_ghost_zones;
    mesh_size_with_ghost_zones *= extents_with_ghost_zone[d];
  }

  // Get interaction rates
  std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>
      emissivity_in_cell;
  std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>
      absorption_opacity;
  std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>
      scattering_opacity;
  std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>
      frac_ka_to_ks;
  DataVector zero_dv_ghost_zones(mesh_size_with_ghost_zones, 0.0);
  for (size_t ns = 0; ns < NeutrinoSpecies; ns++) {
    for (size_t ng = 0; ng < EnergyBins; ng++) {
      gsl::at(gsl::at(emissivity_in_cell, ns), ng) = zero_dv_ghost_zones;
      gsl::at(gsl::at(absorption_opacity, ns), ng) = zero_dv_ghost_zones;
      gsl::at(gsl::at(scattering_opacity, ns), ng) = zero_dv_ghost_zones;
      gsl::at(gsl::at(frac_ka_to_ks, ns), ng) = zero_dv_ghost_zones;
    }
  }

  // Join live points and ghost zones for fluid variables
  // We initialize cell_light_crossing_time_with_ghost to time_step
  // to avoid division by zero on outer boundaries
  Scalar<DataVector> rest_mass_density_with_ghost =
      make_with_value<Scalar<DataVector>>(zero_dv_ghost_zones, 0.0);
  Scalar<DataVector> electron_fraction_with_ghost =
      make_with_value<Scalar<DataVector>>(zero_dv_ghost_zones, 0.0);
  Scalar<DataVector> temperature_with_ghost =
      make_with_value<Scalar<DataVector>>(zero_dv_ghost_zones, 0.0);
  Scalar<DataVector> cell_light_crossing_time_with_ghost =
      make_with_value<Scalar<DataVector>>(zero_dv_ghost_zones, time_step);
  combine_ghost_data(&get(rest_mass_density_with_ghost), mesh, num_ghost_zones,
                     get(rest_mass_density), rest_mass_density_ghost);
  combine_ghost_data(&get(electron_fraction_with_ghost), mesh, num_ghost_zones,
                     get(electron_fraction), electron_fraction_ghost);
  combine_ghost_data(&get(temperature_with_ghost), mesh, num_ghost_zones,
                     get(temperature), temperature_ghost);
  combine_ghost_data(&get(cell_light_crossing_time_with_ghost), mesh,
                     num_ghost_zones, get(cell_light_crossing_time),
                     cell_light_crossing_time_ghost);

  this->implicit_monte_carlo_interaction_rates(
      &emissivity_in_cell, &absorption_opacity, &scattering_opacity,
      &frac_ka_to_ks, cell_light_crossing_time_with_ghost,
      electron_fraction_with_ghost, rest_mass_density_with_ghost,
      temperature_with_ghost, minimum_temperature_table,
      interaction_table, equation_of_state);
  const std::array<double, EnergyBins>& energy_at_bin_center =
      interaction_table.get_neutrino_energies();

  // Bookkeeping tensors for coupling to fluid
  // These also include ghost zones.
  Scalar<DataVector> coupling_tilde_tau =
      make_with_value<Scalar<DataVector>>(zero_dv_ghost_zones, 0.0);
  Scalar<DataVector> coupling_rho_ye =
      make_with_value<Scalar<DataVector>>(zero_dv_ghost_zones, 0.0);
  tnsr::i<DataVector, 3, Frame::Inertial> coupling_tilde_s =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
          zero_dv_ghost_zones, 0.0);

  // Emit new MC packets
  this->emit_packets(
      packets, random_number_generator, &coupling_tilde_tau, &coupling_tilde_s,
      &coupling_rho_ye, start_time, time_step, mesh, num_ghost_zones,
      emissivity_in_cell, *single_packet_energy, energy_at_bin_center,
      lorentz_factor, lower_spatial_four_velocity, inertial_to_fluid_jacobian,
      inertial_to_fluid_inverse_jacobian, cell_proper_four_volume);

  // Propagate packets
  evolve_packets(
      packets, random_number_generator,
      &coupling_tilde_tau, &coupling_tilde_s, &coupling_rho_ye,
      target_end_time, mesh, mesh_coordinates, num_ghost_zones,
      absorption_opacity, scattering_opacity, energy_at_bin_center,
      lorentz_factor, lower_spatial_four_velocity, lapse, shift, d_lapse,
      d_shift, d_inv_spatial_metric, spatial_metric, inv_spatial_metric,
      cell_light_crossing_time,
      mesh_velocity, inverse_jacobian_logical_to_inertial,
      inertial_to_fluid_jacobian, inertial_to_fluid_inverse_jacobian);

  // Couple to hydro needs to be done after communicating coupling terms
  // between ghost zones... so likely outside of this function. Or at
  // least we need to correct for GZ information somehow later.

  // Determine new energy of packets and resample as needed.
}

}  // namespace Particles::MonteCarlo
