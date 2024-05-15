// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Particles/MonteCarlo/TemplatedLocalFunctions.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Particles/MonteCarlo/CellVolume.hpp"
#include "Evolution/Particles/MonteCarlo/NeutrinoInteractionTable.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

namespace Particles::MonteCarlo {

template <size_t EnergyBins, size_t NeutrinoSpecies>
void TemplatedLocalFunctions<EnergyBins, NeutrinoSpecies>::
    take_time_step_on_element(
        const gsl::not_null<std::vector<Packet>*> packets,
        const gsl::not_null<std::mt19937*> random_number_generator,
        const gsl::not_null<std::array<DataVector, NeutrinoSpecies>*>
            single_packet_energy,

        const double start_time, const double target_end_time,
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

        const Mesh<3>& mesh,
        const tnsr::I<DataVector, 3, Frame::ElementLogical>& mesh_coordinates,
        const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coordinates,
        const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
            mesh_velocity,
        const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                              Frame::Inertial>&
            inverse_jacobian_logical_to_inertial,
        const Scalar<DataVector>& det_jacobian_logical_to_inertial,
        const Jacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>&
            inertial_to_fluid_jacobian,
        const InverseJacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>&
            inertial_to_fluid_inverse_jacobian) {
  // Minimum  temperature for use of the NuLib table
  // (Could be an option).
  const double minimum_temperature_table = 0.5;
  const double time_step = target_end_time - start_time;

  // Calculate volume elements
  Scalar<DataVector> cell_proper_four_volume =
      make_with_value<Scalar<DataVector>>(lapse, 0.0);
  Scalar<DataVector> cell_inertial_three_volume =
      make_with_value<Scalar<DataVector>>(lapse, 0.0);
  cell_proper_four_volume_finite_difference(&cell_proper_four_volume,
     lapse, determinant_spatial_metric, time_step,
     mesh, det_jacobian_logical_to_inertial);
  cell_inertial_coordinate_three_volume_finite_difference(
     &cell_inertial_three_volume, mesh,
     det_jacobian_logical_to_inertial);

  // Get interaction rates
  // Replace with high-opacity corrections after merging 5977
  std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>
      emission_in_cell;
  std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>
      absorption_opacity;
  std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>
      scattering_opacity;
  DataVector zero_dv(mesh.number_of_grid_points(), 0.0);
  for (size_t ns = 0; ns < NeutrinoSpecies; ns++) {
    for (size_t ng = 0; ng < EnergyBins; ng++) {
      gsl::at(gsl::at(emission_in_cell, ns), ng) = zero_dv;
      gsl::at(gsl::at(absorption_opacity, ns), ng) = zero_dv;
      gsl::at(gsl::at(scattering_opacity, ns), ng) = zero_dv;
    }
  }
  interaction_table.get_neutrino_matter_interactions(
   &emission_in_cell, &absorption_opacity, &scattering_opacity,
   electron_fraction, rest_mass_density, temperature,
   cell_proper_four_volume, minimum_temperature_table);
  const std::array<double, EnergyBins>& energy_at_bin_center
    = interaction_table.get_neutrino_energies();

  // Bookkeeping tensors for coupling to fluid
  Scalar<DataVector> coupling_tilde_tau
    = make_with_value< Scalar<DataVector> >(lapse, 0.0);
  Scalar<DataVector> coupling_rho_ye
    = make_with_value< Scalar<DataVector> >(lapse, 0.0);
  tnsr::i<DataVector, 3, Frame::Inertial> coupling_tilde_s
    = make_with_value< tnsr::i<DataVector, 3, Frame::Inertial> >(lapse, 0.0);

  // Emit new MC packets
  this->emit_packets(
    packets, random_number_generator,
    &coupling_tilde_tau, &coupling_tilde_s, &coupling_rho_ye,
    start_time, time_step, mesh, emission_in_cell,
    *single_packet_energy, energy_at_bin_center,
    lorentz_factor, lower_spatial_four_velocity,
    inertial_to_fluid_jacobian, inertial_to_fluid_inverse_jacobian);

  // Propagate packets
  evolve_packets(
      packets, random_number_generator,
      &coupling_tilde_tau, &coupling_tilde_s, &coupling_rho_ye,
      target_end_time, mesh, mesh_coordinates, inertial_coordinates,
      absorption_opacity, scattering_opacity, energy_at_bin_center,
      lorentz_factor, lower_spatial_four_velocity, lapse, shift, d_lapse,
      d_shift, d_inv_spatial_metric, spatial_metric, inv_spatial_metric,
      mesh_velocity, inverse_jacobian_logical_to_inertial,
      inertial_to_fluid_jacobian, inertial_to_fluid_inverse_jacobian);

  // Couple to hydro

  // Determine new energy of packets and resample as needed.
}

}  // namespace Particles::MonteCarlo
