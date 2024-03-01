// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Particles/MonteCarlo/EvolvePackets.hpp"
#include "Evolution/Particles/MonteCarlo/Packet.hpp"
#include "Evolution/Particles/MonteCarlo/Scattering.hpp"
#include "Evolution/Particles/MonteCarlo/TemplatedLocalFunctions.hpp"

namespace Particles::MonteCarlo {

template <size_t EnergyBins, size_t NeutrinoSpecies>
void TemplatedLocalFunctions<EnergyBins, NeutrinoSpecies>::
    interpolate_opacities_at_fluid_energy(
        gsl::not_null<double*> absorption_opacity_packet,
        gsl::not_null<double*> scattering_opacity_packet,
        const double fluid_frame_energy, const size_t species,
        const size_t index,
        const std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>&
            absorption_opacity_table,
        const std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>&
            scattering_opacity_table,
        const std::array<double, EnergyBins>& energy_at_bin_center) {
  const auto upper_bracket =
      std::lower_bound(energy_at_bin_center.begin(), energy_at_bin_center.end(),
                       fluid_frame_energy);
  if (upper_bracket == energy_at_bin_center.begin()) {
    *absorption_opacity_packet =
        gsl::at(gsl::at(absorption_opacity_table, species), 0)[index];
    *scattering_opacity_packet =
        gsl::at(gsl::at(scattering_opacity_table, species), 0)[index];
  } else {
    if (upper_bracket == energy_at_bin_center.end()) {
      *absorption_opacity_packet = gsl::at(
          gsl::at(absorption_opacity_table, species), EnergyBins - 1)[index];
      *scattering_opacity_packet = gsl::at(
          gsl::at(scattering_opacity_table, species), EnergyBins - 1)[index];
    } else {
      const auto lower_bracket = std::prev(upper_bracket);
      const size_t lower_index = static_cast<size_t>(
          std::distance(energy_at_bin_center.begin(), lower_bracket));
      const size_t upper_index = lower_index + 1;
      const double inter_coef =
          (fluid_frame_energy - gsl::at(energy_at_bin_center, lower_index)) /
          (gsl::at(energy_at_bin_center, upper_index) -
           gsl::at(energy_at_bin_center, lower_index));
      *absorption_opacity_packet =
          log(std::max(gsl::at(gsl::at(absorption_opacity_table, species),
                               upper_index)[index],
                       opacity_floor)) *
              inter_coef +
          log(std::max(gsl::at(gsl::at(absorption_opacity_table, species),
                               lower_index)[index],
                       opacity_floor)) *
              (1.0 - inter_coef);
      *scattering_opacity_packet =
          log(std::max(gsl::at(gsl::at(scattering_opacity_table, species),
                               upper_index)[index],
                       opacity_floor)) *
              inter_coef +
          log(std::max(gsl::at(gsl::at(scattering_opacity_table, species),
                               lower_index)[index],
                       opacity_floor)) *
              (1.0 - inter_coef);
      *absorption_opacity_packet = exp(*absorption_opacity_packet);
      *scattering_opacity_packet = exp(*scattering_opacity_packet);
    }
  }
  *absorption_opacity_packet =
      std::max(*absorption_opacity_packet, opacity_floor);
  *scattering_opacity_packet =
      std::max(*scattering_opacity_packet, opacity_floor);
}

template <size_t EnergyBins, size_t NeutrinoSpecies>
void TemplatedLocalFunctions<EnergyBins, NeutrinoSpecies>::evolve_packets(
    const gsl::not_null<std::vector<Packet>*> packets,
    const gsl::not_null<std::mt19937*> random_number_generator,
    const double final_time, const Mesh<3>& mesh,
    const tnsr::I<DataVector, 3, Frame::ElementLogical>& mesh_coordinates,
    const std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>&
        absorption_opacity_table,
    const std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>&
        scattering_opacity_table,
    const std::array<double, EnergyBins>& energy_at_bin_center,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::i<DataVector, 3, Frame::Inertial>& lower_spatial_four_velocity,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::i<DataVector, 3, Frame::Inertial>& d_lapse,
    const tnsr::iJ<DataVector, 3, Frame::Inertial>& d_shift,
    const tnsr::iJJ<DataVector, 3, Frame::Inertial>& d_inv_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>& mesh_velocity,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>&
        inverse_jacobian_logical_to_inertial,
    const Jacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>&
        inertial_to_fluid_jacobian,
    const InverseJacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>&
        inertial_to_fluid_inverse_jacobian) {
  // RNG from uniform distribution in [eps=1.e-100,1[
  // Eps used to avoid log(0).
  std::uniform_real_distribution<double> rng_uniform_eps_to_one(1.e-100, 1.0);

  // Struct used for diffusion pre-computations
  const DiffusionMonteCarloParameters diffusion_params;
  auto prefactor_diffusion_time_step = make_with_value<DataVector>(lapse, 0.0);
  auto prefactor_diffusion_four_velocity =
      make_with_value<DataVector>(lapse, 0.0);
  auto prefactor_diffusion_time_vector =
      make_with_value<DataVector>(lapse, 0.0);

  DiffusionPrecomputeForElement(
      &prefactor_diffusion_time_vector, &prefactor_diffusion_four_velocity,
      &prefactor_diffusion_time_step, lorentz_factor,
      lower_spatial_four_velocity, lapse, shift, spatial_metric);

  // Mesh information. Currently assumes uniform grid without map
  const Index<3>& extents = mesh.extents();
  const std::array<double, 3> bottom_coord_mesh{mesh_coordinates.get(0)[0],
                                                mesh_coordinates.get(1)[0],
                                                mesh_coordinates.get(2)[0]};
  const std::array<double, 3> dx_mesh{
      mesh_coordinates.get(0)[1] - bottom_coord_mesh[0],
      mesh_coordinates.get(1)[extents[0]] - bottom_coord_mesh[1],
      mesh_coordinates.get(2)[extents[0] * extents[1]] - bottom_coord_mesh[2]};

  // Temporary variables keeping track of opacities and times to next events
  double fluid_frame_energy = -1.0;
  double absorption_opacity = 0.0;
  double scattering_opacity = 0.0;
  double dt_end_step = -1.0;
  double dt_cell_check = -1.0;
  double dt_absorption = -1.0;
  double dt_scattering = -1.0;
  double dt_min = -1.0;
  double initial_time = -1.0;
  // Loop over packets
  const size_t n_packets = packets->size();
  for (size_t p = 0; p < n_packets; p++) {
    Packet& packet = (*packets)[p];

    initial_time = packet.time;
    dt_end_step = final_time - initial_time;

    // We evolve until at least 95 percent of the desired step.
    // We don't require the full step because diffusion in the fluid
    // frame leads to unpredictable time steps in the inertial frame,
    // and we want to avoid taking a lot of potentially small steps
    // when reaching the end of the desired step.
    while (dt_end_step > 0.05 * (final_time - initial_time)) {
      const size_t& idx = packet.index_of_closest_grid_point;
      // Get fluid frame energy of neutrinos in packet, then retrieve
      // opacities
      packet.renormalize_momentum(inv_spatial_metric, lapse);
      fluid_frame_energy = compute_fluid_frame_energy(
          packet, lorentz_factor, lower_spatial_four_velocity, lapse,
          inv_spatial_metric);
      this->interpolate_opacities_at_fluid_energy(
          &absorption_opacity, &scattering_opacity, fluid_frame_energy,
          packet.species, idx, absorption_opacity_table,
          scattering_opacity_table, energy_at_bin_center);

      // Determine time to next events
      // TO DO: Implement cell check with ghost zone methods
      dt_min = dt_end_step;
      dt_cell_check = 10.0 * dt_end_step;
      dt_min = std::min(dt_cell_check, dt_min);
      // Time step to next absorption is
      // -ln(r)/K_a*p^t/nu
      dt_absorption =
          absorption_opacity > opacity_floor
              ? -log(rng_uniform_eps_to_one(*random_number_generator)) /
                    (absorption_opacity)*packet.momentum_upper_t /
                    fluid_frame_energy
              : 10.0 * dt_end_step;
      dt_min = std::min(dt_absorption, dt_min);
      // Time step to next scattering is
      // -ln(r)/K_s*p^t/nu
      dt_scattering =
          scattering_opacity > opacity_floor
              ? -log(rng_uniform_eps_to_one(*random_number_generator)) /
                    (scattering_opacity)*packet.momentum_upper_t /
                    fluid_frame_energy
              : 10.0 * dt_end_step;
      dt_min = std::min(dt_scattering, dt_min);

      // If absorption is the first event, we just delete
      // the packet (no need to propagate)
      //
      // To remove this packet we swap the current and last packet,
      // then pop the last packet from the end of the vector of packets.
      // We then decrease the counter `p` so that we check the packet
      // from the end that we just swapped into the current slot.
      //
      // Note: This works fine even if p==0 since unsigned ints (size_t)
      // wrap at zero.
      if (dt_min == dt_absorption) {
        std::swap((*packets)[p], (*packets)[n_packets - 1]);
        packets->pop_back();
        p--;
        break;
      }
      // Propagation to the next event, whatever it is
      evolve_single_packet_on_geodesic(&packet, dt_min, lapse, shift, d_lapse,
                                       d_shift, d_inv_spatial_metric,
                                       inv_spatial_metric, mesh_velocity,
                                       inverse_jacobian_logical_to_inertial);
      // If the next event was a scatter, perform that scatter and
      // continue evolution
      if (dt_min == dt_scattering) {
        // Next event is a scatter. Calculate the time step to the next
        // non-scattering event, and the scattering optical depth over
        // that period.
        dt_end_step -= dt_min;
        dt_absorption -= dt_min;
        dt_cell_check -= dt_min;
        dt_min = dt_end_step;
        dt_min = std::min(dt_cell_check, dt_min);
        dt_min = std::min(dt_absorption, dt_min);
        const double scattering_optical_depth = dt_min * scattering_opacity *
                                                get(lapse)[idx] /
                                                get(lorentz_factor)[idx];
        // High optical depth: use approximate diffusion method to move packet
        // The scatterig depth of 3.0 was found to be sufficient for diffusion
        // to be accurate (see Foucart 2018, 10.1093/mnras/sty108)
        if (scattering_optical_depth > 3.0) {
          diffuse_packet(
              &packet, random_number_generator, &fluid_frame_energy, dt_min,
              diffusion_params, scattering_opacity, lorentz_factor,
              lower_spatial_four_velocity, lapse, shift, d_lapse, d_shift,
              d_inv_spatial_metric, spatial_metric, inv_spatial_metric,
              mesh_velocity, inverse_jacobian_logical_to_inertial,
              inertial_to_fluid_jacobian, inertial_to_fluid_inverse_jacobian,
              prefactor_diffusion_time_step, prefactor_diffusion_four_velocity,
              prefactor_diffusion_time_vector);
        } else {
          // Low optical depth; perform scatterings one by one.
          do {
            scatter_packet(&packet, random_number_generator, fluid_frame_energy,
                           inertial_to_fluid_jacobian,
                           inertial_to_fluid_inverse_jacobian);
            // Time to next scattering event.
            dt_scattering =
                scattering_opacity > opacity_floor
                    ? -log(rng_uniform_eps_to_one(*random_number_generator)) /
                          (scattering_opacity)*packet.momentum_upper_t /
                          fluid_frame_energy
                    : 10.0 * dt_end_step;
            dt_min = std::min(dt_scattering, dt_min);
            // Propagation to the next event, whatever it is
            evolve_single_packet_on_geodesic(
                &packet, dt_min, lapse, shift, d_lapse, d_shift,
                d_inv_spatial_metric, inv_spatial_metric, mesh_velocity,
                inverse_jacobian_logical_to_inertial);
            dt_end_step -= dt_min;
            dt_absorption -= dt_min;
            dt_cell_check -= dt_min;
            dt_min = dt_end_step;
            dt_min = std::min(dt_cell_check, dt_min);
            dt_min = std::min(dt_absorption, dt_min);
          } while (dt_min > 0.0);
        }
        // If absorption is the next event; delete the packet
        if (dt_min == dt_absorption) {
          (*packets)[p] = (*packets)[n_packets - 1];
          packets->pop_back();
          p--;
          break;
        }
      }
      // TO DO: Deal with update to opacities to neighboring
      // points when we can handle ghost zones.

      // Update time to end of step
      dt_end_step = final_time - packet.time;
    }

    // Find closest grid point to packet at current time
    // TO DO: Handle ghost zones
    std::array<size_t, 3> closest_point_index_3d{0, 0, 0};
    for (size_t d = 0; d < 3; d++) {
      gsl::at(closest_point_index_3d, d) =
          std::floor((packet.coordinates[d] - gsl::at(bottom_coord_mesh, d)) /
                         gsl::at(dx_mesh, d) +
                     0.5);
    }
    // In SpEC, we update a packet index after a full time step;
    // only the opacities are updated mid-step. Decide whether to do
    // the same here once we handle ghost zone (the main reason not to
    // update is to limit the amount of ghost zone information we
    // store.
    packet.index_of_closest_grid_point =
        closest_point_index_3d[0] +
        extents[0] * (closest_point_index_3d[1] +
                      extents[1] * closest_point_index_3d[2]);
  }
}

}  // namespace Particles::MonteCarlo
