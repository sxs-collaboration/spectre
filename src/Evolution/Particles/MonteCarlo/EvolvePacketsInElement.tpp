// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Particles/MonteCarlo/TemplatedLocalFunctions.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Particles/MonteCarlo/CouplingTermsForPropagation.hpp"
#include "Evolution/Particles/MonteCarlo/EvolvePackets.hpp"
#include "Evolution/Particles/MonteCarlo/Packet.hpp"
#include "Evolution/Particles/MonteCarlo/Scattering.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

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
    const gsl::not_null<Scalar<DataVector>*> coupling_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        coupling_tilde_s,
    const gsl::not_null<Scalar<DataVector>*> coupling_rho_ye,
    const double final_time, const Mesh<3>& mesh,
    const tnsr::I<DataVector, 3, Frame::ElementLogical>& mesh_coordinates,
    const size_t num_ghost_zones,
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
    const Scalar<DataVector>& cell_light_crossing_time,
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

  // Mesh information.
  const Index<3>& extents = mesh.extents();
  const std::array<double, 3> bottom_coord_mesh{mesh_coordinates.get(0)[0],
                                                mesh_coordinates.get(1)[0],
                                                mesh_coordinates.get(2)[0]};
  const std::array<size_t, 3> step{1, extents[0], extents[0] * extents[1]};
  const std::array<size_t, 3> step_with_ghost_zones{1,
    extents[0] + 2 * num_ghost_zones,
    (extents[0] + 2 * num_ghost_zones) * (extents[1] + 2 * num_ghost_zones)};
  const std::array<double, 3> dx_mesh{
      mesh_coordinates.get(0)[step[0]] - bottom_coord_mesh[0],
      mesh_coordinates.get(1)[step[1]] - bottom_coord_mesh[1],
      mesh_coordinates.get(2)[step[2]] - bottom_coord_mesh[2]};

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
  size_t n_packets = packets->size();
  for (size_t p = 0; p < n_packets; p++) {
    Packet& packet = (*packets)[p];

    initial_time = packet.time;
    dt_end_step = final_time - initial_time;

    // Get quantities that we do NOT update if the packet
    // changes cell.
    // local_idx is the index on the mesh without ghost zones
    // extended_idx is the index on the mesh with ghost zones
    // We only update extended_idx during a time step, as we
    // only recompute the opacities during a step (and they
    // live on the extended grid). We also need the extended
    // index for coupling to the fluid.
    const size_t& local_idx = packet.index_of_closest_grid_point;
    Index<3> index_3d{0,0,0};
    size_t extended_idx = local_idx;
    for(size_t d=0; d<3 ; d++){
      index_3d[d] = extended_idx % extents[d];
      extended_idx = (extended_idx - index_3d[d]) / extents[d];
      index_3d[d] += num_ghost_zones;
    }
    extended_idx = index_3d[0] +
      (extents[0] + 2 * num_ghost_zones) *
        ( index_3d[1] +
          ( extents[1] + 2 * num_ghost_zones) *
            index_3d[2] );
    // Bookkeeping variable to know whether opacities should be
    // recomputed.
    size_t previous_extended_idx = extended_idx;

    const double& lapse_packet = get(lapse)[local_idx];
    const double& lorentz_factor_packet = get(lorentz_factor)[local_idx];
    const std::array<double, 3> lower_spatial_four_velocity_packet = {
        lower_spatial_four_velocity.get(0)[local_idx],
        lower_spatial_four_velocity.get(1)[local_idx],
        lower_spatial_four_velocity.get(2)[local_idx]};

    // Estimate light-crossing time in the cell.
    const double& cell_light_crossing_time_packet =
      get(cell_light_crossing_time)[local_idx];

    // Get fluid frame energy of neutrinos in packet, then retrieve
    // opacities at current points and neighboring points. We do
    // not have interactions that modify the fluid frame energy
    // of the packets so far, so we precompute it.
    packet.renormalize_momentum(inv_spatial_metric, lapse);
    fluid_frame_energy = compute_fluid_frame_energy(
        packet, lorentz_factor, lower_spatial_four_velocity, lapse,
        inv_spatial_metric);

    // Find maximum total opacity in current cell and
    // neighbors, to limit time step in high opacity regions
    // Need to properly deal with ghost zones.
    // Opacities calculated at the current location of the packet
    this->interpolate_opacities_at_fluid_energy(
        &absorption_opacity, &scattering_opacity, fluid_frame_energy,
        packet.species, extended_idx, absorption_opacity_table,
        scattering_opacity_table, energy_at_bin_center);
    double max_opacity = absorption_opacity + scattering_opacity;
    for (size_t d = 0; d < 3; d++) {
      double ka_neighbor = 0.0;
      double ks_neighbor = 0.0;
      this->interpolate_opacities_at_fluid_energy(
          &ka_neighbor, &ks_neighbor, fluid_frame_energy, packet.species,
          extended_idx - step_with_ghost_zones[d],
          absorption_opacity_table, scattering_opacity_table,
          energy_at_bin_center);
      max_opacity = std::max(max_opacity, ka_neighbor + ks_neighbor);
      this->interpolate_opacities_at_fluid_energy(
          &ka_neighbor, &ks_neighbor, fluid_frame_energy, packet.species,
          extended_idx + step_with_ghost_zones[d],
          absorption_opacity_table, scattering_opacity_table,
          energy_at_bin_center);
      max_opacity = std::max(max_opacity, ka_neighbor + ks_neighbor);
    }
    // Minimum fraction of light crossing time used for time step.
    // This is a compromise between taking very small steps in high
    // opacity regions close to cell boundaries, and minimizing
    // computational costs.
    const double fmin = std::max(
        0.03, 0.1 / (max_opacity * cell_light_crossing_time_packet
                     + opacity_floor));

    // We evolve until at least 95 percent of the desired step.
    // We don't require the full step because diffusion in the fluid
    // frame leads to unpredictable time steps in the inertial frame,
    // and we want to avoid taking a lot of potentially small steps
    // when reaching the end of the desired step.
    while (dt_end_step > 0.05 * (final_time - initial_time)) {

      // Shortest distance to a grid boundary, in units of the grid spacing.
      // Note that this does not use the extents, so it is safe to use in the
      // ghost zones... if we allow negative indices.
      double frac_min_grid = 1.0;
      {
        std::array<int, 3> closest_point_index_3d{0, 0, 0};
        for (size_t d = 0; d < 3; d++) {
          gsl::at(closest_point_index_3d, d) = std::floor(
              (packet.coordinates[d] - gsl::at(bottom_coord_mesh, d)) /
                  gsl::at(dx_mesh, d) +
              0.5);
          double frac_grid =
              (packet.coordinates[d] - gsl::at(bottom_coord_mesh, d) -
               gsl::at(closest_point_index_3d, d) * gsl::at(dx_mesh, d)) /
                  gsl::at(dx_mesh, d) +
              0.5;
          frac_min_grid = std::min(frac_min_grid, frac_grid + fmin);
          frac_min_grid = std::min(frac_min_grid, 1.0 - frac_grid + fmin);
        }
      }

      // Recompute opacities if needed
      if(previous_extended_idx != extended_idx){
        this->interpolate_opacities_at_fluid_energy(
          &absorption_opacity, &scattering_opacity, fluid_frame_energy,
          packet.species, extended_idx, absorption_opacity_table,
          scattering_opacity_table, energy_at_bin_center);
      }

      // Determine time to next events
      dt_min = dt_end_step;
      // Limit time step close to cell boundary in high-opacity regions
      dt_cell_check = frac_min_grid * cell_light_crossing_time_packet;
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

      // Propagation to the next event, whatever it is
      evolve_single_packet_on_geodesic(&packet, dt_min, lapse, shift, d_lapse,
                                       d_shift, d_inv_spatial_metric,
                                       inv_spatial_metric, mesh_velocity,
                                       inverse_jacobian_logical_to_inertial);
      AddCouplingTermsForPropagation(
          coupling_tilde_tau, coupling_tilde_s, coupling_rho_ye, packet,
          extended_idx, dt_min,
          absorption_opacity, scattering_opacity, fluid_frame_energy,
          lapse_packet, lorentz_factor_packet,
          lower_spatial_four_velocity_packet);

      // If absorption is the first event, we just delete
      // the packet.
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
        n_packets--;
        break;
      }
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
                                         lapse_packet / lorentz_factor_packet;
        // High optical depth: use approximate diffusion method to move packet
        // The scatterig depth of 3.0 was found to be sufficient for diffusion
        // to be accurate (see Foucart 2018, 10.1093/mnras/sty108)
        if (scattering_optical_depth > 3.0) {
          diffuse_packet(
              &packet, random_number_generator, &fluid_frame_energy,
              coupling_tilde_tau, coupling_tilde_s, coupling_rho_ye,
              extended_idx, dt_min,
              diffusion_params, absorption_opacity, scattering_opacity,
              lorentz_factor, lower_spatial_four_velocity, lapse, shift,
              d_lapse, d_shift, d_inv_spatial_metric, spatial_metric,
              inv_spatial_metric, mesh_velocity,
              inverse_jacobian_logical_to_inertial, inertial_to_fluid_jacobian,
              inertial_to_fluid_inverse_jacobian, prefactor_diffusion_time_step,
              prefactor_diffusion_four_velocity,
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
            AddCouplingTermsForPropagation(
                coupling_tilde_tau, coupling_tilde_s, coupling_rho_ye, packet,
                extended_idx, dt_min, absorption_opacity, scattering_opacity,
                fluid_frame_energy, lapse_packet, lorentz_factor_packet,
                lower_spatial_four_velocity_packet);
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
          n_packets--;
          break;
        }
      }

      // Index of the new cells including ghost zones.
      std::array<size_t, 3> closest_point_index_3d{0, 0, 0};
      for (size_t d = 0; d < 3; d++) {
        gsl::at(closest_point_index_3d, d) =
            num_ghost_zones +
            std::floor((packet.coordinates[d] - gsl::at(bottom_coord_mesh, d)) /
                           gsl::at(dx_mesh, d) +
                       0.5);
      }
      previous_extended_idx = extended_idx;
      extended_idx =
          closest_point_index_3d[0] +
          (extents[0] + 2 * num_ghost_zones) * (closest_point_index_3d[1] +
            (extents[1] + 2 * num_ghost_zones) * closest_point_index_3d[2]);

      // Update time to end of step
      dt_end_step = final_time - packet.time;
    }

    // Find closest grid point to packet at current time, using
    // extents for live points only.
    std::array<size_t, 3> closest_point_index_3d{0, 0, 0};
    bool packet_out_of_bounds = false;
    for (size_t d = 0; d < 3; d++) {
      if(packet.coordinates[d]<-1.0 || packet.coordinates[d]>1.0){
        packet_out_of_bounds = true;
        break;
      }
      gsl::at(closest_point_index_3d, d) =
          std::floor((packet.coordinates[d] - gsl::at(bottom_coord_mesh, d)) /
                         gsl::at(dx_mesh, d) +
                     0.5);
    }
    // Update index of packet at the end of a step.
    // Note that we mark out of bounds packets with an out of bound
    // index.
    if(packet_out_of_bounds){
      packet.index_of_closest_grid_point = mesh.number_of_grid_points();
    } else{
      packet.index_of_closest_grid_point =
          closest_point_index_3d[0] +
          extents[0] * (closest_point_index_3d[1] +
                        extents[1] * closest_point_index_3d[2]);
    }
  }
}

}  // namespace Particles::MonteCarlo
