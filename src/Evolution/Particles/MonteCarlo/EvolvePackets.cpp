// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Particles/MonteCarlo/EvolvePackets.hpp"

#include "Evolution/Particles/MonteCarlo/Packet.hpp"

namespace Particles::MonteCarlo {

namespace detail {

// Time derivative of p_i in a packet, according to geodesic equation
void time_derivative_momentum_geodesic(
    gsl::not_null<std::array<double, 3>*> dt_momentum, const Packet& packet,
    const Scalar<DataVector>& lapse,
    const tnsr::i<DataVector, 3, Frame::Inertial>& d_lapse,
    const tnsr::iJ<DataVector, 3, Frame::Inertial>& d_shift,
    const tnsr::iJJ<DataVector, 3, Frame::Inertial>& d_inv_spatial_metric) {
  const size_t& closest_point_index = packet.index_of_closest_grid_point;
  for (size_t i = 0; i < 3; i++) {
    gsl::at(*dt_momentum, i) = (-1.0) * d_lapse.get(i)[closest_point_index] *
                               get(lapse)[closest_point_index] *
                               packet.momentum_upper_t;
    for (size_t j = 0; j < 3; j++) {
      gsl::at(*dt_momentum, i) +=
          d_shift.get(i, j)[closest_point_index] * packet.momentum[j];
      for (size_t k = 0; k < 3; k++) {
        gsl::at(*dt_momentum, i) -=
            0.5 * d_inv_spatial_metric.get(i, j, k)[closest_point_index] *
            packet.momentum[j] * packet.momentum[k] / packet.momentum_upper_t;
      }
    }
  }
}

void evolve_single_packet_on_geodesic(
    gsl::not_null<Packet*> packet, const double& time_step,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::i<DataVector, 3, Frame::Inertial>& d_lapse,
    const tnsr::iJ<DataVector, 3, Frame::Inertial>& d_shift,
    const tnsr::iJJ<DataVector, 3, Frame::Inertial>& d_inv_spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>& mesh_velocity,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian) {
  const size_t& closest_point_index = packet->index_of_closest_grid_point;
  // Temporary variables
  std::array<double, 3> dpdt{0, 0, 0};
  std::array<double, 3> dxdt_inertial{0, 0, 0};
  std::array<double, 3> dxdt_logical{0, 0, 0};
  std::array<double, 3> p0{0, 0, 0};

  // Calculate p^t from normalization of 4-momentum
  packet->renormalize_momentum(inv_spatial_metric, lapse);

  // Calculate time derivative of 3-momentum at beginning of time step
  detail::time_derivative_momentum_geodesic(&dpdt, *packet, lapse, d_lapse,
                                            d_shift, d_inv_spatial_metric);
  // Take half-step (momentum only, as time derivative is independent of
  // position)
  for (size_t i = 0; i < 3; i++) {
    // Store momentum at beginning of step
    gsl::at(p0, i) = packet->momentum[i];
    packet->momentum[i] += gsl::at(dpdt, i) * time_step * 0.5;
  }
  // Calculate p^t from normalization of 4-momentum
  packet->renormalize_momentum(inv_spatial_metric, lapse);

  // Calculate time derivative of 3-momentum and position at half-step
  detail::time_derivative_momentum_geodesic(&dpdt, *packet, lapse, d_lapse,
                                            d_shift, d_inv_spatial_metric);
  for (size_t i = 0; i < 3; i++) {
    gsl::at(dxdt_inertial, i) = (-1.0) * shift.get(i)[closest_point_index];
    if (mesh_velocity.has_value()) {
      gsl::at(dxdt_inertial, i) -=
          mesh_velocity.value().get(i)[closest_point_index];
    }
    for (size_t j = 0; j < 3; j++) {
      gsl::at(dxdt_inertial, i) +=
          inv_spatial_metric.get(i, j)[closest_point_index] *
          packet->momentum[j] / packet->momentum_upper_t;
    }
  }
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      gsl::at(dxdt_logical, i) +=
          gsl::at(dxdt_inertial, j) *
          inverse_jacobian.get(i, j)[closest_point_index];
    }
  }
  // Take full time step
  for (size_t i = 0; i < 3; i++) {
    packet->momentum[i] = gsl::at(p0, i) + gsl::at(dpdt, i) * time_step;
    packet->coordinates[i] += gsl::at(dxdt_logical, i) * time_step;
  }
  packet->time += time_step;
}

// Functions to be implemented to complete implementation of Monte-Carlo
// time step
double compute_fluid_frame_energy(const Packet& /*packet*/) { return -1.0; }
void compute_opacities(const gsl::not_null<double*> absorption_opacity,
                       const gsl::not_null<double*> scattering_opacity,
                       const double& /*fluid_frame_energy*/) {
  *absorption_opacity = 0.0;
  *scattering_opacity = 0.0;
}
void scatter_packet(const gsl::not_null<Packet*> /*packet*/) {
  // To be implemented
}
void diffuse_packet(const gsl::not_null<Packet*> /*packet*/,
                    const double& /*time_step*/) {
  // To be implemented
}

}  // namespace detail

void evolve_packets(
    gsl::not_null<std::vector<Packet>*> packets, const double& final_time,
    const Mesh<3>& mesh,
    const tnsr::I<DataVector, 3, Frame::ElementLogical>& mesh_coordinates,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::i<DataVector, 3, Frame::Inertial>& d_lapse,
    const tnsr::iJ<DataVector, 3, Frame::Inertial>& d_shift,
    const tnsr::iJJ<DataVector, 3, Frame::Inertial>& d_inv_spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>& mesh_velocity,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian) {
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
      // Get fluid frame energy of neutrinos in packet, then retrieve
      // opacities
      fluid_frame_energy = detail::compute_fluid_frame_energy(packet);
      detail::compute_opacities(&absorption_opacity, &scattering_opacity,
                                fluid_frame_energy);

      // Determine time to next events
      // TO DO: Implement when including opacities!
      // Currently we set all timescales > dt_end_step,
      // which guarantees that we only propagate packets
      // along geodesics.
      dt_min = dt_end_step;
      dt_cell_check = 10.0 * dt_end_step;
      dt_min = std::min(dt_cell_check, dt_min);
      dt_absorption = 10.0 * dt_end_step;
      dt_min = std::min(dt_absorption, dt_min);
      dt_scattering = 10.0 * dt_end_step;
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
      detail::evolve_single_packet_on_geodesic(
          &packet, dt_min, lapse, shift, d_lapse, d_shift, d_inv_spatial_metric,
          inv_spatial_metric, mesh_velocity, inverse_jacobian);
      // If the next event was a scatter, perform that scatter and
      // continue evolution
      if (dt_min == dt_scattering) {
        detail::scatter_packet(&packet);
        dt_end_step -= dt_min;
        dt_absorption -= dt_min;
        dt_cell_check -= dt_min;
        dt_min = dt_end_step;
        dt_min = std::min(dt_cell_check, dt_min);
        dt_min = std::min(dt_absorption, dt_min);
        // Correct for relativistic effects when using opacities
        const double scattering_optical_depth = dt_min * scattering_opacity;
        // High optical depth: use approximate diffusion method to move packet
        // The scatterig depth of 3.0 was found to be sufficient for diffusion
        // to be accurate (see Foucart 2018, 10.1093/mnras/sty108)
        if (scattering_optical_depth > 3.0) {
          detail::diffuse_packet(&packet, dt_min);
        }
      }

      // Find closest grid point to packet at current time
      std::array<size_t, 3> closest_point_index_3d{0, 0, 0};
      for (size_t d = 0; d < 3; d++) {
        gsl::at(closest_point_index_3d, d) =
            std::floor((packet.coordinates[d] - gsl::at(bottom_coord_mesh, d)) /
                           gsl::at(dx_mesh, d) +
                       0.5);
      }
      packet.index_of_closest_grid_point =
          closest_point_index_3d[0] +
          extents[0] * (closest_point_index_3d[1] +
                        extents[1] * closest_point_index_3d[2]);
      // Update time to end of step
      dt_end_step = final_time - packet.time;
    }
  }
}

}  // namespace Particles::MonteCarlo
