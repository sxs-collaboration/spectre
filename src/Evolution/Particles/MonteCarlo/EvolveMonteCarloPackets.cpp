// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "EvolveMonteCarloPackets.hpp"

#include "Evolution/Particles/MonteCarlo/MonteCarloPacket.hpp"

namespace Particles::MonteCarlo {

namespace detail {

// Time derivative of p_i in a packet, according to geodesic equation
void time_derivative_momentum_geodesic(
    gsl::not_null<std::array<double, 3>*> dt_momentum, const Packet& packet,
    const Scalar<DataVector>& lapse,
    const tnsr::i<DataVector, 3, Frame::Inertial>& d_lapse,
    const tnsr::iJ<DataVector, 3, Frame::Inertial>& d_shift,
    const tnsr::iJJ<DataVector, 3, Frame::Inertial>& d_inv_spatial_metric,
    const size_t& closest_point_index) {
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
                          Frame::Inertial>& inverse_jacobian,
    const size_t& closest_point_index) {
  // Temporary variables
  std::array<double, 3> dpdt{0, 0, 0};
  std::array<double, 3> dxdt_inertial{0, 0, 0};
  std::array<double, 3> dxdt_logical{0, 0, 0};
  std::array<double, 3> p0{0, 0, 0};

  // Calculate p^t from normalization of 4-momentum
  packet->renormalize_momentum(inv_spatial_metric, lapse, closest_point_index);

  // Calculate time derivative of 3-momentum at beginning of time step
  detail::time_derivative_momentum_geodesic(&dpdt, *packet, lapse, d_lapse,
                                            d_shift, d_inv_spatial_metric,
                                            closest_point_index);
  // Take half-step (momentum only, as time derivative is independent of
  // position)
  for (size_t i = 0; i < 3; i++) {
    // Store momentum at beginning of step
    gsl::at(p0, i) = packet->momentum[i];
    packet->momentum[i] += gsl::at(dpdt, i) * time_step * 0.5;
  }
  // Calculate p^t from normalization of 4-momentum
  packet->renormalize_momentum(inv_spatial_metric, lapse, closest_point_index);

  // Calculate time derivative of 3-momentum and position at half-step
  detail::time_derivative_momentum_geodesic(&dpdt, *packet, lapse, d_lapse,
                                            d_shift, d_inv_spatial_metric,
                                            closest_point_index);
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

}  // namespace detail

void evolve_packets(
    gsl::not_null<std::vector<Packet>*> packets, const double& time_step,
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

  // Loop over packets
  for (size_t p = 0; p < packets->size(); p++) {
    Packet& packet = (*packets)[p];
    // Find closest grid point to packet
    std::array<size_t, 3> closest_point_index_3d{0, 0, 0};
    for (size_t d = 0; d < 3; d++) {
      gsl::at(closest_point_index_3d, d) =
          std::floor((packet.coordinates[d] - gsl::at(bottom_coord_mesh, d)) /
                         gsl::at(dx_mesh, d) +
                     0.5);
    }
    const size_t closest_point_index =
        closest_point_index_3d[0] +
        extents[0] * (closest_point_index_3d[1] +
                      extents[1] * closest_point_index_3d[2]);
    detail::evolve_single_packet_on_geodesic(
        &packet, time_step, lapse, shift, d_lapse, d_shift,
        d_inv_spatial_metric, inv_spatial_metric, mesh_velocity,
        inverse_jacobian, closest_point_index);
  }
}

}  // namespace Particles::MonteCarlo
