// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Particles/MonteCarlo/Packet.hpp"

namespace Particles::MonteCarlo {

void Packet::renormalize_momentum(
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& lapse) {
  momentum_upper_t = 0.0;
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      momentum_upper_t +=
          inv_spatial_metric.get(i, j)[index_of_closest_grid_point] *
          momentum[i] * momentum[j];
    }
  }
  momentum_upper_t =
      sqrt(momentum_upper_t) / get(lapse)[index_of_closest_grid_point];
}

double compute_fluid_frame_energy(
    const Packet& packet, const Scalar<DataVector>& lorentz_factor,
    const tnsr::i<DataVector, 3, Frame::Inertial>& lower_spatial_four_velocity,
    const Scalar<DataVector>& lapse,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric) {
  const size_t idx = packet.index_of_closest_grid_point;
  double fluid_frame_energy =
      get(lorentz_factor)[idx] * get(lapse)[idx] * packet.momentum_upper_t;
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      fluid_frame_energy -= inv_spatial_metric.get(i, j)[idx] *
                            lower_spatial_four_velocity.get(i)[idx] *
                            packet.momentum[j];
    }
  }
  return fluid_frame_energy;
}

}  // namespace Particles::MonteCarlo
