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

}  // namespace Particles::MonteCarlo
