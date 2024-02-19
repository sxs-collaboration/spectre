// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "MonteCarloPacket.hpp"

namespace Particles::MonteCarlo {

void Packet::renormalize_momentum(
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& lapse, const size_t& closest_point_index) {
  momentum_upper_t = 0.0;
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      momentum_upper_t += inv_spatial_metric.get(i, j)[closest_point_index] *
                          momentum[i] * momentum[j];
    }
  }
  momentum_upper_t = sqrt(momentum_upper_t) / get(lapse)[closest_point_index];
}

}  // namespace Particles::MonteCarlo
