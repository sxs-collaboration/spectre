// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Particles/MonteCarlo/EvolvePackets.hpp"
#include "Evolution/Particles/MonteCarlo/Packet.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace {

void check_packet() {
  const double epsilon_approx = 1.e-15;

  const size_t dv_size = 5;
  Scalar<DataVector> lapse(dv_size, 1.0);
  Scalar<DataVector> lorentz_factor(dv_size, 1.0);
  tnsr::i<DataVector, 3, Frame::Inertial> lower_spatial_four_velocity =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  tnsr::II<DataVector, 3, Frame::Inertial> inv_spatial_metric =
      make_with_value<tnsr::II<DataVector, 3, Frame::Inertial>>(lapse, 0.0);

  const size_t pos = 3;
  Particles::MonteCarlo::Packet packet(1, 2.0, pos, 0.0, -1.0, -0.5, -0.2, 0.9,
                                       -0.3, 0.2, 0.1);
  // The values below are wildly inconsistent, but we only want to check the
  // math
  get(lapse)[pos] = 0.8;
  get(lorentz_factor)[pos] = 1.3;
  lower_spatial_four_velocity.get(0)[pos] = 0.3;
  lower_spatial_four_velocity.get(1)[pos] = 0.4;
  lower_spatial_four_velocity.get(2)[pos] = 0.5;
  inv_spatial_metric.get(0, 0) = 1.1;
  inv_spatial_metric.get(1, 1) = 1.2;
  inv_spatial_metric.get(2, 2) = 1.3;
  inv_spatial_metric.get(0, 1) = 0.1;
  inv_spatial_metric.get(0, 2) = 0.2;
  inv_spatial_metric.get(1, 2) = 0.3;

  // p^t = \sqrt{\gamma^{ij} p_i p_j}/\alpha
  const double expected_momentum_upper_t =
      sqrt(1.1 * 0.3 * 0.3 + 1.2 * 0.2 * 0.2 + 1.3 * 0.1 * 0.1 +
           2.0 * 0.1 * (-0.3) * 0.2 + 2.0 * 0.2 * (-0.3) * 0.1 +
           2.0 * 0.3 * 0.2 * 0.1) /
      0.8;
  packet.renormalize_momentum(inv_spatial_metric, lapse);
  CHECK(fabs(packet.momentum_upper_t - expected_momentum_upper_t) <
        epsilon_approx);

  // E = W \alpha p^t - \gamma^{ij} u_i p_j
  const double expected_energy =
      0.8 * 1.3 * expected_momentum_upper_t - 1.1 * (-0.3) * 0.3 -
      1.2 * 0.2 * 0.4 - 1.3 * 0.1 * 0.5 - 0.1 * (-0.3) * 0.4 -
      0.1 * (0.2) * (0.3) - 0.2 * (-0.3) * 0.5 - 0.2 * 0.1 * 0.3 -
      0.3 * 0.2 * 0.5 - 0.3 * 0.1 * 0.4;
  const double computed_energy =
      Particles::MonteCarlo::compute_fluid_frame_energy(
          packet, lorentz_factor, lower_spatial_four_velocity, lapse,
          inv_spatial_metric);
  CHECK(fabs(expected_energy - computed_energy) < epsilon_approx);

  CHECK(packet.species == 1);
  CHECK(packet.coordinates.get(0) == -1.0);
  CHECK(packet.coordinates.get(1) == -0.5);
  CHECK(packet.coordinates.get(2) == -0.2);
  CHECK(packet.momentum.get(0) == -0.3);
  CHECK(packet.momentum.get(1) == 0.2);
  CHECK(packet.momentum.get(2) == 0.1);
  CHECK(packet.time == 0.0);
  CHECK(packet.index_of_closest_grid_point == pos);
  CHECK(packet.number_of_neutrinos == 2.0);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Particles.MonteCarloPacket",
                  "[Unit][Evolution]") {
  check_packet();
}
