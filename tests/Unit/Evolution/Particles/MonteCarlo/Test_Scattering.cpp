// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Particles/MonteCarlo/Packet.hpp"
#include "Evolution/Particles/MonteCarlo/Scattering.hpp"
#include "Framework/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Particles.MonteCarloScattering",
                  "[Unit][Evolution]") {
  MAKE_GENERATOR(generator);

  DataVector zero_dv(8);
  zero_dv = 0.0;
  InverseJacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>
      inverse_jacobian = make_with_value<
          InverseJacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>>(
          zero_dv, 0.0);
  Jacobian<DataVector, 4, Frame::Inertial, Frame::Fluid> jacobian =
      make_with_value<Jacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>>(
          zero_dv, 0.0);

  // Set Jacobian to a Lorentz boost in the y-direction with W=2
  const double v_boost = sqrt(3) / 2.0;
  const double W_boost = 2.0;

  jacobian.get(0, 0) = W_boost;
  jacobian.get(0, 2) = -W_boost * v_boost;
  jacobian.get(1, 1) = 1.0;
  jacobian.get(2, 0) = -W_boost * v_boost;
  jacobian.get(2, 2) = W_boost;
  jacobian.get(3, 3) = 1.0;
  inverse_jacobian.get(0, 0) = W_boost;
  inverse_jacobian.get(0, 2) = W_boost * v_boost;
  inverse_jacobian.get(1, 1) = 1.0;
  inverse_jacobian.get(2, 0) = W_boost * v_boost;
  inverse_jacobian.get(2, 2) = W_boost;
  inverse_jacobian.get(3, 3) = 1.0;

  Particles::MonteCarlo::Packet packet(0, 1.0, 0, 0.0, -1.0, -1.0, -1.0, 1.0,
                                       1.0, 0.0, 0.0);

  auto boosted_frame_energy = [&W_boost, &v_boost, &packet]() {
    return W_boost * (packet.momentum_upper_t - packet.momentum[1] * v_boost);
  };

  const double initial_fluid_frame_energy = boosted_frame_energy();
  for (int step = 0; step < 5; step++) {
    scatter_packet(&packet, &generator,
                   initial_fluid_frame_energy, jacobian, inverse_jacobian);
    const double current_fluid_frame_energy = boosted_frame_energy();
    CHECK(fabs(initial_fluid_frame_energy - current_fluid_frame_energy) <
          1.e-14);
  }
}
