// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Particles/MonteCarlo/Scattering.hpp"

#include "Evolution/Particles/MonteCarlo/Packet.hpp"

namespace Particles::MonteCarlo {

void scatter_packet(
    const gsl::not_null<Packet*> packet,
    const gsl::not_null<std::mt19937*> random_number_generator,
    const double& fluid_frame_energy,
    const Jacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>&
        inertial_to_fluid_jacobian,
    const InverseJacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>&
        inertial_to_fluid_inverse_jacobian) {
  const size_t& idx = packet->index_of_closest_grid_point;
  std::uniform_real_distribution<double> rng_uniform_zero_to_one(0.0, 1.0);
  const double cos_theta =
      2.0 * rng_uniform_zero_to_one(*random_number_generator) - 1.0;
  const double phi =
      2.0 * M_PI * rng_uniform_zero_to_one(*random_number_generator);
  const double sin_theta = sqrt(1.0 - cos_theta * cos_theta);
  std::array<double, 4> four_momentum_fluid{
    fluid_frame_energy,
    sin_theta * cos(phi) * fluid_frame_energy,
    sin_theta * sin(phi) * fluid_frame_energy,
    cos_theta * fluid_frame_energy};

  packet->momentum_upper_t = 0.0;
  for (size_t d = 0; d < 4; d++) {
    packet->momentum_upper_t +=
        inertial_to_fluid_inverse_jacobian.get(0, d)[idx] *
        gsl::at(four_momentum_fluid, d);
  }
  for (size_t d = 0; d < 3; d++) {
    // Multiply by -1.0 because p_t = -p^t in an orthonormal frame
    packet->momentum.get(d) = -gsl::at(four_momentum_fluid, 0) *
                              inertial_to_fluid_jacobian.get(d + 1, 0)[idx];
    for (size_t dd = 0; dd < 3; dd++) {
      packet->momentum.get(d) +=
          gsl::at(four_momentum_fluid, dd + 1) *
          inertial_to_fluid_jacobian.get(d + 1, dd + 1)[idx];
    }
  }
}

void diffuse_packet(const gsl::not_null<Packet*> /*packet*/,
                    const double& /*time_step*/) {
  // To be implemented
}


}  // namespace Particles::MonteCarlo
