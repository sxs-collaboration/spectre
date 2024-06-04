// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Particles/MonteCarlo/TemplatedLocalFunctions.hpp"

#include <cmath>

#include "Evolution/Particles/MonteCarlo/EmitPackets.tpp"
#include "Evolution/Particles/MonteCarlo/EvolvePacketsInElement.tpp"
#include "Evolution/Particles/MonteCarlo/Packet.hpp"
#include "Evolution/Particles/MonteCarlo/TakeTimeStep.tpp"
#include "Utilities/Gsl.hpp"

namespace Particles::MonteCarlo::detail {
// Draw a single packet with homogeneouse spatial distribution
// in [0,1]x[0,1]x[0,1], energy of 1, isotropic momentum
// distribution, and time in [0,1].
void draw_single_packet(
    const gsl::not_null<double*> time,
    const gsl::not_null<std::array<double, 3>*> coord,
    const gsl::not_null<std::array<double, 3>*> momentum,
    const gsl::not_null<std::mt19937*> random_number_generator) {
  std::uniform_real_distribution<double> rng_uniform_zero_to_one(0.0, 1.0);

  *time = rng_uniform_zero_to_one(*random_number_generator);
  for (size_t i = 0; i < 3; i++) {
    gsl::at(*coord, i) = rng_uniform_zero_to_one(*random_number_generator);
  }
  const double cos_theta =
      -1.0 + 2.0 * rng_uniform_zero_to_one(*random_number_generator);
  const double phi =
      2.0 * M_PI * rng_uniform_zero_to_one(*random_number_generator);
  const double sin_theta = sqrt(1.0 - cos_theta * cos_theta);
  gsl::at(*momentum, 0) = sin_theta * cos(phi);
  gsl::at(*momentum, 1) = sin_theta * sin(phi);
  gsl::at(*momentum, 2) = cos_theta;
}
}  // namespace Particles::MonteCarlo::detail

template struct Particles::MonteCarlo::TemplatedLocalFunctions<2, 2>;
template struct Particles::MonteCarlo::TemplatedLocalFunctions<2, 3>;
template struct Particles::MonteCarlo::TemplatedLocalFunctions<4, 3>;
template struct Particles::MonteCarlo::TemplatedLocalFunctions<16, 3>;
