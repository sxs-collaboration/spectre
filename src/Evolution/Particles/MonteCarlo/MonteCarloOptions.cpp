// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Particles/MonteCarlo/MonteCarloOptions.hpp"

#include <cstddef>
#include <pup.h>
#include <pup_stl.h>

namespace Particles::MonteCarlo {

template <size_t NeutrinoSpecies>
PUP::able::PUP_ID MonteCarloOptions<NeutrinoSpecies>::my_PUP_ID = 0;  // NOLINT

template <size_t NeutrinoSpecies>
void MonteCarloOptions<NeutrinoSpecies>::pup(PUP::er& p) {
  PUP::able::pup(p);
  p | initial_packet_energy_;
}

}  // namespace Particles::MonteCarlo

template class Particles::MonteCarlo::MonteCarloOptions<2>;
template class Particles::MonteCarlo::MonteCarloOptions<3>;
