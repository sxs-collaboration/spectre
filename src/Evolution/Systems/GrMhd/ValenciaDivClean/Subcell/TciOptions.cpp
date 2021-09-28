// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TciOptions.hpp"

#include <pup.h>

#include "Parallel/PupStlCpp17.hpp"

namespace grmhd::ValenciaDivClean::subcell {
void TciOptions::pup(PUP::er& p) {
  p | minimum_rest_mass_density_times_lorentz_factor;
  p | minimum_tilde_tau;
  p | atmosphere_density;
  p | safety_factor_for_magnetic_field;
  p | magnetic_field_cutoff;
}
}  // namespace grmhd::ValenciaDivClean::subcell
