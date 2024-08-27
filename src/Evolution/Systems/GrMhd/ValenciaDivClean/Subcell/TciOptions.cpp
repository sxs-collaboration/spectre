// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TciOptions.hpp"

#include <pup.h>

#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace grmhd::ValenciaDivClean::subcell {
TciOptions::TciOptions() = default;
TciOptions::TciOptions(
    const double minimum_rest_mass_density_times_lorentz_factor_in,
    const double minimum_ye_in, const double minimum_tilde_tau_in,
    const double atmosphere_density_in,
    const double safety_factor_for_magnetic_field_in,
    const std::optional<double> magnetic_field_cutoff_in)
    : minimum_rest_mass_density_times_lorentz_factor(
          minimum_rest_mass_density_times_lorentz_factor_in),
      minimum_ye(minimum_ye_in),
      minimum_tilde_tau(minimum_tilde_tau_in),
      atmosphere_density(atmosphere_density_in),
      safety_factor_for_magnetic_field(safety_factor_for_magnetic_field_in),
      magnetic_field_cutoff(magnetic_field_cutoff_in) {}

void TciOptions::pup(PUP::er& p) {
  p | minimum_rest_mass_density_times_lorentz_factor;
  p | minimum_ye;
  p | minimum_tilde_tau;
  p | atmosphere_density;
  p | safety_factor_for_magnetic_field;
  p | magnetic_field_cutoff;
}
}  // namespace grmhd::ValenciaDivClean::subcell
