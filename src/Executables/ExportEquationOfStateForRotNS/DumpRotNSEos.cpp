// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Executables/ExportEquationOfStateForRotNS/DumpRotNSEos.hpp"

#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <string>

#include "Parallel/Printf/Printf.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/FileSystem.hpp"

namespace ExportEosForRotNS {

void dump_eos(const EquationsOfState::EquationOfState<true, 3>& eos,
              const size_t number_of_log10_number_density_points_for_dump,
              const std::string& output_file_name,
              const double lower_bound_rest_mass_density_cgs,
              const double upper_bound_rest_mass_density_cgs) {
  using std::log10;
  using std::pow;
  // Baryon mass, used to go from number density to rest mass
  // density. I.e. `rho_cgs = n_cgs * baryon_mass`, where `n_gcs` is the number
  // density in CGS units. This is the baryon mass that RotNS uses. This
  // might be different from the baryon mass that the EoS uses.
  //
  // https://github.com/sxs-collaboration/spectre/issues/4694
  const double baryon_mass_of_rotns_cgs =
      hydro::units::geometric::default_baryon_mass *
      hydro::units::cgs::mass_unit;
  const double log10_lower_bound_number_density_cgs =
      log10(lower_bound_rest_mass_density_cgs / baryon_mass_of_rotns_cgs);
  const double log10_upper_bound_number_density_cgs =
      log10(upper_bound_rest_mass_density_cgs / baryon_mass_of_rotns_cgs);
  const double delta_log_number_density_cgs =
      (log10_upper_bound_number_density_cgs -
       log10_lower_bound_number_density_cgs) /
      static_cast<double>(number_of_log10_number_density_points_for_dump - 1);

  if (file_system::check_if_file_exists(output_file_name)) {
    ERROR_NO_TRACE("File " << output_file_name
                           << " already exists. Refusing to overwrite.");
  }
  std::ofstream outfile(output_file_name.c_str());

  if (not eos.is_barotropic()) {
    Parallel::printf(
        "Warning: the equation of state is not barotropic but no "
        "thermodynamic profile was provided. The temperature defaults to "
        "the EoS lower bound.\n");
  }

  const double temperature_lower_bound =
      std::max({eos.temperature_lower_bound(), 1.e-100});

  for (size_t log10_number_density_index = 0;
       log10_number_density_index <
       number_of_log10_number_density_points_for_dump;
       ++log10_number_density_index) {
    const double number_density_cgs =
        pow(10.0, log10_lower_bound_number_density_cgs +
                      (static_cast<double>(log10_number_density_index) *
                       delta_log_number_density_cgs));

    // Note: we will want to add the baryon mass to our EOS interface.
    //
    // https://github.com/sxs-collaboration/spectre/issues/4694
    const Scalar<double> rest_mass_density_geometric{
        number_density_cgs * cube(hydro::units::cgs::length_unit) *
        eos.baryon_mass()};
    const Scalar<double> temperature = make_with_value<Scalar<double>>(
        rest_mass_density_geometric, temperature_lower_bound);

    const Scalar<double> electron_fraction =
        eos.equilibrium_electron_fraction_from_density_temperature(
            rest_mass_density_geometric, temperature);

    const Scalar<double> specific_internal_energy_geometric =
        eos.specific_internal_energy_from_density_and_temperature(
            rest_mass_density_geometric, temperature, electron_fraction);

    const Scalar<double> total_energy_density_geometric{
        get(rest_mass_density_geometric) *
        (1. + get(specific_internal_energy_geometric))};

    // Note: the energy density is divided by c^2
    const double total_energy_density_cgs =
        get(total_energy_density_geometric) *
        hydro::units::cgs::rest_mass_density_unit;

    // should be dyne cm^(-3)
    const double pressure_cgs =
        get(eos.pressure_from_density_and_temperature(
            rest_mass_density_geometric, temperature, electron_fraction)) *
        hydro::units::cgs::pressure_unit;

    outfile << std::scientific << std::setw(24) << std::setprecision(14)
            << log10(number_density_cgs) << std::setw(24)
            << std::setprecision(14) << log10(total_energy_density_cgs)
            << std::setw(24) << std::setprecision(14) << get(electron_fraction)
            << std::setw(24) << std::setprecision(14) << log10(pressure_cgs)
            << "\n";
  }
  outfile.close();
}

}  // namespace ExportEosForRotNS
