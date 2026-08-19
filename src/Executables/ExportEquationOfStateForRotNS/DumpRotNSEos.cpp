// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Executables/ExportEquationOfStateForRotNS/DumpRotNSEos.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/Interpolation/PolynomialInterpolation.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"

namespace ExportEosForRotNS {
void interpolate_profile_at_log_density(
    const gsl::not_null<double*> result, const double target_log_density,
    const DataVector& log_rest_mass_density_interp,
    const DataVector& values_interp,
    const double max_variation_for_linear_fallback) {
  constexpr size_t stencil_size = 4;
  const size_t num_density_points = log_rest_mass_density_interp.size();

  // If target_log_density is outside [log_rho[0], log_rho[n-1]], density_index
  // stays at 0 (below) or points past the last stencil start (above). The
  // stencil clamp below keeps the interpolation in-range in both cases.
  // dump_eos asserts the dump range is within the table before calling here,
  // so out-of-range inputs should not occur in normal use.
  size_t density_index = 0;
  for (size_t j = 0; j < num_density_points; ++j) {
    if (log_rest_mass_density_interp[j] > target_log_density) {
      density_index = j - 1;
      break;
    }
  }

  const size_t density_stencil_index = static_cast<size_t>(std::clamp(
      static_cast<int>(density_index) - (static_cast<int>(stencil_size) / 2), 0,
      static_cast<int>(num_density_points - stencil_size)));

  const auto log_density_stencil = gsl::make_span(
      &log_rest_mass_density_interp[density_stencil_index], stencil_size);
  const auto values_stencil =
      gsl::make_span(&values_interp[density_stencil_index], stencil_size);

  double error_y = 0.0;
  const auto [min_iter, max_iter] = alg::minmax_element(values_stencil);
  if (*max_iter - *min_iter > max_variation_for_linear_fallback) {
    std::array<double, 2> log_density_linear{
        {std::numeric_limits<double>::signaling_NaN(),
         std::numeric_limits<double>::signaling_NaN()}};
    std::array<double, 2> values_linear{
        {std::numeric_limits<double>::signaling_NaN(),
         std::numeric_limits<double>::signaling_NaN()}};
    for (size_t k = 0; k < stencil_size - 1; ++k) {
      if (log_density_stencil[k] <= target_log_density and
          target_log_density <= log_density_stencil[k + 1]) {
        log_density_linear[0] = log_density_stencil[k];
        log_density_linear[1] = log_density_stencil[k + 1];
        values_linear[0] = gsl::at(values_stencil, k);
        values_linear[1] = gsl::at(values_stencil, k + 1);
        break;
      }
    }
    intrp::polynomial_interpolation<1>(
        result, make_not_null(&error_y), target_log_density,
        gsl::make_span(values_linear.data(), values_linear.size()),
        gsl::make_span(log_density_linear.data(), log_density_linear.size()));
  } else {
    intrp::polynomial_interpolation<stencil_size - 1>(
        result, make_not_null(&error_y), target_log_density, values_stencil,
        log_density_stencil);
  }
}

void dump_eos(
    const EquationsOfState::EquationOfState<true, 3>& eos,
    const size_t number_of_log10_number_density_points_for_dump,
    const std::string& output_file_name,
    const double lower_bound_rest_mass_density_cgs,
    const double upper_bound_rest_mass_density_cgs,
    const std::optional<std::string>& thermodynamic_profile_filename) {
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

  if (eos.is_barotropic()) {
    if (thermodynamic_profile_filename.has_value()) {
      Parallel::printf(
          "Warning: the equation of state is barotropic, so the supplied "
          "thermodynamic profile will be ignored.\n");
    }
  } else {
    if (not thermodynamic_profile_filename.has_value()) {
      Parallel::printf(
          "Warning: the equation of state is not barotropic but no "
          "thermodynamic profile was provided. The temperature defaults to "
          "the EoS lower bound.\n");
    }
  }

  const double temperature_lower_bound =
      std::max({eos.temperature_lower_bound(), 1.e-100});
  const double temperature_upper_bound = eos.temperature_upper_bound();

  DataVector log_rest_mass_density_interp{};
  DataVector log_temperature_interp{};
  DataVector electron_fraction_interp{};
  bool has_ye_profile = false;
  if (thermodynamic_profile_filename.has_value() and not eos.is_barotropic()) {
    // The file stores raw rho, T, and optionally Y_e values. We convert rho
    // and T to log10 on read and interpolate in log10 space for better
    // accuracy across the many orders of magnitude they span. Y_e is
    // interpolated in linear space.
    if (not file_system::check_if_file_exists(
            thermodynamic_profile_filename.value())) {
      ERROR("Cannot open file " << thermodynamic_profile_filename.value()
                                << ".\n");
    }
    std::ifstream profile_file(thermodynamic_profile_filename.value());

    size_t num_density_points = 0;
    int has_ye_profile_int = 0;
    profile_file >> num_density_points >> has_ye_profile_int;
    has_ye_profile = (has_ye_profile_int != 0);

    log_rest_mass_density_interp = DataVector(num_density_points);
    log_temperature_interp = DataVector(num_density_points);
    if (has_ye_profile) {
      electron_fraction_interp = DataVector(num_density_points);
    }
    for (size_t i = 0; i < num_density_points; i++) {
      double raw_density{};
      double raw_temperature{};
      profile_file >> raw_density >> raw_temperature;
      // Restrict temperature to EoS bounds before taking log
      log_rest_mass_density_interp[i] = log10(raw_density);
      log_temperature_interp[i] = log10(std::clamp(
          raw_temperature, temperature_lower_bound, temperature_upper_bound));
      if (has_ye_profile) {
        profile_file >> electron_fraction_interp[i];
        electron_fraction_interp[i] = std::clamp(
            electron_fraction_interp[i], eos.electron_fraction_lower_bound(),
            eos.electron_fraction_upper_bound());
      }
    }

    if (has_ye_profile and eos.is_equilibrium()) {
      Parallel::printf(
          "Note: the equation of state is in beta-equilibrium, so the "
          "supplied Y_e(rho) profile will be ignored.\n");
    }
    if (not has_ye_profile and not eos.is_equilibrium()) {
      Parallel::printf(
          "Note: no Y_e(rho) profile was provided; assuming "
          "beta-equilibrium for electron fraction.\n");
    }

    const double log10_n_to_rho =
        log10(cube(hydro::units::cgs::length_unit) * eos.baryon_mass());
    const double log_rho_dump_min =
        log10_lower_bound_number_density_cgs + log10_n_to_rho;
    const double log_rho_dump_max =
        log10_upper_bound_number_density_cgs + log10_n_to_rho;
    const double log_rho_table_min = log_rest_mass_density_interp[0];
    const double log_rho_table_max =
        log_rest_mass_density_interp[log_rest_mass_density_interp.size() - 1];
    if (log_rho_dump_min < log_rho_table_min or
        log_rho_dump_max > log_rho_table_max) {
      ERROR("The requested rest mass density range ["
            << pow(10.0, log_rho_dump_min) << ", "
            << pow(10.0, log_rho_dump_max)
            << "] (geometric units) extends outside the thermal profile table "
               "range ["
            << pow(10.0, log_rho_table_min) << ", "
            << pow(10.0, log_rho_table_max)
            << "] (geometric units). Extend the thermal profile table or "
               "narrow the "
               "requested density range.");
    }
  }

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
    Scalar<double> temperature = make_with_value<Scalar<double>>(
        rest_mass_density_geometric, temperature_lower_bound);

    if (thermodynamic_profile_filename.has_value() and
        not eos.is_barotropic()) {
      double log_temperature_result =
          std::numeric_limits<double>::signaling_NaN();
      interpolate_profile_at_log_density(
          make_not_null(&log_temperature_result),
          log10(get(rest_mass_density_geometric)), log_rest_mass_density_interp,
          log_temperature_interp, 2.0);
      get(temperature) = pow(10.0, log_temperature_result);
    }

    Scalar<double> electron_fraction{};
    if (has_ye_profile and not eos.is_equilibrium()) {
      double ye_result = std::numeric_limits<double>::signaling_NaN();
      interpolate_profile_at_log_density(
          make_not_null(&ye_result), log10(get(rest_mass_density_geometric)),
          log_rest_mass_density_interp, electron_fraction_interp);
      electron_fraction = Scalar<double>{ye_result};
    } else {
      electron_fraction =
          eos.equilibrium_electron_fraction_from_density_temperature(
              rest_mass_density_geometric, temperature);
    }

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
