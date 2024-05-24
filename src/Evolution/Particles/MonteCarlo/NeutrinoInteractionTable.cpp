// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Particles/MonteCarlo/NeutrinoInteractionTable.hpp"

#include <cmath>
#include <cstddef>
#include <hdf5.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/CheckH5.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/Helpers.hpp"
#include "IO/H5/OpenGroup.hpp"
#include "IO/H5/Wrappers.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

using hydro::units::cgs::length_unit;
using hydro::units::cgs::mass_unit;
using hydro::units::cgs::rest_mass_density_unit;
using hydro::units::cgs::time_unit;

namespace Particles::MonteCarlo {

namespace {
const double emissivity_NuLib_to_code =
    length_unit * cube(time_unit) / mass_unit * 4.0 * M_PI;
}  // namespace

template <size_t EnergyBins, size_t NeutrinoSpecies>
NeutrinoInteractionTable<EnergyBins, NeutrinoSpecies>::NeutrinoInteractionTable(
    std::vector<double> table_data_,
    const std::array<double, EnergyBins>& table_neutrino_energies_,
    std::vector<double> table_log_density_,
    std::vector<double> table_log_temperature_,
    std::vector<double> table_electron_fraction_)
    : table_data(std::move(table_data_)),
      table_neutrino_energies(table_neutrino_energies_),
      table_log_density(std::move(table_log_density_)),
      table_log_temperature(std::move(table_log_temperature_)),
      table_electron_fraction(std::move(table_electron_fraction_)) {
  initialize_interpolator();
}

template <size_t EnergyBins, size_t NeutrinoSpecies>
NeutrinoInteractionTable<EnergyBins, NeutrinoSpecies>::NeutrinoInteractionTable(
    const std::string& filename) {
  const hid_t file_id =
      H5Fopen(filename.c_str(), H5F_ACC_RDONLY, h5::h5p_default());

  const size_t n_points_density = static_cast<size_t>(
      h5::read_data<1, std::vector<int> >(file_id, "nrho")[0]);
  const size_t n_points_temperature = static_cast<size_t>(
      h5::read_data<1, std::vector<int> >(file_id, "ntemp")[0]);
  const size_t n_points_electron_fraction = static_cast<size_t>(
      h5::read_data<1, std::vector<int> >(file_id, "nye")[0]);
  const size_t n_energy_groups = static_cast<size_t>(
      h5::read_data<1, std::vector<int> >(file_id, "number_groups")[0]);
  const size_t n_neutrino_species = static_cast<size_t>(
      h5::read_data<1, std::vector<int> >(file_id, "number_species")[0]);

  if (n_energy_groups != EnergyBins) {
    ERROR("Table has " << n_energy_groups
                       << " energy bins, while the executable assumed "
                       << EnergyBins);
  }
  if (n_neutrino_species != NeutrinoSpecies) {
    ERROR("Table has " << n_neutrino_species
                       << " neutrino species, while the executable assumed "
                       << NeutrinoSpecies);
  }

  table_log_density =
      h5::read_data<1, std::vector<double> >(file_id, "rho_points");
  if (table_log_density.size() != n_points_density) {
    ERROR("Table has inconsistent number of points in density. Expected "
          << n_points_density << " but got " << table_log_density.size());
  }
  for (size_t i = 0; i < n_points_density; i++) {
    table_log_density[i] = log(table_log_density[i] / rest_mass_density_unit);
  }

  table_log_temperature =
      h5::read_data<1, std::vector<double> >(file_id, "temp_points");
  if (table_log_temperature.size() != n_points_temperature) {
    ERROR("Table has inconsistent number of points in temperature. Expected "
          << n_points_temperature << " but got "
          << table_log_temperature.size());
  }
  for (size_t i = 0; i < n_points_temperature; i++) {
    table_log_temperature[i] = log(table_log_temperature[i]);
  }

  table_electron_fraction =
      h5::read_data<1, std::vector<double> >(file_id, "ye_points");
  if (table_electron_fraction.size() != n_points_electron_fraction) {
    ERROR(
        "Table has inconsistent number of points in electron fraction. "
        "Expected "
        << n_points_electron_fraction << " but got "
        << table_electron_fraction.size());
  }

  const std::vector<double> energy_bin_lower_bound =
      h5::read_data<1, std::vector<double> >(file_id, "bin_bottom");
  const std::vector<double> energy_bin_upper_bound =
      h5::read_data<1, std::vector<double> >(file_id, "bin_top");
  if ((energy_bin_lower_bound.size() != EnergyBins) or
      (energy_bin_upper_bound.size() != EnergyBins)) {
    ERROR("Table has inconsistent number of points in energy. Expected"
          << EnergyBins << " but got " << energy_bin_lower_bound.size()
          << " (lower bound) and " << energy_bin_upper_bound.size()
          << " (upper bound).");
  }
  for (size_t i = 0; i < EnergyBins; i++) {
    gsl::at(table_neutrino_energies, i) =
        0.5 * (energy_bin_lower_bound[i] + energy_bin_upper_bound[i]);
  }

  // 5-dimensional array, with dimensions
  // (NeutrinoSpecies, EnergyBins, n_points_electron_fraction,
  //  n_points_temperature, n_points_density)
  boost::multi_array<double, 5> buffer =
      h5::read_data<5, boost::multi_array<double, 5> >(file_id, "emissivities");

  const size_t n_pts_per_var =
      n_points_density * n_points_temperature * n_points_electron_fraction;
  const size_t number_of_vars = 3 * NeutrinoSpecies * EnergyBins;
  table_data.resize(number_of_vars * n_pts_per_var);
  size_t idx = 0;
  size_t idx_emissivity = 0;
  for (size_t ng = 0; ng < EnergyBins; ng++) {
    for (size_t ns = 0; ns < NeutrinoSpecies; ns++) {
      for (size_t ny = 0; ny < n_points_electron_fraction; ny++) {
        for (size_t nt = 0; nt < n_points_temperature; nt++) {
          for (size_t nr = 0; nr < n_points_density; nr++) {
            idx = (ns * EnergyBins + ng) +
                  (ny * n_points_temperature * n_points_density +
                   nt * n_points_density + nr) *
                      number_of_vars;
            table_data[idx] =
                buffer[ng][ns][ny][nt][nr] * emissivity_NuLib_to_code *
                (energy_bin_upper_bound[ng] - energy_bin_lower_bound[ng]);
          }
        }
      }
    }
  }

  buffer = h5::read_data<5, boost::multi_array<double, 5> >(
      file_id, "absorption_opacity");
  for (size_t ng = 0; ng < EnergyBins; ng++) {
    for (size_t ns = 0; ns < NeutrinoSpecies; ns++) {
      for (size_t ny = 0; ny < n_points_electron_fraction; ny++) {
        for (size_t nt = 0; nt < n_points_temperature; nt++) {
          for (size_t nr = 0; nr < n_points_density; nr++) {
            idx_emissivity = (ns * EnergyBins + ng) +
                             (ny * n_points_temperature * n_points_density +
                              nt * n_points_density + nr) *
                                 number_of_vars;
            idx = (NeutrinoSpecies * EnergyBins) + idx_emissivity;
            table_data[idx] = buffer[ng][ns][ny][nt][nr] * length_unit;
            if (table_data[idx] < min_kappa) {
              table_data[idx_emissivity] = 0.0;
              table_data[idx] = min_kappa;
            }
            if (table_data[idx] > max_kappa) {
              table_data[idx_emissivity] *= max_kappa / table_data[idx];
              table_data[idx] = max_kappa;
            }
          }
        }
      }
    }
  }

  buffer = h5::read_data<5, boost::multi_array<double, 5> >(
      file_id, "scattering_opacity");
  for (size_t ng = 0; ng < EnergyBins; ng++) {
    for (size_t ns = 0; ns < NeutrinoSpecies; ns++) {
      for (size_t ny = 0; ny < n_points_electron_fraction; ny++) {
        for (size_t nt = 0; nt < n_points_temperature; nt++) {
          for (size_t nr = 0; nr < n_points_density; nr++) {
            idx_emissivity = (ns * EnergyBins + ng) +
                             (ny * n_points_temperature * n_points_density +
                              nt * n_points_density + nr) *
                                 number_of_vars;
            idx = 2 * (NeutrinoSpecies * EnergyBins) + idx_emissivity;
            table_data[idx] = buffer[ng][ns][ny][nt][nr] * length_unit;
            table_data[idx] =
              std::clamp(table_data[idx], min_kappa, max_kappa);
          }
        }
      }
    }
  }

  // Correction to emissivity/absorption/scattering at end of table to keep
  // Ye within the neutrino table bounds (in this regime, we cannot evolve Ye
  // according to the true tabulated value; we correct the tabulated
  // values in the hope of at least getting lepton number conservation).
  // Note that this could be insufficient if the EoS table bounds are
  // more restrictive than the neutrino table bounds; but this table
  // does not know the EoS bounds. Ideally, the neutrino and EoS tables
  // would have the same bounds.
  if (NeutrinoSpecies >= 2) {
    // Correction at bounds of table in electron fraction
    // assume that we have at least electron neutrinos and
    // antineutrinos.
    // We correct the emissivity at both bounds, and the
    // absorption only when absorption would drive the
    // electron fraction outside of the table bounds.
    // Note that absorbing an electron neutrino (ns==0)
    // increases the electron fraction.
    for (size_t ng = 0; ng < EnergyBins; ng++) {
      for (size_t nt = 0; nt < n_points_temperature; nt++) {
        for (size_t nr = 0; nr < n_points_density; nr++) {
          idx_emissivity = ng + (nt * n_points_density + nr) * number_of_vars;
          idx = (NeutrinoSpecies * EnergyBins) + idx_emissivity;
          // nu_e
          table_data[idx_emissivity] = 0.0;
          // nu_a
          table_data[idx_emissivity + EnergyBins] = 0.0;
          table_data[idx + EnergyBins] = min_kappa;

          idx += ((n_points_electron_fraction - 1) * n_points_temperature *
                  n_points_density) *
                 number_of_vars;
          idx_emissivity += ((n_points_electron_fraction - 1) *
                             n_points_temperature * n_points_density) *
                            number_of_vars;
          // nu_e
          table_data[idx_emissivity] = 0.0;
          table_data[idx] = min_kappa;
          // nu_a
          table_data[idx_emissivity + EnergyBins] = 0.0;
        }
      }
    }
  }
  initialize_interpolator();
}

template <size_t EnergyBins, size_t NeutrinoSpecies>
void NeutrinoInteractionTable<EnergyBins,
                              NeutrinoSpecies>::initialize_interpolator() {
  // Initialize interpolator
  Index<3> num_x_points;
  // The order is rho, T, Ye
  num_x_points[0] = table_log_density.size();
  num_x_points[1] = table_log_temperature.size();
  num_x_points[2] = table_electron_fraction.size();
  std::array<gsl::span<double const>, 3> independent_data_view;
  independent_data_view[0] =
      gsl::span<double const>{table_log_density.data(), num_x_points[0]};
  independent_data_view[1] =
      gsl::span<double const>{table_log_temperature.data(), num_x_points[1]};
  independent_data_view[2] =
      gsl::span<double const>{table_electron_fraction.data(), num_x_points[2]};
  interpolator_ =
      intrp::UniformMultiLinearSpanInterpolation<3, 3 * EnergyBins *
                                                        NeutrinoSpecies>(
          independent_data_view, {table_data.data(), table_data.size()},
          num_x_points);
}

template <size_t EnergyBins, size_t NeutrinoSpecies>
void NeutrinoInteractionTable<EnergyBins, NeutrinoSpecies>::
    get_neutrino_matter_interactions(
        const gsl::not_null<
            std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>*>
            emissivity_in_cell,
        const gsl::not_null<
            std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>*>
            absorption_opacity,
        const gsl::not_null<
            std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>*>
            scattering_opacity,
        const Scalar<DataVector>& electron_fraction,
        const Scalar<DataVector>& rest_mass_density,
        const Scalar<DataVector>& temperature,
        const Scalar<DataVector>& cell_proper_four_volume,
        const double& minimum_temperature) const {
  const size_t n_rho_points = table_log_density.size();
  const size_t n_temp_points = table_log_temperature.size();
  const size_t n_ye_points = table_electron_fraction.size();
  double temperature_correction_factor = 0.0;

  double ye = 0.0;
  double log_rho = 0.0;
  double log_temp = 0.0;
  for (size_t p = 0; p < get(electron_fraction).size(); p++) {
    ye = get(electron_fraction)[p];
    log_rho = log(get(rest_mass_density)[p]);
    log_temp = get(temperature)[p] > minimum_temperature
                   ? log(get(temperature)[p])
                   : log(minimum_temperature);
    temperature_correction_factor =
        get(temperature)[p] > minimum_temperature
            ? 1.0
            : get(temperature)[p] / minimum_temperature;

    ye = std::clamp(ye, table_electron_fraction[0],
                    table_electron_fraction[n_ye_points - 1]);
    log_rho = std::clamp(log_rho, table_log_density[0],
                         table_log_density[n_rho_points - 1]);
    log_temp = std::clamp(log_temp, table_log_temperature[0],
                          table_log_temperature[n_temp_points - 1]);

    const auto weights = interpolator_.get_weights(log_rho, log_temp, ye);

    for (size_t ng = 0; ng < EnergyBins; ng++) {
      for (size_t ns = 0; ns < NeutrinoSpecies; ns++) {
        gsl::at(gsl::at(*emissivity_in_cell, ns), ng)[p] =
            interpolator_.interpolate(weights, ns * EnergyBins + ng);
        gsl::at(gsl::at(*absorption_opacity, ns), ng)[p] =
            interpolator_.interpolate(
                weights, ns * EnergyBins + ng + EnergyBins * NeutrinoSpecies);
        gsl::at(gsl::at(*scattering_opacity, ns), ng)[p] =
            interpolator_.interpolate(
                weights,
                ns * EnergyBins + ng + EnergyBins * NeutrinoSpecies * 2);
      }
    }
    // Multiply emissivity by cell volume, and apply corrections for
    // low-temperature points.
    for (size_t ng = 0; ng < EnergyBins; ng++) {
      for (size_t ns = 0; ns < NeutrinoSpecies; ns++) {
        gsl::at(gsl::at(*emissivity_in_cell, ns), ng)[p] *=
            get(cell_proper_four_volume)[p] *
            square(cube(temperature_correction_factor));
        gsl::at(gsl::at(*absorption_opacity, ns), ng)[p] *=
            square(square(temperature_correction_factor));
        gsl::at(gsl::at(*scattering_opacity, ns), ng)[p] *=
            square(square(temperature_correction_factor));
      }
    }
  }
}

}  // namespace Particles::MonteCarlo

template class Particles::MonteCarlo::NeutrinoInteractionTable<2, 2>;
template class Particles::MonteCarlo::NeutrinoInteractionTable<2, 3>;
template class Particles::MonteCarlo::NeutrinoInteractionTable<4, 3>;
template class Particles::MonteCarlo::NeutrinoInteractionTable<16, 3>;
