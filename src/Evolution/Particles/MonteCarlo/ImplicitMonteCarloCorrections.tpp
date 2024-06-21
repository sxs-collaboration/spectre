// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Particles/MonteCarlo/TemplatedLocalFunctions.hpp"

#include <array>

#include "Evolution/Particles/MonteCarlo/NeutrinoInteractionTable.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"

using hydro::units::nuclear::proton_mass;

namespace Particles::MonteCarlo {

template <size_t EnergyBins, size_t NeutrinoSpecies>
void TemplatedLocalFunctions<EnergyBins, NeutrinoSpecies>::
    implicit_monte_carlo_interaction_rates(
        gsl::not_null<
            std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>*>
            emissivity_in_cell,
        gsl::not_null<
            std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>*>
            absorption_opacity,
        gsl::not_null<
            std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>*>
            scattering_opacity,
        gsl::not_null<
            std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>*>
            fraction_ka_to_ks,
        const Scalar<DataVector>& cell_light_crossing_time,
        const Scalar<DataVector>& electron_fraction,
        const Scalar<DataVector>& rest_mass_density,
        const Scalar<DataVector>& temperature,
        const double minimum_temperature,
        const NeutrinoInteractionTable<EnergyBins, NeutrinoSpecies>&
            interaction_table,
        const EquationsOfState::EquationOfState<true, 3>& equation_of_state) {
  // Initial guess for interaction rates: no implicit MC corrections
  interaction_table.get_neutrino_matter_interactions(
      emissivity_in_cell, absorption_opacity, scattering_opacity,
      electron_fraction, rest_mass_density, temperature,
      minimum_temperature);

  const std::array<double, EnergyBins>& neutrino_energies =
      interaction_table.get_neutrino_energies();

  // Apply implicit MC corrections as needed
  const size_t dv_size = rest_mass_density.size();

  // Calculate beta parameter (relative change of MC vs fluid variables)
  // For photon transport,
  // beta = d(radiation_energy_density)/d(fluid_energy_density)
  // at constant rest mass density, which is well defined. For
  // neutrinos, because we can vary both temperatures and electron
  // fraction, it is not as clear what should be done. Here, we
  // calculate both
  // d(radiation_energy_density)/d(fluid_energy_density)
  // at constant Ye, and
  // d(neutrino_lepton_number_density)/d(proton_number_density)
  // at constant T as a measure of how 'impactful' neutrino
  // emission/absorption can be on the fluid.
  Scalar<DataVector> beta =
      make_with_value<Scalar<DataVector>>(rest_mass_density, 0.0);
  Scalar<DataVector> fluid_energy_0 =
      make_with_value<Scalar<DataVector>>(rest_mass_density, 0.0);
  Scalar<DataVector> fluid_lepton_number_0 =
      make_with_value<Scalar<DataVector>>(rest_mass_density, 0.0);
  Scalar<DataVector> neutrino_energy_0 =
      make_with_value<Scalar<DataVector>>(rest_mass_density, 0.0);
  Scalar<DataVector> neutrino_lepton_number_0 =
      make_with_value<Scalar<DataVector>>(rest_mass_density, 0.0);
  // Fluid densities at initial T,Ye
  Scalar<DataVector> specific_internal_energy =
      equation_of_state.specific_internal_energy_from_density_and_temperature(
          rest_mass_density, temperature, electron_fraction);
  for (size_t i = 0; i < dv_size; i++) {
    get(fluid_energy_0)[i] =
        get(rest_mass_density)[i] * get(specific_internal_energy)[i];
    get(fluid_lepton_number_0)[i] =
        get(rest_mass_density)[i] * get(electron_fraction)[i] / proton_mass;
    // Radiation densities at initial T,Ye
    for (size_t ns = 0; ns < NeutrinoSpecies; ns++) {
      for (size_t ng = 0; ng < EnergyBins; ng++) {
        get(neutrino_energy_0)[i] +=
            gsl::at(gsl::at(*emissivity_in_cell, ns), ng)[i] /
            gsl::at(gsl::at(*absorption_opacity, ns), ng)[i];
        if (ns < 2) {
          get(neutrino_lepton_number_0)[i] +=
              (ns == 0 ? 1.0 : -1.0) *
              gsl::at(gsl::at(*emissivity_in_cell, ns), ng)[i] /
              gsl::at(gsl::at(*absorption_opacity, ns), ng)[i] /
              neutrino_energies[ns];
        }
      }
    }
  }
  // Calculate energy as we vary temperature
  const double TEMP_EPS = 1.e-6;
  Scalar<DataVector> temperature_1 = temperature;
  get(temperature_1) += TEMP_EPS;
  Scalar<DataVector> fluid_energy_1 =
      make_with_value<Scalar<DataVector>>(rest_mass_density, 0.0);
  Scalar<DataVector> neutrino_energy_1 =
      make_with_value<Scalar<DataVector>>(rest_mass_density, 0.0);
  // Neutrino interaction rates at T+eps
  interaction_table.get_neutrino_matter_interactions(
      emissivity_in_cell, absorption_opacity, scattering_opacity,
      electron_fraction, rest_mass_density, temperature_1,
      minimum_temperature);
  specific_internal_energy =
      equation_of_state.specific_internal_energy_from_density_and_temperature(
          rest_mass_density, temperature_1, electron_fraction);
  for (size_t i = 0; i < dv_size; i++) {
    get(fluid_energy_1)[i] =
        get(rest_mass_density)[i] * get(specific_internal_energy)[i];
    for (size_t ns = 0; ns < NeutrinoSpecies; ns++) {
      for (size_t ng = 0; ng < EnergyBins; ng++) {
        get(neutrino_energy_1)[i] +=
            gsl::at(gsl::at(*emissivity_in_cell, ns), ng)[i] /
            gsl::at(gsl::at(*absorption_opacity, ns), ng)[i];
      }
    }
  }
  // Calculate lepton number as we vary electron fraction
  Scalar<DataVector> fluid_lepton_number_1 =
      make_with_value<Scalar<DataVector>>(rest_mass_density, 0.0);
  Scalar<DataVector> neutrino_lepton_number_1 =
      make_with_value<Scalar<DataVector>>(rest_mass_density, 0.0);
  const double YE_EPS = 1.e-8;
  Scalar<DataVector> electron_fraction_1 = electron_fraction;
  get(electron_fraction_1) += YE_EPS;
  // Neutrino interaction rates at Ye+eps
  interaction_table.get_neutrino_matter_interactions(
      emissivity_in_cell, absorption_opacity, scattering_opacity,
      electron_fraction_1, rest_mass_density, temperature,
      minimum_temperature);
  for (size_t i = 0; i < dv_size; i++) {
    get(fluid_lepton_number_1)[i] =
        get(rest_mass_density)[i] * get(electron_fraction_1)[i] / proton_mass;
    // Radiation densities at initial T,Ye
    for (size_t ns = 0; ns < NeutrinoSpecies; ns++) {
      for (size_t ng = 0; ng < EnergyBins; ng++) {
        if (ns < 2) {
          get(neutrino_lepton_number_1)[i] +=
              (ns == 0 ? 1.0 : -1.0) *
              gsl::at(gsl::at(*emissivity_in_cell, ns), ng)[i] /
              gsl::at(gsl::at(*absorption_opacity, ns), ng)[i] /
              neutrino_energies[ns];
        }
      }
    }
    // We have all we need to calculate beta here. We take the largest of the
    // two variations
    get(beta)[i] = fabs(get(neutrino_energy_1)[i] - get(neutrino_energy_0)[i]) /
                   fabs(get(fluid_energy_1)[i] - get(fluid_energy_0)[i]);
    const double beta_lepton =
        fabs(get(neutrino_lepton_number_1)[i] -
             get(neutrino_lepton_number_0)[i]) /
        fabs(get(fluid_lepton_number_1)[i] - get(fluid_lepton_number_0)[i]);
    get(beta)[i] =
        std::max(get(beta)[i], beta_lepton);
  }

  // Correct interaction rates, group by group
  interaction_table.get_neutrino_matter_interactions(
      emissivity_in_cell, absorption_opacity, scattering_opacity,
      electron_fraction, rest_mass_density, temperature,
      minimum_temperature);
  for (size_t i = 0; i < dv_size; i++) {
    const double max_opacity_for_implicit_mc =
      std::min(100.0 ,1.0/get(cell_light_crossing_time)[i]);
    for (size_t ns = 0; ns < NeutrinoSpecies; ns++) {
      for (size_t ng = 0; ng < EnergyBins; ng++) {
        const double ka = gsl::at(gsl::at(*absorption_opacity, ns), ng)[i];
        // Fraction of the absorption opacity to be moved to
        // scattering opacity.
        double frac_ka_to_ks = (ka > max_opacity_for_implicit_mc)
                                   ? 1.0 - max_opacity_for_implicit_mc / ka
                                   : 0.0;
        const double frac_from_beta =
            1.0 - 1.0 / ka / get(cell_light_crossing_time)[i] /
              (1.0 + get(beta)[i]);
        frac_ka_to_ks = std::max(frac_ka_to_ks, frac_from_beta);
        gsl::at(gsl::at(*scattering_opacity, ns), ng)[i] +=
          frac_ka_to_ks * ka;
        gsl::at(gsl::at(*absorption_opacity, ns), ng)[i] *=
            (1.0 - frac_ka_to_ks);
        gsl::at(gsl::at(*emissivity_in_cell, ns), ng)[i] *=
            (1.0 - frac_ka_to_ks);
        gsl::at(gsl::at(*fraction_ka_to_ks, ns), ng)[i] =
          frac_ka_to_ks;
      }
    }
  }
}

}  // namespace Particles::MonteCarlo
