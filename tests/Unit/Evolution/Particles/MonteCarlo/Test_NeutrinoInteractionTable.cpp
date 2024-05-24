// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Particles/MonteCarlo/NeutrinoInteractionTable.hpp"
#include "Framework/TestingFramework.hpp"
#include "Informer/InfoFromBuild.hpp"

namespace {
void test_explicit_interaction_table() {
  const std::vector<double> table_log_density{-10.0, 0.0};
  const std::vector<double> table_log_temperature{-2.0, 0.0};
  const std::vector<double> table_electron_fraction{0.05, 0.06};
  const size_t n_points_table = 8;
  const size_t EnergyBins = 2;
  const size_t NeutrinoSpecies = 2;
  const size_t number_of_vars = 3 * EnergyBins * NeutrinoSpecies;
  const std::array<double, EnergyBins> table_neutrino_energies = {1.0, 3.0};
  std::vector<double> table_data{};
  table_data.resize(number_of_vars * n_points_table);
  // Fill in fake data for interaction rates. Fastest moving index is density,
  // then temperature, then electron fraction.
  // Here, we set interaction rates that linearly grow in log(density) so
  // we fill all 'high density' points of the table.
  for (size_t p = 1; p < n_points_table; p += 2) {
    // Emissivity
    // nu_e
    table_data[p * number_of_vars] = 1.0;
    table_data[p * number_of_vars + 1] = 9.0;
    // nu_a
    table_data[p * number_of_vars + EnergyBins] = 0.5;
    table_data[p * number_of_vars + EnergyBins + 1] = 4.5;
    // Absorption
    size_t shift = EnergyBins * NeutrinoSpecies;
    // nu_e
    table_data[p * number_of_vars + shift] = 1.0;
    table_data[p * number_of_vars + 1 + shift] = 9.0;
    // nu_a
    table_data[p * number_of_vars + EnergyBins + shift] = 0.5;
    table_data[p * number_of_vars + EnergyBins + 1 + shift] = 4.5;
    // Scattering
    shift = 2 * EnergyBins * NeutrinoSpecies;
    // nu_e
    table_data[p * number_of_vars + shift] = 1.0;
    table_data[p * number_of_vars + 1 + shift] = 9.0;
    // nu_a
    table_data[p * number_of_vars + EnergyBins + shift] = 0.5;
    table_data[p * number_of_vars + EnergyBins + 1 + shift] = 4.5;
  }
  const Particles::MonteCarlo::NeutrinoInteractionTable<2, 2> interaction_table(
      table_data, table_neutrino_energies, table_log_density,
      table_log_temperature, table_electron_fraction);
  CHECK(table_neutrino_energies == std::array<double, EnergyBins>{{1.0, 3.0}});

  const size_t dv_size = 1;
  DataVector zero_dv(dv_size, 0.0);
  std::array<std::array<DataVector, 2>, 2> emission_in_cells = {
      std::array<DataVector, 2>{{zero_dv, zero_dv}},
      std::array<DataVector, 2>{{zero_dv, zero_dv}}};
  std::array<std::array<DataVector, 2>, 2> absorption_opacity = {
      std::array<DataVector, 2>{{zero_dv, zero_dv}},
      std::array<DataVector, 2>{{zero_dv, zero_dv}}};
  std::array<std::array<DataVector, 2>, 2> scattering_opacity = {
      std::array<DataVector, 2>{{zero_dv, zero_dv}},
      std::array<DataVector, 2>{{zero_dv, zero_dv}}};
  const double minimum_temperature = 0.0;
  Scalar<DataVector> baryon_density(dv_size, exp(-5.0));
  Scalar<DataVector> temperature(dv_size, exp(-1.0));
  Scalar<DataVector> electron_fraction(dv_size, 0.055);
  Scalar<DataVector> cell_proper_four_volume(dv_size, 1.5);
  interaction_table.get_neutrino_matter_interactions(
      &emission_in_cells, &absorption_opacity, &scattering_opacity,
      electron_fraction, baryon_density, temperature, cell_proper_four_volume,
      minimum_temperature);

  CHECK(gsl::at(gsl::at(emission_in_cells, 0), 0)[0] == 0.75);
  CHECK(gsl::at(gsl::at(emission_in_cells, 0), 1)[0] == 6.75);
  CHECK(gsl::at(gsl::at(emission_in_cells, 1), 0)[0] == 0.375);
  CHECK(gsl::at(gsl::at(emission_in_cells, 1), 1)[0] == 3.375);
  CHECK(gsl::at(gsl::at(absorption_opacity, 0), 0)[0] == 0.5);
  CHECK(gsl::at(gsl::at(absorption_opacity, 0), 1)[0] == 4.5);
  CHECK(gsl::at(gsl::at(absorption_opacity, 1), 0)[0] == 0.25);
  CHECK(gsl::at(gsl::at(absorption_opacity, 1), 1)[0] == 2.25);
  CHECK(gsl::at(gsl::at(scattering_opacity, 0), 0)[0] == 0.5);
  CHECK(gsl::at(gsl::at(scattering_opacity, 0), 1)[0] == 4.5);
  CHECK(gsl::at(gsl::at(scattering_opacity, 1), 0)[0] == 0.25);
  CHECK(gsl::at(gsl::at(scattering_opacity, 1), 1)[0] == 2.25);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Particles.MonteCarloInteractionTable",
                  "[Unit][Evolution]") {
  const std::string h5_file_name{
      unit_test_src_path() +
      "Evolution/Particles/MonteCarlo/NuLib_TestTable.h5"};
  const Particles::MonteCarlo::NeutrinoInteractionTable<4, 3> interaction_table(
      h5_file_name);
  const double minimum_temperature = 0.0;

  const size_t dv_size = 4;
  DataVector zero_dv(4, 0.0);
  std::array<std::array<DataVector, 4>, 3> emission_in_cells = {
      std::array<DataVector, 4>{{zero_dv, zero_dv, zero_dv, zero_dv}},
      std::array<DataVector, 4>{{zero_dv, zero_dv, zero_dv, zero_dv}},
      std::array<DataVector, 4>{{zero_dv, zero_dv, zero_dv, zero_dv}}};
  std::array<std::array<DataVector, 4>, 3> absorption_opacity = {
      std::array<DataVector, 4>{{zero_dv, zero_dv, zero_dv, zero_dv}},
      std::array<DataVector, 4>{{zero_dv, zero_dv, zero_dv, zero_dv}},
      std::array<DataVector, 4>{{zero_dv, zero_dv, zero_dv, zero_dv}}};
  std::array<std::array<DataVector, 4>, 3> scattering_opacity = {
      std::array<DataVector, 4>{{zero_dv, zero_dv, zero_dv, zero_dv}},
      std::array<DataVector, 4>{{zero_dv, zero_dv, zero_dv, zero_dv}},
      std::array<DataVector, 4>{{zero_dv, zero_dv, zero_dv, zero_dv}}};

  // Point 0, using default values, is below the lower bound of
  // the table in each dimension
  Scalar<DataVector> baryon_density(dv_size, 1.e-16);
  Scalar<DataVector> temperature(dv_size, 1.e-5);
  Scalar<DataVector> electron_fraction(dv_size, 0.0);
  Scalar<DataVector> cell_proper_four_volume(dv_size, 1.5);

  // Point 1, 2 do not require interapolation, and are at points
  // (0,0,0) and (1,1,1) of the  table in rho x temp x ye
  // for the table NuLib_TestTable.h5
  get(baryon_density)[1] = 1.619109365278362e-05;
  get(baryon_density)[2] = 1.619109365278362e-05;
  get(temperature)[1] = 10.0;
  get(temperature)[2] = 10.0;
  get(electron_fraction)[1] = 0.05;
  get(electron_fraction)[2] = 0.06;

  // Point 3 constructed to get 0.3 * (Point 1) + 0.7 * (Point 2)
  // Note that we use interpolation in log density and log temperature, but
  // linearly in electron fraction
  get(baryon_density)[3] = exp(0.3 * log(get(baryon_density)[1]) +
                               0.7 * log(get(baryon_density)[2]));
  get(temperature)[3] =
      exp(0.3 * log(get(temperature)[1]) + 0.7 * log(get(temperature)[2]));
  get(electron_fraction)[3] =
      0.3 * get(electron_fraction)[1] + 0.7 * get(electron_fraction)[2];

  interaction_table.get_neutrino_matter_interactions(
      &emission_in_cells, &absorption_opacity, &scattering_opacity,
      electron_fraction, baryon_density, temperature, cell_proper_four_volume,
      minimum_temperature);

  const std::array<double, 4>& table_neutrino_energies =
      interaction_table.get_neutrino_energies();
  const std::array<double, 4> expected_neutrino_energies = {2, 6, 9, 10.5};

  CHECK(table_neutrino_energies == expected_neutrino_energies);

  for (size_t ng = 0; ng < 4; ng++) {
    for (size_t ns = 0; ns < 3; ns++) {
      CHECK(fabs(gsl::at(gsl::at(emission_in_cells, ns), ng)[0] -
                 gsl::at(gsl::at(emission_in_cells, ns), ng)[1]) <=
            1.e-5 * (gsl::at(gsl::at(emission_in_cells, ns), ng)[0] +
                     gsl::at(gsl::at(emission_in_cells, ns), ng)[1]));
      CHECK(fabs(gsl::at(gsl::at(absorption_opacity, ns), ng)[0] -
                 gsl::at(gsl::at(absorption_opacity, ns), ng)[1]) <=
            1.e-5 * (gsl::at(gsl::at(absorption_opacity, ns), ng)[0] +
                     gsl::at(gsl::at(absorption_opacity, ns), ng)[1]));
      CHECK(fabs(gsl::at(gsl::at(scattering_opacity, ns), ng)[0] -
                 gsl::at(gsl::at(scattering_opacity, ns), ng)[1]) <=
            1.e-5 * (gsl::at(gsl::at(scattering_opacity, ns), ng)[0] +
                     gsl::at(gsl::at(scattering_opacity, ns), ng)[1]));

      CHECK(fabs(gsl::at(gsl::at(emission_in_cells, ns), ng)[3] -
                 0.3 * gsl::at(gsl::at(emission_in_cells, ns), ng)[1] -
                 0.7 * gsl::at(gsl::at(emission_in_cells, ns), ng)[2]) <=
            1.e-5 * (gsl::at(gsl::at(emission_in_cells, ns), ng)[1] +
                     gsl::at(gsl::at(emission_in_cells, ns), ng)[2]));
      CHECK(fabs(gsl::at(gsl::at(absorption_opacity, ns), ng)[3] -
                 0.3 * gsl::at(gsl::at(absorption_opacity, ns), ng)[1] -
                 0.7 * gsl::at(gsl::at(absorption_opacity, ns), ng)[2]) <=
            1.e-5 * (gsl::at(gsl::at(absorption_opacity, ns), ng)[1] +
                     gsl::at(gsl::at(absorption_opacity, ns), ng)[2]));
      CHECK(fabs(gsl::at(gsl::at(scattering_opacity, ns), ng)[3] -
                 0.3 * gsl::at(gsl::at(scattering_opacity, ns), ng)[1] -
                 0.7 * gsl::at(gsl::at(scattering_opacity, ns), ng)[2]) <=
            1.e-5 * (gsl::at(gsl::at(scattering_opacity, ns), ng)[1] +
                     gsl::at(gsl::at(scattering_opacity, ns), ng)[2]));
    }
  }

  test_explicit_interaction_table();
}
