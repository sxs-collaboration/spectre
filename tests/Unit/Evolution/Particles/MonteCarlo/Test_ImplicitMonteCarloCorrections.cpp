// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Particles/MonteCarlo/NeutrinoInteractionTable.hpp"
#include "Evolution/Particles/MonteCarlo/TemplatedLocalFunctions.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/Hydro/EquationsOfState/TestHelpers.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Tabulated3d.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Particles.MonteCarloImplicitCorrections",
                  "[Unit][Evolution]") {
  const size_t dv_size = 1;

  DataVector zero_dv(dv_size, 0.0);
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
  std::array<std::array<DataVector, 4>, 3> fraction_ka_to_ks = {
      std::array<DataVector, 4>{{zero_dv, zero_dv, zero_dv, zero_dv}},
      std::array<DataVector, 4>{{zero_dv, zero_dv, zero_dv, zero_dv}},
      std::array<DataVector, 4>{{zero_dv, zero_dv, zero_dv, zero_dv}}};

  Scalar<DataVector> baryon_density(dv_size, 1.619109365278362e-05);
  Scalar<DataVector> temperature(dv_size, 10.0);
  Scalar<DataVector> electron_fraction(dv_size, 0.06);
  Scalar<DataVector> cell_light_crossing_time(dv_size, 1.1);
  const double minimum_temperature = 0.0;

  const std::string h5_file_name_nulib{
      unit_test_src_path() +
      "Evolution/Particles/MonteCarlo/NuLib_TestTable.h5"};
  const Particles::MonteCarlo::NeutrinoInteractionTable<4, 3> interaction_table(
      h5_file_name_nulib);

  std::string h5_file_name_compose{
      unit_test_src_path() +
      "PointwiseFunctions/Hydro/EquationsOfState/dd2_unit_test.h5"};
  EquationsOfState::Tabulated3D<true> equation_of_state(h5_file_name_compose,
                                                        "/dd2");

  const Mesh<3> mesh(1, Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered);
  Particles::MonteCarlo::TemplatedLocalFunctions<4, 3> MonteCarloStruct;
  MonteCarloStruct.implicit_monte_carlo_interaction_rates(
      &emission_in_cells, &absorption_opacity, &scattering_opacity,
      &fraction_ka_to_ks, cell_light_crossing_time, electron_fraction,
      baryon_density, temperature, minimum_temperature, interaction_table,
      equation_of_state);

  std::array<std::array<DataVector, 4>, 3> emission_in_cells_0 = {
      std::array<DataVector, 4>{{zero_dv, zero_dv, zero_dv, zero_dv}},
      std::array<DataVector, 4>{{zero_dv, zero_dv, zero_dv, zero_dv}},
      std::array<DataVector, 4>{{zero_dv, zero_dv, zero_dv, zero_dv}}};
  std::array<std::array<DataVector, 4>, 3> absorption_opacity_0 = {
      std::array<DataVector, 4>{{zero_dv, zero_dv, zero_dv, zero_dv}},
      std::array<DataVector, 4>{{zero_dv, zero_dv, zero_dv, zero_dv}},
      std::array<DataVector, 4>{{zero_dv, zero_dv, zero_dv, zero_dv}}};
  std::array<std::array<DataVector, 4>, 3> scattering_opacity_0 = {
      std::array<DataVector, 4>{{zero_dv, zero_dv, zero_dv, zero_dv}},
      std::array<DataVector, 4>{{zero_dv, zero_dv, zero_dv, zero_dv}},
      std::array<DataVector, 4>{{zero_dv, zero_dv, zero_dv, zero_dv}}};
  interaction_table.get_neutrino_matter_interactions(
      &emission_in_cells_0, &absorption_opacity_0, &scattering_opacity_0,
      electron_fraction, baryon_density, temperature,
      minimum_temperature);

  CHECK_ITERABLE_CUSTOM_APPROX(absorption_opacity_0 + scattering_opacity_0,
                               absorption_opacity + scattering_opacity,
                               Approx::custom().epsilon(1.e-15).scale(1.0));
  for (size_t ng = 0; ng < 4; ng++) {
    for (size_t ns = 0; ns < 3; ns++) {
      CHECK_ITERABLE_CUSTOM_APPROX(
          gsl::at(gsl::at(scattering_opacity, ns), ng),
          gsl::at(gsl::at(scattering_opacity_0, ns), ng) +
              gsl::at(gsl::at(absorption_opacity_0, ns), ng) *
                  gsl::at(gsl::at(fraction_ka_to_ks, ns), ng),
          Approx::custom().epsilon(1.e-15).scale(1.0));
      CHECK_ITERABLE_CUSTOM_APPROX(
          gsl::at(gsl::at(emission_in_cells, ns), ng) /
              gsl::at(gsl::at(absorption_opacity, ns), ng),
          gsl::at(gsl::at(emission_in_cells_0, ns), ng) /
              gsl::at(gsl::at(absorption_opacity_0, ns), ng),
          Approx::custom().epsilon(1.e-15).scale(1.0));
    }
  }
}
