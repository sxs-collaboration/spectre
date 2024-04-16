// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Particles/MonteCarlo/Packet.hpp"
#include "Evolution/Particles/MonteCarlo/TemplatedLocalFunctions.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Particles.MonteCarloInterpolateOpacities",
                  "[Unit][Evolution]") {
  const double eps_check = 1.e-14;

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<double> rng_uniform_zero_to_one(0.0, 1.0);

  // Zero vector for tensor creations
  DataVector zero_dv(8, 0.0);

  std::array<std::array<DataVector, 2>, 3> absorption_opacity_table = {
      std::array<DataVector, 2>{{zero_dv, zero_dv}},
      std::array<DataVector, 2>{{zero_dv, zero_dv}},
      std::array<DataVector, 2>{{zero_dv, zero_dv}}};
  std::array<std::array<DataVector, 2>, 3> scattering_opacity_table = {
      std::array<DataVector, 2>{{zero_dv, zero_dv}},
      std::array<DataVector, 2>{{zero_dv, zero_dv}},
      std::array<DataVector, 2>{{zero_dv, zero_dv}}};
  const std::array<double, 2> energy_at_bin_center = {2.0, 5.0};

  // Set random interaction rates
  for (size_t s = 0; s < 3; s++) {
    for (size_t i = 0; i < zero_dv.size(); i++) {
      for (size_t g = 0; g < 2; g++) {
        gsl::at(gsl::at(absorption_opacity_table, s), g)[i] =
            rng_uniform_zero_to_one(generator);
        gsl::at(gsl::at(scattering_opacity_table, s), g)[i] =
            rng_uniform_zero_to_one(generator);
      }
    }
  }

  // Get interaction rates from MC code and directly here, compare results.
  Particles::MonteCarlo::TemplatedLocalFunctions<2, 3> MonteCarloStruct;
  double absorption_opacity = 0.0;
  double scattering_opacity = 0.0;
  double expected_absorption_opacity = 0.0;
  double expected_scattering_opacity = 0.0;
  double fluid_frame_energy = 0.0;
  for (size_t p = 0; p < zero_dv.size(); p++) {
    fluid_frame_energy = 1.0 + 5.0 * rng_uniform_zero_to_one(generator);
    MonteCarloStruct.interpolate_opacities_at_fluid_energy(
        &absorption_opacity, &scattering_opacity, fluid_frame_energy, p % 3, p,
        absorption_opacity_table, scattering_opacity_table,
        energy_at_bin_center);
    // The actual calculation has a floor on the opacity, which
    // the test might run into.
    gsl::at(gsl::at(absorption_opacity_table, p % 3), 0)[p] =
        std::max(gsl::at(gsl::at(absorption_opacity_table, p % 3), 0)[p],
                 MonteCarloStruct.opacity_floor);
    gsl::at(gsl::at(absorption_opacity_table, p % 3), 1)[p] =
        std::max(gsl::at(gsl::at(absorption_opacity_table, p % 3), 1)[p],
                 MonteCarloStruct.opacity_floor);
    gsl::at(gsl::at(scattering_opacity_table, p % 3), 0)[p] =
        std::max(gsl::at(gsl::at(scattering_opacity_table, p % 3), 0)[p],
                 MonteCarloStruct.opacity_floor);
    gsl::at(gsl::at(scattering_opacity_table, p % 3), 1)[p] =
        std::max(gsl::at(gsl::at(scattering_opacity_table, p % 3), 1)[p],
                 MonteCarloStruct.opacity_floor);
    if (fluid_frame_energy <= gsl::at(energy_at_bin_center, 0)) {
      expected_absorption_opacity =
          gsl::at(gsl::at(absorption_opacity_table, p % 3), 0)[p];
      expected_scattering_opacity =
          gsl::at(gsl::at(scattering_opacity_table, p % 3), 0)[p];
    } else {
      if (fluid_frame_energy >= gsl::at(energy_at_bin_center, 1)) {
        expected_absorption_opacity =
            gsl::at(gsl::at(absorption_opacity_table, p % 3), 1)[p];
        expected_scattering_opacity =
            gsl::at(gsl::at(scattering_opacity_table, p % 3), 1)[p];
      } else {
        double coef = (fluid_frame_energy - gsl::at(energy_at_bin_center, 0)) /
                      (gsl::at(energy_at_bin_center, 1) -
                       gsl::at(energy_at_bin_center, 0));
        expected_absorption_opacity =
            log(gsl::at(gsl::at(absorption_opacity_table, p % 3), 1)[p]) *
                coef +
            log(gsl::at(gsl::at(absorption_opacity_table, p % 3), 0)[p]) *
                (1.0 - coef);
        expected_scattering_opacity =
            log(gsl::at(gsl::at(scattering_opacity_table, p % 3), 1)[p]) *
                coef +
            log(gsl::at(gsl::at(scattering_opacity_table, p % 3), 0)[p]) *
                (1.0 - coef);
        expected_absorption_opacity = exp(expected_absorption_opacity);
        expected_scattering_opacity = exp(expected_scattering_opacity);
      }
    }
    CHECK(fabs(expected_absorption_opacity - absorption_opacity) < eps_check);
    CHECK(fabs(expected_scattering_opacity - scattering_opacity) < eps_check);
  }
}
