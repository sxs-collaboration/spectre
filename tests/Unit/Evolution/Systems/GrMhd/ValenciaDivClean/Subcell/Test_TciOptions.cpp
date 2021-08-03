// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TciOptions.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.Subcell.TciOptions",
                  "[Unit][GrMhd]") {
  const auto tci_options_from_opts = TestHelpers::test_option_tag<
      grmhd::ValenciaDivClean::subcell::OptionTags::TciOptions>(
      "MinimumValueOfD: 1.0e-18\n"
      "MinimumValueOfTildeTau: 1.0e-38\n"
      "AtmosphereDensity: 1.1e-12\n"
      "SafetyFactorForB: 1.0e-12\n");
  const auto tci_options = serialize_and_deserialize(tci_options_from_opts);
  CHECK(tci_options.minimum_rest_mass_density_times_lorentz_factor == 1.0e-18);
  CHECK(tci_options.minimum_tilde_tau == 1.0e-38);
  CHECK(tci_options.atmosphere_density == 1.1e-12);
  CHECK(tci_options.safety_factor_for_magnetic_field == 1.0e-12);
}
