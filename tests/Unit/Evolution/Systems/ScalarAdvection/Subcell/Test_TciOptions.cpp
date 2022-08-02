// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/ScalarAdvection/Subcell/TciOptions.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.ScalarAdvection.Subcell.TciOptions",
                  "[Unit][Evolution]") {
  const auto tci_options_from_opts = TestHelpers::test_option_tag<
      ScalarAdvection::subcell::OptionTags::TciOptions>("UCutoff: 1.0e-10\n");
  const auto tci_options = serialize_and_deserialize(tci_options_from_opts);
  CHECK(tci_options.u_cutoff == 1.0e-10);
}
