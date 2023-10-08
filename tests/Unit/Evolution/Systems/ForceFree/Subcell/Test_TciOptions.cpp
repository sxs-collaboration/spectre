// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/ForceFree/Subcell/TciOptions.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ForceFree.Subcell.TciOptions",
                  "[Unit][Evolution]") {
  const auto tci_options_from_opts =
      TestHelpers::test_option_tag<ForceFree::subcell::OptionTags::TciOptions>(
          "TildeQCutoff: 1.0e-10\n");
  const auto tci_options = serialize_and_deserialize(tci_options_from_opts);
  CHECK(tci_options.tilde_q_cutoff  == 1.0e-10);
}
