// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/Tensor/IndexType.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/RadiationTransport/M1Grey/Tags.hpp"
#include "Evolution/Systems/RadiationTransport/Tags.hpp"

/// Test tags used in M1Grey code
SPECTRE_TEST_CASE("Evolution.Systems.RadiationTransport.M1Grey.Tags",
                  "[Unit][M1Grey]") {
  // (1) Tags for neutrino species
  const neutrinos::ElectronNeutrinos<3> nue;
  const neutrinos::ElectronAntiNeutrinos<3> nua;
  const neutrinos::HeavyLeptonNeutrinos<3> nux;

  CHECK(neutrinos::get_name(nue) == "ElectronNeutrinos3");
  CHECK(neutrinos::get_name(nua) == "ElectronAntiNeutrinos3");
  CHECK(neutrinos::get_name(nux) == "HeavyLeptonNeutrinos3");

  // (2) Tags for evolved variables
  CHECK(RadiationTransport::M1Grey::Tags::TildeE<
            Frame::Inertial, neutrinos::ElectronNeutrinos<1> >::name() ==
        "TildeE_ElectronNeutrinos1");
  CHECK(RadiationTransport::M1Grey::Tags::TildeS<
            Frame::Grid, neutrinos::ElectronAntiNeutrinos<2> >::name() ==
        "Grid_TildeS_ElectronAntiNeutrinos2");
}
