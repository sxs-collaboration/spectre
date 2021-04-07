// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/Tensor/IndexType.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/RadiationTransport/M1Grey/Characteristics.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Tags.hpp"
#include "Evolution/Systems/RadiationTransport/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

// Test tags used in M1Grey code
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
  TestHelpers::db::test_simple_tag<RadiationTransport::M1Grey::Tags::TildeE<
      Frame::Grid, neutrinos::ElectronNeutrinos<1>>>(
      "Grid_TildeE_ElectronNeutrinos1");
  TestHelpers::db::test_simple_tag<RadiationTransport::M1Grey::Tags::TildeS<
      Frame::Grid, neutrinos::ElectronAntiNeutrinos<2>>>(
      "Grid_TildeS_ElectronAntiNeutrinos2");
  TestHelpers::db::test_simple_tag<RadiationTransport::M1Grey::Tags::TildeP<
      Frame::Grid, neutrinos::ElectronAntiNeutrinos<2>>>(
      "Grid_TildeP_ElectronAntiNeutrinos2");
  TestHelpers::db::test_simple_tag<
      RadiationTransport::M1Grey::Tags::TildeSVector<Frame::Grid>>(
      "Grid_TildeSVector");
  TestHelpers::db::test_simple_tag<
      RadiationTransport::M1Grey::Tags::ClosureFactor<
          neutrinos::ElectronNeutrinos<1>>>("ClosureFactor_ElectronNeutrinos1");
  TestHelpers::db::test_simple_tag<
      RadiationTransport::M1Grey::Tags::TildeJ<
          neutrinos::ElectronNeutrinos<1>>>("TildeJ_ElectronNeutrinos1");
  TestHelpers::db::test_simple_tag<
      RadiationTransport::M1Grey::Tags::TildeHNormal<
          neutrinos::ElectronNeutrinos<1>>>("TildeHNormal_ElectronNeutrinos1");
  TestHelpers::db::test_simple_tag<
      RadiationTransport::M1Grey::Tags::TildeHSpatial<
          Frame::Grid, neutrinos::ElectronNeutrinos<1>>>(
      "Grid_TildeHSpatial_ElectronNeutrinos1");
  TestHelpers::db::test_simple_tag<
      RadiationTransport::M1Grey::Tags::GreyEmissivity<
          neutrinos::ElectronNeutrinos<1>>>(
      "GreyEmissivity_ElectronNeutrinos1");
  TestHelpers::db::test_simple_tag<
      RadiationTransport::M1Grey::Tags::GreyAbsorptionOpacity<
          neutrinos::ElectronNeutrinos<1>>>(
      "GreyAbsorptionOpacity_ElectronNeutrinos1");
  TestHelpers::db::test_simple_tag<
      RadiationTransport::M1Grey::Tags::GreyScatteringOpacity<
          neutrinos::ElectronNeutrinos<1>>>(
      "GreyScatteringOpacity_ElectronNeutrinos1");
  TestHelpers::db::test_simple_tag<
      RadiationTransport::M1Grey::Tags::M1HydroCouplingNormal<
          neutrinos::ElectronNeutrinos<1>>>(
      "M1HydroCouplingNormal_ElectronNeutrinos1");
  TestHelpers::db::test_simple_tag<
      RadiationTransport::M1Grey::Tags::M1HydroCouplingSpatial<
          Frame::Grid, neutrinos::ElectronNeutrinos<1>>>(
      "Grid_M1HydroCouplingSpatial_ElectronNeutrinos1");
  TestHelpers::db::test_simple_tag<
      RadiationTransport::M1Grey::Tags::CharacteristicSpeeds>(
      "CharacteristicSpeeds");
  TestHelpers::db::test_compute_tag<
      RadiationTransport::M1Grey::Tags::CharacteristicSpeedsCompute>(
      "CharacteristicSpeeds");

}
