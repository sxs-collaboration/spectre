// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/Tensor/IndexType.hpp"  // IWYU pragma: keep
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"

class DataVector;
template <bool IsRelativistic>
class IdealFluid;

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Hydro.Tags", "[Unit][Hydro]") {
  TestHelpers::db::test_simple_tag<hydro::Tags::AlfvenSpeedSquared<DataVector>>(
      "AlfvenSpeedSquared");
  TestHelpers::db::test_simple_tag<
      hydro::Tags::ComovingMagneticField<DataVector, 3, Frame::Logical>>(
      "Logical_ComovingMagneticField");
  TestHelpers::db::test_simple_tag<
      hydro::Tags::ComovingMagneticFieldSquared<DataVector>>(
      "ComovingMagneticFieldSquared");
  TestHelpers::db::test_simple_tag<
      hydro::Tags::DivergenceCleaningField<DataVector>>(
      "DivergenceCleaningField");
  TestHelpers::db::test_base_tag<hydro::Tags::EquationOfStateBase>(
      "EquationOfStateBase");
  TestHelpers::db::test_simple_tag<
      hydro::Tags::EquationOfState<IdealFluid<true>>>("EquationOfState");
  TestHelpers::db::test_simple_tag<hydro::Tags::LorentzFactor<DataVector>>(
      "LorentzFactor");
  TestHelpers::db::test_simple_tag<
      hydro::Tags::LorentzFactorSquared<DataVector>>("LorentzFactorSquared");
  TestHelpers::db::test_simple_tag<
      hydro::Tags::MagneticField<DataVector, 3, Frame::Distorted>>(
      "Distorted_MagneticField");
  TestHelpers::db::test_simple_tag<
      hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>>(
      "MagneticFieldDotSpatialVelocity");
  TestHelpers::db::test_simple_tag<
      hydro::Tags::MagneticFieldOneForm<DataVector, 3, Frame::Logical>>(
      "Logical_MagneticFieldOneForm");
  TestHelpers::db::test_simple_tag<
      hydro::Tags::MagneticFieldSquared<DataVector>>("MagneticFieldSquared");
  TestHelpers::db::test_simple_tag<hydro::Tags::MagneticPressure<DataVector>>(
      "MagneticPressure");
  TestHelpers::db::test_simple_tag<hydro::Tags::Pressure<DataVector>>(
      "Pressure");
  TestHelpers::db::test_simple_tag<hydro::Tags::RestMassDensity<DataVector>>(
      "RestMassDensity");
  TestHelpers::db::test_simple_tag<hydro::Tags::SoundSpeedSquared<DataVector>>(
      "SoundSpeedSquared");
  // [prefix_example]
  TestHelpers::db::test_simple_tag<
      hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Grid>>(
      "Grid_SpatialVelocity");
  TestHelpers::db::test_simple_tag<
      hydro::Tags::SpatialVelocityOneForm<DataVector, 3, Frame::Logical>>(
      "Logical_SpatialVelocityOneForm");
  // [prefix_example]
  TestHelpers::db::test_simple_tag<hydro::Tags::SpatialVelocitySquared<double>>(
      "SpatialVelocitySquared");
  TestHelpers::db::test_simple_tag<
      hydro::Tags::SpatialVelocitySquared<DataVector>>(
      "SpatialVelocitySquared");
  TestHelpers::db::test_simple_tag<hydro::Tags::SpecificEnthalpy<double>>(
      "SpecificEnthalpy");
  TestHelpers::db::test_simple_tag<hydro::Tags::SpecificEnthalpy<DataVector>>(
      "SpecificEnthalpy");
  TestHelpers::db::test_simple_tag<hydro::Tags::SpecificInternalEnergy<double>>(
      "SpecificInternalEnergy");
  TestHelpers::db::test_simple_tag<
      hydro::Tags::SpecificInternalEnergy<DataVector>>(
      "SpecificInternalEnergy");
  TestHelpers::db::test_simple_tag<
      hydro::Tags::MassFlux<DataVector, 3, Frame::Logical>>("Logical_MassFlux");
}
