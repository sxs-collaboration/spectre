// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/Tensor/IndexType.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/Tags.hpp"

class DataVector;
template <bool IsRelativistic>
class IdealFluid;

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Hydro.Tags", "[Unit][Hydro]") {
  CHECK(hydro::Tags::AlfvenSpeedSquared<DataVector>::name() ==
        "AlfvenSpeedSquared");
  CHECK(hydro::Tags::ComovingMagneticField<DataVector, 3,
                                           Frame::Inertial>::name() ==
        "ComovingMagneticField");
  CHECK(hydro::Tags::ComovingMagneticField<DataVector, 3,
                                           Frame::Logical>::name() ==
        "Logical_ComovingMagneticField");
  CHECK(hydro::Tags::ComovingMagneticFieldSquared<DataVector>::name() ==
        "ComovingMagneticFieldSquared");
  CHECK(hydro::Tags::DivergenceCleaningField<DataVector>::name() ==
        "DivergenceCleaningField");
  CHECK(hydro::Tags::EquationOfState<IdealFluid<true>>::name() ==
        "EquationOfState");
  CHECK(hydro::Tags::LorentzFactor<DataVector>::name() == "LorentzFactor");
  CHECK(hydro::Tags::LorentzFactorSquared<DataVector>::name() ==
        "LorentzFactorSquared");
  CHECK(hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>::name() ==
        "MagneticField");
  CHECK(hydro::Tags::MagneticField<DataVector, 3, Frame::Distorted>::name() ==
        "Distorted_MagneticField");
  CHECK(hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>::name() ==
        "MagneticFieldDotSpatialVelocity");
  CHECK(hydro::Tags::MagneticFieldOneForm<DataVector, 3,
                                          Frame::Inertial>::name() ==
        "MagneticFieldOneForm");
  CHECK(hydro::Tags::MagneticFieldSquared<DataVector>::name() ==
        "MagneticFieldSquared");
  CHECK(hydro::Tags::MagneticPressure<DataVector>::name() ==
        "MagneticPressure");
  CHECK(hydro::Tags::Pressure<DataVector>::name() == "Pressure");
  CHECK(hydro::Tags::RestMassDensity<DataVector>::name() == "RestMassDensity");
  CHECK(hydro::Tags::SoundSpeedSquared<DataVector>::name() ==
        "SoundSpeedSquared");
  /// [prefix_example]
  CHECK(hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>::name() ==
        "SpatialVelocity");
  CHECK(hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Grid>::name() ==
        "Grid_SpatialVelocity");
  CHECK(hydro::Tags::SpatialVelocityOneForm<DataVector, 3,
                                            Frame::Inertial>::name() ==
        "SpatialVelocityOneForm");
  CHECK(hydro::Tags::SpatialVelocityOneForm<DataVector, 3,
                                            Frame::Logical>::name() ==
        "Logical_SpatialVelocityOneForm");
  /// [prefix_example]
  CHECK(hydro::Tags::SpatialVelocitySquared<double>::name() ==
        "SpatialVelocitySquared");
  CHECK(hydro::Tags::SpatialVelocitySquared<DataVector>::name() ==
        "SpatialVelocitySquared");
  CHECK(hydro::Tags::SpecificEnthalpy<double>::name() == "SpecificEnthalpy");
  CHECK(hydro::Tags::SpecificEnthalpy<DataVector>::name() ==
        "SpecificEnthalpy");
  CHECK(hydro::Tags::SpecificInternalEnergy<double>::name() ==
        "SpecificInternalEnergy");
  CHECK(hydro::Tags::SpecificInternalEnergy<DataVector>::name() ==
        "SpecificInternalEnergy");
}
