// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/Tensor/IndexType.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/GrMhd/Tags.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GrMhd.Tags", "[Unit][GrMhd]") {
  CHECK(grmhd::Tags::RestMassDensity::name() == "RestMassDensity");
  CHECK(grmhd::Tags::SpecificInternalEnergy::name() ==
        "SpecificInternalEnergy");
  /// [prefix_example]
  CHECK(grmhd::Tags::SpatialVelocity<Frame::Inertial>::name() ==
        "SpatialVelocity");
  CHECK(grmhd::Tags::SpatialVelocity<Frame::Grid>::name() ==
        "Grid_SpatialVelocity");
  CHECK(grmhd::Tags::MagneticField<Frame::Inertial>::name() == "MagneticField");
  CHECK(grmhd::Tags::MagneticField<Frame::Distorted>::name() ==
        "Distorted_MagneticField");
  /// [prefix_example]
  CHECK(grmhd::Tags::DivergenceCleaningField::name() ==
        "DivergenceCleaningField");
  CHECK(grmhd::Tags::Pressure::name() == "Pressure");
  CHECK(grmhd::Tags::SpecificEnthalpy::name() == "SpecificEnthalpy");
  CHECK(grmhd::Tags::ComovingMagneticField<Frame::Inertial>::name() ==
        "ComovingMagneticField");
  CHECK(grmhd::Tags::ComovingMagneticField<Frame::Logical>::name() ==
        "Logical_ComovingMagneticField");
  CHECK(grmhd::Tags::MagneticPressure::name() == "MagneticPressure");
  CHECK(grmhd::Tags::LorentzFactor::name() == "LorentzFactor");
}
