// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Elliptic/Systems/BnsInitialData/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Elliptic.Systems.BnsInitialData.Tags",
                  "[Unit][Elliptic]") {
  TestHelpers::db::test_simple_tag<
      BnsInitialData::Tags::VelocityPotential<DataVector>>(
      std::string{"VelocityPotential"});
  TestHelpers::db::test_simple_tag<
      BnsInitialData::Tags::RotationalShift<DataVector>>(
      std::string{"RotationalShift"});
  TestHelpers::db::test_simple_tag<
      BnsInitialData::Tags::RotationalShiftStress<DataVector>>(
      std::string{"RotationalShiftStress"});
  TestHelpers::db::test_simple_tag<
      BnsInitialData::Tags::DerivLogLapseTimesDensityOverSpecificEnthalpy<
          DataVector>>(
      std::string{"DerivLogLapseTimesDensityOverSpecificEnthalpy"});
  TestHelpers::db::test_simple_tag<
      BnsInitialData::Tags::SpatialRotationalKillingVector<DataVector>>(
      std::string{"SpatialRotationalKillingVector"});
  TestHelpers::db::test_simple_tag<
      BnsInitialData::Tags::DerivSpatialRotationalKillingVector<DataVector>>(
      std::string{"DerivSpatialRotationalKillingVector"});
  TestHelpers::db::test_simple_tag<BnsInitialData::Tags::EulerEnthalpyConstant>(
      std::string{"EulerEnthalpyConstant"});
}
