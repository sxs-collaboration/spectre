// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/Cce/BoundaryDataTags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.BoundaryDataTags",
                  "[Unit][Cce]") {
  TestHelpers::db::test_simple_tag<Cce::Tags::detail::CosPhi>("CosPhi");
  TestHelpers::db::test_simple_tag<Cce::Tags::detail::CosTheta>("CosTheta");
  TestHelpers::db::test_simple_tag<Cce::Tags::detail::SinPhi>("SinPhi");
  TestHelpers::db::test_simple_tag<Cce::Tags::detail::SinTheta>("SinTheta");
  TestHelpers::db::test_simple_tag<Cce::Tags::detail::CartesianCoordinates>(
      "CartesianCoordinates");
  TestHelpers::db::test_simple_tag<
      Cce::Tags::detail::CartesianToSphericalJacobian>(
      "CartesianToSphericalJacobian");
  TestHelpers::db::test_simple_tag<
      Cce::Tags::detail::InverseCartesianToSphericalJacobian>(
      "InverseCartesianToSphericalJacobian");
  TestHelpers::db::test_simple_tag<Cce::Tags::detail::WorldtubeNormal>(
      "WorldtubeNormal");
  TestHelpers::db::test_simple_tag<Cce::Tags::detail::UpDyad>("UpDyad");
  TestHelpers::db::test_simple_tag<Cce::Tags::detail::DownDyad>("DownDyad");
  TestHelpers::db::test_simple_tag<Cce::Tags::detail::RealBondiR>("RealBondiR");
  TestHelpers::db::test_simple_tag<Cce::Tags::detail::AngularDNullL>(
      "AngularDNullL");
  TestHelpers::db::test_simple_tag<Cce::Tags::detail::NullL>("NullL");
  TestHelpers::db::test_simple_tag<
      Cce::Tags::detail::DLambda<Cce::Tags::detail::RealBondiR>>("DLambda");
}
