// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Elliptic/Systems/SelfForce/Scalar/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace ScalarSelfForce {

SPECTRE_TEST_CASE(
    "Unit.Elliptic.Systems.ScalarSelfForce.Tags", "[Unit][Elliptic]") {
  TestHelpers::db::test_simple_tag<Tags::MMode>("MMode");
  TestHelpers::db::test_simple_tag<Tags::Alpha>("Alpha");
  TestHelpers::db::test_simple_tag<Tags::Beta>("Beta");
  TestHelpers::db::test_simple_tag<Tags::Gamma>("Gamma");
  TestHelpers::db::test_simple_tag<Tags::FieldIsRegularized>(
      "FieldIsRegularized");
  TestHelpers::db::test_simple_tag<Tags::SingularField>("SingularField");
  TestHelpers::db::test_simple_tag<Tags::BoyerLindquistRadius>(
      "BoyerLindquistRadius");
  TestHelpers::db::test_simple_tag<Tags::BoostFunction>("BoostFunction");
  TestHelpers::db::test_simple_tag<Tags::BoostFunctionDeriv>(
      "BoostFunctionDeriv");
}

}  // namespace ScalarSelfForce
