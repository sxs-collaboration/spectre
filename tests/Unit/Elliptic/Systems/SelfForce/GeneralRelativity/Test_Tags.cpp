// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Elliptic/Systems/SelfForce/GeneralRelativity/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace GrSelfForce {

SPECTRE_TEST_CASE("Unit.Elliptic.Systems.Xcts.Tags", "[Unit][Elliptic]") {
  TestHelpers::db::test_simple_tag<Tags::MMode>("MMode");
  TestHelpers::db::test_simple_tag<Tags::Alpha>("Alpha");
  TestHelpers::db::test_simple_tag<Tags::Beta>("Beta");
  TestHelpers::db::test_simple_tag<Tags::GammaRstar>("GammaRstar");
  TestHelpers::db::test_simple_tag<Tags::GammaTheta>("GammaTheta");
  TestHelpers::db::test_simple_tag<Tags::FieldIsRegularized>(
      "FieldIsRegularized");
  TestHelpers::db::test_simple_tag<Tags::SingularField>("SingularField");
  TestHelpers::db::test_simple_tag<Tags::BoyerLindquistRadius>(
      "BoyerLindquistRadius");
}

}  // namespace GrSelfForce
