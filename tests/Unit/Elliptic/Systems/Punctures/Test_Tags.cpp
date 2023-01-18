// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Elliptic/Systems/Punctures/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace Punctures {

SPECTRE_TEST_CASE("Unit.Elliptic.Systems.Punctures.Tags", "[Unit][Elliptic]") {
  TestHelpers::db::test_simple_tag<Tags::Field>("Field");
  TestHelpers::db::test_simple_tag<Tags::Alpha>("Alpha");
  TestHelpers::db::test_simple_tag<Tags::Beta>("Beta");
  TestHelpers::db::test_simple_tag<Tags::TracelessConformalExtrinsicCurvature>(
      "TracelessConformalExtrinsicCurvature");
}

}  // namespace Punctures
