// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/Tags.hpp"
#include "tests/Unit/DataStructures/DataBox/TestHelpers.hpp"

namespace {
struct SomeConstitutiveRelation;
}  // namespace

SPECTRE_TEST_CASE("Unit.Elasticity.Tags", "[Unit][Elasticity]") {
  TestHelpers::db::test_base_tag<Elasticity::Tags::ConstitutiveRelationBase>(
      "ConstitutiveRelationBase");
  TestHelpers::db::test_simple_tag<
      Elasticity::Tags::ConstitutiveRelation<SomeConstitutiveRelation>>(
      "Material");
}
