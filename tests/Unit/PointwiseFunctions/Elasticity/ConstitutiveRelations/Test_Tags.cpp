// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/Tags.hpp"

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
