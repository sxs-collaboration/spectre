// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/Tags.hpp"

namespace Elasticity::ConstitutiveRelations {

SPECTRE_TEST_CASE("Unit.Elasticity.ConstitutiveRelations.Tags",
                  "[Unit][Elasticity]") {
  TestHelpers::db::test_simple_tag<Tags::ConstitutiveRelation<3>>(
      "ConstitutiveRelation");
}

}  // namespace Elasticity::ConstitutiveRelations
