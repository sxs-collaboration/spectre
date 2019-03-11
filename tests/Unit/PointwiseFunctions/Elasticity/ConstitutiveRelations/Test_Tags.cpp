// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/Tags.hpp"

namespace {
struct SomeConstitutiveRelation;
}  // namespace

SPECTRE_TEST_CASE("Unit.Elasticity.Tags", "[Unit][Elasticity]") {
  CHECK(Elasticity::Tags::ConstitutiveRelation<
            SomeConstitutiveRelation>::name() == "Material");
}
