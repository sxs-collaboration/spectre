// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Parallel/CreateFromOptions.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/Tags.hpp"

namespace Elasticity::ConstitutiveRelations {

namespace {
struct Provider {
  const IsotropicHomogeneous<3>& constitutive_relation() const {
    return constitutive_relation_;
  }

 private:
  IsotropicHomogeneous<3> constitutive_relation_{1., 2.};
};

struct ProviderTag : db::SimpleTag {
  using type = Provider;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Elasticity.ConstitutiveRelations.Tags",
                  "[Unit][Elasticity]") {
  TestHelpers::db::test_simple_tag<Tags::ConstitutiveRelation<3>>(
      "ConstitutiveRelation");
  CHECK(db::tag_name<Tags::ConstitutiveRelation<3>>() ==
        "ConstitutiveRelation");
  const auto box = db::create<
      db::AddSimpleTags<ProviderTag>,
      db::AddComputeTags<Tags::ConstitutiveRelationReference<3, ProviderTag>>>(
      Provider{});
  CHECK(&db::get<Tags::ConstitutiveRelation<3>>(box) ==
        &db::get<ProviderTag>(box).constitutive_relation());
}

}  // namespace Elasticity::ConstitutiveRelations
