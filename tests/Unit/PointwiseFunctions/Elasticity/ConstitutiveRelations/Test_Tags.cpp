// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Parallel/CreateFromOptions.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/Tags.hpp"

namespace Elasticity {
namespace ConstitutiveRelations {

namespace {
struct Provider {
  using constitutive_relation_type = IsotropicHomogeneous<3>;
  static constitutive_relation_type constitutive_relation() { return {1., 2.}; }
};

struct ProviderOptionTag {
  using type = Provider;
};

struct Metavariables {
  using constitutive_relation_provider_option_tag = ProviderOptionTag;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Elasticity.ConstitutiveRelations.Tags",
                  "[Unit][Elasticity]") {
  TestHelpers::db::test_base_tag<Elasticity::Tags::ConstitutiveRelationBase>(
      "ConstitutiveRelationBase");
  TestHelpers::db::test_simple_tag<
      Elasticity::Tags::ConstitutiveRelation<IsotropicHomogeneous<3>>>(
      "ConstitutiveRelation");
  {
    INFO("Constitutive relation from provider option");
    // Fake some output of option-parsing
    const tuples::TaggedTuple<ProviderOptionTag> options{Provider{}};
    // Dispatch the `create_from_options` function call that constructs the
    // constitutive relation from the options.
    const auto constructed_data = Parallel::create_from_options<Metavariables>(
        options,
        tmpl::list<
            Elasticity::Tags::ConstitutiveRelation<IsotropicHomogeneous<3>>>{});
    // Since the result is a tagged tuple we can't use base tags to retrieve it
    const auto& constructed_constitutive_relation = tuples::get<
        Elasticity::Tags::ConstitutiveRelation<IsotropicHomogeneous<3>>>(
        constructed_data);
    CHECK(constructed_constitutive_relation == IsotropicHomogeneous<3>{1., 2.});
  }
}

}  // namespace ConstitutiveRelations
}  // namespace Elasticity
