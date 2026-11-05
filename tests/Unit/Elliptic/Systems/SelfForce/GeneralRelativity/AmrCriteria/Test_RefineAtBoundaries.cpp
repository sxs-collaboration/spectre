// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/AmrCriteria/RefineAtBoundary.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criterion.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace GrSelfForce::AmrCriteria {

namespace {

struct Metavariables {
  static constexpr size_t volume_dim = 2;
  using component_list = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        amr::Criterion,
        tmpl::list<GrSelfForce::AmrCriteria::RefineAtBoundary<2, 1>>>>;
  };
};

void test_criterion() {
  const auto created =
      TestHelpers::test_creation<std::unique_ptr<amr::Criterion>,
                                 Metavariables>("RefineAtBoundaryY");
  REQUIRE(dynamic_cast<const RefineAtBoundary<2, 1>*>(created.get()) !=
          nullptr);
  const auto& criterion = serialize_and_deserialize(
      dynamic_cast<const RefineAtBoundary<2, 1>&>(*created));

  {
    INFO("Evaluate");
    // Domain in (r_*, theta) coordinates. Refinement level 2 gives 4 segments
    // per axis; y-indices 0 and 3 are at the angular boundaries theta~0, pi.
    const domain::creators::Rectangle domain_creator{
        {{10., 0.}}, {{15., 2.}}, {{2, 2}}, {{3, 3}}, {{false, false}}};
    const auto domain = domain_creator.create_domain();
    const std::vector<std::array<size_t, 2>> refinement_levels{{2, 2}};
    Parallel::GlobalCache<Metavariables> empty_cache{};

    const auto test_element = [&](const ElementId<2>& element_id,
                                  const std::array<amr::Flag, 2>& expected) {
      auto element = domain::create_initial_element(element_id, domain.blocks(),
                                                    refinement_levels);
      auto databox =
          db::create<tmpl::list<domain::Tags::Element<2>>>(std::move(element));
      const ObservationBox<tmpl::list<>, decltype(databox)> box{
          make_not_null(&databox)};
      CHECK(criterion.evaluate(box, empty_cache, element_id) == expected);
    };

    // RefineAtBoundary<2,1> splits only in y; x is always DoNothing.
    auto split_y = make_array<2>(amr::Flag::DoNothing);
    split_y[1] = amr::Flag::Split;
    const auto do_nothing = make_array<2>(amr::Flag::DoNothing);

    {
      INFO("Elements at angular boundary");
      // y-index 0: lower y-boundary (theta ~ 0)
      // y-index 3: upper y-boundary (theta ~ pi)
      for (const auto& element_id : {ElementId<2>{0, {{{2, 0}, {2, 0}}}},
                                     ElementId<2>{0, {{{2, 2}, {2, 0}}}},
                                     ElementId<2>{0, {{{2, 0}, {2, 3}}}},
                                     ElementId<2>{0, {{{2, 2}, {2, 3}}}}}) {
        test_element(element_id, split_y);
      }
    }
    {
      INFO("Elements away from angular boundary");
      for (const auto& element_id : {ElementId<2>{0, {{{2, 0}, {2, 1}}}},
                                     ElementId<2>{0, {{{2, 2}, {2, 2}}}}}) {
        test_element(element_id, do_nothing);
      }
    }
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.GrSelfForce.AmrCriteria.RefineAtBoundary",
                  "[Unit][ParallelAlgorithms]") {
  register_factory_classes_with_charm<Metavariables>();
  test_criterion();
}

}  // namespace GrSelfForce::AmrCriteria
