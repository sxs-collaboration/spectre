// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/AmrCriteria/RefineAtBoundary.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criterion.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace GrSelfForce::AmrCriteria {

namespace {

struct Metavariables {
  static constexpr size_t volume_dim = 2;
  using component_list = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<amr::Criterion, tmpl::list<RefineAtBoundary<2, 1>>>>;
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
    const ElementId<2> element_id{0};
    auto databox = db::create<tmpl::list<domain::Tags::Element<2>>>(
            Element<2>{element_id, DirectionMap<2, Neighbors<2>>{}});
    const ObservationBox<tmpl::list<>,
                         db::DataBox<tmpl::list<domain::Tags::Element<2>>>>
        box{make_not_null(&databox)};

    Parallel::GlobalCache<Metavariables> empty_cache{};

    {
      INFO("Element with boundary");
      const auto expected_flags =
          std::array<amr::Flag, 2>{{amr::Flag::DoNothing, amr::Flag::Split}};
      auto flags = criterion.evaluate(box, empty_cache, element_id);
      CHECK(flags == expected_flags);
    }
    {
      INFO("Element without boundary");
      DirectionMap<2, Neighbors<2>> neighbors{};
      const ElementId<2> neighbor_id{0};
      const OrientationMap<2> identity_orientation{std::array<Direction<2>, 2>{
          {Direction<2>{0, Side::Upper}, Direction<2>{1, Side::Upper}}}};

      neighbors.emplace(Direction<2>::lower_eta(),
                        Neighbors<2>{{neighbor_id}, identity_orientation});
      neighbors.emplace(Direction<2>::upper_eta(),
                        Neighbors<2>{{neighbor_id}, identity_orientation});
auto internal_databox = db::create<tmpl::list<domain::Tags::Element<2>>>(
          Element<2>{element_id, std::move(neighbors)});
const ObservationBox<tmpl::list<>,
                           db::DataBox<tmpl::list<domain::Tags::Element<2>>>>
          internal_box{make_not_null(&internal_databox)};
      const auto expected_internal_flags = make_array<2>(amr::Flag::DoNothing);
      CHECK(criterion.evaluate(internal_box, empty_cache, element_id) ==
            expected_internal_flags);
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
