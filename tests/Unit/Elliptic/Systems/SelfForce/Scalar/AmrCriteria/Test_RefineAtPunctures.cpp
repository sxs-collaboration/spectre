// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/AmrCriteria/RefineAtPuncture.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/AnalyticData/CircularOrbit.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Tags/Criteria.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarSelfForce::AmrCriteria {

namespace {

struct Metavariables {
  static constexpr size_t volume_dim = 2;
  using component_list = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<amr::Criterion,
                   tmpl::list<ScalarSelfForce::AmrCriteria::RefineAtPuncture>>>;
  };
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
struct OtherBackground : elliptic::analytic_data::Background,
                         elliptic::analytic_data::InitialGuess {
  OtherBackground() = default;
  explicit OtherBackground(CkMigrateMessage* /*m*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(OtherBackground);
};

PUP::able::PUP_ID OtherBackground::my_PUP_ID = 0;  // NOLINT
#pragma GCC diagnostic pop

void test_criterion(
    std::unique_ptr<elliptic::analytic_data::Background> background) {
  const auto created =
      TestHelpers::test_creation<std::unique_ptr<amr::Criterion>,
                                 Metavariables>("RefineAtPuncture");
  REQUIRE(dynamic_cast<const RefineAtPuncture*>(created.get()) != nullptr);
  const auto& criterion = serialize_and_deserialize(
      dynamic_cast<const RefineAtPuncture&>(*created));

  {
    INFO("Evaluate");
    using background_tag =
        elliptic::Tags::Background<elliptic::analytic_data::Background>;

    const domain::creators::Rectangle domain_creator{
        {{10., 0.}}, {{15., 1.}}, {{2, 2}}, {{3, 3}}, {{false, false}}};
    auto databox =
        db::create<tmpl::list<background_tag, domain::Tags::Domain<2>>>(
            std::move(background), domain_creator.create_domain());
    const ObservationBox<
        tmpl::list<>,
        db::DataBox<tmpl::list<background_tag, domain::Tags::Domain<2>>>>
        box{make_not_null(&databox)};
    Parallel::GlobalCache<Metavariables> empty_cache{};

    {
      INFO("Element with puncture within");
      const ElementId<2> element_id{0, {{{2, 2}, {2, 0}}}};
      const auto expected_flags = make_array<2>(amr::Flag::Split);
      auto flags = criterion.evaluate(box, empty_cache, element_id);
      CHECK(flags == expected_flags);
    }
    {
      INFO("Elements without puncture");
      const auto expected_flags = make_array<2>(amr::Flag::DoNothing);
      for (const auto& element_id : {ElementId<2>{0, {{{2, 1}, {2, 0}}}},
                                     ElementId<2>{0, {{{2, 3}, {2, 3}}}},
                                     ElementId<2>{0, {{{2, 3}, {2, 0}}}},
                                     ElementId<2>{0, {{{2, 0}, {2, 3}}}}}) {
        auto flags = criterion.evaluate(box, empty_cache, element_id);
        CHECK(flags == expected_flags);
      }
    }
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.ScalarSelfForce.AmrCriteria.RefineAtPuncture",
                  "[Unit][ParallelAlgorithms]") {
  register_factory_classes_with_charm<Metavariables>();
  test_criterion(std::make_unique<ScalarSelfForce::AnalyticData::CircularOrbit>(
      // Only orbital radius is relevant for the test
      1.0, 0.5, /* orbital radius */ 10.0, 2, std::nullopt, false));
  CHECK_THROWS_WITH(test_criterion(std::make_unique<OtherBackground>()),
                    Catch::Matchers::ContainsSubstring(
                        "RefineAtPuncture only works with 'CircularOrbit'."));
}

}  // namespace ScalarSelfForce::AmrCriteria
