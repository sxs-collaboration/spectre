// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <deque>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Shell.hpp"
#include "Domain/Domain.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

/// \cond
class DataVector;
namespace Tags {
template <size_t Dim, typename Frame>
struct Domain;
}  // namespace Tags
/// \endcond

namespace {

template <typename Metavariables, typename InterpolationTargetTag>
struct mock_interpolation_target {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tag_list = tmpl::list<>;
  using action_list = tmpl::list<>;
  using initial_databox = db::compute_databox_type<
      typename ::intrp::Actions::InitializeInterpolationTarget<
          InterpolationTargetTag>::template return_tag_list<Metavariables>>;
};

struct MockMetavariables {
  struct InterpolationTargetA {
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
  };
  using temporal_id = ::Tags::TimeId;
  using domain_frame = Frame::Inertial;
  static constexpr size_t domain_dim = 3;

  using component_list = tmpl::list<
      mock_interpolation_target<MockMetavariables, InterpolationTargetA>>;
  using const_global_cache_tag_list = tmpl::list<>;
  enum class Phase { Initialize, Exit };
};

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.InterpolationTarget.Initialize",
                  "[Unit]") {
  using metavars = MockMetavariables;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  using TupleOfMockDistributedObjects =
      MockRuntimeSystem::TupleOfMockDistributedObjects;
  TupleOfMockDistributedObjects dist_objects{};
  using MockDistributedObjectsTag =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          mock_interpolation_target<metavars, metavars::InterpolationTargetA>>;
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(0,
               ActionTesting::MockDistributedObject<mock_interpolation_target<
                   metavars, metavars::InterpolationTargetA>>{});
  MockRuntimeSystem runner{{}, std::move(dist_objects)};

  const auto domain_creator =
      domain::creators::Shell<Frame::Inertial>(0.9, 4.9, 1, {{5, 5}}, false);

  runner.simple_action<
      mock_interpolation_target<metavars, metavars::InterpolationTargetA>,
      ::intrp::Actions::InitializeInterpolationTarget<
          metavars::InterpolationTargetA>>(0, domain_creator.create_domain());

  const auto& box =
      runner
          .template algorithms<mock_interpolation_target<
              metavars, metavars::InterpolationTargetA>>()
          .at(0)
          .template get_databox<typename mock_interpolation_target<
              metavars, metavars::InterpolationTargetA>::initial_databox>();

  CHECK(db::get<::intrp::Tags::IndicesOfFilledInterpPoints>(box).empty());
  CHECK(db::get<::intrp::Tags::TemporalIds<metavars>>(box).empty());

  CHECK(db::get<::Tags::Domain<3, Frame::Inertial>>(box) ==
        domain_creator.create_domain());

  const auto test_vars = db::item_type<
      ::Tags::Variables<tmpl::list<gr::Tags::Lapse<DataVector>>>>{};
  CHECK(db::get<::Tags::Variables<typename metavars::InterpolationTargetA::
                                      vars_to_interpolate_to_target>>(box) ==
        test_vars);

  CHECK(::intrp::Tags::IndicesOfFilledInterpPoints::name() ==
        "IndicesOfFilledInterpPoints");
  CHECK(::intrp::Tags::TemporalIds<metavars>::name() == "TemporalIds");
}

}  // namespace
