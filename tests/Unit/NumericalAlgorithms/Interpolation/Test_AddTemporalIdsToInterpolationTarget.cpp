// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <deque>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/Creators/Shell.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Interpolation/AddTemporalIdsToInterpolationTarget.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InitializeInterpolationTarget.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Rational.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
// IWYU pragma: no_forward_declare db::DataBox

// IWYU pragma: no_include <boost/variant/get.hpp>

/// \cond
class DataVector;
namespace intrp::Tags {
template <typename TemporalId>
struct IndicesOfFilledInterpPoints;
template <typename TemporalId>
struct TemporalIds;
}  // namespace intrp::Tags
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
/// \endcond

namespace {

struct MockSendPointsToInterpolator {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex, typename TemporalId,
            Requires<tmpl::list_contains_v<
                DbTags, intrp::Tags::TemporalIds<TemporalId>>> = nullptr>
  static void apply(db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const TemporalId& temporal_id) noexcept {
    // Put something in IndicesOfFilledInterpPts so we can check later whether
    // this function was called.  This isn't the usual usage of
    // IndicesOfFilledInterpPoints.
    db::mutate<::intrp::Tags::IndicesOfFilledInterpPoints<TemporalId>>(
        make_not_null(&box),
        [&temporal_id](
            const gsl::not_null<std::unordered_map<TemporalId,
                                                   std::unordered_set<size_t>>*>
                indices) noexcept {
          (*indices)[temporal_id].insert((*indices)[temporal_id].size() + 1);
        });
  }
};

template <typename Metavariables, typename InterpolationTargetTag>
struct mock_interpolation_target {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<3>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<intrp::Actions::InitializeInterpolationTarget<
              Metavariables, InterpolationTargetTag>>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing, tmpl::list<>>>;
  using replace_these_simple_actions = tmpl::list<
      intrp::Actions::SendPointsToInterpolator<InterpolationTargetTag>>;
  using with_these_simple_actions = tmpl::list<MockSendPointsToInterpolator>;
};

template <typename IsSequential>
struct MockComputeTargetPoints {
  using is_sequential = IsSequential;
};

template <typename IsSequential>
struct MockMetavariables {
  struct InterpolationTargetA {
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
    using compute_items_on_target = tmpl::list<>;
    using compute_target_points = MockComputeTargetPoints<IsSequential>;
  };
  using temporal_id = ::Tags::TimeStepId;

  using component_list = tmpl::list<
      mock_interpolation_target<MockMetavariables, InterpolationTargetA>>;
  enum class Phase { Initialization, Testing, Exit };
};

template <typename IsSequential>
void test_add_temporal_ids() {
  using metavars = MockMetavariables<IsSequential>;
  using temporal_id_type = typename metavars::temporal_id::type;
  using target_component =
      mock_interpolation_target<metavars,
                                typename metavars::InterpolationTargetA>;

  const auto domain_creator =
      domain::creators::Shell(0.9, 4.9, 1, {{5, 5}}, false);

  ActionTesting::MockRuntimeSystem<metavars> runner{
      {domain_creator.create_domain()}};
  ActionTesting::emplace_component<target_component>(&runner, 0);
  ActionTesting::next_action<target_component>(make_not_null(&runner), 0);
  ActionTesting::set_phase(make_not_null(&runner), metavars::Phase::Testing);

  CHECK(ActionTesting::get_databox_tag<
            target_component, ::intrp::Tags::TemporalIds<temporal_id_type>>(
            runner, 0)
            .empty());

  Slab slab(0.0, 1.0);
  const std::vector<TimeStepId> temporal_ids = {
      TimeStepId(true, 0, Time(slab, 0)),
      TimeStepId(true, 0, Time(slab, Rational(1, 3)))};

  runner.template simple_action<
      target_component, ::intrp::Actions::AddTemporalIdsToInterpolationTarget<
                            typename metavars::InterpolationTargetA>>(
      0, temporal_ids);

  CHECK(ActionTesting::get_databox_tag<
            target_component, ::intrp::Tags::TemporalIds<temporal_id_type>>(
            runner, 0) ==
        std::deque<TimeStepId>(temporal_ids.begin(), temporal_ids.end()));

  // Add the same temporal_ids again, which should do nothing...
  runner.template simple_action<
      target_component, ::intrp::Actions::AddTemporalIdsToInterpolationTarget<
                            typename metavars::InterpolationTargetA>>(
      0, temporal_ids);
  // ...and check that it indeed did nothing.
  CHECK(ActionTesting::get_databox_tag<
            target_component, ::intrp::Tags::TemporalIds<temporal_id_type>>(
            runner, 0) ==
        std::deque<TimeStepId>(temporal_ids.begin(), temporal_ids.end()));

  if (IsSequential::value) {
    // Only one simple action should be queued, for the first temporal_id.
    runner.template invoke_queued_simple_action<target_component>(0);
  } else {
    // Two simple actions should be queued, one for each temporal_id.
    runner.template invoke_queued_simple_action<target_component>(0);
    runner.template invoke_queued_simple_action<target_component>(0);
  }

  // Check that MockSendPointsToInterpolator was called.
  CHECK(ActionTesting::get_databox_tag<
            target_component,
            ::intrp::Tags::IndicesOfFilledInterpPoints<temporal_id_type>>(
            runner, 0)
            .at(temporal_ids[0])
            .size() == 1);
  if (not IsSequential::value) {
    // MockSendPointsToInterpolator should have been called twice
    CHECK(ActionTesting::get_databox_tag<
              target_component,
              ::intrp::Tags::IndicesOfFilledInterpPoints<temporal_id_type>>(
              runner, 0)
              .at(temporal_ids[1])
              .size() == 1);
  }

  // Call again.
  // If sequential, it should not call MockSendPointsToInterpolator this time.
  // Otherwise it should call MockSendPointsToInterpolator twice.
  const std::vector<TimeStepId> temporal_ids_2 = {
      TimeStepId(true, 0, Time(slab, Rational(2, 3))),
      TimeStepId(true, 0, Time(slab, Rational(3, 3)))};
  runner.template simple_action<
      target_component, ::intrp::Actions::AddTemporalIdsToInterpolationTarget<
                            typename metavars::InterpolationTargetA>>(
      0, temporal_ids_2);

  if (not IsSequential::value) {
    // For non-sequential, there should be two queued_simple_actions,
    // each of which will call MockSendPointsToInterpolator for one of the
    // new temporal_ids.
    for (size_t i = 0; i < 2; ++i) {
      runner.template invoke_queued_simple_action<target_component>(0);
      CHECK(ActionTesting::get_databox_tag<
                target_component,
                ::intrp::Tags::IndicesOfFilledInterpPoints<temporal_id_type>>(
                runner, 0)
                .at(temporal_ids_2[i])
                .size() == 1);
    }
  }

  // Check that there are no queued simple actions.
  CHECK(runner.template is_simple_action_queue_empty<target_component>(0));
}
SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.InterpolationTarget.AddTemporalIds",
                  "[Unit]") {
  test_add_temporal_ids<std::true_type>();
  test_add_temporal_ids<std::false_type>();
}
}  // namespace
