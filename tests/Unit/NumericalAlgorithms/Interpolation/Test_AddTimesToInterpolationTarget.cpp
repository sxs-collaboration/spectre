// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <deque>
#include <memory>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Shell.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/UniformTranslation.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Interpolation/AddTimesToInterpolationTarget.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InitializeInterpolationTarget.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
// IWYU pragma: no_forward_declare db::DataBox

// IWYU pragma: no_include <boost/variant/get.hpp>

class DataVector;
namespace intrp::Tags {
struct IndicesOfFilledInterpPoints;
struct Times;
struct PendingTimes;
}  // namespace intrp::Tags
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel

namespace {

struct MockSendPointsToInterpolator {
  template <
      typename ParallelComponent, typename DbTags, typename Metavariables,
      typename ArrayIndex,
      Requires<tmpl::list_contains_v<DbTags, intrp::Tags::Times>> = nullptr>
  static void apply(db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const double time) noexcept {
    // Put something in IndicesOfFilledInterpPts so we can check later whether
    // this function was called.  This isn't the usual usage of
    // IndicesOfFilledInterpPoints.
    db::mutate<::intrp::Tags::IndicesOfFilledInterpPoints>(
        make_not_null(&box),
        [&time](const gsl::not_null<
                std::unordered_map<double, std::unordered_set<size_t>>*>
                    indices) noexcept {
          (*indices)[time].insert((*indices)[time].size() + 1);
        });
  }
};

template <typename Metavariables, typename InterpolationTargetTag>
struct mock_interpolation_target {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<3>>;
  using mutable_global_cache_tags =
      tmpl::conditional_t<metavariables::use_time_dependent_maps,
                          tmpl::list<domain::Tags::FunctionsOfTime>,
                          tmpl::list<>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<Actions::SetupDataBox,
                     intrp::Actions::InitializeInterpolationTarget<
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
  using frame = ::Frame::Inertial;
};

template <typename IsSequential, typename IsTimeDependent>
struct MockMetavariables {
  struct InterpolationTargetA {
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
    using compute_items_on_target = tmpl::list<>;
    using compute_target_points = MockComputeTargetPoints<IsSequential>;
  };
  static constexpr bool use_time_dependent_maps = IsTimeDependent::value;
  static constexpr size_t volume_dim = 3;

  using component_list = tmpl::list<
      mock_interpolation_target<MockMetavariables, InterpolationTargetA>>;
  enum class Phase { Initialization, Testing, Exit };
};

template <typename IsSequential>
void test_add_times() {
  using metavars = MockMetavariables<IsSequential, std::false_type>;
  using target_component =
      mock_interpolation_target<metavars,
                                typename metavars::InterpolationTargetA>;

  const auto domain_creator =
      domain::creators::Shell(0.9, 4.9, 1, {{5, 5}}, false);

  ActionTesting::MockRuntimeSystem<metavars> runner{
      {domain_creator.create_domain()}};
  ActionTesting::emplace_component<target_component>(&runner, 0);
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<target_component>(make_not_null(&runner), 0);
  }
  ActionTesting::set_phase(make_not_null(&runner), metavars::Phase::Testing);

  // Both PendingTimes and Times should be initially empty.
  CHECK(ActionTesting::get_databox_tag<target_component,
                                       ::intrp::Tags::PendingTimes>(runner, 0)
            .empty());
  CHECK(ActionTesting::get_databox_tag<target_component, ::intrp::Tags::Times>(
            runner, 0)
            .empty());

  const std::vector<double> times = {0.0, 1.0 / 3.0};

  ActionTesting::simple_action<target_component,
                               ::intrp::Actions::AddTimesToInterpolationTarget<
                                   typename metavars::InterpolationTargetA>>(
      make_not_null(&runner), 0, times);

  // Times should still be empty, but PendingTimes should
  // have been filled.
  CHECK(ActionTesting::get_databox_tag<target_component,
                                       ::intrp::Tags::PendingTimes>(
            runner, 0) == std::deque<double>(times.begin(), times.end()));
  CHECK(ActionTesting::get_databox_tag<target_component, ::intrp::Tags::Times>(
            runner, 0)
            .empty());

  // Should be only one queued simple action: VerifyTimesAndSendPoints
  CHECK(ActionTesting::number_of_queued_simple_actions<target_component>(
            runner, 0) == 1);

  // Add the same times again, which should do nothing...
  ActionTesting::simple_action<target_component,
                               ::intrp::Actions::AddTimesToInterpolationTarget<
                                   typename metavars::InterpolationTargetA>>(
      make_not_null(&runner), 0, times);
  // ...and check that it indeed did nothing.
  CHECK(ActionTesting::get_databox_tag<target_component,
                                       ::intrp::Tags::PendingTimes>(
            runner, 0) == std::deque<double>(times.begin(), times.end()));
  CHECK(ActionTesting::get_databox_tag<target_component, ::intrp::Tags::Times>(
            runner, 0)
            .empty());

  // At this point, there should be only one queued simple action,
  // VerifyTimesAndSendPoints, and triggering this action
  // should move the PendingTimes to Times and invoke
  // other simple actions.
  CHECK(ActionTesting::number_of_queued_simple_actions<target_component>(
            runner, 0) == 1);
  ActionTesting::invoke_queued_simple_action<target_component>(
      make_not_null(&runner), 0);

  // Now the PendingTimes should be empty, and the
  // Times should contain the two IDs.
  CHECK(ActionTesting::get_databox_tag<target_component, ::intrp::Tags::Times>(
            runner, 0) == std::deque<double>(times.begin(), times.end()));
  CHECK(ActionTesting::get_databox_tag<target_component,
                                       ::intrp::Tags::PendingTimes>(runner, 0)
            .empty());

  // Add the same times yet again, which should do nothing...
  ActionTesting::simple_action<target_component,
                               ::intrp::Actions::AddTimesToInterpolationTarget<
                                   typename metavars::InterpolationTargetA>>(
      make_not_null(&runner), 0, times);
  // ...and check that it indeed did nothing.
  CHECK(ActionTesting::get_databox_tag<target_component, ::intrp::Tags::Times>(
            runner, 0) == std::deque<double>(times.begin(), times.end()));
  CHECK(ActionTesting::get_databox_tag<target_component,
                                       ::intrp::Tags::PendingTimes>(runner, 0)
            .empty());

  if (IsSequential::value) {
    // Only one simple action should be queued, for the first time.
    CHECK(ActionTesting::number_of_queued_simple_actions<target_component>(
              runner, 0) == 1);
    ActionTesting::invoke_queued_simple_action<target_component>(
        make_not_null(&runner), 0);
  } else {
    // Two simple actions should be queued, one for each time.
    CHECK(ActionTesting::number_of_queued_simple_actions<target_component>(
              runner, 0) == 2);
    ActionTesting::invoke_queued_simple_action<target_component>(
        make_not_null(&runner), 0);
    ActionTesting::invoke_queued_simple_action<target_component>(
        make_not_null(&runner), 0);
  }

  // Check that MockSendPointsToInterpolator was called.
  CHECK(ActionTesting::get_databox_tag<
            target_component, ::intrp::Tags::IndicesOfFilledInterpPoints>(
            runner, 0)
            .at(times[0])
            .size() == 1);
  if (not IsSequential::value) {
    // MockSendPointsToInterpolator should have been called twice
    CHECK(ActionTesting::get_databox_tag<
              target_component, ::intrp::Tags::IndicesOfFilledInterpPoints>(
              runner, 0)
              .at(times[1])
              .size() == 1);
  }

  // Check that there are no queued simple actions.
  CHECK(
      ActionTesting::is_simple_action_queue_empty<target_component>(runner, 0));

  // Call again.
  // If sequential, it should not call MockSendPointsToInterpolator this time.
  // Otherwise it should call MockSendPointsToInterpolator twice.
  const std::vector<double> times_2 = {2.0 / 3.0, 1.0};
  ActionTesting::simple_action<target_component,
                               ::intrp::Actions::AddTimesToInterpolationTarget<
                                   typename metavars::InterpolationTargetA>>(
      make_not_null(&runner), 0, times_2);

  if (not IsSequential::value) {
    // For non-sequential, there should be only one queued simple action,
    // VerifyTimesAndSendPoints.
    CHECK(ActionTesting::number_of_queued_simple_actions<target_component>(
              runner, 0) == 1);
    ActionTesting::invoke_queued_simple_action<target_component>(
        make_not_null(&runner), 0);

    // Now there should be two queued_simple_actions,
    // each of which will call MockSendPointsToInterpolator for one of the
    // new times.
    for (size_t i = 0; i < 2; ++i) {
      ActionTesting::invoke_queued_simple_action<target_component>(
          make_not_null(&runner), 0);
      CHECK(ActionTesting::get_databox_tag<
                target_component, ::intrp::Tags::IndicesOfFilledInterpPoints>(
                runner, 0)
                .at(times_2[i])
                .size() == 1);
    }
  }

  // Check that there are no queued simple actions.
  CHECK(
      ActionTesting::is_simple_action_queue_empty<target_component>(runner, 0));
}

// For updating the expiration time in the FunctionOfTimes.
template <size_t Multiplier>
struct MyFunctionOfTimeUpdater {
  static void apply(gsl::not_null<typename domain::Tags::FunctionsOfTime::type*>
                        functions_of_time) {
    for (auto& name_and_function_of_time : *functions_of_time) {
      auto* function_of_time =
          dynamic_cast<domain::FunctionsOfTime::PiecewisePolynomial<2>*>(
              name_and_function_of_time.second.get());
      REQUIRE(function_of_time != nullptr);
      function_of_time->reset_expiration_time(0.5 * Multiplier);
    }
  }
};

template <typename IsSequential>
void test_add_times_time_dependent() {
  using metavars = MockMetavariables<IsSequential, std::true_type>;
  using target_component =
      mock_interpolation_target<metavars,
                                typename metavars::InterpolationTargetA>;

  const double expiration_time = 0.1;
  // Create a Domain with time-dependence. For this test we don't care
  // what the Domain actually is, we care only that it has time-dependence.
  const auto domain_creator = domain::creators::Brick(
      {{-1.2, 3.0, 2.5}}, {{0.8, 5.0, 3.0}}, {{1, 1, 1}}, {{5, 4, 3}},
      {{false, false, false}},
      std::make_unique<
          domain::creators::time_dependence::UniformTranslation<3>>(
          0.0, expiration_time, std::array<double, 3>({{0.1, 0.2, 0.3}})));

  ActionTesting::MockRuntimeSystem<metavars> runner{
      {domain_creator.create_domain()}, {domain_creator.functions_of_time()}};
  ActionTesting::emplace_component<target_component>(&runner, 0);
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<target_component>(make_not_null(&runner), 0);
  }
  ActionTesting::set_phase(make_not_null(&runner), metavars::Phase::Testing);

  // Both PendingTimes and Times should be initially empty.
  CHECK(ActionTesting::get_databox_tag<target_component,
                                       ::intrp::Tags::PendingTimes>(runner, 0)
            .empty());
  CHECK(ActionTesting::get_databox_tag<target_component, ::intrp::Tags::Times>(
            runner, 0)
            .empty());

  // Two of the times are before the expiration_time, the
  // others are afterwards.  Later we will update the FunctionOfTimes
  // so that the later times become valid.
  const std::vector<double> times = {0.0,       1.0 / 20.0, 1.0 / 4.0,
                                     1.0 / 3.0, 2.0 / 3.0,  3.0 / 4.0};

  ActionTesting::simple_action<target_component,
                               ::intrp::Actions::AddTimesToInterpolationTarget<
                                   typename metavars::InterpolationTargetA>>(
      make_not_null(&runner), 0, times);

  // Times should still be empty, but PendingTimes should
  // have been filled.
  CHECK(ActionTesting::get_databox_tag<target_component,
                                       ::intrp::Tags::PendingTimes>(
            runner, 0) == std::deque<double>(times.begin(), times.end()));
  CHECK(ActionTesting::get_databox_tag<target_component, ::intrp::Tags::Times>(
            runner, 0)
            .empty());

  // Should be only one queued simple action: VerifyTimesAndSendPoints
  CHECK(ActionTesting::number_of_queued_simple_actions<target_component>(
            runner, 0) == 1);

  // Add the same times again, which should do nothing...
  ActionTesting::simple_action<target_component,
                               ::intrp::Actions::AddTimesToInterpolationTarget<
                                   typename metavars::InterpolationTargetA>>(
      make_not_null(&runner), 0, times);
  // ...and check that it indeed did nothing.
  CHECK(ActionTesting::get_databox_tag<target_component,
                                       ::intrp::Tags::PendingTimes>(
            runner, 0) == std::deque<double>(times.begin(), times.end()));
  CHECK(ActionTesting::get_databox_tag<target_component, ::intrp::Tags::Times>(
            runner, 0)
            .empty());

  // At this point, there should still be only one queued simple
  // action, VerifyTimesAndSendPoints.
  CHECK(ActionTesting::number_of_queued_simple_actions<target_component>(
            runner, 0) == 1);

  // Now invoke VerifyTimesAndSendPoints.
  ActionTesting::invoke_queued_simple_action<target_component>(
      make_not_null(&runner), 0);

  // Two of the times are before the expiration_time, but the
  // other times are after the expiration_time.  So only the
  // first two times should have been moved from
  // PendingTimes to Times.
  CHECK(ActionTesting::get_databox_tag<target_component, ::intrp::Tags::Times>(
            runner, 0) == std::deque<double>({times[0], times[1]}));
  CHECK(
      ActionTesting::get_databox_tag<target_component,
                                     ::intrp::Tags::PendingTimes>(runner, 0) ==
      std::deque<double>({times[2], times[3], times[4], times[5]}));

  if (IsSequential::value) {
    // Only one simple action should be queued,
    // MockSendPointsToInterpolator for the first time.
    CHECK(ActionTesting::number_of_queued_simple_actions<target_component>(
              runner, 0) == 1);
  } else {
    // Three simple actions should be queued,
    // MockSendPointsToInterpolator for each of the first two
    // times, and VerifyTimesAndSendPoints because there
    // are remaining times that are still pending.
    CHECK(ActionTesting::number_of_queued_simple_actions<target_component>(
              runner, 0) == 3);
  }

  // Add the same times yet again, which should do nothing...
  ActionTesting::simple_action<target_component,
                               ::intrp::Actions::AddTimesToInterpolationTarget<
                                   typename metavars::InterpolationTargetA>>(
      make_not_null(&runner), 0, times);
  // ...and check that it indeed did nothing. That is, check that the
  // Times and PendingTimes and the number of queued actions are as
  // before.
  CHECK(ActionTesting::get_databox_tag<target_component, ::intrp::Tags::Times>(
            runner, 0) == std::deque<double>({times[0], times[1]}));
  CHECK(
      ActionTesting::get_databox_tag<target_component,
                                     ::intrp::Tags::PendingTimes>(runner, 0) ==
      std::deque<double>({times[2], times[3], times[4], times[5]}));
  if (IsSequential::value) {
    CHECK(ActionTesting::number_of_queued_simple_actions<target_component>(
              runner, 0) == 1);
  } else {
    CHECK(ActionTesting::number_of_queued_simple_actions<target_component>(
              runner, 0) == 3);
  }

  // Now call all of the queued actions, so that we are now waiting on
  // the FunctionOfTime to update.
  if (IsSequential::value) {
    ActionTesting::invoke_queued_simple_action<target_component>(
        make_not_null(&runner), 0);
  } else {
    ActionTesting::invoke_queued_simple_action<target_component>(
        make_not_null(&runner), 0);
    ActionTesting::invoke_queued_simple_action<target_component>(
        make_not_null(&runner), 0);
    ActionTesting::invoke_queued_simple_action<target_component>(
        make_not_null(&runner), 0);
  }

  // Check that MockSendPointsToInterpolator was called for the first
  // time.
  CHECK(ActionTesting::get_databox_tag<
            target_component, ::intrp::Tags::IndicesOfFilledInterpPoints>(
            runner, 0)
            .at(times[0])
            .size() == 1);
  if (not IsSequential::value) {
    // Check that MockSendPointsToInterpolator was called for the second
    // time, in the non-sequential case.
    CHECK(ActionTesting::get_databox_tag<
              target_component, ::intrp::Tags::IndicesOfFilledInterpPoints>(
              runner, 0)
              .at(times[1])
              .size() == 1);
  }

  // Check that there are no queued simple actions.
  CHECK(
      ActionTesting::is_simple_action_queue_empty<target_component>(runner, 0));

  // So now mutate the FunctionsOfTime.  For the non-sequential case,
  // there should be a callback waiting on a modification of the
  // FunctionOfTimes in the GlobalCache.  For the sequential case,
  // there should be no callback because the next interpolation is
  // started when the previous interpolation is finished
  // (and that code is not included in this test).
  auto& cache = ActionTesting::cache<target_component>(runner, 0_st);
  Parallel::mutate<domain::Tags::FunctionsOfTime, MyFunctionOfTimeUpdater<1>>(
      cache);

  if (IsSequential::value) {
    // Check that there are no queued simple actions.
    CHECK(ActionTesting::is_simple_action_queue_empty<target_component>(runner,
                                                                        0));
  } else {
    // The callback should have queued a single simple action,
    // VerifyTimesAndSendPoints.
    CHECK(ActionTesting::number_of_queued_simple_actions<target_component>(
              runner, 0) == 1);

    ActionTesting::invoke_queued_simple_action<target_component>(
        make_not_null(&runner), 0);

    // The first 4 times should now be in Times,
    // and the last 2 should be in PendingTimes.
    CHECK(
        ActionTesting::get_databox_tag<target_component, ::intrp::Tags::Times>(
            runner, 0) ==
        std::deque<double>({times[0], times[1], times[2], times[3]}));
    CHECK(ActionTesting::get_databox_tag<target_component,
                                         ::intrp::Tags::PendingTimes>(
              runner, 0) == std::deque<double>({times[4], times[5]}));

    // Now there should be three queued simple actions.  One is
    // VerifyTimesAndSendPoints, which was invoked by the last
    // invocation of VerifyTimesAndSendPoints because there are
    // still pending times.
    // The other two are MockSendPointsToInterpolator for each
    // of the newly-added times.
    CHECK(ActionTesting::number_of_queued_simple_actions<target_component>(
              runner, 0) == 3);

    // Invoke the first two simple actions (the ones that call
    // MockSendPointsToInterpolator), and verify that
    // MockSendPointsToInterpolator was indeed called for the
    // times.
    for (size_t i = 0; i < 2; ++i) {
      ActionTesting::invoke_queued_simple_action<target_component>(
          make_not_null(&runner), 0);
      CHECK(ActionTesting::get_databox_tag<
                target_component, ::intrp::Tags::IndicesOfFilledInterpPoints>(
                runner, 0)
                .at(times[i + 2])
                .size() == 1);
    }
  }

  // Earlier in this test we mutated the FunctionsOfTime when there were
  // no more simple_actions in the queue.  Now we mutate the
  // FunctionsOfTime while there is still (for the nonsequential case) a
  // VerifyTimesAndSendPoints queued.
  Parallel::mutate<domain::Tags::FunctionsOfTime, MyFunctionOfTimeUpdater<2>>(
      cache);

  if (IsSequential::value) {
    // Check that there are no queued simple actions.
    CHECK(ActionTesting::is_simple_action_queue_empty<target_component>(runner,
                                                                        0));
  } else {
    // There should still be a single simple action in the queue,
    // VerifyTimesAndSendPoints, because there was no callback.
    CHECK(ActionTesting::number_of_queued_simple_actions<target_component>(
              runner, 0) == 1);

    ActionTesting::invoke_queued_simple_action<target_component>(
        make_not_null(&runner), 0);

    // All times should now be in Times,
    // and none should be in PendingTimes.
    CHECK(
        ActionTesting::get_databox_tag<target_component, ::intrp::Tags::Times>(
            runner, 0) == std::deque<double>(times.begin(), times.end()));
    CHECK(ActionTesting::get_databox_tag<target_component,
                                         ::intrp::Tags::PendingTimes>(runner, 0)
              .empty());

    // Now there should be two queued simple actions, the
    // MockSendPointsToInterpolator ones.
    CHECK(ActionTesting::number_of_queued_simple_actions<target_component>(
              runner, 0) == 2);

    // Invoke the two simple actions, and verify that
    // MockSendPointsToInterpolator was indeed called for the
    // times.
    for (size_t i = 0; i < 2; ++i) {
      ActionTesting::invoke_queued_simple_action<target_component>(
          make_not_null(&runner), 0);
      CHECK(ActionTesting::get_databox_tag<
                target_component, ::intrp::Tags::IndicesOfFilledInterpPoints>(
                runner, 0)
                .at(times[i + 4])
                .size() == 1);
    }
  }

  // Check that there are no queued simple actions.
  CHECK(
      ActionTesting::is_simple_action_queue_empty<target_component>(runner, 0));
}

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.InterpolationTarget.AddTimes",
                  "[Unit]") {
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  domain::FunctionsOfTime::register_derived_with_charm();
  test_add_times<std::true_type>();
  test_add_times<std::false_type>();
  test_add_times_time_dependent<std::true_type>();
  test_add_times_time_dependent<std::false_type>();
}
}  // namespace
