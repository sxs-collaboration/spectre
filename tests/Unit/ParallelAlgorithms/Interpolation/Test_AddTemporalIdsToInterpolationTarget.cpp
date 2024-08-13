// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <deque>
#include <memory>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "ControlSystem/UpdateFunctionOfTime.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/UniformTranslation.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/ParallelAlgorithms/Interpolation/InterpolationTargetTestHelpers.hpp"
#include "IO/Observer/Initialize.hpp"
#include "IO/Observer/Tags.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/AddTemporalIdsToInterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolationTargetReceiveVars.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorReceivePoints.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/VerifyTemporalIdsAndSendPoints.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolatedVars.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/ComputeTargetPoints.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Rational.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

class DataVector;
namespace intrp::Tags {
template <typename TemporalId>
struct IndicesOfFilledInterpPoints;
template <typename TemporalId>
struct TemporalIds;
template <typename TemporalId>
struct PendingTemporalIds;
}  // namespace intrp::Tags
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel

namespace {
template <typename Metavariables>
struct mock_observer_writer {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockNodeGroupChare;
  using array_index = size_t;
  using component_being_mocked = observers::ObserverWriter<Metavariables>;
  using const_global_cache_tags =
      tmpl::list<observers::Tags::VolumeFileName,
                 observers::Tags::ReductionFileName>;

  using simple_tags =
      typename observers::Actions::InitializeWriter<Metavariables>::simple_tags;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<observers::Actions::InitializeWriter<Metavariables>>>>;
};

template <typename Metavariables>
struct mock_interpolator {
  using TargetTag = typename Metavariables::InterpolationTargetA;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<intrp::Actions::InitializeInterpolator<
              intrp::Tags::VolumeVarsInfo<Metavariables,
                                          typename TargetTag::temporal_id>,
              intrp::Tags::InterpolatedVarsHolders<Metavariables>>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<>>>;
  using replace_these_simple_actions =
      tmpl::list<intrp::Actions::ReceivePoints<TargetTag>>;
  using with_these_simple_actions =
      tmpl::list<InterpTargetTestHelpers::MockReceivePoints<TargetTag>>;

  using component_being_mocked = intrp::Interpolator<Metavariables>;
};

template <typename Metavariables, typename InterpolationTargetTag>
struct mock_interpolation_target {
  static_assert(
      tt::assert_conforms_to_v<InterpolationTargetTag,
                               intrp::protocols::InterpolationTargetTag>);
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<3>>;
  using mutable_global_cache_tags =
      tmpl::conditional_t<metavariables::use_time_dependent_maps,
                          tmpl::list<domain::Tags::FunctionsOfTimeInitialize>,
                          tmpl::list<>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<intrp::Actions::InitializeInterpolationTarget<
              Metavariables, InterpolationTargetTag>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<>>>;

  using component_being_mocked =
      intrp::InterpolationTarget<Metavariables, InterpolationTargetTag>;
};

struct MockComputeTargetPoints
    : tt::ConformsTo<intrp::protocols::ComputeTargetPoints> {
  using is_sequential = std::true_type;
  using frame = ::Frame::Inertial;
  template <typename Metavariables, typename DbTags, typename TemporalId>
  static tnsr::I<DataVector, 3, Frame::Inertial> points(
      const db::DataBox<DbTags>& /*box*/,
      const tmpl::type_<Metavariables>& /*meta*/,
      const TemporalId& /*temporal_id*/) {
    return tnsr::I<DataVector, 3, Frame::Inertial>{};
  }
};

template <typename IsTimeDependent>
struct MockMetavariables {
  struct InterpolationTargetA
      : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::TimeStepId;
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
    using compute_items_on_target = tmpl::list<>;
    using post_interpolation_callbacks =
        tmpl::list<intrp::callbacks::ObserveTimeSeriesOnSurface<
            tmpl::list<>, InterpolationTargetA>>;
    using compute_target_points = MockComputeTargetPoints;
  };
  static constexpr bool use_time_dependent_maps = IsTimeDependent::value;
  static constexpr size_t volume_dim = 3;

  using interpolation_target_tags = tmpl::list<InterpolationTargetA>;
  using interpolator_source_vars = tmpl::list<>;
  using observed_reduction_data_tags = tmpl::list<>;

  using component_list = tmpl::list<
      mock_interpolation_target<MockMetavariables, InterpolationTargetA>,
      mock_interpolator<MockMetavariables>,
      mock_observer_writer<MockMetavariables>>;
};

void test_add_temporal_ids() {
  using metavars = MockMetavariables<std::false_type>;
  using temporal_id_type =
      typename metavars::InterpolationTargetA::temporal_id::type;
  using interpolator_component = mock_interpolator<metavars>;
  using target_component =
      mock_interpolation_target<metavars,
                                typename metavars::InterpolationTargetA>;

  const auto domain_creator = domain::creators::Sphere(
      0.9, 4.9, domain::creators::Sphere::Excision{}, 1_st, 5_st, false);

  ActionTesting::MockRuntimeSystem<metavars> runner{
      {domain_creator.create_domain(), "UnusedVolumeFileName",
       "UnusedReductionFilename"}};
  ActionTesting::emplace_component<target_component>(&runner, 0);
  for (int i = 0; i < 2; ++i) {
    ActionTesting::next_action<target_component>(make_not_null(&runner), 0);
  }
  ActionTesting::emplace_array_component<interpolator_component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0}, 0);
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  auto& target_box =
      ActionTesting::get_databox<target_component>(make_not_null(&runner), 0);

  // Both PendingTemporalIds and TemporalIds should be initially empty.
  CHECK(ActionTesting::get_databox_tag<
            target_component,
            ::intrp::Tags::PendingTemporalIds<temporal_id_type>>(runner, 0)
            .empty());
  CHECK(ActionTesting::get_databox_tag<
            target_component, ::intrp::Tags::TemporalIds<temporal_id_type>>(
            runner, 0)
            .empty());

  const Slab slab(0.0, 1.0);
  const TimeStepId temporal_id_1{true, 0, Time(slab, 0)};
  const TimeStepId temporal_id_2{true, 0, Time(slab, Rational(1, 3))};
  const std::deque<TimeStepId> deque_of_ids{temporal_id_1, temporal_id_2};

  const auto add_id_to_target = [&](const TimeStepId& id) {
    ActionTesting::simple_action<
        target_component, ::intrp::Actions::AddTemporalIdsToInterpolationTarget<
                              typename metavars::InterpolationTargetA>>(
        make_not_null(&runner), 0, id);
  };

  add_id_to_target(temporal_id_1);

  // Because of sequential calls to other actions within
  // AddTemporalIdsToInterpolationTarget, temporal_id_1 should have been placed
  // in PendingTemporalIds, moved to TemporalIds, and removed from
  // PendingTemporalIds
  CHECK(ActionTesting::get_databox_tag<
            target_component,
            ::intrp::Tags::PendingTemporalIds<temporal_id_type>>(runner, 0)
            .empty());
  CHECK(ActionTesting::get_databox_tag<
            target_component, ::intrp::Tags::TemporalIds<temporal_id_type>>(
            runner, 0) == std::deque<TimeStepId>{temporal_id_1});

  // Add the same temporal_id again, which should do nothing...
  add_id_to_target(temporal_id_1);

  // ...and check that it indeed did nothing.
  CHECK(ActionTesting::get_databox_tag<
            target_component,
            ::intrp::Tags::PendingTemporalIds<temporal_id_type>>(runner, 0)
            .empty());
  CHECK(ActionTesting::get_databox_tag<
            target_component, ::intrp::Tags::TemporalIds<temporal_id_type>>(
            runner, 0) == std::deque<TimeStepId>{temporal_id_1});

  // Send the next temporal id
  add_id_to_target(temporal_id_2);

  // temporal_id_2 should be pending because temporal_id_1 hasn't finished.
  // Otherwise, both should be in TemporalIds
  CHECK(ActionTesting::get_databox_tag<
            target_component,
            ::intrp::Tags::PendingTemporalIds<temporal_id_type>>(runner, 0) ==
        std::deque<TimeStepId>{temporal_id_2});
  CHECK(ActionTesting::get_databox_tag<
            target_component, ::intrp::Tags::TemporalIds<temporal_id_type>>(
            runner, 0) == std::deque<TimeStepId>{temporal_id_1});

  // Should be one queued simple action on the Interpolator (ReceivePoints)
  // because one id is still pending
  CHECK(ActionTesting::number_of_queued_simple_actions<interpolator_component>(
            runner, 0) == 1);
  ActionTesting::invoke_queued_simple_action<interpolator_component>(
      make_not_null(&runner), 0);

  // Add the same temporal_id yet again, which should do nothing...
  add_id_to_target(temporal_id_2);
  // ...and check that it indeed did nothing.
  CHECK(ActionTesting::get_databox_tag<
            target_component,
            ::intrp::Tags::PendingTemporalIds<temporal_id_type>>(runner, 0) ==
        std::deque<TimeStepId>{temporal_id_2});
  CHECK(ActionTesting::get_databox_tag<
            target_component, ::intrp::Tags::TemporalIds<temporal_id_type>>(
            runner, 0) == std::deque<TimeStepId>{temporal_id_1});

  // Move both ids to completed
  db::mutate<::intrp::Tags::TemporalIds<TimeStepId>,
             ::intrp::Tags::CompletedTemporalIds<TimeStepId>,
             ::intrp::Tags::PendingTemporalIds<TimeStepId>>(
      [&](const gsl::not_null<std::deque<TimeStepId>*> ids,
          const gsl::not_null<std::deque<TimeStepId>*> completed_ids,
          const gsl::not_null<std::deque<TimeStepId>*> pending_ids) {
        completed_ids->emplace_back(temporal_id_1);
        completed_ids->emplace_back(temporal_id_2);
        pending_ids->clear();
        ids->clear();
      },
      make_not_null(&target_box));

  // Check that there are no queued simple actions.
  CHECK(
      ActionTesting::is_simple_action_queue_empty<target_component>(runner, 0));

  // Call with some more out of order
  const TimeStepId temporal_id_3{true, 0, Time(slab, Rational(2, 3))};
  const TimeStepId temporal_id_4{true, 0, Time(slab, Rational(3, 3))};
  add_id_to_target(temporal_id_4);
  add_id_to_target(temporal_id_3);

  CHECK(ActionTesting::get_databox_tag<
            target_component,
            ::intrp::Tags::PendingTemporalIds<temporal_id_type>>(runner, 0) ==
        std::deque<TimeStepId>{temporal_id_3});
  CHECK(ActionTesting::get_databox_tag<
            target_component, ::intrp::Tags::TemporalIds<temporal_id_type>>(
            runner, 0) == std::deque<TimeStepId>{temporal_id_4});

  // For temporal_id_3
  CHECK(ActionTesting::number_of_queued_simple_actions<interpolator_component>(
            runner, 0) == 1);
}

void test_add_temporal_ids_time_dependent() {
  using metavars = MockMetavariables<std::true_type>;
  using interpolator_component = mock_interpolator<metavars>;
  using target_tag = typename metavars::InterpolationTargetA;
  using temporal_id_type = typename target_tag::temporal_id::type;
  using target_component = mock_interpolation_target<metavars, target_tag>;

  // Create a Domain with time-dependence. For this test we don't care
  // what the Domain actually is, we care only that it has time-dependence.
  const domain::creators::Brick domain_creator(
      {{-1.2, 3.0, 2.5}}, {{0.8, 5.0, 3.0}}, {{1, 1, 1}}, {{5, 4, 3}},
      {{false, false, false}}, {},
      std::make_unique<
          domain::creators::time_dependence::UniformTranslation<3>>(
          0.0, std::array<double, 3>({{0.1, 0.2, 0.3}})));

  // This name must match the hard coded one in UniformTranslation
  const std::string f_of_t_name = "Translation";
  std::unordered_map<std::string, double> initial_expiration_times{};
  initial_expiration_times[f_of_t_name] = 0.1;
  const double new_expiration_time = 1.0;
  ActionTesting::MockRuntimeSystem<metavars> runner{
      {domain_creator.create_domain(), "UnusedVolumeFileName",
       "UnusedReductionFilename"},
      {domain_creator.functions_of_time(initial_expiration_times)}};
  ActionTesting::emplace_component<target_component>(&runner, 0);
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<target_component>(make_not_null(&runner), 0);
  }
  ActionTesting::emplace_array_component<interpolator_component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0}, 0);
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  auto& target_box =
      ActionTesting::get_databox<target_component>(make_not_null(&runner), 0);

  // Both PendingTemporalIds and TemporalIds should be initially empty.
  CHECK(ActionTesting::get_databox_tag<
            target_component,
            ::intrp::Tags::PendingTemporalIds<temporal_id_type>>(runner, 0)
            .empty());
  CHECK(ActionTesting::get_databox_tag<
            target_component, ::intrp::Tags::TemporalIds<temporal_id_type>>(
            runner, 0)
            .empty());

  const auto add_id_to_target = [&](const TimeStepId& id) {
    ActionTesting::simple_action<
        target_component,
        ::intrp::Actions::AddTemporalIdsToInterpolationTarget<target_tag>>(
        make_not_null(&runner), 0, id);
  };

  // Two of the temporal_ids are before the expiration_time, the
  // others are afterwards.  Later we will update the FunctionOfTimes
  // so that the later temporal_ids become valid.
  const Slab slab(0.0, 1.0);
  const std::vector<TimeStepId> before_expr_ids = {
      TimeStepId(true, 0, Time(slab, 0)),
      TimeStepId(true, 0, Time(slab, Rational(1, 20)))};
  const std::vector<TimeStepId> after_expr_ids = {
      TimeStepId(true, 0, Time(slab, Rational(1, 4))),
      TimeStepId(true, 0, Time(slab, Rational(3, 4)))};

  for (const auto& id : before_expr_ids) {
    add_id_to_target(id);
  }

  // First id should be in TemporalIds and the second should be in pending
  CHECK(ActionTesting::get_databox_tag<
            target_component,
            ::intrp::Tags::PendingTemporalIds<temporal_id_type>>(runner, 0) ==
        std::deque<TimeStepId>{before_expr_ids.back()});
  CHECK(ActionTesting::get_databox_tag<
            target_component, ::intrp::Tags::TemporalIds<temporal_id_type>>(
            runner, 0) == std::deque<TimeStepId>{before_expr_ids.front()});

  // Should be one queued simple action on the Interpolator (ReceivePoints)
  // because one id is still pending
  CHECK(ActionTesting::number_of_queued_simple_actions<interpolator_component>(
            runner, 0) == 1);
  ActionTesting::invoke_queued_simple_action<interpolator_component>(
      make_not_null(&runner), 0);

  // Comes from MockReceivePoints in InterpolationTargetTestHelpers
  const auto check_interpolator_for_id = [&](const TimeStepId& id,
                                             const bool contains) {
    CAPTURE(id);
    CHECK(get<intrp::Vars::HolderTag<target_tag, metavars>>(
              ActionTesting::get_databox_tag<
                  interpolator_component,
                  intrp::Tags::InterpolatedVarsHolders<metavars>>(runner, 0))
              .infos.contains(id) == contains);
  };

  // The interpolator should have received the first id
  check_interpolator_for_id(before_expr_ids.front(), true);

  // Add the same temporal_ids again, which should do nothing...
  for (const auto& id : before_expr_ids) {
    add_id_to_target(id);
  }
  // ...and check that it indeed did nothing.
  CHECK(ActionTesting::get_databox_tag<
            target_component,
            ::intrp::Tags::PendingTemporalIds<temporal_id_type>>(runner, 0) ==
        std::deque<TimeStepId>{before_expr_ids.back()});
  CHECK(ActionTesting::get_databox_tag<
            target_component, ::intrp::Tags::TemporalIds<temporal_id_type>>(
            runner, 0) == std::deque<TimeStepId>{before_expr_ids.front()});

  // Add the ids after the expiration time
  for (const auto& id : after_expr_ids) {
    add_id_to_target(id);
  }

  // All of the new temporal ids should have been put in pending. No callbacks
  // have been registered because the above didn't call
  // VerifyTemporalIdsAndSendPoints
  CHECK(ActionTesting::get_databox_tag<
            target_component, ::intrp::Tags::TemporalIds<temporal_id_type>>(
            runner, 0) == std::deque<TimeStepId>{before_expr_ids.front()});
  {
    std::deque<TimeStepId> expected{after_expr_ids.begin(),
                                    after_expr_ids.end()};
    expected.push_front(before_expr_ids.back());
    CHECK(ActionTesting::get_databox_tag<
              target_component,
              ::intrp::Tags::PendingTemporalIds<temporal_id_type>>(runner, 0) ==
          expected);
  }

  // No simple actions should have been queued on either the target or the
  // interpolator
  CHECK(
      ActionTesting::is_simple_action_queue_empty<target_component>(runner, 0));
  CHECK(ActionTesting::is_simple_action_queue_empty<interpolator_component>(
      runner, 0));

  // Move the before expiration ids to completed so we can continue
  db::mutate<intrp::Tags::CompletedTemporalIds<TimeStepId>,
             intrp::Tags::PendingTemporalIds<TimeStepId>,
             intrp::Tags::TemporalIds<TimeStepId>>(
      [&](const gsl::not_null<std::deque<TimeStepId>*> completed_ids,
          const gsl::not_null<std::deque<TimeStepId>*> pending_ids,
          const gsl::not_null<std::deque<TimeStepId>*> ids) {
        pending_ids->pop_front();
        ids->pop_front();
        completed_ids->insert(completed_ids->begin(), before_expr_ids.begin(),
                              before_expr_ids.end());
      },
      make_not_null(&target_box));

  // Call VerifyTemporalIdsAndSendPoints so we register some callbacks
  ActionTesting::simple_action<
      target_component,
      intrp::Actions::VerifyTemporalIdsAndSendPoints<target_tag>>(
      make_not_null(&runner), 0);

  // This should do nothing
  for (const auto& id : after_expr_ids) {
    add_id_to_target(id);
  }

  // Check so
  CHECK(ActionTesting::get_databox_tag<
            target_component,
            ::intrp::Tags::PendingTemporalIds<temporal_id_type>>(runner, 0) ==
        std::deque<TimeStepId>{after_expr_ids.begin(), after_expr_ids.end()});
  CHECK(ActionTesting::get_databox_tag<
            target_component, ::intrp::Tags::TemporalIds<temporal_id_type>>(
            runner, 0)
            .empty());

  // Check that there are no queued simple actions.
  CHECK(
      ActionTesting::is_simple_action_queue_empty<target_component>(runner, 0));
  CHECK(ActionTesting::is_simple_action_queue_empty<interpolator_component>(
      runner, 0));

  // So now mutate the FunctionsOfTime. Since we cleared the temporal ids above
  // in the db::mutate, one callback should be registered
  auto& cache = ActionTesting::cache<target_component>(runner, 0_st);
  double current_expiration_time = initial_expiration_times[f_of_t_name];
  Parallel::mutate<domain::Tags::FunctionsOfTime,
                   control_system::UpdateSingleFunctionOfTime>(
      cache, f_of_t_name, current_expiration_time, DataVector{3, 0.0},
      new_expiration_time);

  // Check that there is one simple action queued
  // (VerifyTemporalIdsAndSendPoints)
  CHECK(ActionTesting::number_of_queued_simple_actions<target_component>(
            runner, 0) == 1);
  ActionTesting::invoke_queued_simple_action<target_component>(
      make_not_null(&runner), 0);

  // The all ids of "after expr time" ids are now valid, they should be in
  // TemporalIds and nothing in Pending
  CHECK(ActionTesting::get_databox_tag<
            target_component,
            ::intrp::Tags::PendingTemporalIds<temporal_id_type>>(runner, 0)
            .empty());
  CHECK(ActionTesting::get_databox_tag<
            target_component, ::intrp::Tags::TemporalIds<temporal_id_type>>(
            runner, 0) ==
        std::deque<TimeStepId>{after_expr_ids.begin(), after_expr_ids.end()});

  // And there should be a MockReceivePoints simple action queued on the
  // interpolator
  CHECK(ActionTesting::number_of_queued_simple_actions<interpolator_component>(
            runner, 0) == 1);
  ActionTesting::invoke_queued_simple_action<interpolator_component>(
      make_not_null(&runner), 0);

  // But only the first ready id should have been inserted
  for (const auto& id : after_expr_ids) {
    check_interpolator_for_id(id, id == after_expr_ids.front());
  }
}

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.InterpolationTarget.AddTemporalIds",
                  "[Unit]") {
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  domain::FunctionsOfTime::register_derived_with_charm();
  test_add_temporal_ids();
  test_add_temporal_ids_time_dependent();
}
}  // namespace
