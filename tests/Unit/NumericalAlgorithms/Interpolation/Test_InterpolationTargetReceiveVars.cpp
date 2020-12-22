// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Shell.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolator.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InterpolatedVars.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InterpolationTargetReceiveVars.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Rational.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include <boost/variant/get.hpp>

/// \cond
// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables
namespace intrp::Actions {
template <typename InterpolationTargetTag>
struct CleanUpInterpolator;
}  // namespace intrp::Actions
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
namespace intrp::Tags {
template <typename Metavariables>
struct IndicesOfFilledInterpPoints;
struct NumberOfElements;
template <typename TemporalId>
struct TemporalIds;
template <typename TemporalId>
struct CompletedTemporalIds;
}  // namespace intrp::Tags
/// \endcond

namespace {

// In the test, we don't care what SendPointsToInterpolator actually does;
// we care only that SendPointsToInterpolator is called with the
// correct arguments.
template <typename InterpolationTargetTag>
struct MockSendPointsToInterpolator {
  template <
      typename ParallelComponent, typename DbTags, typename Metavariables,
      typename ArrayIndex,
      Requires<tmpl::list_contains_v<
          DbTags, intrp::Tags::TemporalIds<
                      typename Metavariables::temporal_id::type>>> = nullptr>
  static void apply(
      db::DataBox<DbTags>& box,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const typename Metavariables::temporal_id::type& temporal_id) noexcept {
    using temporal_id_type = typename Metavariables::temporal_id::type;
    Slab slab(0.0, 1.0);
    CHECK(temporal_id == TimeStepId(true, 0, Time(slab, Rational(14, 15))));
    // Increment IndicesOfFilledInterpPoints so we can check later
    // whether this function was called.  This isn't the usual usage
    // of IndicesOfFilledInterpPoints; this is done only for the test.
    db::mutate<intrp::Tags::IndicesOfFilledInterpPoints<temporal_id_type>>(
        make_not_null(&box),
        [&temporal_id](
            const gsl::not_null<std::unordered_map<temporal_id_type,
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
  using component_being_mocked =
      intrp::InterpolationTarget<Metavariables, InterpolationTargetTag>;
  using const_global_cache_tags =
      tmpl::list<domain::Tags::Domain<Metavariables::volume_dim>>;
  using simple_tags =
      db::get_items<typename intrp::Actions::InitializeInterpolationTarget<
          Metavariables, InterpolationTargetTag>::return_tag_list>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<
              simple_tags,
              typename InterpolationTargetTag::compute_items_on_target>>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing, tmpl::list<>>>;
  using replace_these_simple_actions = tmpl::list<
      intrp::Actions::SendPointsToInterpolator<InterpolationTargetTag>>;
  using with_these_simple_actions =
      tmpl::list<MockSendPointsToInterpolator<InterpolationTargetTag>>;
};

template <typename InterpolationTargetTag>
struct MockCleanUpInterpolator {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex,
            Requires<tmpl::list_contains_v<
                DbTags, typename intrp::Tags::NumberOfElements>> = nullptr>
  static void apply(
      db::DataBox<DbTags>& box,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const typename Metavariables::temporal_id::type& temporal_id) noexcept {
    Slab slab(0.0, 1.0);
    CHECK(temporal_id == TimeStepId(true, 0, Time(slab, Rational(13, 15))));
    // Put something in NumberOfElements so we can check later whether
    // this function was called.  This isn't the usual usage of
    // NumberOfElements.
    db::mutate<intrp::Tags::NumberOfElements>(
        make_not_null(&box),
        [](const gsl::not_null<size_t*> number_of_elements) noexcept {
          ++(*number_of_elements);
        });
  }
};

// In the test, MockComputeTargetPoints is used only for the
// is_sequential typedef; normally compute_target_points has a
// points() function, but that function isn't called or needed in the test.
struct MockComputeTargetPoints {
  using is_sequential = std::true_type;
};

// Simple DataBoxItems to test.
namespace Tags {
struct Square : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct SquareCompute : Square, db::ComputeTag {
  static void function(gsl::not_null<Scalar<DataVector>*> result,
                       const Scalar<DataVector>& x) noexcept {
    get(*result) = square(get(x));
  }
  using argument_tags = tmpl::list<gr::Tags::Lapse<DataVector>>;
  using base = Square;
  using return_type = Scalar<DataVector>;
};
}  // namespace Tags

template <typename DbTags, typename TemporalId>
void callback_impl(const db::DataBox<DbTags>& box,
                   const TemporalId& temporal_id) noexcept {
  Slab slab(0.0, 1.0);
  CHECK(temporal_id == TimeStepId(true, 0, Time(slab, Rational(13, 15))));
  // The result should be the square of the first 10 integers, in
  // a Scalar<DataVector>.
  const Scalar<DataVector> expected{
      DataVector{{0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0}}};
  CHECK_ITERABLE_APPROX(expected, db::get<Tags::Square>(box));
}

struct MockPostInterpolationCallback {
  template <typename DbTags, typename Metavariables>
  static void apply(
      const db::DataBox<DbTags>& box,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const typename Metavariables::temporal_id::type& temporal_id) noexcept {
    callback_impl(box, temporal_id);
  }
};

// The sole purpose of this class is to check
// the overload of the callback, and to prevent
// cleanup.  The only time that one would actually want
// to prevent cleanup is in the AH finder, and it is tested there,
//  but this is a more direct test.
struct MockPostInterpolationCallbackNoCleanup {
  template <typename DbTags, typename Metavariables>
  static bool apply(
      const gsl::not_null<db::DataBox<DbTags>*> box,
      const gsl::not_null<Parallel::GlobalCache<Metavariables>*> /*cache*/,
      const typename Metavariables::temporal_id::type& temporal_id) noexcept {
    callback_impl(*box, temporal_id);
    return false;
  }
};

template <size_t NumberOfInvalidInterpolationPoints>
struct MockPostInterpolationCallbackWithInvalidPoints {
  static constexpr double fill_invalid_points_with = 15.0;

  template <typename DbTags, typename Metavariables>
  static void apply(
      const db::DataBox<DbTags>& box,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const typename Metavariables::temporal_id::type& temporal_id) noexcept {
    Slab slab(0.0, 1.0);
    CHECK(temporal_id == TimeStepId(true, 0, Time(slab, Rational(13, 15))));

    // The result should be the square of the first 10 integers, in
    // a Scalar<DataVector>, followed by several 225s.
    DataVector expected_dv(NumberOfInvalidInterpolationPoints + 10);
    for (size_t i = 0; i < 10; ++i) {
      expected_dv[i] = double(i * i);
    }
    for (size_t i = 10; i < 10 + NumberOfInvalidInterpolationPoints; ++i) {
      expected_dv[i] = 225.0;
    }
    const Scalar<DataVector> expected{expected_dv};
    CHECK_ITERABLE_APPROX(expected, db::get<Tags::Square>(box));
  }
};

template <typename Metavariables>
struct mock_interpolator {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<Actions::SetupDataBox,
                     intrp::Actions::InitializeInterpolator<
                         intrp::Tags::VolumeVarsInfo<Metavariables>,
                         intrp::Tags::InterpolatedVarsHolders<Metavariables>>>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing, tmpl::list<>>>;

  using component_being_mocked = intrp::Interpolator<Metavariables>;
  using replace_these_simple_actions =
      tmpl::list<intrp::Actions::CleanUpInterpolator<
          typename Metavariables::InterpolationTargetA>>;
  using with_these_simple_actions = tmpl::list<
      MockCleanUpInterpolator<typename Metavariables::InterpolationTargetA>>;
};

template <typename MockCallBackType>
struct MockMetavariables {
  struct InterpolationTargetA {
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
    using compute_target_points = MockComputeTargetPoints;
    using post_interpolation_callback = MockCallBackType;
    using compute_items_on_target = tmpl::list<Tags::SquareCompute>;
  };
  using interpolator_source_vars = tmpl::list<gr::Tags::Lapse<DataVector>>;
  using interpolation_target_tags = tmpl::list<InterpolationTargetA>;
  using temporal_id = ::Tags::TimeStepId;
  static constexpr size_t volume_dim = 3;

  using component_list = tmpl::list<
      mock_interpolation_target<MockMetavariables, InterpolationTargetA>,
      mock_interpolator<MockMetavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};

template <typename MockCallbackType, size_t NumberOfExpectedCleanUpActions,
          size_t NumberOfInvalidPointsToAdd>
void test_interpolation_target_receive_vars() noexcept {
  using metavars = MockMetavariables<MockCallbackType>;
  using temporal_id_type = typename metavars::temporal_id::type;
  using interp_component = mock_interpolator<metavars>;
  using target_component =
      mock_interpolation_target<metavars,
                                typename metavars::InterpolationTargetA>;

  const auto domain_creator =
      domain::creators::Shell(0.9, 4.9, 1, {{5, 5}}, false);
  Slab slab(0.0, 1.0);
  const size_t num_points = 10;
  const TimeStepId first_temporal_id(true, 0, Time(slab, Rational(13, 15)));
  const TimeStepId second_temporal_id(true, 0, Time(slab, Rational(14, 15)));

  // Add indices of invalid points (if there are any) at the end.
  std::unordered_map<temporal_id_type, std::unordered_set<size_t>>
      invalid_indices{};
  for (size_t index = num_points;
       index < num_points + NumberOfInvalidPointsToAdd; ++index) {
    invalid_indices[first_temporal_id].insert(index);
  }

  // Type alias for better readability below.
  using vars_type = Variables<
      typename metavars::InterpolationTargetA::vars_to_interpolate_to_target>;

  ActionTesting::MockRuntimeSystem<metavars> runner{
      {domain_creator.create_domain()}};
  ActionTesting::emplace_component<interp_component>(&runner, 0);
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<interp_component>(make_not_null(&runner), 0);
  }
  ActionTesting::emplace_component_and_initialize<target_component>(
      &runner, 0,
      {std::unordered_map<temporal_id_type, std::unordered_set<size_t>>{},
       std::unordered_map<temporal_id_type, std::unordered_set<size_t>>{
           invalid_indices},
       std::deque<temporal_id_type>{first_temporal_id, second_temporal_id},
       std::deque<temporal_id_type>{},
       std::unordered_map<temporal_id_type,
                          Variables<typename metavars::InterpolationTargetA::
                                        vars_to_interpolate_to_target>>{
           {first_temporal_id,
            vars_type{num_points + NumberOfInvalidPointsToAdd}}},
       // Default-constructed Variables cause problems, so below
       // we construct the Variables with a single point.
       vars_type{1}});
  ActionTesting::set_phase(make_not_null(&runner), metavars::Phase::Testing);

  // Now set up the vars.
  std::vector<typename ::Variables<
      typename metavars::InterpolationTargetA::vars_to_interpolate_to_target>>
      vars_src;
  std::vector<std::vector<size_t>> global_offsets;

  // Adds more data to vars_src and global_offsets.
  auto add_to_vars_src = [&vars_src, &global_offsets](
                             const std::vector<double>& lapse_vals,
                             const std::vector<size_t>& offset_vals) {
    vars_src.emplace_back(
        ::Variables<typename metavars::InterpolationTargetA::
                        vars_to_interpolate_to_target>{lapse_vals.size()});
    global_offsets.emplace_back(offset_vals);
    auto& lapse = get<gr::Tags::Lapse<DataVector>>(vars_src.back());
    for (size_t i = 0; i < lapse_vals.size(); ++i) {
      get<>(lapse)[i] = lapse_vals[i];
    }
  };

  add_to_vars_src({{3.0, 6.0}}, {{3, 6}});
  add_to_vars_src({{2.0, 7.0}}, {{2, 7}});

  ActionTesting::simple_action<target_component,
                               intrp::Actions::InterpolationTargetReceiveVars<
                                   typename metavars::InterpolationTargetA>>(
      make_not_null(&runner), 0, vars_src, global_offsets, first_temporal_id);

  // It should have interpolated 4 points by now.
  CHECK(
      ActionTesting::get_databox_tag<
          target_component,
          intrp::Tags::IndicesOfFilledInterpPoints<temporal_id_type>>(runner, 0)
          .at(first_temporal_id)
          .size() == 4);

  // Should be no queued simple action until we get num_points points.
  CHECK(
      ActionTesting::is_simple_action_queue_empty<target_component>(runner, 0));
  CHECK(
      ActionTesting::is_simple_action_queue_empty<interp_component>(runner, 0));
  CHECK(ActionTesting::get_databox_tag<
            target_component, intrp::Tags::TemporalIds<temporal_id_type>>(
            runner, 0)
            .size() == 2);

  vars_src.clear();
  global_offsets.clear();
  add_to_vars_src({{1.0, 888888.0}},
                  {{1, 6}});  // 6 is repeated, point will be ignored.
  add_to_vars_src({{8.0, 0.0, 4.0}}, {{8, 0, 4}});

  ActionTesting::simple_action<target_component,
                               intrp::Actions::InterpolationTargetReceiveVars<
                                   typename metavars::InterpolationTargetA>>(
      make_not_null(&runner), 0, vars_src, global_offsets, first_temporal_id);

  // It should have interpolated 8 points by now. (The ninth point had
  // a repeated global_offsets so it should be ignored)
  CHECK(
      ActionTesting::get_databox_tag<
          target_component,
          intrp::Tags::IndicesOfFilledInterpPoints<temporal_id_type>>(runner, 0)
          .at(first_temporal_id)
          .size() == 8);

  // Should be no queued simple action until we have added 10 points.
  CHECK(
      ActionTesting::is_simple_action_queue_empty<target_component>(runner, 0));
  CHECK(
      ActionTesting::is_simple_action_queue_empty<mock_interpolator<metavars>>(
          runner, 0));

  vars_src.clear();
  global_offsets.clear();
  add_to_vars_src({{9.0, 5.0}}, {{9, 5}});

  // This will call InterpolationTargetA::post_interpolation_callback
  // where we check that the points are correct.
  ActionTesting::simple_action<target_component,
                               intrp::Actions::InterpolationTargetReceiveVars<
                                   typename metavars::InterpolationTargetA>>(
      make_not_null(&runner), 0, vars_src, global_offsets, first_temporal_id);

  if (NumberOfExpectedCleanUpActions == 0) {
    // We called the function without cleanup, as a test, so there should
    // be no queued simple actions (tested below outside the if-else).

    // It should have interpolated all the points by now.
    // But those points should have not been cleaned up.
    CHECK(ActionTesting::get_databox_tag<
              target_component,
              intrp::Tags::IndicesOfFilledInterpPoints<temporal_id_type>>(
              runner, 0)
              .at(first_temporal_id)
              .size() == num_points);

    // Check that MockCleanUpInterpolator was NOT called.  If called, it resets
    // the (fake) number of elements, specifically so we can test it here.
    CHECK(ActionTesting::get_databox_tag<interp_component,
                                         intrp::Tags::NumberOfElements>(
              runner, 0) == 0);

    // Also, there should be 2 TemporalIds left because we did not
    // clean up one of them.
    CHECK(ActionTesting::get_databox_tag<
              target_component, intrp::Tags::TemporalIds<temporal_id_type>>(
              runner, 0)
              .size() == 2);

    // And there should be 0 CompletedTemporalIds because we did not
    // clean up TemporalIds.
    CHECK(ActionTesting::get_databox_tag<
              target_component,
              intrp::Tags::CompletedTemporalIds<temporal_id_type>>(runner, 0)
              .empty());

  } else {
    // This is the (usual) case where we want a cleanup.

    // It should have interpolated all the points by now,
    // and the list of points should have been cleaned up.
    CHECK(ActionTesting::get_databox_tag<
              target_component,
              intrp::Tags::IndicesOfFilledInterpPoints<temporal_id_type>>(
              runner, 0)
              .count(first_temporal_id) == 0);

    // Now there should be a queued simple action, which is
    // CleanUpInterpolator, which here we mock.
    ActionTesting::invoke_queued_simple_action<interp_component>(
        make_not_null(&runner), 0);

    // Check that MockCleanUpInterpolator was called.  It resets the
    // (fake) number of elements, specifically so we can test it here.
    CHECK(ActionTesting::get_databox_tag<interp_component,
                                         intrp::Tags::NumberOfElements>(
              runner, 0) == 1);

    // Also, there should be only 1 TemporalId left.
    // And its value should be second_temporal_id.
    CHECK(ActionTesting::get_databox_tag<
              target_component, intrp::Tags::TemporalIds<temporal_id_type>>(
              runner, 0)
              .size() == 1);
    CHECK(ActionTesting::get_databox_tag<
              target_component, intrp::Tags::TemporalIds<temporal_id_type>>(
              runner, 0)
              .front() == second_temporal_id);

    // And there should be 1 CompletedTemporalId, and its value
    // should be the value of the initial TemporalId.
    CHECK(ActionTesting::get_databox_tag<
              target_component,
              intrp::Tags::CompletedTemporalIds<temporal_id_type>>(runner, 0)
              .size() == 1);
    CHECK(ActionTesting::get_databox_tag<
              target_component,
              intrp::Tags::CompletedTemporalIds<temporal_id_type>>(runner, 0)
              .front() == first_temporal_id);

    // Check that MockSendPointsToInterpolator was not yet called.
    // MockSendPointsToInterpolator sets a (fake) value of
    // IndicesOfFilledInterpPoints for the express purpose of this
    // check.
    const auto& indices_to_check = ActionTesting::get_databox_tag<
        target_component,
        intrp::Tags::IndicesOfFilledInterpPoints<temporal_id_type>>(runner, 0);
    CHECK(indices_to_check.find(second_temporal_id) == indices_to_check.end());

    // And there is yet one more simple action, SendPointsToInterpolator,
    // which here we mock just to check that it is called.
    ActionTesting::invoke_queued_simple_action<target_component>(
        make_not_null(&runner), 0);

    // Check that MockSendPointsToInterpolator was called.
    // MockSendPointsToInterpolator sets a (fake) value of
    // IndicesOfFilledInterpPoints for the express purpose of this check.
    CHECK(ActionTesting::get_databox_tag<
              target_component,
              intrp::Tags::IndicesOfFilledInterpPoints<temporal_id_type>>(
              runner, 0)
              .at(second_temporal_id)
              .size() == 1);
  }

  // There should be no more queued actions; verify this.
  CHECK(
      ActionTesting::is_simple_action_queue_empty<target_component>(runner, 0));
  CHECK(
      ActionTesting::is_simple_action_queue_empty<interp_component>(runner, 0));
}

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.InterpolationTarget.ReceiveVars",
                  "[Unit]") {
  domain::creators::register_derived_with_charm();
  test_interpolation_target_receive_vars<MockPostInterpolationCallback, 1, 0>();
  test_interpolation_target_receive_vars<MockPostInterpolationCallbackNoCleanup,
                                         0, 0>();
  test_interpolation_target_receive_vars<
      MockPostInterpolationCallbackWithInvalidPoints<3>, 1, 3>();
}
}  // namespace
