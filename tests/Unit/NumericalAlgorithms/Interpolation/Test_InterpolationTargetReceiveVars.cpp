// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Domain/Creators/Shell.hpp"
#include "Domain/Domain.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolator.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InterpolatedVars.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InterpolationTargetReceiveVars.hpp"
#include "Parallel/AddOptionsToDataBox.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Rational.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_include <boost/variant/get.hpp>

/// \cond
// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables
namespace intrp {
namespace Actions {
template <typename InterpolationTargetTag>
struct CleanUpInterpolator;
}  // namespace Actions
}  // namespace intrp
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
namespace intrp {
namespace Tags {
struct IndicesOfFilledInterpPoints;
struct NumberOfElements;
template <typename Metavariables>
struct TemporalIds;
template <typename Metavariables>
struct CompletedTemporalIds;
}  // namespace Tags
}  // namespace intrp
/// \endcond

namespace {

template <typename Metavariables, typename InterpolationTargetTag>
struct mock_interpolation_target {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using component_being_mocked =
      intrp::InterpolationTarget<Metavariables, InterpolationTargetTag>;
  using const_global_cache_tag_list = tmpl::list<>;
  using simple_tags =
      db::get_items<typename intrp::Actions::InitializeInterpolationTarget<
          InterpolationTargetTag>::template return_tag_list<Metavariables>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing, tmpl::list<>>>;
  using add_options_to_databox =
      typename intrp::Actions::InitializeInterpolationTarget<
          InterpolationTargetTag>::template AddOptionsToDataBox<Metavariables>;
};

template <typename InterpolationTargetTag>
struct MockCleanUpInterpolator {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex,
            Requires<tmpl::list_contains_v<
                DbTags, typename intrp::Tags::NumberOfElements>> = nullptr>
  static void apply(
      db::DataBox<DbTags>& box,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const typename Metavariables::temporal_id::type& temporal_id) noexcept {
    Slab slab(0.0, 1.0);
    CHECK(temporal_id == TimeId(true, 0, Time(slab, Rational(13, 15))));
    // Put something in NumberOfElements so we can check later whether
    // this function was called.  This isn't the usual usage of
    // NumberOfElements.
    db::mutate<intrp::Tags::NumberOfElements>(
        make_not_null(&box),
        [](const gsl::not_null<db::item_type<intrp::Tags::NumberOfElements>*>
               number_of_elements) noexcept { ++(*number_of_elements); });
  }
};

struct MockComputeTargetPoints {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex,
            Requires<tmpl::list_contains_v<
                DbTags, intrp::Tags::TemporalIds<Metavariables>>> = nullptr>
  static void apply(
      db::DataBox<DbTags>& box,
      Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const typename Metavariables::temporal_id::type& temporal_id) noexcept {
    Slab slab(0.0, 1.0);
    CHECK(temporal_id == TimeId(true, 0, Time(slab, Rational(14, 15))));
    // Increment IndicesOfFilledInterpPoints so we can check later
    // whether this function was called.  This isn't the usual usage
    // of IndicesOfFilledInterpPoints; this is done only for the test.
    db::mutate<intrp::Tags::IndicesOfFilledInterpPoints>(
        make_not_null(&box),
        [](const gsl::not_null<
            db::item_type<intrp::Tags::IndicesOfFilledInterpPoints>*>
               indices) noexcept { indices->insert(indices->size() + 1); });
  }
};

// Simple DataBoxItems to test.
namespace Tags {
struct Square : db::SimpleTag {
  static std::string name() noexcept { return "Square"; }
  using type = Scalar<DataVector>;
};
struct SquareComputeItem : Square, db::ComputeTag {
  static std::string name() noexcept { return "Square"; }
  static Scalar<DataVector> function(const Scalar<DataVector>& x) noexcept {
    auto result = make_with_value<Scalar<DataVector>>(x, 0.0);
    get<>(result) = square(get<>(x));
    return result;
  }
  using argument_tags = tmpl::list<gr::Tags::Lapse<DataVector>>;
};
}  // namespace Tags

template <typename DbTags, typename TemporalId>
void callback_impl(const db::DataBox<DbTags>& box,
                   const TemporalId& temporal_id) noexcept {
  Slab slab(0.0, 1.0);
  CHECK(temporal_id == TimeId(true, 0, Time(slab, Rational(13, 15))));
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
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
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
      const gsl::not_null<Parallel::ConstGlobalCache<Metavariables>*> /*cache*/,
      const typename Metavariables::temporal_id::type& temporal_id) noexcept {
    callback_impl(*box, temporal_id);
    return false;
  }
};

template <typename Metavariables>
struct mock_interpolator {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tag_list = tmpl::list<>;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<intrp::Actions::InitializeInterpolator>>,
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
  struct InterpolationTargetA : db::SimpleTag {
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
    using compute_target_points = MockComputeTargetPoints;
    using post_interpolation_callback = MockCallBackType;
    using compute_items_on_target = tmpl::list<Tags::SquareComputeItem>;
    using container_tag = InterpolationTargetA;
    static std::string name() noexcept { return "InterpolationTargetA"; }
  };
  using interpolator_source_vars = tmpl::list<gr::Tags::Lapse<DataVector>>;
  using interpolation_target_tags = tmpl::list<InterpolationTargetA>;
  using temporal_id = ::Tags::TimeId;
  using domain_frame = Frame::Inertial;
  static constexpr size_t domain_dim = 3;

  using component_list = tmpl::list<
      mock_interpolation_target<MockMetavariables, InterpolationTargetA>,
      mock_interpolator<MockMetavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;
  enum class Phase { Initialization, Testing, Exit };
};

template <typename MockCallbackType, size_t NumberOfExpectedCleanUpActions>
void test_interpolation_target_receive_vars() noexcept {
  using metavars = MockMetavariables<MockCallbackType>;
  using interp_component = mock_interpolator<metavars>;
  using target_component =
      mock_interpolation_target<metavars,
                                typename metavars::InterpolationTargetA>;

  const auto domain_creator =
      domain::creators::Shell<Frame::Inertial>(0.9, 4.9, 1, {{5, 5}}, false);
  Slab slab(0.0, 1.0);
  const size_t num_points = 10;

  ActionTesting::MockRuntimeSystem<metavars> runner{{}};
  ActionTesting::emplace_component<interp_component>(&runner, 0);
  ActionTesting::next_action<interp_component>(make_not_null(&runner), 0);
  ActionTesting::emplace_component_and_initialize<target_component>(
      &runner, 0,
      {db::item_type<intrp::Tags::IndicesOfFilledInterpPoints>{},
       db::item_type<typename intrp::Tags::TemporalIds<metavars>>{
           TimeId(true, 0, Time(slab, Rational(13, 15))),
           TimeId(true, 0, Time(slab, Rational(14, 15)))},
       db::item_type<typename intrp::Tags::CompletedTemporalIds<metavars>>{},
       db::item_type<::Tags::Variables<typename metavars::InterpolationTargetA::
                                           vars_to_interpolate_to_target>>{
           num_points}},
      domain_creator.create_domain());
  runner.set_phase(metavars::Phase::Testing);

  // Now set up the vars.
  std::vector<db::item_type<::Tags::Variables<
      typename metavars::InterpolationTargetA::vars_to_interpolate_to_target>>>
      vars_src;
  std::vector<std::vector<size_t>> global_offsets;

  // Adds more data to vars_src and global_offsets.
  auto add_to_vars_src = [&vars_src, &global_offsets](
                             const std::vector<double>& lapse_vals,
                             const std::vector<size_t>& offset_vals) {
    vars_src.emplace_back(
        db::item_type<
            ::Tags::Variables<typename metavars::InterpolationTargetA::
                                  vars_to_interpolate_to_target>>{
            lapse_vals.size()});
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
      make_not_null(&runner), 0, vars_src, global_offsets);

  // It should have interpolated 4 points by now.
  CHECK(
      ActionTesting::get_databox_tag<target_component,
                                     intrp::Tags::IndicesOfFilledInterpPoints>(
          runner, 0)
          .size() == 4);

  // Should be no queued simple action until we get num_points points.
  CHECK(
      ActionTesting::is_simple_action_queue_empty<target_component>(runner, 0));
  CHECK(
      ActionTesting::is_simple_action_queue_empty<interp_component>(runner, 0));
  CHECK(ActionTesting::get_databox_tag<target_component,
                                       intrp::Tags::TemporalIds<metavars>>(
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
      make_not_null(&runner), 0, vars_src, global_offsets);

  // It should have interpolated 8 points by now. (The ninth point had
  // a repeated global_offsets so it should be ignored)
  CHECK(
      ActionTesting::get_databox_tag<target_component,
                                     intrp::Tags::IndicesOfFilledInterpPoints>(
          runner, 0)
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
      make_not_null(&runner), 0, vars_src, global_offsets);

  // It should have interpolated all the points by now.
  CHECK(
      ActionTesting::get_databox_tag<target_component,
                                     intrp::Tags::IndicesOfFilledInterpPoints>(
          runner, 0)
          .size() == num_points);

  if (NumberOfExpectedCleanUpActions == 0) {
    // We called the function without cleanup, as a test, so there should
    // be no queued simple actions (tested below outside the if-else).

    // Check that MockCleanUpInterpolator was NOT called.  If called, it resets
    // the (fake) number of elements, specifically so we can test it here.
    CHECK(ActionTesting::get_databox_tag<interp_component,
                                         intrp::Tags::NumberOfElements>(
              runner, 0) == 0);

    // Also, there should be 2 TemporalIds left because we did not
    // clean up one of them.
    CHECK(ActionTesting::get_databox_tag<target_component,
                                         intrp::Tags::TemporalIds<metavars>>(
              runner, 0)
              .size() == 2);

    // And there should be 0 CompletedTemporalIds because we did not
    // clean up TemporalIds.
    CHECK(ActionTesting::get_databox_tag<
              target_component, intrp::Tags::CompletedTemporalIds<metavars>>(
              runner, 0)
              .empty());

  } else {
    // This is the (usual) case where we want a cleanup.

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
    CHECK(ActionTesting::get_databox_tag<target_component,
                                         intrp::Tags::TemporalIds<metavars>>(
              runner, 0)
              .size() == 1);

    // And there should be 1 CompletedTemporalId, and its value
    // should be the value of the initial TemporalId.
    CHECK(ActionTesting::get_databox_tag<
              target_component, intrp::Tags::CompletedTemporalIds<metavars>>(
              runner, 0)
              .size() == 1);
    CHECK(ActionTesting::get_databox_tag<
              target_component, intrp::Tags::CompletedTemporalIds<metavars>>(
              runner, 0)
              .front() == TimeId(true, 0, Time(slab, Rational(13, 15))));

    // And there is yet one more simple action, compute_target_points,
    // which here we mock just to check that it is called.
    ActionTesting::invoke_queued_simple_action<target_component>(
        make_not_null(&runner), 0);

    // Check that MockComputeTargetPoints was called.
    // It sets a (fake) value of IndicesOfFilledInterpPoints for the express
    // purpose of this check.
    CHECK(ActionTesting::get_databox_tag<
              target_component, intrp::Tags::IndicesOfFilledInterpPoints>(
              runner, 0)
              .size() == num_points + 1);
  }

  // There should be no more queued actions; verify this.
  CHECK(
      ActionTesting::is_simple_action_queue_empty<target_component>(runner, 0));
  CHECK(
      ActionTesting::is_simple_action_queue_empty<interp_component>(runner, 0));
}

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.InterpolationTarget.ReceiveVars",
                  "[Unit]") {
  test_interpolation_target_receive_vars<MockPostInterpolationCallback, 1>();
  test_interpolation_target_receive_vars<MockPostInterpolationCallbackNoCleanup,
                                         0>();
}
}  // namespace
