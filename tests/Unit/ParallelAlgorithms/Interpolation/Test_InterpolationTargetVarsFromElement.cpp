// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RotationAboutZAxis.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolationTargetVarsFromElement.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/ComputeTargetPoints.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/PostInterpolationCallback.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Rational.hpp"

namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel

namespace {

namespace Tags {
// Simple Variables tag for test.
struct TestSolution : db::SimpleTag {
  using type = Scalar<DataVector>;
};
// Simple DataBoxItem to test.
struct Square : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct SquareCompute : Square, db::ComputeTag {
  static void function(gsl::not_null<Scalar<DataVector>*> result,
                       const Scalar<DataVector>& x) {
    get(*result) = square(get(x));
  }
  using argument_tags = tmpl::list<TestSolution>;
  using base = Square;
  using return_type = Scalar<DataVector>;
};
}  // namespace Tags

struct ResetFoT {
  static void apply(
      const gsl::not_null<std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>*>
          f_of_t_list,
      const std::string& f_of_t_name, const double new_expiration_time) {
    const double initial_time = f_of_t_list->at(f_of_t_name)->time_bounds()[0];
    std::array<DataVector, 4> init_func_and_2_derivs{
        DataVector{3, 0.0}, DataVector{3, 0.0}, DataVector{3, 0.0},
        DataVector{3, 0.0}};

    f_of_t_list->erase(f_of_t_name);

    (*f_of_t_list)[f_of_t_name] =
        std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<3>>(
            initial_time, std::move(init_func_and_2_derivs),
            new_expiration_time);
  }
};

struct MockComputeTargetPoints
    : tt::ConformsTo<intrp::protocols::ComputeTargetPoints> {
  using is_sequential = std::false_type;
  using frame = ::Frame::Inertial;
  template <typename Metavariables, typename DbTags>
  static tnsr::I<DataVector, 3, Frame::Inertial> points(
      const db::DataBox<DbTags>& /*box*/,
      const tmpl::type_<Metavariables>& /*meta*/) {
    // Need 10 points to agree with test.
    const size_t num_pts = 10;
    // Doesn't matter what the points are; they are not used except
    // that they need to be inside the domain.
    tnsr::I<DataVector, 3, Frame::Inertial> target_points(num_pts);
    for (size_t n = 0; n < num_pts; ++n) {
      for (size_t d = 0; d < 3; ++d) {
        target_points.get(d)[n] = 1.0 + 0.01 * n + 0.5 * d;
      }
    }
    return target_points;
  }

  template <typename Metavariables, typename DbTags, typename TemporalId>
  static tnsr::I<DataVector, 3, Frame::Inertial> points(
      const db::DataBox<DbTags>& box, const tmpl::type_<Metavariables>& meta,
      const TemporalId& /*temporal_id*/) {
    return points(box, meta);
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
size_t num_callback_calls = 0;

struct MockPostInterpolationCallback
    : tt::ConformsTo<intrp::protocols::PostInterpolationCallback> {
  template <typename DbTags, typename Metavariables, typename TemporalId>
  static void apply(const db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const TemporalId& temporal_id) {
    static_assert(std::is_same_v<TemporalId, TimeStepId>,
                  "MockPostInterpolationCallback expects a TimeStepId as its "
                  "temporal ID.");
    // This callback simply checks that the points are as expected.
    Slab slab(0.0, 1.0);
    const TimeStepId first_temporal_id(true, 0, Time(slab, Rational(13, 15)));
    const TimeStepId second_temporal_id(true, 0, Time(slab, Rational(14, 15)));
    if (temporal_id == first_temporal_id) {
      const Scalar<DataVector> expected{
          DataVector{{0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0}}};
      CHECK_ITERABLE_APPROX(expected, db::get<Tags::Square>(box));
    } else if (temporal_id == second_temporal_id) {
      const Scalar<DataVector> expected{DataVector{
          {0.0, 4.0, 16.0, 36.0, 64.0, 100.0, 144.0, 196.0, 256.0, 324.0}}};
      CHECK_ITERABLE_APPROX(expected, db::get<Tags::Square>(box));
    } else {
      // Should never get here.
      CHECK(false);
    }

    ++num_callback_calls;
  }
};

struct MockPostInterpolationCallbackWithInvalidPoints
    : public MockPostInterpolationCallback {
  static constexpr double fill_invalid_points_with = 0.0;
};

template <typename Metavariables, typename InterpolationTargetTag>
struct mock_interpolation_target {
  static_assert(
      tt::assert_conforms_to_v<InterpolationTargetTag,
                               intrp::protocols::InterpolationTargetTag>);
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using component_being_mocked =
      intrp::InterpolationTarget<Metavariables, InterpolationTargetTag>;
  using const_global_cache_tags =
      tmpl::list<domain::Tags::Domain<Metavariables::volume_dim>,
                 intrp::Tags::Verbosity>;
  using simple_tags = typename intrp::Actions::InitializeInterpolationTarget<
      Metavariables, InterpolationTargetTag>::simple_tags;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<
          simple_tags,
          typename InterpolationTargetTag::compute_items_on_target>>>>;
};

template <bool UseTimeDepMaps>
struct MockMetavariables {
  struct InterpolationTargetA
      : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::TimeStepId;
    using vars_to_interpolate_to_target = tmpl::list<Tags::TestSolution>;
    using compute_items_on_target = tmpl::list<Tags::SquareCompute>;
    using compute_target_points = MockComputeTargetPoints;
    using post_interpolation_callbacks =
        tmpl::list<MockPostInterpolationCallback,
                   MockPostInterpolationCallbackWithInvalidPoints>;
  };
  static constexpr size_t volume_dim = 3;
  using interpolation_target_tags = tmpl::list<InterpolationTargetA>;
  using mutable_global_cache_tags =
      tmpl::conditional_t<UseTimeDepMaps,
                          tmpl::list<domain::Tags::FunctionsOfTimeInitialize>,
                          tmpl::list<>>;

  using component_list = tmpl::list<
      mock_interpolation_target<MockMetavariables, InterpolationTargetA>>;
};

template <bool UseTimeDepMaps>
void test() {
  domain::creators::register_derived_with_charm();
  num_callback_calls = 0_st;

  if constexpr (UseTimeDepMaps) {
    domain::creators::time_dependence::register_derived_with_charm();
    domain::FunctionsOfTime::register_derived_with_charm();
  }

  using metavars = MockMetavariables<UseTimeDepMaps>;
  using temporal_id_type =
      typename metavars::InterpolationTargetA::temporal_id::type;
  using target_component =
      mock_interpolation_target<metavars,
                                typename metavars::InterpolationTargetA>;

  Slab slab(0.0, 1.0);
  const TimeStepId first_temporal_id(true, 0, Time(slab, Rational(13, 15)));
  const TimeStepId second_temporal_id(true, 0, Time(slab, Rational(14, 15)));
  const domain::creators::Sphere domain_creator = [&slab]() {
    if constexpr (UseTimeDepMaps) {
      return domain::creators::Sphere(
          0.9, 4.9, domain::creators::Sphere::Excision{}, 1_st, 5_st, false,
          std::nullopt, {}, {domain::CoordinateMaps::Distribution::Linear},
          ShellWedges::All,
          // Doesn't actually have to move. Just needs to go through the time
          // dependent code
          std::make_unique<
              domain::creators::time_dependence::RotationAboutZAxis<3>>(
              slab.start().value(), 0.0, 0.0, 0.0));
    } else {
      (void)slab;
      return domain::creators::Sphere(
          0.9, 4.9, domain::creators::Sphere::Excision{}, 1_st, 5_st, false);
    }
  }();

  // Type alias for better readability below.
  using vars_type = Variables<
      typename metavars::InterpolationTargetA::vars_to_interpolate_to_target>;

  std::unordered_map<std::string, double> init_expr_times{};
  const std::string name = "Rotation";
  init_expr_times[name] = slab.end().value();

  // Initialization
  ActionTesting::MockRuntimeSystem<metavars> runner = [&domain_creator,
                                                       &init_expr_times]() {
    if constexpr (UseTimeDepMaps) {
      return ActionTesting::MockRuntimeSystem<metavars>(
          {domain_creator.create_domain(), ::Verbosity::Silent},
          {domain_creator.functions_of_time(init_expr_times)});
    } else {
      (void)init_expr_times;
      return ActionTesting::MockRuntimeSystem<metavars>(
          {domain_creator.create_domain(), ::Verbosity::Silent});
    }
  }();
  ActionTesting::emplace_component_and_initialize<target_component>(
      &runner, 0,
      {std::unordered_map<temporal_id_type, std::unordered_set<size_t>>{},
       std::unordered_map<temporal_id_type, std::unordered_set<size_t>>{},
       std::deque<temporal_id_type>{}, std::deque<temporal_id_type>{},
       std::deque<temporal_id_type>{},
       std::unordered_map<temporal_id_type,
                          Variables<typename metavars::InterpolationTargetA::
                                        vars_to_interpolate_to_target>>{},
       // Default-constructed Variables cause problems, so below
       // we construct the Variables with a single point.
       vars_type{1}});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  // Now set up the vars and global offsets
  std::vector<::Variables<
      typename metavars::InterpolationTargetA::vars_to_interpolate_to_target>>
      vars_src;
  std::vector<std::vector<size_t>> global_offsets;

  // Lambda that adds more data to vars_src and global_offsets.
  auto add_to_vars_src = [&vars_src, &global_offsets](
                             const std::vector<double>& vals,
                             const std::vector<size_t>& offset_vals) {
    vars_src.emplace_back(
        ::Variables<typename metavars::InterpolationTargetA::
                        vars_to_interpolate_to_target>{vals.size()});
    global_offsets.emplace_back(offset_vals);
    auto& scalar = get<Tags::TestSolution>(vars_src.back());
    for (size_t i = 0; i < vals.size(); ++i) {
      get<>(scalar)[i] = vals[i];
    }
  };

  // Compute block_logical coordinates (which will not be used except
  // for its size and for determining which points are valid/invalid).
  // For this test, there are 10 points and all are valid; the call
  // to intrp::InterpolationTarget_detail::block_logical_coords should
  // set up those coordinates appropriately.
  auto& target_box =
      ActionTesting::get_databox<target_component>(make_not_null(&runner), 0);

  using BlockLogicalCoordType = std::vector<std::optional<
      IdPair<domain::BlockId, tnsr::I<double, 3, ::Frame::BlockLogical>>>>;
  BlockLogicalCoordType block_logical_coords{};
  if constexpr (UseTimeDepMaps) {
    block_logical_coords =
        intrp::InterpolationTarget_detail::block_logical_coords<
            typename metavars::InterpolationTargetA>(
            target_box, ActionTesting::cache<target_component>(runner, 0),
            first_temporal_id);
  } else {
    block_logical_coords =
        intrp::InterpolationTarget_detail::block_logical_coords<
            typename metavars::InterpolationTargetA>(target_box,
                                                     tmpl::type_<metavars>{});
  }

  // Add points at first_temporal_id
  add_to_vars_src({{3.0, 6.0}}, {{3, 6}});
  add_to_vars_src({{2.0, 7.0}}, {{2, 7}});
  ActionTesting::simple_action<
      target_component, intrp::Actions::InterpolationTargetVarsFromElement<
                            typename metavars::InterpolationTargetA>>(
      make_not_null(&runner), 0, vars_src, block_logical_coords, global_offsets,
      first_temporal_id);

  // It should know about only one temporal_id
  CHECK(ActionTesting::get_databox_tag<
            target_component, intrp::Tags::TemporalIds<temporal_id_type>>(
            runner, 0)
            .size() == 1);
  // It should have accumulated 4 points by now.
  CHECK(
      ActionTesting::get_databox_tag<
          target_component,
          intrp::Tags::IndicesOfFilledInterpPoints<temporal_id_type>>(runner, 0)
          .at(first_temporal_id)
          .size() == 4);
  CHECK(num_callback_calls == 0_st);

  // Add some more points at first_temporal_id
  vars_src.clear();
  global_offsets.clear();
  add_to_vars_src({{1.0, 888888.0}},
                  {{1, 6}});  // 6 is repeated, point will be ignored.
  add_to_vars_src({{8.0, 0.0, 4.0}}, {{8, 0, 4}});
  ActionTesting::simple_action<
      target_component, intrp::Actions::InterpolationTargetVarsFromElement<
                            typename metavars::InterpolationTargetA>>(
      make_not_null(&runner), 0, vars_src, block_logical_coords, global_offsets,
      first_temporal_id);

  // It should have interpolated 8 points by now. (The ninth point had
  // a repeated global_offsets so it should be ignored)
  CHECK(
      ActionTesting::get_databox_tag<
          target_component,
          intrp::Tags::IndicesOfFilledInterpPoints<temporal_id_type>>(runner, 0)
          .at(first_temporal_id)
          .size() == 8);

  // Add points at second_temporal_id
  vars_src.clear();
  global_offsets.clear();
  add_to_vars_src({{10.0, 16.0}}, {{5, 8}});
  add_to_vars_src({{6.0, 8.0}}, {{3, 4}});
  ActionTesting::simple_action<
      target_component, intrp::Actions::InterpolationTargetVarsFromElement<
                            typename metavars::InterpolationTargetA>>(
      make_not_null(&runner), 0, vars_src, block_logical_coords, global_offsets,
      second_temporal_id);

  // It should know about two temporal_ids
  CHECK(ActionTesting::get_databox_tag<
            target_component, intrp::Tags::TemporalIds<temporal_id_type>>(
            runner, 0)
            .size() == 2);
  // It should have accumulated 4 points for second_temporal_id.
  CHECK(
      ActionTesting::get_databox_tag<
          target_component,
          intrp::Tags::IndicesOfFilledInterpPoints<temporal_id_type>>(runner, 0)
          .at(second_temporal_id)
          .size() == 4);
  // ... and still have 8 points for first_temporal_id.
  CHECK(
      ActionTesting::get_databox_tag<
          target_component,
          intrp::Tags::IndicesOfFilledInterpPoints<temporal_id_type>>(runner, 0)
          .at(first_temporal_id)
          .size() == 8);
  CHECK(num_callback_calls == 0_st);

  // Add more points at second_temporal_id
  vars_src.clear();
  global_offsets.clear();
  add_to_vars_src({{2.0, 888888.0}},
                  {{1, 5}});  // 5 is repeated, point will be ignored.
  add_to_vars_src({{4.0, 0.0, 12.0}}, {{2, 0, 6}});
  ActionTesting::simple_action<
      target_component, intrp::Actions::InterpolationTargetVarsFromElement<
                            typename metavars::InterpolationTargetA>>(
      make_not_null(&runner), 0, vars_src, block_logical_coords, global_offsets,
      second_temporal_id);

  // It should have interpolated 8 points by now. (The ninth point had
  // a repeated global_offsets so it should be ignored)
  CHECK(
      ActionTesting::get_databox_tag<
          target_component,
          intrp::Tags::IndicesOfFilledInterpPoints<temporal_id_type>>(runner, 0)
          .at(second_temporal_id)
          .size() == 8);

  // Reset FoT expiration to something before the current slab value
  if constexpr (UseTimeDepMaps) {
    Parallel::mutate<domain::Tags::FunctionsOfTime, ResetFoT>(
        ActionTesting::cache<target_component>(runner, 0), name,
        first_temporal_id.substep_time() / 2.0);
  }

  // Now add enough points at second_temporal_id so it triggers the
  // callback (if the FoTs are ready).  (The first temporal id doesn't yet have
  // enough points, so here we also test asynchronicity).
  vars_src.clear();
  global_offsets.clear();
  add_to_vars_src({{18.0, 14.0}}, {{9, 7}});
  ActionTesting::simple_action<
      target_component, intrp::Actions::InterpolationTargetVarsFromElement<
                            typename metavars::InterpolationTargetA>>(
      make_not_null(&runner), 0, vars_src, block_logical_coords, global_offsets,
      second_temporal_id);

  if constexpr (UseTimeDepMaps) {
    // The interpolation should have finished, but the callback shouldn't have
    // been called and the target shouldn't have been cleaned up. So we should
    // still have two temporal ids, and all of the filled points. And the first
    // temporal id should not have been touched
    CHECK(ActionTesting::get_databox_tag<
              target_component,
              intrp::Tags::CompletedTemporalIds<temporal_id_type>>(runner, 0)
              .size() == 0);
    CHECK(ActionTesting::get_databox_tag<
              target_component, intrp::Tags::TemporalIds<temporal_id_type>>(
              runner, 0)
              .size() == 2);
    CHECK(ActionTesting::get_databox_tag<
              target_component,
              intrp::Tags::IndicesOfFilledInterpPoints<temporal_id_type>>(
              runner, 0)
              .count(second_temporal_id) == 1);
    CHECK(ActionTesting::get_databox_tag<
              target_component,
              intrp::Tags::IndicesOfFilledInterpPoints<temporal_id_type>>(
              runner, 0)
              .at(second_temporal_id)
              .size() == 10);
    CHECK(ActionTesting::get_databox_tag<
              target_component,
              intrp::Tags::IndicesOfFilledInterpPoints<temporal_id_type>>(
              runner, 0)
              .count(first_temporal_id) == 1);
    CHECK(ActionTesting::get_databox_tag<
              target_component,
              intrp::Tags::IndicesOfFilledInterpPoints<temporal_id_type>>(
              runner, 0)
              .at(first_temporal_id)
              .size() == 8);
    CHECK(num_callback_calls == 0_st);

    // Change the expiration back to the original value, which will cause a
    // simple action to run on the target.
    Parallel::mutate<domain::Tags::FunctionsOfTime, ResetFoT>(
        ActionTesting::cache<target_component>(runner, 0), name,
        init_expr_times.at(name));
    // Should be a queued simple action on the target component
    CHECK(ActionTesting::number_of_queued_simple_actions<target_component>(
              runner, 0) == 1);
    ActionTesting::invoke_queued_simple_action<target_component>(
        make_not_null(&runner), 0);
  }

  // It should have interpolated all the points by now,
  // and the list of points should have been cleaned up.
  CHECK(
      ActionTesting::get_databox_tag<
          target_component,
          intrp::Tags::IndicesOfFilledInterpPoints<temporal_id_type>>(runner, 0)
          .count(second_temporal_id) == 0);
  // There should be only 1 temporal_id left.
  // And its value should be first_temporal_id.
  CHECK(ActionTesting::get_databox_tag<
            target_component, intrp::Tags::TemporalIds<temporal_id_type>>(
            runner, 0)
            .size() == 1);
  CHECK(ActionTesting::get_databox_tag<
            target_component, intrp::Tags::TemporalIds<temporal_id_type>>(
            runner, 0)
            .front() == first_temporal_id);
  // There should be 1 CompletedTemporalId, and its value
  // should be second_temporal_id.
  CHECK(ActionTesting::get_databox_tag<
            target_component,
            intrp::Tags::CompletedTemporalIds<temporal_id_type>>(runner, 0)
            .size() == 1);
  CHECK(ActionTesting::get_databox_tag<
            target_component,
            intrp::Tags::CompletedTemporalIds<temporal_id_type>>(runner, 0)
            .front() == second_temporal_id);
  // There are two callbacks
  CHECK(num_callback_calls == 2_st);

  // Now add enough points at first_temporal_id so it triggers the
  // callback.
  vars_src.clear();
  global_offsets.clear();
  add_to_vars_src({{9.0, 5.0}}, {{9, 5}});
  ActionTesting::simple_action<
      target_component, intrp::Actions::InterpolationTargetVarsFromElement<
                            typename metavars::InterpolationTargetA>>(
      make_not_null(&runner), 0, vars_src, block_logical_coords, global_offsets,
      first_temporal_id);

  // It should have interpolated all the points by now,
  // and the list of points should have been cleaned up.
  CHECK(
      ActionTesting::get_databox_tag<
          target_component,
          intrp::Tags::IndicesOfFilledInterpPoints<temporal_id_type>>(runner, 0)
          .count(first_temporal_id) == 0);
  // There should be no temporal_ids left.
  CHECK(ActionTesting::get_databox_tag<
            target_component, intrp::Tags::TemporalIds<temporal_id_type>>(
            runner, 0)
            .empty());
  // There should be 2 CompletedTemporalIds
  CHECK(ActionTesting::get_databox_tag<
            target_component,
            intrp::Tags::CompletedTemporalIds<temporal_id_type>>(runner, 0)
            .size() == 2);
  CHECK(ActionTesting::get_databox_tag<
            target_component,
            intrp::Tags::CompletedTemporalIds<temporal_id_type>>(runner, 0)
            .at(0) == second_temporal_id);
  CHECK(ActionTesting::get_databox_tag<
            target_component,
            intrp::Tags::CompletedTemporalIds<temporal_id_type>>(runner, 0)
            .at(1) == first_temporal_id);
  // There are two more callbacks
  CHECK(num_callback_calls == 4_st);

  // Should be no queued simple action.
  CHECK(
      ActionTesting::is_simple_action_queue_empty<target_component>(runner, 0));
}

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Interpolator.TargetVarsFromElement",
                  "[Unit]") {
  test<false>();
  test<true>();
}
}  // namespace
