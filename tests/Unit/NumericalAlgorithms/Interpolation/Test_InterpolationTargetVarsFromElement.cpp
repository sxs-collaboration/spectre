// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Shell.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Interpolation/Actions/InterpolationTargetVarsFromElement.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolationTarget.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Utilities/MakeWithValue.hpp"

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
                       const Scalar<DataVector>& x) noexcept {
    get(*result) = square(get(x));
  }
  using argument_tags = tmpl::list<TestSolution>;
  using base = Square;
  using return_type = Scalar<DataVector>;
};
}  // namespace Tags

struct MockComputeTargetPoints {
  using is_sequential = std::false_type;
  template <typename Metavariables, typename DbTags>
  static tnsr::I<DataVector, 3, Frame::Inertial> points(
      const db::DataBox<DbTags>& /*box*/,
      const tmpl::type_<Metavariables>& /*meta*/) noexcept {
    // Need 10 points to agree with test.
    const size_t num_pts = 10;
    // Doesn't matter what the points are; they are not used except
    // that they need to be inside the domain.
    tnsr::I<DataVector, 3, Frame::Inertial> target_points(num_pts);
    for (size_t n = 0; n < num_pts; ++n) {
      for (size_t d=0;d<3;++d) {
        target_points.get(d)[n] = 1.0 + 0.01 * n + 0.5 * d;
      }
    }
    return target_points;
  }
};

struct MockPostInterpolationCallback {
  template <typename DbTags, typename Metavariables>
  static void apply(const db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const double time) noexcept {
    // This callback simply checks that the points are as expected.
    const double first_time = 13.0 / 15.0;
    const double second_time = 14.0 / 15.0;
    if (time == first_time) {
      const Scalar<DataVector> expected{
          DataVector{{0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0}}};
      CHECK_ITERABLE_APPROX(expected, db::get<Tags::Square>(box));
    } else if (time == second_time) {
      const Scalar<DataVector> expected{DataVector{
          {0.0, 4.0, 16.0, 36.0, 64.0, 100.0, 144.0, 196.0, 256.0, 324.0}}};
      CHECK_ITERABLE_APPROX(expected, db::get<Tags::Square>(box));
    } else {
      // Should never get here.
      CHECK(false);
    }
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
  using simple_tags = typename intrp::Actions::InitializeInterpolationTarget<
      Metavariables, InterpolationTargetTag>::simple_tags;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<
          simple_tags,
          typename InterpolationTargetTag::compute_items_on_target>>>>;
};

struct MockMetavariables {
  struct InterpolationTargetA {
    using vars_to_interpolate_to_target = tmpl::list<Tags::TestSolution>;
    using compute_items_on_target = tmpl::list<Tags::SquareCompute>;
    using compute_target_points = MockComputeTargetPoints;
    using post_interpolation_callback = MockPostInterpolationCallback;
  };
  static constexpr size_t volume_dim = 3;
  using interpolation_target_tags = tmpl::list<InterpolationTargetA>;

  using component_list = tmpl::list<
      mock_interpolation_target<MockMetavariables, InterpolationTargetA>>;
  enum class Phase { Initialization, Testing, Exit };
};

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Interpolator.TargetVarsFromElement",
                  "[Unit]") {
  domain::creators::register_derived_with_charm();

  using metavars = MockMetavariables;
  using target_component =
      mock_interpolation_target<metavars,
                                typename metavars::InterpolationTargetA>;

  const double first_time = 13.0 / 15.0;
  const double second_time = 14.0 / 15.0;
  const auto domain_creator =
      domain::creators::Shell(0.9, 4.9, 1, {{5, 5}}, false);

  // Type alias for better readability below.
  using vars_type = Variables<
      typename metavars::InterpolationTargetA::vars_to_interpolate_to_target>;

  // Initialization
  ActionTesting::MockRuntimeSystem<metavars> runner{
      {domain_creator.create_domain()}};
  ActionTesting::emplace_component_and_initialize<target_component>(
      &runner, 0,
      {std::unordered_map<double, std::unordered_set<size_t>>{},
       std::unordered_map<double, std::unordered_set<size_t>>{},
       std::deque<double>{}, std::deque<double>{}, std::deque<double>{},
       std::unordered_map<double,
                          Variables<typename metavars::InterpolationTargetA::
                                        vars_to_interpolate_to_target>>{},
       // Default-constructed Variables cause problems, so below
       // we construct the Variables with a single point.
       vars_type{1}});
  ActionTesting::set_phase(make_not_null(&runner), metavars::Phase::Testing);

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

  // Add points at first_time
  add_to_vars_src({{3.0, 6.0}}, {{3, 6}});
  add_to_vars_src({{2.0, 7.0}}, {{2, 7}});
  ActionTesting::simple_action<
      target_component, intrp::Actions::InterpolationTargetVarsFromElement<
                            typename metavars::InterpolationTargetA>>(
      make_not_null(&runner), 0, vars_src, global_offsets, first_time);

  // It should know about only one time
  CHECK(ActionTesting::get_databox_tag<target_component, intrp::Tags::Times>(
            runner, 0)
            .size() == 1);
  // It should have accumulated 4 points by now.
  CHECK(
      ActionTesting::get_databox_tag<target_component,
                                     intrp::Tags::IndicesOfFilledInterpPoints>(
          runner, 0)
          .at(first_time)
          .size() == 4);

  // Add some more points at first_time
  vars_src.clear();
  global_offsets.clear();
  add_to_vars_src({{1.0, 888888.0}},
                  {{1, 6}});  // 6 is repeated, point will be ignored.
  add_to_vars_src({{8.0, 0.0, 4.0}}, {{8, 0, 4}});
  ActionTesting::simple_action<
      target_component, intrp::Actions::InterpolationTargetVarsFromElement<
                            typename metavars::InterpolationTargetA>>(
      make_not_null(&runner), 0, vars_src, global_offsets, first_time);

  // It should have interpolated 8 points by now. (The ninth point had
  // a repeated global_offsets so it should be ignored)
  CHECK(
      ActionTesting::get_databox_tag<target_component,
                                     intrp::Tags::IndicesOfFilledInterpPoints>(
          runner, 0)
          .at(first_time)
          .size() == 8);

  // Add points at second_time
  vars_src.clear();
  global_offsets.clear();
  add_to_vars_src({{10.0, 16.0}}, {{5, 8}});
  add_to_vars_src({{6.0, 8.0}}, {{3, 4}});
  ActionTesting::simple_action<
      target_component, intrp::Actions::InterpolationTargetVarsFromElement<
                            typename metavars::InterpolationTargetA>>(
      make_not_null(&runner), 0, vars_src, global_offsets, second_time);

  // It should know about two times
  CHECK(ActionTesting::get_databox_tag<target_component, intrp::Tags::Times>(
            runner, 0)
            .size() == 2);
  // It should have accumulated 4 points for second_time.
  CHECK(
      ActionTesting::get_databox_tag<target_component,
                                     intrp::Tags::IndicesOfFilledInterpPoints>(
          runner, 0)
          .at(second_time)
          .size() == 4);
  // ... and still have 8 points for first_time.
  CHECK(
      ActionTesting::get_databox_tag<target_component,
                                     intrp::Tags::IndicesOfFilledInterpPoints>(
          runner, 0)
          .at(first_time)
          .size() == 8);

  // Add more points at second_time
  vars_src.clear();
  global_offsets.clear();
  add_to_vars_src({{2.0, 888888.0}},
                  {{1, 5}});  // 5 is repeated, point will be ignored.
  add_to_vars_src({{4.0, 0.0, 12.0}}, {{2, 0, 6}});
  ActionTesting::simple_action<
      target_component, intrp::Actions::InterpolationTargetVarsFromElement<
                            typename metavars::InterpolationTargetA>>(
      make_not_null(&runner), 0, vars_src, global_offsets, second_time);

  // It should have interpolated 8 points by now. (The ninth point had
  // a repeated global_offsets so it should be ignored)
  CHECK(
      ActionTesting::get_databox_tag<target_component,
                                     intrp::Tags::IndicesOfFilledInterpPoints>(
          runner, 0)
          .at(second_time)
          .size() == 8);

  // Now add enough points at second_time so it triggers the
  // callback.  (The first time doesn't yet have enough points, so
  // here we also test asynchronicity).
  vars_src.clear();
  global_offsets.clear();
  add_to_vars_src({{18.0, 14.0}}, {{9, 7}});
  ActionTesting::simple_action<
      target_component, intrp::Actions::InterpolationTargetVarsFromElement<
                            typename metavars::InterpolationTargetA>>(
      make_not_null(&runner), 0, vars_src, global_offsets, second_time);

  // It should have interpolated all the points by now,
  // and the list of points should have been cleaned up.
  CHECK(
      ActionTesting::get_databox_tag<target_component,
                                     intrp::Tags::IndicesOfFilledInterpPoints>(
          runner, 0)
          .count(second_time) == 0);
  // There should be only 1 time left.
  // And its value should be first_time.
  CHECK(ActionTesting::get_databox_tag<target_component, intrp::Tags::Times>(
            runner, 0)
            .size() == 1);
  CHECK(ActionTesting::get_databox_tag<target_component, intrp::Tags::Times>(
            runner, 0)
            .front() == first_time);
  // There should be 1 CompletedTime, and its value
  // should be second_time.
  CHECK(ActionTesting::get_databox_tag<target_component,
                                       intrp::Tags::CompletedTimes>(runner, 0)
            .size() == 1);
  CHECK(ActionTesting::get_databox_tag<target_component,
                                       intrp::Tags::CompletedTimes>(runner, 0)
            .front() == second_time);

  // Now add enough points at first_time so it triggers the
  // callback.
  vars_src.clear();
  global_offsets.clear();
  add_to_vars_src({{9.0, 5.0}}, {{9, 5}});
  ActionTesting::simple_action<
      target_component, intrp::Actions::InterpolationTargetVarsFromElement<
                            typename metavars::InterpolationTargetA>>(
      make_not_null(&runner), 0, vars_src, global_offsets, first_time);

  // It should have interpolated all the points by now,
  // and the list of points should have been cleaned up.
  CHECK(
      ActionTesting::get_databox_tag<target_component,
                                     intrp::Tags::IndicesOfFilledInterpPoints>(
          runner, 0)
          .count(first_time) == 0);
  // There should be no times left.
  CHECK(ActionTesting::get_databox_tag<target_component, intrp::Tags::Times>(
            runner, 0)
            .empty());
  // There should be 2 CompletedTimes
  CHECK(ActionTesting::get_databox_tag<target_component,
                                       intrp::Tags::CompletedTimes>(runner, 0)
            .size() == 2);
  CHECK(ActionTesting::get_databox_tag<target_component,
                                       intrp::Tags::CompletedTimes>(runner, 0)
            .at(0) == second_time);
  CHECK(ActionTesting::get_databox_tag<target_component,
                                       intrp::Tags::CompletedTimes>(runner, 0)
            .at(1) == first_time);

  // Should be no queued simple action.
  CHECK(
      ActionTesting::is_simple_action_queue_empty<target_component>(runner, 0));
}
}  // namespace
