// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <deque>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainCreators/Shell.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatedVars.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InterpolationTargetReceiveVars.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Rational.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

/// \cond
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables
namespace intrp {
namespace Actions {
template <typename InterpolationTargetTag, size_t VolumeDim>
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
      intrp::InterpolationTarget<Metavariables, InterpolationTargetTag, 3>;
  using const_global_cache_tag_list = tmpl::list<>;
  using action_list = tmpl::list<>;
  using initial_databox = db::compute_databox_type<
      typename intrp::Actions::InitializeInterpolationTarget<
          InterpolationTargetTag>::template return_tag_list<Metavariables, 3>>;
};

template <typename InterpolationTargetTag, size_t VolumeDim>
struct MockCleanUpInterpolator {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl::list_contains_v<
                DbTags, typename intrp::Tags::NumberOfElements>> = nullptr>
  static void apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/,
      const typename Metavariables::temporal_id& temporal_id) noexcept {
    Slab slab(0.0, 1.0);
    CHECK(temporal_id == Time(slab, Rational(13, 15)));
    // Put something in NumberOfElements so we can check later whether
    // this function was called.  This isn't the usual usage of
    // NumberOfElements.
    db::mutate<intrp::Tags::NumberOfElements>(
        make_not_null(&box), [](const gsl::not_null<
                                 db::item_type<intrp::Tags::NumberOfElements>*>
                                    number_of_elements) noexcept {
          ++(*number_of_elements);
        });
  }
};

struct MockComputeTargetPoints {
  template <
      typename DbTags, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<
          DbTags, typename intrp::Tags::TemporalIds<Metavariables>>> = nullptr>
  static void apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/,
      const typename Metavariables::temporal_id& temporal_id) noexcept {
    Slab slab(0.0, 1.0);
    CHECK(temporal_id == Time(slab, Rational(14, 15)));
    // Increment IndicesOfFilledInterpPoints so we can check later
    // whether this function was called.  This isn't the usual usage
    // of IndicesOfFilledInterpPoints; this is done only for the test.
    db::mutate<intrp::Tags::IndicesOfFilledInterpPoints>(
        make_not_null(&box), [](const gsl::not_null<db::item_type<
                                    intrp::Tags::IndicesOfFilledInterpPoints>*>
                                    indices) noexcept {
          indices->insert(indices->size() + 1);
        });
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

struct MockPostInterpolationCallback {
  template <typename DbTags, typename Metavariables>
  static void apply(
      const db::DataBox<DbTags>& box,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const typename Metavariables::temporal_id& temporal_id) noexcept {
    Slab slab(0.0, 1.0);
    CHECK(temporal_id == Time(slab, Rational(13, 15)));
    // The result should be the square of the first 10 integers, in
    // a Scalar<DataVector>.
    const Scalar<DataVector> expected{
        DataVector{{0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0}}};
    CHECK_ITERABLE_APPROX(expected, db::get<Tags::Square>(box));
  }
};

template <typename Metavariables, size_t VolumeDim>
struct mock_interpolator {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tag_list = tmpl::list<>;
  using action_list = tmpl::list<>;
  using initial_databox =
      db::compute_databox_type<typename intrp::Actions::InitializeInterpolator<
          VolumeDim>::template return_tag_list<Metavariables>>;
  using component_being_mocked = intrp::Interpolator<Metavariables, VolumeDim>;
  using replace_these_simple_actions =
      tmpl::list<intrp::Actions::CleanUpInterpolator<
          typename Metavariables::InterpolationTargetA, 3>>;
  using with_these_simple_actions = tmpl::list<
      MockCleanUpInterpolator<typename Metavariables::InterpolationTargetA, 3>>;
};

struct MockMetavariables {
  struct InterpolationTargetA {
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
    using compute_target_points = MockComputeTargetPoints;
    using post_interpolation_callback = MockPostInterpolationCallback;
    using compute_items_on_target = tmpl::list<Tags::SquareComputeItem>;
  };
  using interpolator_source_vars = tmpl::list<gr::Tags::Lapse<DataVector>>;
  using interpolation_target_tags = tmpl::list<InterpolationTargetA>;
  using temporal_id = Time;
  using domain_frame = Frame::Inertial;

  using component_list = tmpl::list<
      mock_interpolation_target<MockMetavariables, InterpolationTargetA>,
      mock_interpolator<MockMetavariables, 3>>;
  using const_global_cache_tag_list = tmpl::list<>;
  enum class Phase { Initialize, Exit };
};

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.InterpolationTarget.ReceiveVars",
                  "[Unit]") {
  using metavars = MockMetavariables;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  using TupleOfMockDistributedObjects =
      MockRuntimeSystem::TupleOfMockDistributedObjects;
  TupleOfMockDistributedObjects dist_objects{};
  using MockDistributedObjectsTagTarget =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          mock_interpolation_target<metavars, metavars::InterpolationTargetA>>;
  using MockDistributedObjectsTagInterpolator =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          mock_interpolator<metavars, 3>>;

  const auto domain_creator =
      DomainCreators::Shell<Frame::Inertial>(0.9, 4.9, 1, {{5, 5}}, false);
  Slab slab(0.0, 1.0);
  const size_t num_points = 10;

  // Set databox to contain two temporal_ids and a vars of num_points points.
  tuples::get<MockDistributedObjectsTagTarget>(dist_objects)
      .emplace(
          0,
          ActionTesting::MockDistributedObject<mock_interpolation_target<
              metavars, metavars::InterpolationTargetA>>{
              db::create<
                  db::get_items<intrp::Actions::InitializeInterpolationTarget<
                      metavars::InterpolationTargetA>::return_tag_list<metavars,
                                                                       3>>>(
                  db::item_type<intrp::Tags::IndicesOfFilledInterpPoints>{},
                  db::item_type<intrp::Tags::TemporalIds<metavars>>{
                      Time(slab, Rational(13, 15)),
                      Time(slab, Rational(14, 15))},
                  domain_creator.create_domain(),
                  db::item_type<
                      ::Tags::Variables<metavars::InterpolationTargetA::
                                            vars_to_interpolate_to_target>>{
                      num_points})});

  tuples::get<MockDistributedObjectsTagInterpolator>(dist_objects)
      .emplace(0, ActionTesting::MockDistributedObject<
                      mock_interpolator<metavars, 3>>{});

  MockRuntimeSystem runner{{}, std::move(dist_objects)};

  runner.simple_action<mock_interpolator<metavars, 3>,
                       intrp::Actions::InitializeInterpolator<3>>(0);

  const auto& box_target =
      runner
          .template algorithms<mock_interpolation_target<
              metavars, metavars::InterpolationTargetA>>()
          .at(0)
          .template get_databox<typename mock_interpolation_target<
              metavars, metavars::InterpolationTargetA>::initial_databox>();

  const auto& box_interpolator =
      runner.template algorithms<mock_interpolator<metavars, 3>>()
          .at(0)
          .template get_databox<
              typename mock_interpolator<metavars, 3>::initial_databox>();

  // Now set up the vars.
  std::vector<db::item_type<::Tags::Variables<
      metavars::InterpolationTargetA::vars_to_interpolate_to_target>>>
      vars_src;
  std::vector<std::vector<size_t>> global_offsets;

  // Adds more data to vars_src and global_offsets.
  auto add_to_vars_src = [&vars_src, &global_offsets](
                             const std::vector<double>& lapse_vals,
                             const std::vector<size_t>& offset_vals) {
    vars_src.emplace_back(
        db::item_type<::Tags::Variables<
            metavars::InterpolationTargetA::vars_to_interpolate_to_target>>{
            lapse_vals.size()});
    global_offsets.emplace_back(offset_vals);
    auto& lapse = get<gr::Tags::Lapse<DataVector>>(vars_src.back());
    for (size_t i = 0; i < lapse_vals.size(); ++i) {
      get<>(lapse)[i] = lapse_vals[i];
    }
  };

  add_to_vars_src({{3.0, 6.0}}, {{3, 6}});
  add_to_vars_src({{2.0, 7.0}}, {{2, 7}});

  runner.simple_action<
      mock_interpolation_target<metavars, metavars::InterpolationTargetA>,
      intrp::Actions::InterpolationTargetReceiveVars<
          metavars::InterpolationTargetA, 3>>(0, vars_src, global_offsets);

  // It should have interpolated 4 points by now.
  CHECK(db::get<intrp::Tags::IndicesOfFilledInterpPoints>(box_target).size() ==
        4);

  // Should be no queued simple action until we get num_points points.
  CHECK(runner.is_simple_action_queue_empty<
        mock_interpolation_target<metavars, metavars::InterpolationTargetA>>(
      0));
  CHECK(runner.is_simple_action_queue_empty<mock_interpolator<metavars, 3>>(0));
  CHECK(db::get<intrp::Tags::TemporalIds<metavars>>(box_target).size() == 2);

  vars_src.clear();
  global_offsets.clear();
  add_to_vars_src({{1.0, 888888.0}},
                  {{1, 6}});  // 6 is repeated, point will be ignored.
  add_to_vars_src({{8.0, 0.0, 4.0}}, {{8, 0, 4}});

  runner.simple_action<
      mock_interpolation_target<metavars, metavars::InterpolationTargetA>,
      intrp::Actions::InterpolationTargetReceiveVars<
          metavars::InterpolationTargetA, 3>>(0, vars_src, global_offsets);

  // It should have interpolated 8 points by now. (The ninth point had
  // a repeated global_offsets so it should be ignored)
  CHECK(db::get<intrp::Tags::IndicesOfFilledInterpPoints>(box_target).size() ==
        8);

  // Should be no queued simple action until we have added 10 points.
  CHECK(runner.is_simple_action_queue_empty<
        mock_interpolation_target<metavars, metavars::InterpolationTargetA>>(
      0));
  CHECK(runner.is_simple_action_queue_empty<mock_interpolator<metavars, 3>>(0));

  vars_src.clear();
  global_offsets.clear();
  add_to_vars_src({{9.0, 5.0}}, {{9, 5}});

  // This will call InterpolationTargetA::post_interpolation_callback
  // where we check that the points are correct.
  runner.simple_action<
      mock_interpolation_target<metavars, metavars::InterpolationTargetA>,
      intrp::Actions::InterpolationTargetReceiveVars<
          metavars::InterpolationTargetA, 3>>(0, vars_src, global_offsets);

  // It should have interpolated all the points by now.
  CHECK(db::get<intrp::Tags::IndicesOfFilledInterpPoints>(box_target).size() ==
        num_points);

  // Now there should be a queued simple action, which is
  // CleanUpInterpolator, which here we mock.
  runner.invoke_queued_simple_action<mock_interpolator<metavars, 3>>(0);

  // Check that MockCleanUpInterpolator was called.  It resets the
  // (fake) number of elements, specifically so we can test it here.
  CHECK(db::get<intrp::Tags::NumberOfElements>(box_interpolator) == 1);

  // Also, there should be only 1 TemporalId left.
  CHECK(db::get<intrp::Tags::TemporalIds<metavars>>(box_target).size() == 1);

  // And there is yet one more simple action, compute_target_points,
  // which here we mock just to check that it is called.
  runner.invoke_queued_simple_action<
      mock_interpolation_target<metavars, metavars::InterpolationTargetA>>(0);

  // Check that MockComputeTargetPoints was called.
  // It sets a (fake) value of IndicesOfFilledInterpPoints for the express
  // purpose of this check.
  CHECK(db::get<intrp::Tags::IndicesOfFilledInterpPoints>(box_target).size() ==
        num_points + 1);

  // There should be no more queued actions; verify this.
  CHECK(runner.is_simple_action_queue_empty<
        mock_interpolation_target<metavars, metavars::InterpolationTargetA>>(
      0));
  CHECK(runner.is_simple_action_queue_empty<mock_interpolator<metavars, 3>>(0));
}

}  // namespace
