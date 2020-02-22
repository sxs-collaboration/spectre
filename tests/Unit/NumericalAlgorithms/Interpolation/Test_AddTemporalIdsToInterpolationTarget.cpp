// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <deque>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/Creators/Shell.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Tags.hpp"
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
#include "tests/Unit/ActionTesting.hpp"
// IWYU pragma: no_forward_declare db::DataBox

// IWYU pragma: no_include <boost/variant/get.hpp>

/// \cond
class DataVector;
namespace intrp {
namespace Tags {
struct IndicesOfFilledInterpPoints;
template <typename Metavariables>
struct TemporalIds;
}  // namespace Tags
}  // namespace intrp
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
/// \endcond

namespace {

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
    CHECK(temporal_id == TimeStepId(true, 0, Time(slab, 0)));
    // Put something in IndicesOfFilledInterpPts so we can check later whether
    // this function was called.  This isn't the usual usage of
    // IndicesOfFilledInterpPoints.
    db::mutate<::intrp::Tags::IndicesOfFilledInterpPoints>(
        make_not_null(&box),
        [](const gsl::not_null<
            db::item_type<::intrp::Tags::IndicesOfFilledInterpPoints>*>
               indices) noexcept { indices->insert(indices->size() + 1); });
  }
};

struct MockMetavariables {
  struct InterpolationTargetA {
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
    using compute_items_on_target = tmpl::list<>;
    using compute_target_points = MockComputeTargetPoints;
  };
  using temporal_id = ::Tags::TimeStepId;

  using component_list = tmpl::list<
      mock_interpolation_target<MockMetavariables, InterpolationTargetA>>;
  enum class Phase { Initialization, Testing, Exit };
};

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.InterpolationTarget.AddTemporalIds",
                  "[Unit]") {
  using metavars = MockMetavariables;
  using target_component =
      mock_interpolation_target<metavars, metavars::InterpolationTargetA>;

  const auto domain_creator =
      domain::creators::Shell(0.9, 4.9, 1, {{5, 5}}, false);

  ActionTesting::MockRuntimeSystem<metavars> runner{
      {domain_creator.create_domain()}};
  ActionTesting::emplace_component<target_component>(&runner, 0);
  ActionTesting::next_action<target_component>(make_not_null(&runner), 0);
  runner.set_phase(metavars::Phase::Testing);

  CHECK(ActionTesting::get_databox_tag<target_component,
                                       ::intrp::Tags::TemporalIds<metavars>>(
            runner, 0)
            .empty());

  Slab slab(0.0, 1.0);
  const std::vector<TimeStepId> temporal_ids = {
      TimeStepId(true, 0, Time(slab, 0)),
      TimeStepId(true, 0, Time(slab, Rational(1, 3)))};

  runner.simple_action<target_component,
                       ::intrp::Actions::AddTemporalIdsToInterpolationTarget<
                           metavars::InterpolationTargetA>>(0, temporal_ids);

  CHECK(ActionTesting::get_databox_tag<target_component,
                                       ::intrp::Tags::TemporalIds<metavars>>(
            runner, 0) ==
        std::deque<TimeStepId>(temporal_ids.begin(), temporal_ids.end()));

  // Add the same temporal_ids again, which should do nothing...
  runner.simple_action<target_component,
                       ::intrp::Actions::AddTemporalIdsToInterpolationTarget<
                           metavars::InterpolationTargetA>>(0, temporal_ids);
  // ...and check that it indeed did nothing.
  CHECK(ActionTesting::get_databox_tag<target_component,
                                       ::intrp::Tags::TemporalIds<metavars>>(
            runner, 0) ==
        std::deque<TimeStepId>(temporal_ids.begin(), temporal_ids.end()));

  runner.invoke_queued_simple_action<target_component>(0);

  // Check that MockComputeTargetPoints was called.
  CHECK(ActionTesting::get_databox_tag<
            target_component, ::intrp::Tags::IndicesOfFilledInterpPoints>(
            runner, 0)
            .size() == 1);

  // Call again; it should not call MockComputeTargetPoints this time.
  const std::vector<TimeStepId> temporal_ids_2 = {
      TimeStepId(true, 0, Time(slab, Rational(2, 3))),
      TimeStepId(true, 0, Time(slab, Rational(3, 3)))};
  runner.simple_action<target_component,
                       ::intrp::Actions::AddTemporalIdsToInterpolationTarget<
                           metavars::InterpolationTargetA>>(0, temporal_ids_2);

  // Check that MockComputeTargetPoints was not called.
  CHECK(runner.is_simple_action_queue_empty<target_component>(0));
}
}  // namespace
