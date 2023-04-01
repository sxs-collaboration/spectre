// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <deque>
#include <string>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Domain.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/LineSegment.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

class DataVector;

namespace {

template <typename Metavariables, typename InterpolationTargetTag>
struct mock_interpolation_target {
  static_assert(
      tt::assert_conforms_to_v<InterpolationTargetTag,
                               intrp::protocols::InterpolationTargetTag>);
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<3>>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<intrp::Actions::InitializeInterpolationTarget<
          Metavariables, InterpolationTargetTag>>>>;
};

struct Metavariables {
  struct InterpolationTargetA
      : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::TimeStepId;
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
    using compute_items_on_target = tmpl::list<>;
    using post_interpolation_callback =
        intrp::callbacks::ObserveTimeSeriesOnSurface<tmpl::list<>,
                                                     InterpolationTargetA>;
    using compute_target_points =
        ::intrp::TargetPoints::LineSegment<InterpolationTargetA, 3,
                                           Frame::Inertial>;
  };

  using component_list = tmpl::list<
      mock_interpolation_target<Metavariables, InterpolationTargetA>>;
};

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.InterpolationTarget.Initialize",
                  "[Unit]") {
  domain::creators::register_derived_with_charm();
  using metavars = Metavariables;
  using temporal_id_type =
      typename metavars::InterpolationTargetA::temporal_id::type;
  using component =
      mock_interpolation_target<metavars,
                                typename metavars::InterpolationTargetA>;
  const auto domain_creator = domain::creators::Sphere(
      0.9, 4.9, domain::creators::Sphere::Excision{}, 1_st, 5_st, false);

  ActionTesting::MockRuntimeSystem<metavars> runner{
      {domain_creator.create_domain()}};
  ActionTesting::set_phase(make_not_null(&runner),
                           Parallel::Phase::Initialization);
  ActionTesting::emplace_component<component>(&runner, 0);
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<component>(make_not_null(&runner), 0);
  }
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  CHECK(ActionTesting::get_databox_tag<
            component,
            ::intrp::Tags::IndicesOfFilledInterpPoints<temporal_id_type>>(
            runner, 0)
            .empty());
  CHECK(ActionTesting::get_databox_tag<
            component, ::intrp::Tags::TemporalIds<temporal_id_type>>(runner, 0)
            .empty());

  const auto& cache = ActionTesting::cache<component>(runner, 0_st);
  CHECK(Parallel::get<domain::Tags::Domain<3>>(cache) ==
        domain_creator.create_domain());

  CHECK(ActionTesting::get_databox_tag<
            component, ::intrp::Tags::InterpolatedVars<
                           metavars::InterpolationTargetA, temporal_id_type>>(
            runner, 0)
            .empty());
}

}  // namespace
