// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/ParallelAlgorithms/Interpolation/InterpolateOnElementTestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "ParallelAlgorithms/Interpolation/Events/InterpolateWithoutInterpComponent.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/ComputeVarsToInterpolate.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/LineSegment.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {

template <typename Metavariables>
struct mock_element {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Metavariables::volume_dim>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
  using initial_databox = db::compute_databox_type<db::AddSimpleTags<>>;
};

template <typename Metavariables, typename ElemComponent>
struct initialize_elements_and_queue_simple_actions {
  template <typename InterpPointInfo, typename Runner, typename TemporalId>
  void operator()(const DomainCreator<3>& domain_creator,
                  const Domain<3>& domain,
                  const std::vector<ElementId<3>>& element_ids,
                  const InterpPointInfo& interp_point_info, Runner& runner,
                  const TemporalId& temporal_id) {
    using metavars = Metavariables;
    using elem_component = ElemComponent;
    // Emplace elements.
    for (const auto& element_id : element_ids) {
      ActionTesting::emplace_component<elem_component>(&runner, element_id);
      ActionTesting::next_action<elem_component>(make_not_null(&runner),
                                                 element_id);
    }
    ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

    // Create event.
    typename metavars::event event{};

    CHECK(event.needs_evolved_variables());

    // Run event on all elements.
    for (const auto& element_id : element_ids) {
      // 1. Get vars and mesh
      const auto& [vars, mesh] =
          InterpolateOnElementTestHelpers::make_volume_data_and_mesh<
              ElemComponent, Metavariables::use_time_dependent_maps>(
              domain_creator, runner, domain, element_id, temporal_id);

      // 2. Make a box
      const auto box = db::create<db::AddSimpleTags<
          Parallel::Tags::MetavariablesImpl<metavars>,
          typename metavars::InterpolationTargetA::temporal_id,
          intrp::Tags::InterpPointInfo<metavars>,
          domain::Tags::Mesh<metavars::volume_dim>,
          ::Tags::Variables<
              typename std::remove_reference_t<decltype(vars)>::tags_list>>>(
          metavars{}, temporal_id, interp_point_info, mesh, vars);

      // 3. Run the event.  This will invoke simple actions on
      // InterpolationTarget.
      event.run(
          make_observation_box<
              typename metavars::event::compute_tags_for_observation_box>(box),
          ActionTesting::cache<elem_component>(runner, element_id), element_id,
          std::add_pointer_t<elem_component>{});
    }
  }
};

template <bool HaveComputeVarsToInterpolate, bool UseTimeDependentMaps>
struct MockMetavariables {
  static constexpr bool use_time_dependent_maps = UseTimeDependentMaps;
  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<3>>;
  using mutable_global_cache_tags =
      tmpl::conditional_t<use_time_dependent_maps,
                          tmpl::list<domain::Tags::FunctionsOfTimeInitialize>,
                          tmpl::list<>>;
  struct InterpolationTargetAWithComputeVarsToInterpolate
      : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::TimeStepId;
    using compute_items_on_target = tmpl::list<>;
    using vars_to_interpolate_to_target =
        tmpl::list<InterpolateOnElementTestHelpers::Tags::MultiplyByTwo>;
    using compute_vars_to_interpolate =
        InterpolateOnElementTestHelpers::ComputeMultiplyByTwo;
    // The following are not used in this test, but must be there to
    // conform to the protocol.
    using compute_target_points = ::intrp::TargetPoints::LineSegment<
        InterpolationTargetAWithComputeVarsToInterpolate, 3, Frame::Inertial>;
    using post_interpolation_callback =
        intrp::callbacks::ObserveTimeSeriesOnSurface<
            tmpl::list<>, InterpolationTargetAWithComputeVarsToInterpolate>;
  };
  struct InterpolationTargetAWithoutComputeVarsToInterpolate
      : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::TimeStepId;
    using compute_items_on_target = tmpl::list<>;
    using vars_to_interpolate_to_target =
        tmpl::list<InterpolateOnElementTestHelpers::Tags::TestSolution>;
    // The following are not used in this test, but must be there to
    // conform to the protocol.
    using compute_target_points = ::intrp::TargetPoints::LineSegment<
        InterpolationTargetAWithoutComputeVarsToInterpolate, 3,
        Frame::Inertial>;
    using post_interpolation_callback =
        intrp::callbacks::ObserveTimeSeriesOnSurface<
            tmpl::list<>, InterpolationTargetAWithoutComputeVarsToInterpolate>;
  };
  using InterpolationTargetA =
      tmpl::conditional_t<HaveComputeVarsToInterpolate,
                          InterpolationTargetAWithComputeVarsToInterpolate,
                          InterpolationTargetAWithoutComputeVarsToInterpolate>;
  static constexpr size_t volume_dim = 3;
  using interpolation_target_tags = tmpl::list<InterpolationTargetA>;

  using component_list =
      tmpl::list<InterpolateOnElementTestHelpers::mock_interpolation_target<
                     MockMetavariables, InterpolationTargetA>,
                 mock_element<MockMetavariables>>;

  using event = intrp::Events::InterpolateWithoutInterpComponent<
      volume_dim, InterpolationTargetA, MockMetavariables,
      tmpl::list<InterpolateOnElementTestHelpers::Tags::TestSolution>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<Event, tmpl::list<event>>>;
  };
};

template <typename MockMetavariables>
void run_test() {
  using metavars = MockMetavariables;
  using elem_component = mock_element<metavars>;
  InterpolateOnElementTestHelpers::test_interpolate_on_element<metavars,
                                                               elem_component>(
      initialize_elements_and_queue_simple_actions<metavars, elem_component>{});
}

SPECTRE_TEST_CASE(
    "Unit.NumericalAlgorithms.Interpolator.InterpolateEventNoInterpolator",
    "[Unit]") {
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  domain::FunctionsOfTime::register_derived_with_charm();
  run_test<MockMetavariables<false, false>>();
  run_test<MockMetavariables<true, false>>();
  run_test<MockMetavariables<false, true>>();
  run_test<MockMetavariables<true, true>>();
}
}  // namespace
