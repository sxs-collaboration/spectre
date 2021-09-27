// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "Framework/ActionTesting.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolator.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorRegisterElement.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/Interpolation/InterpolatedVars.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include <boost/variant/get.hpp>

class DataVector;
namespace intrp::Tags {
struct NumberOfElements;
}  // namespace intrp::Tags

namespace {

template <typename Metavariables>
struct mock_interpolator {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockGroupChare;
  using array_index = size_t;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<Actions::SetupDataBox,
                     ::intrp::Actions::InitializeInterpolator<
                         intrp::Tags::VolumeVarsInfo<Metavariables,
                                                     ::Tags::TimeStepId>,
                         intrp::Tags::InterpolatedVarsHolders<Metavariables>>>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Registration, tmpl::list<>>>;
  using initial_databox = db::compute_databox_type<
      typename ::intrp::Actions::InitializeInterpolator<
          intrp::Tags::VolumeVarsInfo<Metavariables, ::Tags::TimeStepId>,
          intrp::Tags::InterpolatedVarsHolders<Metavariables>>::
          return_tag_list>;
  using component_being_mocked = intrp::Interpolator<Metavariables>;
};

template <typename Metavariables>
struct mock_element {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             tmpl::list<>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Registration,
          tmpl::list<intrp::Actions::RegisterElementWithInterpolator>>>;
  using initial_databox = db::compute_databox_type<tmpl::list<>>;
};

struct MockMetavariables {
  struct InterpolatorTargetA {
    using temporal_id = ::Tags::TimeStepId;
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::Lapse<DataVector>>;
    using compute_items_on_target = tmpl::list<>;
  };
  static constexpr size_t volume_dim = 3;
  using interpolator_source_vars = tmpl::list<gr::Tags::Lapse<DataVector>>;
  using interpolation_target_tags = tmpl::list<InterpolatorTargetA>;

  using component_list = tmpl::list<mock_interpolator<MockMetavariables>,
                                    mock_element<MockMetavariables>>;
  enum class Phase { Initialization, Registration, Testing, Exit };
};

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Interpolator.RegisterElement",
                  "[Unit]") {
  using metavars = MockMetavariables;
  using interp_component = mock_interpolator<metavars>;
  using elem_component = mock_element<metavars>;
  ActionTesting::MockRuntimeSystem<metavars> runner{{}};
  ActionTesting::set_phase(make_not_null(&runner),
                           metavars::Phase::Initialization);
  ActionTesting::emplace_group_component<interp_component>(&runner);
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<interp_component>(make_not_null(&runner), 0);
  }
  ActionTesting::emplace_component<elem_component>(&runner, 0);
  // There is no next_action on elem_component, so we don't call it here.
  ActionTesting::set_phase(make_not_null(&runner),
                           metavars::Phase::Registration);

  CHECK(ActionTesting::get_databox_tag<interp_component,
                                       ::intrp::Tags::NumberOfElements>(
            runner, 0) == 0);

  runner.simple_action<interp_component, ::intrp::Actions::RegisterElement>(0);

  CHECK(ActionTesting::get_databox_tag<interp_component,
                                       ::intrp::Tags::NumberOfElements>(
            runner, 0) == 1);

  runner.simple_action<interp_component, ::intrp::Actions::RegisterElement>(0);

  CHECK(ActionTesting::get_databox_tag<interp_component,
                                       ::intrp::Tags::NumberOfElements>(
            runner, 0) == 2);

  // Call RegisterElementWithInterpolator from element, check if
  // it gets registered.
  ActionTesting::next_action<elem_component>(make_not_null(&runner), 0);
  ActionTesting::set_phase(make_not_null(&runner), metavars::Phase::Testing);

  runner.invoke_queued_simple_action<interp_component>(0);

  CHECK(ActionTesting::get_databox_tag<interp_component,
                                       ::intrp::Tags::NumberOfElements>(
            runner, 0) == 3);

  // No more queued simple actions.
  CHECK(runner.is_simple_action_queue_empty<interp_component>(0));
  CHECK(runner.is_simple_action_queue_empty<elem_component>(0));

  {
    INFO("Deregistration");
    intrp::Actions::RegisterElementWithInterpolator::
        template perform_deregistration<elem_component>(
            ActionTesting::get_databox<elem_component, tmpl::list<>>(
                make_not_null(&runner), 0_st),
            ActionTesting::cache<elem_component>(runner, 0_st), 0_st);
    ActionTesting::invoke_queued_simple_action<interp_component>(
        make_not_null(&runner), 0);
    // No more queued simple actions.
    CHECK(runner.is_simple_action_queue_empty<interp_component>(0));
    CHECK(runner.is_simple_action_queue_empty<elem_component>(0));

    CHECK(ActionTesting::get_databox_tag<interp_component,
          ::intrp::Tags::NumberOfElements>(
               runner, 0) == 2);
    runner.simple_action<interp_component, ::intrp::Actions::DeregisterElement>(
        0);
    CHECK(ActionTesting::get_databox_tag<interp_component,
          ::intrp::Tags::NumberOfElements>(
               runner, 0) == 1);
    runner.simple_action<interp_component, ::intrp::Actions::DeregisterElement>(
        0);
    CHECK(ActionTesting::get_databox_tag<interp_component,
                                         ::intrp::Tags::NumberOfElements>(
              runner, 0) == 0);
  }
}

}  // namespace
