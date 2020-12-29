// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <utility>
#include <vector>

#include "Domain/Structure/ElementId.hpp"
#include "Evolution/Systems/Cce/Actions/SendNextTimeToCce.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhLocalTimeStepping.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/Interpolation/InterpolateOnElementTestHelpers.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"

namespace {

struct test_send_to_evolution {
  template <typename ParallelComponent, typename... DbTags, typename ArrayIndex,
            typename Metavariables>
  static void apply(
      const db::DataBox<tmpl::list<DbTags...>>& /*box*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const TimeStepId& /*time*/,
      const Cce::InterfaceManagers::GhInterfaceManager::gh_variables&
      /*gh_variables*/) noexcept {}
};

TimeStepId next_time_received;
struct test_receive_next_element_time {
  template <typename ParallelComponent, typename... DbTags, typename ArrayIndex,
            typename Metavariables>
  static void apply(const db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const TimeStepId& /*time*/,
                    const TimeStepId& next_time) noexcept {
    next_time_received = next_time;
  }
};

template <typename Metavariables>
struct mock_element {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Metavariables::volume_dim>;
  using simple_tags =
      tmpl::list<::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>,
                 ::Tags::TimeStep,
                 domain::Tags::Mesh<Metavariables::volume_dim>,
                 intrp::Tags::InterpPointInfo<Metavariables>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<simple_tags, tmpl::list<>>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<Cce::Actions::SendNextTimeToCce<
              typename Metavariables::InterpolationTargetA>>>>;
};

template <typename Metavariables>
struct mock_gh_worldtube_boundary {
  using component_being_mocked = Cce::GhWorldtubeBoundary<Metavariables>;
  using replace_these_simple_actions =
      tmpl::list<Cce::Actions::SendToEvolution<
                     Cce::GhWorldtubeBoundary<Metavariables>,
                     Cce::CharacteristicEvolution<Metavariables>>,
                 Cce::Actions::ReceiveNextElementTime>;
  using with_these_simple_actions =
      tmpl::list<test_send_to_evolution, test_receive_next_element_time>;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using simple_tags = tmpl::list<Cce::Tags::GhInterfaceManager>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags, tmpl::list<>>>>>;
};

struct test_metavariables {
  static constexpr size_t volume_dim = 3;
  struct InterpolationTargetA {
    using vars_to_interpolate_to_target = tmpl::list<>;
    using compute_items_on_source = tmpl::list<>;
  };
  using interpolation_target_tags = tmpl::list<InterpolationTargetA>;
  using component_list =
      tmpl::list<mock_gh_worldtube_boundary<test_metavariables>,
                 mock_element<test_metavariables>,
                 InterpolateOnElementTestHelpers::mock_interpolation_target<
                     test_metavariables, InterpolationTargetA>>;
  using cce_boundary_component = mock_gh_worldtube_boundary<test_metavariables>;
  enum class Phase { Initialization, Testing, Exit };
};

template <typename Metavariables, typename ElemComponent>
struct initialize_elements_and_queue_simple_actions {
  initialize_elements_and_queue_simple_actions(
      const TimeStepId& time_step_id, const TimeStepId& next_time_step_id,
      const TimeDelta& time_step, const TimeStepId& expected_next_time_step,
      const bool is_substep) noexcept
      : time_step_id_{time_step_id},
        next_time_step_id_{next_time_step_id},
        time_step_{time_step},
        expected_next_time_step_{expected_next_time_step},
        is_substep_{is_substep} {}

  template <typename InterpPointInfo, typename Runner, typename TemporalId>
  void operator()(const DomainCreator<3>& domain_creator,
                  const Domain<3>& domain,
                  const std::vector<ElementId<3>>& element_ids,
                  const InterpPointInfo& interp_point_info, Runner& runner,
                  const TemporalId& /*temporal_id*/) noexcept {
    using elem_component = ElemComponent;

    ActionTesting::emplace_component_and_initialize<
        mock_gh_worldtube_boundary<test_metavariables>>(
        &runner, 0_st,
        {std::make_unique<Cce::InterfaceManagers::GhLocalTimeStepping>(3_st)});

    // Emplace elements.
    for (const auto& element_id : element_ids) {
      // 1. Get mesh
      auto mesh =
          get<1>(InterpolateOnElementTestHelpers::make_volume_data_and_mesh(
              domain_creator, domain, element_id));

      // 2. emplace element.
      ActionTesting::emplace_component_and_initialize<elem_component>(
          &runner, element_id,
          {time_step_id_, next_time_step_id_, time_step_, mesh,
           interp_point_info});
    }
    ActionTesting::set_phase(make_not_null(&runner),
                             test_metavariables::Phase::Testing);

    // Call the action on all the elements.
    for (const auto& element_id : element_ids) {
      ActionTesting::next_action<elem_component>(make_not_null(&runner),
                                                 element_id);
    }
    if (not is_substep_) {
      // run the (single) simple receive action on the boundary
      runner.template invoke_queued_simple_action<
          mock_gh_worldtube_boundary<test_metavariables>>(0_st);

      CHECK(next_time_received == expected_next_time_step_);
    }

    // check that there aren't any more simple actions queued
    CHECK(ActionTesting::is_simple_action_queue_empty<
          mock_gh_worldtube_boundary<test_metavariables>>(runner, 0));
  }

 private:
  TimeStepId time_step_id_;
  TimeStepId next_time_step_id_;
  TimeDelta time_step_;
  TimeStepId expected_next_time_step_;
  bool is_substep_;
};

void test_send_time_to_cce(const bool substep) noexcept {
  Parallel::register_derived_classes_with_charm<
      Cce::InterfaceManagers::GhInterfaceManager>();
  Parallel::register_derived_classes_with_charm<TimeStepper>();
  using metavars = test_metavariables;
  using elem_component = mock_element<metavars>;
  TimeStepId time_step_id;
  TimeStepId next_time_step_id;
  TimeStepId expected_next_step;
  if (substep) {
    time_step_id = TimeStepId{
        true, 0_st, {{0.0, 0.1}, {1, 2}}, 1_st, {{0.0, 0.1}, {1, 1}}};
    next_time_step_id = TimeStepId{
        true, 0_st, {{0.0, 0.1}, {1, 2}}, 2_st, {{0.0, 0.1}, {3, 4}}};
  } else {
    time_step_id = TimeStepId{true, 0_st, {{0.0, 0.1}, {1, 2}}};
    next_time_step_id = TimeStepId{
        true, 0_st, {{0.0, 0.1}, {1, 2}}, 1_st, {{0.0, 0.1}, {1, 1}}};
    expected_next_step = TimeStepId{true, 1_st, {{0.1, 0.2}, {0, 2}}};
  }
  const TimeDelta time_step{{0.0, 0.1}, {1, 2}};
  InterpolateOnElementTestHelpers::test_interpolate_on_element<metavars,
                                                               elem_component>(
      initialize_elements_and_queue_simple_actions<metavars, elem_component>{
          time_step_id, next_time_step_id, time_step, expected_next_step,
          substep},
      std::make_unique<::TimeSteppers::RungeKutta3>());
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.Actions.SendNextTimeToCce",
                  "[Unit][Cce]") {
  test_send_time_to_cce(true);
  test_send_time_to_cce(false);
}
}  // namespace
