// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <utility>

#include "Evolution/Systems/Cce/Actions/TimeManagement.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Components/CharacteristicEvolution.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/Evolution/Systems/Cce/BoundaryTestHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace Cce {

namespace {
template <typename Metavariables>
struct mock_characteristic_evolution {
  using component_being_mocked = CharacteristicEvolution<Metavariables>;
  using replace_these_simple_actions = tmpl::list<>;
  using with_these_simple_actions = tmpl::list<>;

  using simple_tags = db::AddSimpleTags<::Tags::TimeStepId, Tags::EndTime>;
  using compute_tags = db::AddComputeTags<>;

  using initialize_action_list =
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags, compute_tags>>;
  using initialization_tags =
      Parallel::get_initialization_tags<initialize_action_list>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             initialize_action_list>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Evolve,
                             tmpl::list<Actions::ExitIfEndTimeReached>>>;
  using const_global_cache_tags =
      Parallel::get_const_global_cache_tags_from_actions<
          phase_dependent_action_list>;
};

struct metavariables {
  using component_list =
      tmpl::list<mock_characteristic_evolution<metavariables>>;
  enum class Phase { Initialization, Evolve, Exit };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.Actions.TimeManagement",
                  "[Unit][Cce]") {
  using component = mock_characteristic_evolution<metavariables>;
  ActionTesting::MockRuntimeSystem<metavariables> runner{{}};

  const TimeStepId current_id{true, 0, Time{{0.0, 1.0}, {0, 1}}};
  const double end_time = 2.0;

  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0, {std::move(current_id), end_time});  // NOLINT
  runner.set_phase(metavariables::Phase::Evolve);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  CHECK_FALSE(ActionTesting::get_terminate<component>(runner, 0));

  auto& box =
      ActionTesting::get_databox<component,
                                 tmpl::list<::Tags::TimeStepId, Tags::EndTime>>(
          make_not_null(&runner), 0);
  db::mutate<::Tags::TimeStepId>(
      make_not_null(&box), [](const gsl::not_null<TimeStepId*> time_step_id) {
        *time_step_id = TimeStepId{true, 0, Time{{1.5, 2.5}, {3, 4}}};
      });

  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  CHECK(ActionTesting::get_terminate<component>(runner, 0));
}
}  // namespace Cce
