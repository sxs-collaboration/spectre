// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DenseVector.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/ParallelAlgorithms/LinearSolver/ResidualMonitorActionsTestHelpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/Criteria.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/Convergence/Reason.hpp"
#include "ParallelAlgorithms/LinearSolver/ConjugateGradient/ResidualMonitor.hpp"
#include "ParallelAlgorithms/LinearSolver/ConjugateGradient/ResidualMonitorActions.hpp"
#include "ParallelAlgorithms/LinearSolver/Observe.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel

namespace helpers = ResidualMonitorActionsTestHelpers;

namespace {

struct TestLinearSolver {};

struct VectorTag : db::SimpleTag {
  using type = DenseVector<double>;
};

using fields_tag = VectorTag;
using residual_square_tag = LinearSolver::Tags::MagnitudeSquare<
    LinearSolver::Tags::Residual<fields_tag>>;
using initial_residual_magnitude_tag = ::Tags::Initial<
    LinearSolver::Tags::Magnitude<LinearSolver::Tags::Residual<fields_tag>>>;

template <typename Metavariables>
struct MockResidualMonitor {
  using component_being_mocked =
      LinearSolver::cg::detail::ResidualMonitor<Metavariables, fields_tag,
                                                TestLinearSolver>;
  using metavariables = Metavariables;
  // We represent the singleton as an array with only one element for the action
  // testing framework
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags =
      typename LinearSolver::cg::detail::ResidualMonitor<
          Metavariables, fields_tag, TestLinearSolver>::const_global_cache_tags;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<LinearSolver::cg::detail::InitializeResidualMonitor<
          fields_tag, TestLinearSolver>>>>;
};

// This is used to receive action calls from the residual monitor
template <typename Metavariables>
struct MockElementArray {
  using component_being_mocked = void;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<>>>;
  using inbox_tags = tmpl::list<
      LinearSolver::cg::detail::Tags::InitialHasConverged<TestLinearSolver>,
      LinearSolver::cg::detail::Tags::Alpha<TestLinearSolver>,
      LinearSolver::cg::detail::Tags::ResidualRatioAndHasConverged<
          TestLinearSolver>>;
};

struct Metavariables {
  using component_list = tmpl::list<MockResidualMonitor<Metavariables>,
                                    MockElementArray<Metavariables>,
                                    helpers::MockObserverWriter<Metavariables>>;
  enum class Phase { Initialization, RegisterWithObserver, Testing, Exit };
};
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Parallel.LinearSolver.ConjugateGradient.ResidualMonitorActions",
    "[Unit][ParallelAlgorithms][LinearSolver][Actions]") {
  using residual_monitor = MockResidualMonitor<Metavariables>;
  using element_array = MockElementArray<Metavariables>;
  using observer_writer = helpers::MockObserverWriter<Metavariables>;

  const Convergence::Criteria convergence_criteria{2, 0., 0.5};
  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {::Verbosity::Verbose, convergence_criteria}};

  // Setup mock residual monitor
  ActionTesting::emplace_component<residual_monitor>(&runner, 0);
  ActionTesting::next_action<residual_monitor>(make_not_null(&runner), 0);

  // Setup mock element array
  ActionTesting::emplace_component<element_array>(&runner, 0);

  // Setup mock observer writer
  ActionTesting::emplace_component_and_initialize<observer_writer>(
      &runner, 0,
      {observers::ObservationId{}, std::string{}, std::vector<std::string>{},
       std::tuple<size_t, double>{
           0, std::numeric_limits<double>::signaling_NaN()}});

  // DataBox shortcuts
  const auto get_residual_monitor_tag =
      [&runner](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<residual_monitor, tag>(runner, 0);
  };
  const auto get_element_inbox_tag = [&runner](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_inbox_tag<element_array, tag>(runner, 0);
  };
  const auto get_observer_writer_tag = [&runner](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<observer_writer, tag>(runner, 0);
  };

  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);

  SECTION("InitializeResidual") {
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::cg::detail::InitializeResidual<
                              fields_tag, TestLinearSolver, element_array>>(
        make_not_null(&runner), 0, 4.);
    ActionTesting::invoke_queued_threaded_action<observer_writer>(
        make_not_null(&runner), 0);
    // Test residual monitor state
    CHECK(get_residual_monitor_tag(residual_square_tag{}) == 4.);
    CHECK(get_residual_monitor_tag(initial_residual_magnitude_tag{}) == 2.);
    // Test element state
    CHECK_FALSE(get_element_inbox_tag(
                    LinearSolver::cg::detail::Tags::InitialHasConverged<
                        TestLinearSolver>{})
                    .at(0));
    // Test observer writer state
    CHECK(
        get_observer_writer_tag(helpers::CheckObservationIdTag{}) ==
        observers::ObservationId{0, "(anonymous namespace)::TestLinearSolver"});
    CHECK(get_observer_writer_tag(helpers::CheckSubfileNameTag{}) ==
          "/TestLinearSolverResiduals");
    CHECK(get_observer_writer_tag(helpers::CheckReductionNamesTag{}) ==
          std::vector<std::string>{"Iteration", "Residual"});
    CHECK(get<0>(get_observer_writer_tag(helpers::CheckReductionDataTag{})) ==
          0);
    CHECK(get<1>(get_observer_writer_tag(helpers::CheckReductionDataTag{})) ==
          approx(2.));
  }

  SECTION("InitializeResidualAndConverge") {
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::cg::detail::InitializeResidual<
                              fields_tag, TestLinearSolver, element_array>>(
        make_not_null(&runner), 0, 0.);
    // Test residual monitor state
    CHECK(get_residual_monitor_tag(residual_square_tag{}) == 0.);
    CHECK(get_residual_monitor_tag(initial_residual_magnitude_tag{}) == 0.);
    // Test element state
    CHECK(get_element_inbox_tag(
              LinearSolver::cg::detail::Tags::InitialHasConverged<
                  TestLinearSolver>{})
              .at(0));
  }

  SECTION("ComputeAlpha") {
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::cg::detail::InitializeResidual<
                              fields_tag, TestLinearSolver, element_array>>(
        make_not_null(&runner), 0, 1.);
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::cg::detail::ComputeAlpha<
                              fields_tag, TestLinearSolver, element_array>>(
        make_not_null(&runner), 0, 0_st, 2.);
    // Test residual monitor state
    CHECK(get_residual_monitor_tag(residual_square_tag{}) == 1.);
    // Test element state
    CHECK(get_element_inbox_tag(
              LinearSolver::cg::detail::Tags::Alpha<TestLinearSolver>{})
              .at(0) == 0.5);
  }

  SECTION("UpdateResidual") {
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::cg::detail::InitializeResidual<
                              fields_tag, TestLinearSolver, element_array>>(
        make_not_null(&runner), 0, 9.);
    ActionTesting::invoke_queued_threaded_action<observer_writer>(
        make_not_null(&runner), 0);
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::cg::detail::UpdateResidual<
                              fields_tag, TestLinearSolver, element_array>>(
        make_not_null(&runner), 0, 0_st, 4.);
    ActionTesting::invoke_queued_threaded_action<observer_writer>(
        make_not_null(&runner), 0);
    // Test residual monitor state
    CHECK(get_residual_monitor_tag(residual_square_tag{}) == 4.);
    CHECK(get_residual_monitor_tag(initial_residual_magnitude_tag{}) == 3.);
    // Test element state
    const auto& element_inbox =
        get_element_inbox_tag(
            LinearSolver::cg::detail::Tags::ResidualRatioAndHasConverged<
                TestLinearSolver>{})
            .at(0);
    CHECK(get<0>(element_inbox) == approx(4. / 9.));
    const auto& has_converged = get<1>(element_inbox);
    CHECK_FALSE(has_converged);
    // Test observer writer state
    CHECK(
        get_observer_writer_tag(helpers::CheckObservationIdTag{}) ==
        observers::ObservationId{1, "(anonymous namespace)::TestLinearSolver"});
    CHECK(get_observer_writer_tag(helpers::CheckSubfileNameTag{}) ==
          "/TestLinearSolverResiduals");
    CHECK(get_observer_writer_tag(helpers::CheckReductionNamesTag{}) ==
          std::vector<std::string>{"Iteration", "Residual"});
    CHECK(get<0>(get_observer_writer_tag(helpers::CheckReductionDataTag{})) ==
          1);
    CHECK(get<1>(get_observer_writer_tag(helpers::CheckReductionDataTag{})) ==
          approx(2.));
  }

  SECTION("ConvergeByAbsoluteResidual") {
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::cg::detail::InitializeResidual<
                              fields_tag, TestLinearSolver, element_array>>(
        make_not_null(&runner), 0, 1.);
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::cg::detail::UpdateResidual<
                              fields_tag, TestLinearSolver, element_array>>(
        make_not_null(&runner), 0, 0_st, 0.);
    // Test residual monitor state
    CHECK(get_residual_monitor_tag(residual_square_tag{}) == 0.);
    CHECK(get_residual_monitor_tag(initial_residual_magnitude_tag{}) == 1.);
    // Test element state
    const auto& element_inbox =
        get_element_inbox_tag(
            LinearSolver::cg::detail::Tags::ResidualRatioAndHasConverged<
                TestLinearSolver>{})
            .at(0);
    CHECK(get<0>(element_inbox) == 0.);
    const auto& has_converged = get<1>(element_inbox);
    REQUIRE(has_converged);
    CHECK(has_converged.reason() == Convergence::Reason::AbsoluteResidual);
  }

  SECTION("ConvergeByMaxIterations") {
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::cg::detail::InitializeResidual<
                              fields_tag, TestLinearSolver, element_array>>(
        make_not_null(&runner), 0, 1.);
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::cg::detail::UpdateResidual<
                              fields_tag, TestLinearSolver, element_array>>(
        make_not_null(&runner), 0, 1_st, 1.);
    // Test residual monitor state
    CHECK(get_residual_monitor_tag(residual_square_tag{}) == 1.);
    CHECK(get_residual_monitor_tag(initial_residual_magnitude_tag{}) == 1.);
    // Test element state
    const auto& element_inbox =
        get_element_inbox_tag(
            LinearSolver::cg::detail::Tags::ResidualRatioAndHasConverged<
                TestLinearSolver>{})
            .at(1);
    CHECK(get<0>(element_inbox) == 1.);
    const auto& has_converged = get<1>(element_inbox);
    REQUIRE(has_converged);
    CHECK(has_converged.reason() == Convergence::Reason::MaxIterations);
  }

  SECTION("ConvergeByRelativeResidual") {
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::cg::detail::InitializeResidual<
                              fields_tag, TestLinearSolver, element_array>>(
        make_not_null(&runner), 0, 1.);
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::cg::detail::UpdateResidual<
                              fields_tag, TestLinearSolver, element_array>>(
        make_not_null(&runner), 0, 0_st, 0.25);
    // Test residual monitor state
    CHECK(get_residual_monitor_tag(residual_square_tag{}) == 0.25);
    CHECK(get_residual_monitor_tag(initial_residual_magnitude_tag{}) == 1.);
    // Test element state
    const auto& element_inbox =
        get_element_inbox_tag(
            LinearSolver::cg::detail::Tags::ResidualRatioAndHasConverged<
                TestLinearSolver>{})
            .at(0);
    CHECK(get<0>(element_inbox) == 0.25);
    const auto& has_converged = get<1>(element_inbox);
    REQUIRE(has_converged);
    CHECK(has_converged.reason() == Convergence::Reason::RelativeResidual);
  }
}
