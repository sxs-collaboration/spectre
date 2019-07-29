// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DenseVector.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/Criteria.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/Convergence/Reason.hpp"
#include "NumericalAlgorithms/LinearSolver/ConjugateGradient/ResidualMonitor.hpp"
#include "NumericalAlgorithms/LinearSolver/ConjugateGradient/ResidualMonitorActions.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearSolver/Observe.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "ParallelBackend/AddOptionsToDataBox.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/NumericalAlgorithms/LinearSolver/ResidualMonitorActionsTestHelpers.hpp"

// IWYU pragma: no_forward_declare db::DataBox

namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
namespace LinearSolver {
namespace cg_detail {
struct InitializeHasConverged;
struct UpdateFieldValues;
struct UpdateOperand;
}  // namespace cg_detail
}  // namespace LinearSolver

namespace helpers = ResidualMonitorActionsTestHelpers;

namespace {

struct VectorTag : db::SimpleTag {
  using type = DenseVector<double>;
  static std::string name() noexcept { return "VectorTag"; }
};

using residual_square_tag = db::add_tag_prefix<
    LinearSolver::Tags::MagnitudeSquare,
    db::add_tag_prefix<LinearSolver::Tags::Residual, VectorTag>>;
using initial_residual_magnitude_tag = db::add_tag_prefix<
    LinearSolver::Tags::Initial,
    db::add_tag_prefix<
        LinearSolver::Tags::Magnitude,
        db::add_tag_prefix<LinearSolver::Tags::Residual, VectorTag>>>;

struct CheckValueTag : db::SimpleTag {
  using type = double;
  static std::string name() noexcept { return "CheckValueTag"; }
};

using CheckConvergedTag = LinearSolver::Tags::HasConverged;

using check_tags = tmpl::list<CheckValueTag, CheckConvergedTag>;

template <typename Metavariables>
struct MockResidualMonitor {
  using component_being_mocked =
      LinearSolver::cg_detail::ResidualMonitor<Metavariables>;
  using metavariables = Metavariables;
  // We represent the singleton as an array with only one element for the action
  // testing framework
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tag_list =
      typename LinearSolver::cg_detail::ResidualMonitor<
          Metavariables>::const_global_cache_tag_list;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<
          LinearSolver::cg_detail::InitializeResidualMonitor<Metavariables>>>>;
};

struct MockInitializeHasConverged {
  template <
      typename ParallelComponent, typename DbTagsList, typename Metavariables,
      typename ArrayIndex,
      Requires<tmpl::list_contains_v<DbTagsList, CheckConvergedTag>> = nullptr>
  static void apply(db::DataBox<DbTagsList>& box,  // NOLINT
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const db::item_type<LinearSolver::Tags::HasConverged>&
                        has_converged) noexcept {
    db::mutate<CheckConvergedTag>(
        make_not_null(&box), [&has_converged](const gsl::not_null<
                                              db::item_type<CheckConvergedTag>*>
                                                  check_converged) noexcept {
          *check_converged = has_converged;
        });
  }
};

struct MockUpdateFieldValues {
  template <
      typename ParallelComponent, typename DbTagsList, typename Metavariables,
      typename ArrayIndex,
      Requires<tmpl::list_contains_v<DbTagsList, CheckValueTag>> = nullptr>
  static void apply(db::DataBox<DbTagsList>& box,  // NOLINT
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const double alpha) noexcept {
    db::mutate<CheckValueTag>(
        make_not_null(&box), [alpha](const gsl::not_null<double*>
                                         check_value) noexcept {
          *check_value = alpha;
        });
  }
};

struct MockUpdateOperand {
  template <
      typename ParallelComponent, typename DbTagsList, typename Metavariables,
      typename ArrayIndex,
      Requires<tmpl::list_contains_v<DbTagsList, CheckValueTag>> = nullptr>
  static void apply(db::DataBox<DbTagsList>& box,  // NOLINT
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, const double res_ratio,
                    const db::item_type<LinearSolver::Tags::HasConverged>&
                        has_converged) noexcept {
    db::mutate<CheckValueTag, CheckConvergedTag>(
        make_not_null(&box),
        [ res_ratio, &has_converged ](
            const gsl::not_null<double*> check_value,
            const gsl::not_null<db::item_type<CheckConvergedTag>*>
                check_converged) noexcept {
          *check_value = res_ratio;
          *check_converged = has_converged;
        });
  }
};

// This is used to receive action calls from the residual monitor
template <typename Metavariables>
struct MockElementArray {
  using component_being_mocked = void;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tag_list = tmpl::list<>;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<check_tags>>>>;

  using replace_these_simple_actions =
      tmpl::list<LinearSolver::cg_detail::InitializeHasConverged,
                 LinearSolver::cg_detail::UpdateFieldValues,
                 LinearSolver::cg_detail::UpdateOperand>;
  using with_these_simple_actions =
      tmpl::list<MockInitializeHasConverged, MockUpdateFieldValues,
                 MockUpdateOperand>;
};

struct System {
  using fields_tag = VectorTag;
};

struct Metavariables {
  using component_list = tmpl::list<MockResidualMonitor<Metavariables>,
                                    MockElementArray<Metavariables>,
                                    helpers::MockObserverWriter<Metavariables>>;
  using system = System;
  using const_global_cache_tag_list = tmpl::list<>;
  enum class Phase { Initialization, RegisterWithObserver, Testing, Exit };
};
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Numerical.LinearSolver.ConjugateGradient.ResidualMonitorActions",
    "[Unit][NumericalAlgorithms][LinearSolver][Actions]") {
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
  ActionTesting::emplace_component_and_initialize<element_array>(
      &runner, 0,
      {std::numeric_limits<double>::signaling_NaN(),
       db::item_type<CheckConvergedTag>{}});

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
  const auto get_element_tag = [&runner](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner, 0);
  };
  const auto get_observer_writer_tag = [&runner](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<observer_writer, tag>(runner, 0);
  };

  runner.set_phase(Metavariables::Phase::Testing);

  SECTION("InitializeResidual") {
    ActionTesting::simple_action<
        residual_monitor,
        LinearSolver::cg_detail::InitializeResidual<element_array>>(
        make_not_null(&runner), 0, 4.);
    ActionTesting::invoke_queued_threaded_action<observer_writer>(
        make_not_null(&runner), 0);
    ActionTesting::invoke_queued_simple_action<element_array>(
        make_not_null(&runner), 0);
    // Test residual monitor state
    CHECK(get_residual_monitor_tag(residual_square_tag{}) == 4.);
    CHECK(get_residual_monitor_tag(initial_residual_magnitude_tag{}) == 2.);
    CHECK(get_residual_monitor_tag(LinearSolver::Tags::IterationId{}) == 0);
    CHECK_FALSE(get_residual_monitor_tag(LinearSolver::Tags::HasConverged{}));
    // Test element state
    CHECK_FALSE(get_element_tag(CheckConvergedTag{}));
    // Test observer writer state
    CHECK(get_observer_writer_tag(helpers::CheckObservationIdTag{}) ==
          observers::ObservationId{
              0, LinearSolver::observe_detail::ObservationType{}});
    CHECK(get_observer_writer_tag(helpers::CheckSubfileNameTag{}) ==
          "/linear_residuals");
    CHECK(get_observer_writer_tag(helpers::CheckReductionNamesTag{}) ==
          std::vector<std::string>{"Iteration", "Residual"});
    CHECK(get<0>(get_observer_writer_tag(helpers::CheckReductionDataTag{})) ==
          0);
    CHECK(get<1>(get_observer_writer_tag(helpers::CheckReductionDataTag{})) ==
          approx(2.));
  }

  SECTION("InitializeResidualAndConverge") {
    ActionTesting::simple_action<
        residual_monitor,
        LinearSolver::cg_detail::InitializeResidual<element_array>>(
        make_not_null(&runner), 0, 0.);
    ActionTesting::invoke_queued_simple_action<element_array>(
        make_not_null(&runner), 0);
    // Test residual monitor state
    CHECK(get_residual_monitor_tag(residual_square_tag{}) == 0.);
    CHECK(get_residual_monitor_tag(initial_residual_magnitude_tag{}) == 0.);
    CHECK(get_residual_monitor_tag(LinearSolver::Tags::IterationId{}) == 0);
    CHECK(get_residual_monitor_tag(LinearSolver::Tags::HasConverged{}));
    // Test element state
    CHECK(get_element_tag(CheckConvergedTag{}));
  }

  SECTION("ComputeAlpha") {
    ActionTesting::simple_action<
        residual_monitor,
        LinearSolver::cg_detail::InitializeResidual<element_array>>(
        make_not_null(&runner), 0, 1.);
    ActionTesting::invoke_queued_simple_action<element_array>(
        make_not_null(&runner), 0);
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::cg_detail::ComputeAlpha<element_array>>(
        make_not_null(&runner), 0, 2.);
    ActionTesting::invoke_queued_simple_action<element_array>(
        make_not_null(&runner), 0);
    // Test residual monitor state
    CHECK(get_residual_monitor_tag(residual_square_tag{}) == 1.);
    CHECK(get_residual_monitor_tag(LinearSolver::Tags::IterationId{}) == 0);
    // Test element state
    CHECK(get_element_tag(CheckValueTag{}) == 0.5);
  }

  SECTION("UpdateResidual") {
    ActionTesting::simple_action<
        residual_monitor,
        LinearSolver::cg_detail::InitializeResidual<element_array>>(
        make_not_null(&runner), 0, 9.);
    ActionTesting::invoke_queued_threaded_action<observer_writer>(
        make_not_null(&runner), 0);
    ActionTesting::invoke_queued_simple_action<element_array>(
        make_not_null(&runner), 0);
    ActionTesting::simple_action<
        residual_monitor,
        LinearSolver::cg_detail::UpdateResidual<element_array>>(
        make_not_null(&runner), 0, 4.);
    ActionTesting::invoke_queued_simple_action<element_array>(
        make_not_null(&runner), 0);
    ActionTesting::invoke_queued_threaded_action<observer_writer>(
        make_not_null(&runner), 0);
    // Test residual monitor state
    CHECK(get_residual_monitor_tag(residual_square_tag{}) == 4.);
    CHECK(get_residual_monitor_tag(initial_residual_magnitude_tag{}) == 3.);
    CHECK(get_residual_monitor_tag(LinearSolver::Tags::IterationId{}) == 1);
    CHECK_FALSE(get_residual_monitor_tag(LinearSolver::Tags::HasConverged{}));
    // Test element state
    CHECK(get_element_tag(CheckValueTag{}) == approx(4. / 9.));
    CHECK(get_element_tag(CheckConvergedTag{}) ==
          get_residual_monitor_tag(LinearSolver::Tags::HasConverged{}));
    // Test observer writer state
    CHECK(get_observer_writer_tag(helpers::CheckObservationIdTag{}) ==
          observers::ObservationId{
              1, LinearSolver::observe_detail::ObservationType{}});
    CHECK(get_observer_writer_tag(helpers::CheckSubfileNameTag{}) ==
          "/linear_residuals");
    CHECK(get_observer_writer_tag(helpers::CheckReductionNamesTag{}) ==
          std::vector<std::string>{"Iteration", "Residual"});
    CHECK(get<0>(get_observer_writer_tag(helpers::CheckReductionDataTag{})) ==
          1);
    CHECK(get<1>(get_observer_writer_tag(helpers::CheckReductionDataTag{})) ==
          approx(2.));
  }

  SECTION("ConvergeByAbsoluteResidual") {
    ActionTesting::simple_action<
        residual_monitor,
        LinearSolver::cg_detail::InitializeResidual<element_array>>(
        make_not_null(&runner), 0, 1.);
    ActionTesting::invoke_queued_simple_action<element_array>(
        make_not_null(&runner), 0);
    ActionTesting::simple_action<
        residual_monitor,
        LinearSolver::cg_detail::UpdateResidual<element_array>>(
        make_not_null(&runner), 0, 0.);
    ActionTesting::invoke_queued_simple_action<element_array>(
        make_not_null(&runner), 0);
    // Test residual monitor state
    CHECK(get_residual_monitor_tag(residual_square_tag{}) == 0.);
    CHECK(get_residual_monitor_tag(initial_residual_magnitude_tag{}) == 1.);
    CHECK(get_residual_monitor_tag(LinearSolver::Tags::IterationId{}) == 1);
    CHECK(get_residual_monitor_tag(LinearSolver::Tags::HasConverged{}));
    CHECK(
        get_residual_monitor_tag(LinearSolver::Tags::HasConverged{}).reason() ==
        Convergence::Reason::AbsoluteResidual);
    // Test element state
    CHECK(get_element_tag(CheckValueTag{}) == 0.);
    CHECK(get_element_tag(CheckConvergedTag{}) ==
          get_residual_monitor_tag(LinearSolver::Tags::HasConverged{}));
  }

  SECTION("ConvergeByMaxIterations") {
    ActionTesting::simple_action<
        residual_monitor,
        LinearSolver::cg_detail::InitializeResidual<element_array>>(
        make_not_null(&runner), 0, 1.);
    ActionTesting::invoke_queued_simple_action<element_array>(
        make_not_null(&runner), 0);
    // Perform 2 mock iterations
    ActionTesting::simple_action<
        residual_monitor,
        LinearSolver::cg_detail::UpdateResidual<element_array>>(
        make_not_null(&runner), 0, 1.);
    ActionTesting::invoke_queued_simple_action<element_array>(
        make_not_null(&runner), 0);
    ActionTesting::simple_action<
        residual_monitor,
        LinearSolver::cg_detail::UpdateResidual<element_array>>(
        make_not_null(&runner), 0, 1.);
    ActionTesting::invoke_queued_simple_action<element_array>(
        make_not_null(&runner), 0);
    // Test residual monitor state
    CHECK(get_residual_monitor_tag(residual_square_tag{}) == 1.);
    CHECK(get_residual_monitor_tag(initial_residual_magnitude_tag{}) == 1.);
    CHECK(get_residual_monitor_tag(LinearSolver::Tags::IterationId{}) == 2);
    CHECK(get_residual_monitor_tag(LinearSolver::Tags::HasConverged{}));
    CHECK(
        get_residual_monitor_tag(LinearSolver::Tags::HasConverged{}).reason() ==
        Convergence::Reason::MaxIterations);
    // Test element state
    CHECK(get_element_tag(CheckValueTag{}) == 1.);
    CHECK(get_element_tag(CheckConvergedTag{}) ==
          get_residual_monitor_tag(LinearSolver::Tags::HasConverged{}));
  }

  SECTION("ConvergeByRelativeResidual") {
    ActionTesting::simple_action<
        residual_monitor,
        LinearSolver::cg_detail::InitializeResidual<element_array>>(
        make_not_null(&runner), 0, 1.);
    ActionTesting::invoke_queued_simple_action<element_array>(
        make_not_null(&runner), 0);
    ActionTesting::simple_action<
        residual_monitor,
        LinearSolver::cg_detail::UpdateResidual<element_array>>(
        make_not_null(&runner), 0, 0.25);
    ActionTesting::invoke_queued_simple_action<element_array>(
        make_not_null(&runner), 0);
    // Test residual monitor state
    CHECK(get_residual_monitor_tag(residual_square_tag{}) == 0.25);
    CHECK(get_residual_monitor_tag(initial_residual_magnitude_tag{}) == 1.);
    CHECK(get_residual_monitor_tag(LinearSolver::Tags::IterationId{}) == 1);
    CHECK(get_residual_monitor_tag(LinearSolver::Tags::HasConverged{}));
    CHECK(
        get_residual_monitor_tag(LinearSolver::Tags::HasConverged{}).reason() ==
        Convergence::Reason::RelativeResidual);
    // Test element state
    CHECK(get_element_tag(CheckValueTag{}) == 0.25);
    CHECK(get_element_tag(CheckConvergedTag{}) ==
          get_residual_monitor_tag(LinearSolver::Tags::HasConverged{}));
  }
}
