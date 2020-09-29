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
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/DenseVector.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/ParallelAlgorithms/LinearSolver/ResidualMonitorActionsTestHelpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/Criteria.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/Convergence/Reason.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/ResidualMonitor.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/ResidualMonitorActions.hpp"
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
using initial_residual_magnitude_tag = ::Tags::Initial<
    LinearSolver::Tags::Magnitude<LinearSolver::Tags::Residual<fields_tag>>>;
using orthogonalization_history_tag =
    LinearSolver::Tags::OrthogonalizationHistory<fields_tag>;

template <typename Metavariables>
struct MockResidualMonitor {
  using component_being_mocked =
      LinearSolver::gmres::detail::ResidualMonitor<Metavariables, fields_tag,
                                                   TestLinearSolver>;
  using metavariables = Metavariables;
  // We represent the singleton as an array with only one element for the action
  // testing framework
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags =
      typename LinearSolver::gmres::detail::ResidualMonitor<
          Metavariables, fields_tag, TestLinearSolver>::const_global_cache_tags;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<LinearSolver::gmres::detail::InitializeResidualMonitor<
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
      LinearSolver::gmres::detail::Tags::InitialOrthogonalization<
          TestLinearSolver>,
      LinearSolver::gmres::detail::Tags::Orthogonalization<TestLinearSolver>,
      LinearSolver::gmres::detail::Tags::FinalOrthogonalization<
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
    "Unit.ParallelAlgorithms.LinearSolver.Gmres.ResidualMonitorActions",
    "[Unit][ParallelAlgorithms][LinearSolver][Actions]") {
  using residual_monitor = MockResidualMonitor<Metavariables>;
  using element_array = MockElementArray<Metavariables>;
  using observer_writer = helpers::MockObserverWriter<Metavariables>;

  const Convergence::Criteria convergence_criteria{2, 0., 0.5};
  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {::Verbosity::Verbose, convergence_criteria}};

  // Setup mock residual monitor
  ActionTesting::emplace_component<residual_monitor>(make_not_null(&runner), 0);
  ActionTesting::next_action<residual_monitor>(make_not_null(&runner), 0);

  // Setup mock element array
  ActionTesting::emplace_component<element_array>(make_not_null(&runner), 0);

  // Setup mock observer writer
  ActionTesting::emplace_component_and_initialize<observer_writer>(
      make_not_null(&runner), 0,
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

  SECTION("InitializeResidualMagnitude") {
    ActionTesting::simple_action<
        residual_monitor,
        LinearSolver::gmres::detail::InitializeResidualMagnitude<
            fields_tag, TestLinearSolver, element_array>>(
        make_not_null(&runner), 0, 2.);
    ActionTesting::invoke_queued_threaded_action<observer_writer>(
        make_not_null(&runner), 0);
    // Test residual monitor state
    CHECK(get_residual_monitor_tag(initial_residual_magnitude_tag{}) == 2.);
    // Test element state
    const auto& element_inbox =
        get_element_inbox_tag(
            LinearSolver::gmres::detail::Tags::InitialOrthogonalization<
                TestLinearSolver>{})
            .at(0);
    CHECK(get<0>(element_inbox) == 2.);
    const auto& has_converged = get<1>(element_inbox);
    CHECK_FALSE(has_converged);
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

  SECTION("InitializeResidualMagnitudeAndConverge") {
    ActionTesting::simple_action<
        residual_monitor,
        LinearSolver::gmres::detail::InitializeResidualMagnitude<
            fields_tag, TestLinearSolver, element_array>>(
        make_not_null(&runner), 0, 0.);
    // Test residual monitor state
    CHECK(get_residual_monitor_tag(initial_residual_magnitude_tag{}) == 0.);
    // Test element state
    const auto& element_inbox =
        get_element_inbox_tag(
            LinearSolver::gmres::detail::Tags::InitialOrthogonalization<
                TestLinearSolver>{})
            .at(0);
    CHECK(get<0>(element_inbox) == 0.);
    const auto& has_converged = get<1>(element_inbox);
    REQUIRE(has_converged);
    CHECK(has_converged.reason() == Convergence::Reason::AbsoluteResidual);
  }

  SECTION("StoreOrthogonalization") {
    ActionTesting::simple_action<
        residual_monitor,
        LinearSolver::gmres::detail::InitializeResidualMagnitude<
            fields_tag, TestLinearSolver, element_array>>(
        make_not_null(&runner), 0, 1.);
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::gmres::detail::StoreOrthogonalization<
                              fields_tag, TestLinearSolver, element_array>>(
        make_not_null(&runner), 0, 0_st, 0_st, 2.);
    // Test residual monitor state
    CHECK(get_residual_monitor_tag(orthogonalization_history_tag{})(0, 0) ==
          2.);
    // Test element state
    CHECK(get_element_inbox_tag(
              LinearSolver::gmres::detail::Tags::Orthogonalization<
                  TestLinearSolver>{})
              .at(0) == 2.);
  }

  SECTION("StoreOrthogonalization (final)") {
    ActionTesting::simple_action<
        residual_monitor,
        LinearSolver::gmres::detail::InitializeResidualMagnitude<
            fields_tag, TestLinearSolver, element_array>>(
        make_not_null(&runner), 0, 2.);
    ActionTesting::invoke_queued_threaded_action<observer_writer>(
        make_not_null(&runner), 0);
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::gmres::detail::StoreOrthogonalization<
                              fields_tag, TestLinearSolver, element_array>>(
        make_not_null(&runner), 0, 0_st, 0_st, 3.);
    // Test intermediate residual monitor state
    CHECK(get_residual_monitor_tag(orthogonalization_history_tag{})(0, 0) ==
          3.);
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::gmres::detail::StoreOrthogonalization<
                              fields_tag, TestLinearSolver, element_array>>(
        make_not_null(&runner), 0, 0_st, 1_st, 4.);
    ActionTesting::invoke_queued_threaded_action<observer_writer>(
        make_not_null(&runner), 0);
    // Test residual monitor state
    // H = [[3.], [2.]]
    CHECK(get_residual_monitor_tag(orthogonalization_history_tag{}) ==
          DenseMatrix<double>({{3.}, {2.}}));
    // Test element state
    const auto& element_inbox =
        get_element_inbox_tag(
            LinearSolver::gmres::detail::Tags::FinalOrthogonalization<
                TestLinearSolver>{})
            .at(0);
    // beta = [2., 0.]
    // minres = inv(qr_R(H)) * trans(qr_Q(H)) * beta = [0.4615384615384615]
    const auto& minres = get<1>(element_inbox);
    CHECK(minres.size() == 1);
    CHECK_ITERABLE_APPROX(minres, DenseVector<double>({0.4615384615384615}));
    // r = beta - H * minres = [0.6153846153846154, -0.923076923076923]
    // |r| = 1.1094003924504583
    const double residual_magnitude = 1.1094003924504583;
    const auto& has_converged = get<2>(element_inbox);
    CHECK_FALSE(has_converged);
    CHECK(get<0>(element_inbox) == approx(2.));
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
          approx(residual_magnitude));
  }

  SECTION("ConvergeByAbsoluteResidual") {
    ActionTesting::simple_action<
        residual_monitor,
        LinearSolver::gmres::detail::InitializeResidualMagnitude<
            fields_tag, TestLinearSolver, element_array>>(
        make_not_null(&runner), 0, 2.);
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::gmres::detail::StoreOrthogonalization<
                              fields_tag, TestLinearSolver, element_array>>(
        make_not_null(&runner), 0, 0_st, 0_st, 1.);
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::gmres::detail::StoreOrthogonalization<
                              fields_tag, TestLinearSolver, element_array>>(
        make_not_null(&runner), 0, 0_st, 1_st, 0.);
    // Test residual monitor state
    // H = [[1.], [0.]]
    CHECK(get_residual_monitor_tag(orthogonalization_history_tag{}) ==
          DenseMatrix<double>({{1.}, {0.}}));
    // Test element state
    const auto& element_inbox =
        get_element_inbox_tag(
            LinearSolver::gmres::detail::Tags::FinalOrthogonalization<
                TestLinearSolver>{})
            .at(0);
    // beta = [2., 0.]
    // minres = inv(qr_R(H)) * trans(qr_Q(H)) * beta = [2.]
    const auto& minres = get<1>(element_inbox);
    CHECK(minres.size() == 1);
    CHECK_ITERABLE_APPROX(minres, DenseVector<double>({2.}));
    // r = beta - H * minres = [0., 0.]
    // |r| = 0.
    const auto& has_converged = get<2>(element_inbox);
    REQUIRE(has_converged);
    CHECK(has_converged.reason() == Convergence::Reason::AbsoluteResidual);
    CHECK(get<0>(element_inbox) == 0.);
  }

  SECTION("ConvergeByMaxIterations") {
    ActionTesting::simple_action<
        residual_monitor,
        LinearSolver::gmres::detail::InitializeResidualMagnitude<
            fields_tag, TestLinearSolver, element_array>>(
        make_not_null(&runner), 0, 1.);
    // Perform 2 mock iterations
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::gmres::detail::StoreOrthogonalization<
                              fields_tag, TestLinearSolver, element_array>>(
        make_not_null(&runner), 0, 0_st, 0_st, 1.);
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::gmres::detail::StoreOrthogonalization<
                              fields_tag, TestLinearSolver, element_array>>(
        make_not_null(&runner), 0, 0_st, 1_st, 4.);
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::gmres::detail::StoreOrthogonalization<
                              fields_tag, TestLinearSolver, element_array>>(
        make_not_null(&runner), 0, 1_st, 0_st, 3.);
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::gmres::detail::StoreOrthogonalization<
                              fields_tag, TestLinearSolver, element_array>>(
        make_not_null(&runner), 0, 1_st, 1_st, 4.);
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::gmres::detail::StoreOrthogonalization<
                              fields_tag, TestLinearSolver, element_array>>(
        make_not_null(&runner), 0, 1_st, 2_st, 25.);
    // Test residual monitor state
    CHECK(get_residual_monitor_tag(orthogonalization_history_tag{}) ==
          DenseMatrix<double>({{1., 3.}, {2., 4.}, {0., 5.}}));
    // Test element state
    const auto& element_inbox =
        get_element_inbox_tag(
            LinearSolver::gmres::detail::Tags::FinalOrthogonalization<
                TestLinearSolver>{})
            .at(1);
    // beta = [1., 0., 0.]
    // minres = inv(qr_R(H)) * trans(qr_Q(H)) * beta = [0.13178295, 0.03100775]
    const auto& minres = get<1>(element_inbox);
    CHECK(minres.size() == 2);
    CHECK_ITERABLE_APPROX(
        minres, DenseVector<double>({0.1317829457364342, 0.0310077519379845}));
    // r = beta - H * minres = [0.77519, -0.38759, -0.15503]
    // |r| = 0.8804509063256237
    const auto& has_converged = get<2>(element_inbox);
    CHECK(has_converged);
    CHECK(has_converged.reason() == Convergence::Reason::MaxIterations);
  }

  SECTION("ConvergeByRelativeResidual") {
    ActionTesting::simple_action<
        residual_monitor,
        LinearSolver::gmres::detail::InitializeResidualMagnitude<
            fields_tag, TestLinearSolver, element_array>>(
        make_not_null(&runner), 0, 2.);
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::gmres::detail::StoreOrthogonalization<
                              fields_tag, TestLinearSolver, element_array>>(
        make_not_null(&runner), 0, 0_st, 0_st, 3.);
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::gmres::detail::StoreOrthogonalization<
                              fields_tag, TestLinearSolver, element_array>>(
        make_not_null(&runner), 0, 0_st, 1_st, 1.);
    // Test residual monitor state
    // H = [[3.], [1.]]
    CHECK(get_residual_monitor_tag(orthogonalization_history_tag{}) ==
          DenseMatrix<double>({{3.}, {1.}}));
    // Test element state
    const auto& element_inbox =
        get_element_inbox_tag(
            LinearSolver::gmres::detail::Tags::FinalOrthogonalization<
                TestLinearSolver>{})
            .at(0);
    // beta = [2., 0.]
    // minres = inv(qr_R(H)) * trans(qr_Q(H)) * beta = [0.6]
    const auto& minres = get<1>(element_inbox);
    CHECK(minres.size() == 1);
    CHECK_ITERABLE_APPROX(minres, DenseVector<double>({0.6}));
    // r = beta - H * minres = [0.2, -0.6]
    // |r| = 0.6324555320336759
    // |r| / |r_initial| = 0.31622776601683794
    const auto& has_converged = get<2>(element_inbox);
    REQUIRE(has_converged);
    CHECK(has_converged.reason() == Convergence::Reason::RelativeResidual);
  }
}
