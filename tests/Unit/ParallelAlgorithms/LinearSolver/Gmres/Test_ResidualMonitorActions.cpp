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
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/DenseVector.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/Criteria.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/Convergence/Reason.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/ResidualMonitor.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/ResidualMonitorActions.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/LinearSolver/Observe.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/ParallelAlgorithms/LinearSolver/ResidualMonitorActionsTestHelpers.hpp"

// IWYU pragma: no_include <boost/variant/get.hpp>

// IWYU pragma: no_forward_declare db::DataBox

namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
namespace LinearSolver {
namespace gmres_detail {
template <typename FieldsTag>
struct NormalizeInitialOperand;
template <typename FieldsTag>
struct OrthogonalizeOperand;
template <typename FieldsTag>
struct NormalizeOperandAndUpdateField;
}  // namespace gmres_detail
}  // namespace LinearSolver

namespace helpers = ResidualMonitorActionsTestHelpers;

namespace {

struct VectorTag : db::SimpleTag {
  using type = DenseVector<double>;
};

using fields_tag = VectorTag;
using residual_magnitude_tag = db::add_tag_prefix<
    LinearSolver::Tags::Magnitude,
    db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>>;
using orthogonalization_iteration_id_tag =
    db::add_tag_prefix<LinearSolver::Tags::Orthogonalization,
                       LinearSolver::Tags::IterationId>;
using orthogonalization_history_tag =
    db::add_tag_prefix<LinearSolver::Tags::OrthogonalizationHistory,
                       fields_tag>;

struct CheckValueTag : db::SimpleTag {
  using type = double;
};

struct CheckVectorTag : db::SimpleTag {
  using type = DenseVector<double>;
};

using CheckConvergedTag = LinearSolver::Tags::HasConverged;

using check_tags = tmpl::list<CheckValueTag, CheckVectorTag, CheckConvergedTag>;

template <typename Metavariables>
struct MockResidualMonitor {
  using component_being_mocked =
      LinearSolver::gmres_detail::ResidualMonitor<Metavariables, fields_tag>;
  using metavariables = Metavariables;
  // We represent the singleton as an array with only one element for the action
  // testing framework
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags =
      typename LinearSolver::gmres_detail::ResidualMonitor<
          Metavariables, fields_tag>::const_global_cache_tags;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<
          LinearSolver::gmres_detail::InitializeResidualMonitor<fields_tag>>>>;
};

struct MockNormalizeInitialOperand {
  template <
      typename ParallelComponent, typename DbTagsList, typename Metavariables,
      typename ArrayIndex,
      Requires<tmpl::list_contains_v<DbTagsList, CheckValueTag>> = nullptr>
  static void apply(db::DataBox<DbTagsList>& box,  // NOLINT
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const double residual_magnitude,
                    const db::const_item_type<LinearSolver::Tags::HasConverged>&
                        has_converged) noexcept {
    db::mutate<CheckValueTag, CheckConvergedTag>(
        make_not_null(&box),
        [ residual_magnitude, &
          has_converged ](const gsl::not_null<double*> check_value,
                          const gsl::not_null<db::item_type<CheckConvergedTag>*>
                              check_converged) noexcept {
          *check_value = residual_magnitude;
          *check_converged = has_converged;
        });
  }
};

struct MockOrthogonalizeOperand {
  template <
      typename ParallelComponent, typename DbTagsList, typename Metavariables,
      typename ArrayIndex,
      Requires<tmpl::list_contains_v<DbTagsList, CheckValueTag>> = nullptr>
  static void apply(db::DataBox<DbTagsList>& box,  // NOLINT
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const double orthogonalization) noexcept {
    db::mutate<CheckValueTag>(
        make_not_null(&box), [orthogonalization](const gsl::not_null<double*>
                                                     check_value) noexcept {
          *check_value = orthogonalization;
        });
  }
};

struct MockNormalizeOperandAndUpdateField {
  template <
      typename ParallelComponent, typename DbTagsList, typename Metavariables,
      typename ArrayIndex,
      Requires<tmpl::list_contains_v<DbTagsList, CheckValueTag>> = nullptr>
  static void apply(db::DataBox<DbTagsList>& box,  // NOLINT
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const double normalization,
                    const DenseVector<double>& minres,
                    const db::const_item_type<LinearSolver::Tags::HasConverged>&
                        has_converged) noexcept {
    db::mutate<CheckValueTag, CheckVectorTag, CheckConvergedTag>(
        make_not_null(&box),
        [ normalization, &minres, &has_converged ](
            const gsl::not_null<double*> check_value,
            const gsl::not_null<DenseVector<double>*> check_vector,
            const gsl::not_null<db::item_type<CheckConvergedTag>*>
                check_converged) noexcept {
          *check_value = normalization;
          *check_vector = minres;
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
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<check_tags>>>>;

  using replace_these_simple_actions = tmpl::list<
      LinearSolver::gmres_detail::NormalizeInitialOperand<fields_tag>,
      LinearSolver::gmres_detail::OrthogonalizeOperand<fields_tag>,
      LinearSolver::gmres_detail::NormalizeOperandAndUpdateField<fields_tag>>;
  using with_these_simple_actions =
      tmpl::list<MockNormalizeInitialOperand, MockOrthogonalizeOperand,
                 MockNormalizeOperandAndUpdateField>;
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
  ActionTesting::emplace_component_and_initialize<element_array>(
      make_not_null(&runner), 0,
      {std::numeric_limits<double>::signaling_NaN(), DenseVector<double>{},
       db::item_type<CheckConvergedTag>{}});

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
  const auto get_element_tag = [&runner](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner, 0);
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
        LinearSolver::gmres_detail::InitializeResidualMagnitude<fields_tag,
                                                                element_array>>(
        make_not_null(&runner), 0, 2.);
    ActionTesting::invoke_queued_threaded_action<observer_writer>(
        make_not_null(&runner), 0);
    ActionTesting::invoke_queued_simple_action<element_array>(
        make_not_null(&runner), 0);
    // Test residual monitor state
    CHECK(get_residual_monitor_tag(residual_magnitude_tag{}) == 2.);
    CHECK(get_residual_monitor_tag(LinearSolver::Tags::IterationId{}) == 0);
    CHECK_FALSE(get_residual_monitor_tag(LinearSolver::Tags::HasConverged{}));
    // Test element state
    CHECK(get_element_tag(CheckValueTag{}) == 2.);
    CHECK(get_element_tag(CheckConvergedTag{}) ==
          get_residual_monitor_tag(LinearSolver::Tags::HasConverged{}));
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

  SECTION("InitializeResidualMagnitudeAndConverge") {
    ActionTesting::simple_action<
        residual_monitor,
        LinearSolver::gmres_detail::InitializeResidualMagnitude<fields_tag,
                                                                element_array>>(
        make_not_null(&runner), 0, 0.);
    ActionTesting::invoke_queued_simple_action<element_array>(
        make_not_null(&runner), 0);
    // Test residual monitor state
    CHECK(get_residual_monitor_tag(residual_magnitude_tag{}) == 0.);
    CHECK(get_residual_monitor_tag(LinearSolver::Tags::IterationId{}) == 0);
    CHECK(get_residual_monitor_tag(LinearSolver::Tags::HasConverged{}));
    CHECK(
        get_residual_monitor_tag(LinearSolver::Tags::HasConverged{}).reason() ==
        Convergence::Reason::AbsoluteResidual);
    // Test element state
    CHECK(get_element_tag(CheckValueTag{}) == 0.);
    CHECK(get_element_tag(CheckConvergedTag{}) ==
          get_residual_monitor_tag(LinearSolver::Tags::HasConverged{}));
  }

  SECTION("StoreOrthogonalization") {
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::gmres_detail::StoreOrthogonalization<
                              fields_tag, element_array>>(
        make_not_null(&runner), 0, 2.);
    ActionTesting::invoke_queued_simple_action<element_array>(
        make_not_null(&runner), 0);
    // Test residual monitor state
    CHECK(get_residual_monitor_tag(orthogonalization_history_tag{})(0, 0) ==
          2.);
    CHECK(get_residual_monitor_tag(LinearSolver::Tags::IterationId{}) == 0);
    CHECK(get_residual_monitor_tag(orthogonalization_iteration_id_tag{}) == 1);
    // Test element state
    CHECK(get_element_tag(CheckValueTag{}) == 2.);
  }

  SECTION("StoreFinalOrthogonalization") {
    ActionTesting::simple_action<
        residual_monitor,
        LinearSolver::gmres_detail::InitializeResidualMagnitude<fields_tag,
                                                                element_array>>(
        make_not_null(&runner), 0, 2.);
    ActionTesting::invoke_queued_threaded_action<observer_writer>(
        make_not_null(&runner), 0);
    ActionTesting::invoke_queued_simple_action<element_array>(
        make_not_null(&runner), 0);
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::gmres_detail::StoreOrthogonalization<
                              fields_tag, element_array>>(
        make_not_null(&runner), 0, 3.);
    ActionTesting::invoke_queued_simple_action<element_array>(
        make_not_null(&runner), 0);
    // Test intermediate residual monitor state
    CHECK(get_residual_monitor_tag(residual_magnitude_tag{}) == 2.);
    CHECK(get_residual_monitor_tag(orthogonalization_history_tag{}) ==
          DenseMatrix<double>({{3.}, {0.}}));
    CHECK(get_residual_monitor_tag(LinearSolver::Tags::IterationId{}) == 0);
    CHECK(get_residual_monitor_tag(orthogonalization_iteration_id_tag{}) == 1);
    ActionTesting::simple_action<
        residual_monitor,
        LinearSolver::gmres_detail::StoreFinalOrthogonalization<fields_tag,
                                                                element_array>>(
        make_not_null(&runner), 0, 4.);
    ActionTesting::invoke_queued_threaded_action<observer_writer>(
        make_not_null(&runner), 0);
    ActionTesting::invoke_queued_simple_action<element_array>(
        make_not_null(&runner), 0);
    // Test residual monitor state
    // Iteration ids should be prepared for next iteration
    CHECK(get_residual_monitor_tag(LinearSolver::Tags::IterationId{}) == 1);
    CHECK(get_residual_monitor_tag(orthogonalization_iteration_id_tag{}) == 0);
    // H = [[3.], [2.]] and added a zero row and column
    CHECK(get_residual_monitor_tag(orthogonalization_history_tag{}) ==
          DenseMatrix<double>({{3., 0.}, {2., 0.}, {0., 0.}}));
    // Test element state
    // beta = [2., 0.]
    // minres = inv(qr_R(H)) * trans(qr_Q(H)) * beta = [0.4615384615384615]
    CHECK(get_element_tag(CheckVectorTag{}).size() == 1);
    CHECK_ITERABLE_APPROX(get_element_tag(CheckVectorTag{}),
                          DenseVector<double>({0.4615384615384615}));
    // r = beta - H * minres = [0.6153846153846154, -0.923076923076923]
    // |r| = 1.1094003924504583
    CHECK(get_residual_monitor_tag(residual_magnitude_tag{}) ==
          approx(1.1094003924504583));
    CHECK_FALSE(get_residual_monitor_tag(LinearSolver::Tags::HasConverged{}));
    CHECK(get_element_tag(CheckConvergedTag{}) ==
          get_residual_monitor_tag(LinearSolver::Tags::HasConverged{}));
    CHECK(get_element_tag(CheckValueTag{}) == approx(2.));
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
          approx(1.1094003924504583));
  }

  SECTION("ConvergeByAbsoluteResidual") {
    ActionTesting::simple_action<
        residual_monitor,
        LinearSolver::gmres_detail::InitializeResidualMagnitude<fields_tag,
                                                                element_array>>(
        make_not_null(&runner), 0, 2.);
    ActionTesting::invoke_queued_simple_action<element_array>(
        make_not_null(&runner), 0);
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::gmres_detail::StoreOrthogonalization<
                              fields_tag, element_array>>(
        make_not_null(&runner), 0, 1.);
    ActionTesting::invoke_queued_simple_action<element_array>(
        make_not_null(&runner), 0);
    ActionTesting::simple_action<
        residual_monitor,
        LinearSolver::gmres_detail::StoreFinalOrthogonalization<fields_tag,
                                                                element_array>>(
        make_not_null(&runner), 0, 0.);
    ActionTesting::invoke_queued_simple_action<element_array>(
        make_not_null(&runner), 0);
    // Test residual monitor state
    // H = [[1.], [0.]] and added a zero row and column
    CHECK(get_residual_monitor_tag(orthogonalization_history_tag{}) ==
          DenseMatrix<double>({{1., 0.}, {0., 0.}, {0., 0.}}));
    // Test element state
    // beta = [2., 0.]
    // minres = inv(qr_R(H)) * trans(qr_Q(H)) * beta = [2.]
    CHECK(get_element_tag(CheckVectorTag{}).size() == 1);
    CHECK_ITERABLE_APPROX(get_element_tag(CheckVectorTag{}),
                          DenseVector<double>({2.}));
    // r = beta - H * minres = [0., 0.]
    // |r| = 0.
    CHECK(get_residual_monitor_tag(residual_magnitude_tag{}) == approx(0.));
    CHECK(get_residual_monitor_tag(LinearSolver::Tags::HasConverged{}));
    CHECK(
        get_residual_monitor_tag(LinearSolver::Tags::HasConverged{}).reason() ==
        Convergence::Reason::AbsoluteResidual);
    CHECK(get_element_tag(CheckConvergedTag{}) ==
          get_residual_monitor_tag(LinearSolver::Tags::HasConverged{}));
  }

  SECTION("ConvergeByMaxIterations") {
    ActionTesting::simple_action<
        residual_monitor,
        LinearSolver::gmres_detail::InitializeResidualMagnitude<fields_tag,
                                                                element_array>>(
        make_not_null(&runner), 0, 1.);
    ActionTesting::invoke_queued_simple_action<element_array>(
        make_not_null(&runner), 0);
    // Perform 2 mock iterations
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::gmres_detail::StoreOrthogonalization<
                              fields_tag, element_array>>(
        make_not_null(&runner), 0, 1.);
    ActionTesting::invoke_queued_simple_action<element_array>(
        make_not_null(&runner), 0);
    ActionTesting::simple_action<
        residual_monitor,
        LinearSolver::gmres_detail::StoreFinalOrthogonalization<fields_tag,
                                                                element_array>>(
        make_not_null(&runner), 0, 4.);
    ActionTesting::invoke_queued_simple_action<element_array>(
        make_not_null(&runner), 0);
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::gmres_detail::StoreOrthogonalization<
                              fields_tag, element_array>>(
        make_not_null(&runner), 0, 3.);
    ActionTesting::invoke_queued_simple_action<element_array>(
        make_not_null(&runner), 0);
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::gmres_detail::StoreOrthogonalization<
                              fields_tag, element_array>>(
        make_not_null(&runner), 0, 4.);
    ActionTesting::invoke_queued_simple_action<element_array>(
        make_not_null(&runner), 0);
    ActionTesting::simple_action<
        residual_monitor,
        LinearSolver::gmres_detail::StoreFinalOrthogonalization<fields_tag,
                                                                element_array>>(
        make_not_null(&runner), 0, 25.);
    ActionTesting::invoke_queued_simple_action<element_array>(
        make_not_null(&runner), 0);
    // Test residual monitor state
    CHECK(get_residual_monitor_tag(orthogonalization_history_tag{}) ==
          DenseMatrix<double>(
              {{1., 3., 0.}, {2., 4., 0.}, {0., 5., 0.}, {0., 0., 0.}}));
    // Test element state
    // beta = [1., 0., 0.]
    // minres = inv(qr_R(H)) * trans(qr_Q(H)) * beta = [0.13178295, 0.03100775]
    CHECK(get_element_tag(CheckVectorTag{}).size() == 2);
    CHECK_ITERABLE_APPROX(
        get_element_tag(CheckVectorTag{}),
        DenseVector<double>({0.1317829457364342, 0.0310077519379845}));
    // r = beta - H * minres = [0.77519, -0.38759, -0.15503]
    // |r| = 0.8804509063256237
    CHECK(get_residual_monitor_tag(residual_magnitude_tag{}) ==
          approx(0.8804509063256237));
    const auto& has_converged =
        get_residual_monitor_tag(LinearSolver::Tags::HasConverged{});
    CHECK(has_converged);
    CHECK(has_converged.reason() == Convergence::Reason::MaxIterations);
    CHECK(get_element_tag(CheckConvergedTag{}) == has_converged);
  }

  SECTION("ConvergeByRelativeResidual") {
    ActionTesting::simple_action<
        residual_monitor,
        LinearSolver::gmres_detail::InitializeResidualMagnitude<fields_tag,
                                                                element_array>>(
        make_not_null(&runner), 0, 2.);
    ActionTesting::invoke_queued_simple_action<element_array>(
        make_not_null(&runner), 0);
    ActionTesting::simple_action<
        residual_monitor, LinearSolver::gmres_detail::StoreOrthogonalization<
                              fields_tag, element_array>>(
        make_not_null(&runner), 0, 3.);
    ActionTesting::invoke_queued_simple_action<element_array>(
        make_not_null(&runner), 0);
    ActionTesting::simple_action<
        residual_monitor,
        LinearSolver::gmres_detail::StoreFinalOrthogonalization<fields_tag,
                                                                element_array>>(
        make_not_null(&runner), 0, 1.);
    ActionTesting::invoke_queued_simple_action<element_array>(
        make_not_null(&runner), 0);
    // Test residual monitor state
    // H = [[3.], [1.]] and added a zero row and column
    CHECK(get_residual_monitor_tag(orthogonalization_history_tag{}) ==
          DenseMatrix<double>({{3., 0.}, {1., 0.}, {0., 0.}}));
    // Test element state
    // beta = [2., 0.]
    // minres = inv(qr_R(H)) * trans(qr_Q(H)) * beta = [0.6]
    CHECK(get_element_tag(CheckVectorTag{}).size() == 1);
    CHECK_ITERABLE_APPROX(get_element_tag(CheckVectorTag{}),
                          DenseVector<double>({0.6}));
    // r = beta - H * minres = [0.2, -0.6]
    // |r| = 0.6324555320336759
    CHECK(get_residual_monitor_tag(residual_magnitude_tag{}) ==
          approx(0.6324555320336759));
    // |r| / |r_initial| = 0.31622776601683794
    CHECK(get_residual_monitor_tag(LinearSolver::Tags::HasConverged{}));
    CHECK(
        get_residual_monitor_tag(LinearSolver::Tags::HasConverged{}).reason() ==
        Convergence::Reason::RelativeResidual);
    CHECK(get_element_tag(CheckConvergedTag{}) ==
          get_residual_monitor_tag(LinearSolver::Tags::HasConverged{}));
  }
}
