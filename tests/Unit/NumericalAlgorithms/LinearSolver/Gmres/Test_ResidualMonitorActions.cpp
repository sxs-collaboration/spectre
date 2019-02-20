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
// IWYU pragma: no_include <boost/variant/get.hpp>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/DenseVector.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/LinearSolver/Convergence.hpp"
#include "NumericalAlgorithms/LinearSolver/Gmres/ResidualMonitor.hpp"
#include "NumericalAlgorithms/LinearSolver/Gmres/ResidualMonitorActions.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearSolver/IterationId.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/Gsl.hpp"
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
namespace gmres_detail {
struct NormalizeInitialOperand;
struct OrthogonalizeOperand;
struct NormalizeOperandAndUpdateField;
}  // namespace gmres_detail
}  // namespace LinearSolver

// NOLINTNEXTLINE(google-build-using-namespace)
using namespace ResidualMonitorActionsTestHelpers;

namespace {

struct VectorTag : db::SimpleTag {
  using type = DenseVector<double>;
  static std::string name() noexcept { return "VectorTag"; }
};

using residual_magnitude_tag = db::add_tag_prefix<
    LinearSolver::Tags::Magnitude,
    db::add_tag_prefix<LinearSolver::Tags::Residual, VectorTag>>;
using orthogonalization_iteration_id_tag =
    db::add_tag_prefix<LinearSolver::Tags::Orthogonalization,
                       LinearSolver::Tags::IterationId>;
using orthogonalization_history_tag =
    db::add_tag_prefix<LinearSolver::Tags::OrthogonalizationHistory, VectorTag>;

struct CheckValueTag : db::SimpleTag {
  using type = double;
  static std::string name() noexcept { return "CheckValueTag"; }
};

struct CheckVectorTag : db::SimpleTag {
  using type = DenseVector<double>;
  static std::string name() noexcept { return "CheckVectorTag"; }
};

using CheckConvergedTag = LinearSolver::Tags::HasConverged;

using check_tags = tmpl::list<CheckValueTag, CheckVectorTag, CheckConvergedTag>;

template <typename Metavariables>
using residual_monitor_tags =
    tmpl::append<typename LinearSolver::gmres_detail::InitializeResidualMonitor<
                     Metavariables>::simple_tags,
                 typename LinearSolver::gmres_detail::InitializeResidualMonitor<
                     Metavariables>::compute_tags>;

template <typename Metavariables>
struct MockResidualMonitor {
  using component_being_mocked =
      LinearSolver::gmres_detail::ResidualMonitor<Metavariables>;
  using metavariables = Metavariables;
  // We represent the singleton as an array with only one element for the action
  // testing framework
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tag_list =
      tmpl::list<LinearSolver::OptionTags::ResidualMonitorOptions>;
  using action_list = tmpl::list<>;
  using initial_databox =
      db::compute_databox_type<residual_monitor_tags<Metavariables>>;
};

struct MockNormalizeInitialOperand {
  template <typename... InboxTags, typename Metavariables, typename ActionList,
            typename ParallelComponent, typename ArrayIndex>
  static void apply(db::DataBox<check_tags>& box,  // NOLINT
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const double residual_magnitude,
                    const db::item_type<LinearSolver::Tags::HasConverged>&
                        has_converged) noexcept {
    db::mutate<CheckValueTag, CheckConvergedTag>(
        make_not_null(&box),
        [ residual_magnitude, &has_converged ](
            const gsl::not_null<double*> check_value,
            const gsl::not_null<db::item_type<CheckConvergedTag>*>
                check_converged) noexcept {
          *check_value = residual_magnitude;
          *check_converged = has_converged;
        });
  }
};

struct MockOrthogonalizeOperand {
  template <typename... InboxTags, typename Metavariables, typename ActionList,
            typename ParallelComponent, typename ArrayIndex>
  static void apply(db::DataBox<check_tags>& box,  // NOLINT
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const double orthogonalization) noexcept {
    db::mutate<CheckValueTag>(
        make_not_null(&box), [orthogonalization](const gsl::not_null<double*>
                                                     check_value) noexcept {
          *check_value = orthogonalization;
        });
  }
};

struct MockNormalizeOperandAndUpdateField {
  template <typename... InboxTags, typename Metavariables, typename ActionList,
            typename ParallelComponent, typename ArrayIndex>
  static void apply(db::DataBox<check_tags>& box,  // NOLINT
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const double normalization,
                    const DenseVector<double>& minres,
                    const db::item_type<LinearSolver::Tags::HasConverged>&
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
  using const_global_cache_tag_list = tmpl::list<>;
  using action_list = tmpl::list<>;
  using initial_databox = db::compute_databox_type<check_tags>;

  using replace_these_simple_actions =
      tmpl::list<LinearSolver::gmres_detail::NormalizeInitialOperand,
                 LinearSolver::gmres_detail::OrthogonalizeOperand,
                 LinearSolver::gmres_detail::NormalizeOperandAndUpdateField>;
  using with_these_simple_actions =
      tmpl::list<MockNormalizeInitialOperand, MockOrthogonalizeOperand,
                 MockNormalizeOperandAndUpdateField>;
};

struct System {
  using fields_tag = VectorTag;
};

struct Metavariables {
  using component_list = tmpl::list<MockResidualMonitor<Metavariables>,
                                    MockElementArray<Metavariables>,
                                    MockObserverWriter<Metavariables>>;
  using system = System;
  using const_global_cache_tag_list = tmpl::list<>;
};

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearSolver.Gmres.ResidualMonitorActions",
                  "[Unit][NumericalAlgorithms][LinearSolver][Actions]") {
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  MockRuntimeSystem::TupleOfMockDistributedObjects dist_objects{};

  // Setup mock residual monitor
  const LinearSolver::OptionTags::ResidualMonitorOptions options{
      Verbosity::Verbose, LinearSolver::ConvergenceCriteria{2, 0., 0.5}};
  using MockSingletonObjectsTag = MockRuntimeSystem::MockDistributedObjectsTag<
      MockResidualMonitor<Metavariables>>;
  const int singleton_id{0};
  tuples::get<MockSingletonObjectsTag>(dist_objects)
      .emplace(
          singleton_id,
          db::create<
              typename LinearSolver::gmres_detail::InitializeResidualMonitor<
                  Metavariables>::simple_tags,
              typename LinearSolver::gmres_detail::InitializeResidualMonitor<
                  Metavariables>::compute_tags>(
              get<LinearSolver::Tags::ConvergenceCriteria>(options),
              std::numeric_limits<double>::signaling_NaN(),
              std::numeric_limits<double>::signaling_NaN(),
              LinearSolver::IterationId{0}, LinearSolver::IterationId{0},
              DenseMatrix<double>{2, 1, 0.}));

  // Setup mock element array
  using MockDistributedObjectsTag =
      MockRuntimeSystem::MockDistributedObjectsTag<
          MockElementArray<Metavariables>>;
  const int element_id{0};
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(element_id,
               db::create<check_tags>(
                   std::numeric_limits<double>::signaling_NaN(),
                   DenseVector<double>{}, db::item_type<CheckConvergedTag>{}));

  // Setup mock observer writer
  using MockObserverObjectsTag = MockRuntimeSystem::MockDistributedObjectsTag<
      MockObserverWriter<Metavariables>>;
  tuples::get<MockObserverObjectsTag>(dist_objects)
      .emplace(singleton_id,
               db::create<observer_writer_tags>(
                   observers::ObservationId{}, std::string{},
                   std::vector<std::string>{},
                   std::tuple<size_t, double>{
                       0, std::numeric_limits<double>::signaling_NaN()}));

  MockRuntimeSystem runner{{options}, std::move(dist_objects)};

  // DataBox shortcuts
  const auto get_box = [&runner, &singleton_id]() -> decltype(auto) {
    return runner.algorithms<MockResidualMonitor<Metavariables>>()
        .at(singleton_id)
        .get_databox<
            db::compute_databox_type<residual_monitor_tags<Metavariables>>>();
  };
  const auto get_mock_element_box = [&runner, &element_id]() -> decltype(auto) {
    return runner.algorithms<MockElementArray<Metavariables>>()
        .at(element_id)
        .get_databox<db::compute_databox_type<check_tags>>();
  };
  const auto get_mock_observer_writer_box =
      [&runner, &singleton_id]() -> decltype(auto) {
    return runner.algorithms<MockObserverWriter<Metavariables>>()
        .at(singleton_id)
        .get_databox<db::compute_databox_type<observer_writer_tags>>();
  };

  SECTION("InitializeResidualMagnitude") {
    runner
        .simple_action<MockResidualMonitor<Metavariables>,
                       LinearSolver::gmres_detail::InitializeResidualMagnitude<
                           MockElementArray<Metavariables>>>(singleton_id, 2.);
    runner.invoke_queued_threaded_action<MockObserverWriter<Metavariables>>(
        singleton_id);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    const auto& box = get_box();
    CHECK(db::get<residual_magnitude_tag>(box) == 2.);
    CHECK(db::get<LinearSolver::Tags::IterationId>(box).step_number == 0);
    CHECK_FALSE(db::get<LinearSolver::Tags::HasConverged>(box));
    const auto& mock_element_box = get_mock_element_box();
    CHECK(db::get<CheckValueTag>(mock_element_box) == 2.);
    CHECK(db::get<CheckConvergedTag>(mock_element_box) ==
          db::get<LinearSolver::Tags::HasConverged>(box));
    const auto& mock_observer_writer_box = get_mock_observer_writer_box();
    CHECK(db::get<CheckObservationIdTag>(mock_observer_writer_box) ==
          observers::ObservationId{LinearSolver::IterationId{0}});
    CHECK(db::get<CheckSubfileNameTag>(mock_observer_writer_box) ==
          "/linear_residuals");
    CHECK(db::get<CheckReductionNamesTag>(mock_observer_writer_box) ==
          std::vector<std::string>{"Iteration", "Residual"});
    CHECK(get<0>(db::get<CheckReductionDataTag>(mock_observer_writer_box)) ==
          0);
    CHECK(get<1>(db::get<CheckReductionDataTag>(mock_observer_writer_box)) ==
          approx(2.));
  }

  SECTION("InitializeResidualMagnitudeAndConverge") {
    runner
        .simple_action<MockResidualMonitor<Metavariables>,
                       LinearSolver::gmres_detail::InitializeResidualMagnitude<
                           MockElementArray<Metavariables>>>(singleton_id, 0.);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    const auto& box = get_box();
    CHECK(db::get<residual_magnitude_tag>(box) == 0.);
    CHECK(db::get<LinearSolver::Tags::IterationId>(box).step_number == 0);
    CHECK(db::get<LinearSolver::Tags::HasConverged>(box));
    CHECK(db::get<LinearSolver::Tags::HasConverged>(box).reason() ==
          LinearSolver::ConvergenceReason::AbsoluteResidual);
    const auto& mock_element_box = get_mock_element_box();
    CHECK(db::get<CheckValueTag>(mock_element_box) == 0.);
    CHECK(db::get<CheckConvergedTag>(mock_element_box) ==
          db::get<LinearSolver::Tags::HasConverged>(box));
  }

  SECTION("StoreOrthogonalization") {
    runner.simple_action<MockResidualMonitor<Metavariables>,
                         LinearSolver::gmres_detail::StoreOrthogonalization<
                             MockElementArray<Metavariables>>>(singleton_id,
                                                               2.);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    const auto& box = get_box();
    CHECK(db::get<orthogonalization_history_tag>(box)(0, 0) == 2.);
    CHECK(db::get<LinearSolver::Tags::IterationId>(box).step_number == 0);
    CHECK(db::get<orthogonalization_iteration_id_tag>(box).step_number == 1);
    const auto& mock_element_box = get_mock_element_box();
    CHECK(db::get<CheckValueTag>(mock_element_box) == 2.);
  }

  SECTION("StoreFinalOrthogonalization") {
    runner
        .simple_action<MockResidualMonitor<Metavariables>,
                       LinearSolver::gmres_detail::InitializeResidualMagnitude<
                           MockElementArray<Metavariables>>>(singleton_id, 2.);
    runner.invoke_queued_threaded_action<MockObserverWriter<Metavariables>>(
        singleton_id);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    runner.simple_action<MockResidualMonitor<Metavariables>,
                         LinearSolver::gmres_detail::StoreOrthogonalization<
                             MockElementArray<Metavariables>>>(singleton_id,
                                                               3.);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    const auto& intermediate_box = get_box();
    CHECK(db::get<residual_magnitude_tag>(intermediate_box) == 2.);
    CHECK(db::get<orthogonalization_history_tag>(intermediate_box) ==
          DenseMatrix<double>({{3.}, {0.}}));
    CHECK(db::get<LinearSolver::Tags::IterationId>(intermediate_box)
              .step_number == 0);
    CHECK(db::get<orthogonalization_iteration_id_tag>(intermediate_box)
              .step_number == 1);
    runner
        .simple_action<MockResidualMonitor<Metavariables>,
                       LinearSolver::gmres_detail::StoreFinalOrthogonalization<
                           MockElementArray<Metavariables>>>(singleton_id, 4.);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    runner.invoke_queued_threaded_action<MockObserverWriter<Metavariables>>(
        singleton_id);
    const auto& box = get_box();
    // Iteration ids should be prepared for next iteration
    CHECK(db::get<LinearSolver::Tags::IterationId>(box).step_number == 1);
    CHECK(db::get<orthogonalization_iteration_id_tag>(box).step_number == 0);
    // H = [[3.], [2.]] and added a zero row and column
    CHECK(db::get<orthogonalization_history_tag>(box) ==
          DenseMatrix<double>({{3., 0.}, {2., 0.}, {0., 0.}}));
    const auto& mock_element_box = get_mock_element_box();
    // beta = [2., 0.]
    // minres = inv(qr_R(H)) * trans(qr_Q(H)) * beta = [0.4615384615384615]
    CHECK(db::get<CheckVectorTag>(mock_element_box).size() == 1);
    CHECK_ITERABLE_APPROX(db::get<CheckVectorTag>(mock_element_box),
                          DenseVector<double>({0.4615384615384615}));
    // r = beta - H * minres = [0.6153846153846154, -0.923076923076923]
    // |r| = 1.1094003924504583
    CHECK(get<residual_magnitude_tag>(box) == approx(1.1094003924504583));
    CHECK_FALSE(db::get<LinearSolver::Tags::HasConverged>(box));
    CHECK(db::get<CheckConvergedTag>(mock_element_box) ==
          db::get<LinearSolver::Tags::HasConverged>(box));
    CHECK(db::get<CheckValueTag>(mock_element_box) == approx(2.));
    const auto& mock_observer_writer_box = get_mock_observer_writer_box();
    CHECK(db::get<CheckObservationIdTag>(mock_observer_writer_box) ==
          observers::ObservationId{LinearSolver::IterationId{1}});
    CHECK(db::get<CheckSubfileNameTag>(mock_observer_writer_box) ==
          "/linear_residuals");
    CHECK(db::get<CheckReductionNamesTag>(mock_observer_writer_box) ==
          std::vector<std::string>{"Iteration", "Residual"});
    CHECK(get<0>(db::get<CheckReductionDataTag>(mock_observer_writer_box)) ==
          1);
    CHECK(get<1>(db::get<CheckReductionDataTag>(mock_observer_writer_box)) ==
          approx(1.1094003924504583));
  }

  SECTION("ConvergeByAbsoluteResidual") {
    runner
        .simple_action<MockResidualMonitor<Metavariables>,
                       LinearSolver::gmres_detail::InitializeResidualMagnitude<
                           MockElementArray<Metavariables>>>(singleton_id, 2.);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    runner.simple_action<MockResidualMonitor<Metavariables>,
                         LinearSolver::gmres_detail::StoreOrthogonalization<
                             MockElementArray<Metavariables>>>(singleton_id,
                                                               1.);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    runner
        .simple_action<MockResidualMonitor<Metavariables>,
                       LinearSolver::gmres_detail::StoreFinalOrthogonalization<
                           MockElementArray<Metavariables>>>(singleton_id, 0.);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    const auto& box = get_box();
    // H = [[1.], [0.]] and added a zero row and column
    CHECK(db::get<orthogonalization_history_tag>(box) ==
          DenseMatrix<double>({{1., 0.}, {0., 0.}, {0., 0.}}));
    const auto& mock_element_box = get_mock_element_box();
    // beta = [2., 0.]
    // minres = inv(qr_R(H)) * trans(qr_Q(H)) * beta = [2.]
    CHECK(db::get<CheckVectorTag>(mock_element_box).size() == 1);
    CHECK_ITERABLE_APPROX(db::get<CheckVectorTag>(mock_element_box),
                          DenseVector<double>({2.}));
    // r = beta - H * minres = [0., 0.]
    // |r| = 0.
    CHECK(get<residual_magnitude_tag>(box) == approx(0.));
    CHECK(db::get<LinearSolver::Tags::HasConverged>(box));
    CHECK(db::get<LinearSolver::Tags::HasConverged>(box).reason() ==
          LinearSolver::ConvergenceReason::AbsoluteResidual);
    CHECK(db::get<CheckConvergedTag>(mock_element_box) ==
          db::get<LinearSolver::Tags::HasConverged>(box));
  }

  SECTION("ConvergeByMaxIterations") {
    runner
        .simple_action<MockResidualMonitor<Metavariables>,
                       LinearSolver::gmres_detail::InitializeResidualMagnitude<
                           MockElementArray<Metavariables>>>(singleton_id, 1.);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    // Perform 2 mock iterations
    runner.simple_action<MockResidualMonitor<Metavariables>,
                         LinearSolver::gmres_detail::StoreOrthogonalization<
                             MockElementArray<Metavariables>>>(singleton_id,
                                                               1.);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    runner
        .simple_action<MockResidualMonitor<Metavariables>,
                       LinearSolver::gmres_detail::StoreFinalOrthogonalization<
                           MockElementArray<Metavariables>>>(singleton_id, 4.);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    runner.simple_action<MockResidualMonitor<Metavariables>,
                         LinearSolver::gmres_detail::StoreOrthogonalization<
                             MockElementArray<Metavariables>>>(singleton_id,
                                                               3.);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    runner.simple_action<MockResidualMonitor<Metavariables>,
                         LinearSolver::gmres_detail::StoreOrthogonalization<
                             MockElementArray<Metavariables>>>(singleton_id,
                                                               4.);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    runner
        .simple_action<MockResidualMonitor<Metavariables>,
                       LinearSolver::gmres_detail::StoreFinalOrthogonalization<
                           MockElementArray<Metavariables>>>(singleton_id, 25.);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    const auto& box = get_box();
    CHECK(db::get<orthogonalization_history_tag>(box) ==
          DenseMatrix<double>(
              {{1., 3., 0.}, {2., 4., 0.}, {0., 5., 0.}, {0., 0., 0.}}));
    const auto& mock_element_box = get_mock_element_box();
    // beta = [1., 0., 0.]
    // minres = inv(qr_R(H)) * trans(qr_Q(H)) * beta = [0.13178295, 0.03100775]
    CHECK(db::get<CheckVectorTag>(mock_element_box).size() == 2);
    CHECK_ITERABLE_APPROX(
        db::get<CheckVectorTag>(mock_element_box),
        DenseVector<double>({0.1317829457364342, 0.0310077519379845}));
    // r = beta - H * minres = [0.77519, -0.38759, -0.15503]
    // |r| = 0.8804509063256237
    CHECK(get<residual_magnitude_tag>(box) == approx(0.8804509063256237));
    const auto& has_converged = get<LinearSolver::Tags::HasConverged>(box);
    CHECK(has_converged);
    CHECK(has_converged.reason() ==
          LinearSolver::ConvergenceReason::MaxIterations);
    CHECK(db::get<CheckConvergedTag>(mock_element_box) == has_converged);
  }

  SECTION("ConvergeByRelativeResidual") {
    runner
        .simple_action<MockResidualMonitor<Metavariables>,
                       LinearSolver::gmres_detail::InitializeResidualMagnitude<
                           MockElementArray<Metavariables>>>(singleton_id, 2.);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    runner.simple_action<MockResidualMonitor<Metavariables>,
                         LinearSolver::gmres_detail::StoreOrthogonalization<
                             MockElementArray<Metavariables>>>(singleton_id,
                                                               3.);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    runner
        .simple_action<MockResidualMonitor<Metavariables>,
                       LinearSolver::gmres_detail::StoreFinalOrthogonalization<
                           MockElementArray<Metavariables>>>(singleton_id, 1.);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    const auto& box = get_box();
    // H = [[3.], [1.]] and added a zero row and column
    CHECK(db::get<orthogonalization_history_tag>(box) ==
          DenseMatrix<double>({{3., 0.}, {1., 0.}, {0., 0.}}));
    const auto& mock_element_box = get_mock_element_box();
    // beta = [2., 0.]
    // minres = inv(qr_R(H)) * trans(qr_Q(H)) * beta = [0.6]
    CHECK(db::get<CheckVectorTag>(mock_element_box).size() == 1);
    CHECK_ITERABLE_APPROX(db::get<CheckVectorTag>(mock_element_box),
                          DenseVector<double>({0.6}));
    // r = beta - H * minres = [0.2, -0.6]
    // |r| = 0.6324555320336759
    CHECK(get<residual_magnitude_tag>(box) == approx(0.6324555320336759));
    // |r| / |r_initial| = 0.31622776601683794
    CHECK(db::get<LinearSolver::Tags::HasConverged>(box));
    CHECK(db::get<LinearSolver::Tags::HasConverged>(box).reason() ==
          LinearSolver::ConvergenceReason::RelativeResidual);
    CHECK(db::get<CheckConvergedTag>(mock_element_box) ==
          db::get<LinearSolver::Tags::HasConverged>(box));
  }
}
