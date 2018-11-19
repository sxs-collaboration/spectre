// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/DenseVector.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/LinearSolver/Gmres/ResidualMonitor.hpp"
#include "NumericalAlgorithms/LinearSolver/Gmres/ResidualMonitorActions.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearSolver/IterationId.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"
// IWYU pragma: no_forward_declare db::DataBox

namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
namespace LinearSolver {
namespace gmres_detail {
struct NormalizeInitialOperand;
struct OrthogonalizeOperand;
struct NormalizeOperand;
struct UpdateFieldAndTerminate;
}  // namespace gmres_detail
}  // namespace LinearSolver

namespace {

struct VectorTag : db::SimpleTag {
  using type = DenseVector<double>;
  static std::string name() noexcept { return "VectorTag"; }
};

using residual_magnitude_tag = db::add_tag_prefix<
    LinearSolver::Tags::Magnitude,
    db::add_tag_prefix<LinearSolver::Tags::Residual, VectorTag>>;
using source_magnitude_tag =
    db::add_tag_prefix<LinearSolver::Tags::Magnitude,
                       db::add_tag_prefix<::Tags::Source, VectorTag>>;
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

template <typename Metavariables>
using residual_monitor_tags =
    tmpl::append<typename LinearSolver::gmres_detail::InitializeResidualMonitor<
                     Metavariables>::simple_tags,
                 typename LinearSolver::gmres_detail::InitializeResidualMonitor<
                     Metavariables>::compute_tags>;

template <typename Metavariables>
struct MockResidualMonitor {
  using metavariables = Metavariables;
  // We represent the singleton as an array with only one element for the action
  // testing framework
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tag_list = tmpl::list<>;
  using action_list = tmpl::list<>;
  using initial_databox =
      db::compute_databox_type<residual_monitor_tags<Metavariables>>;
};

struct MockNormalizeInitialOperand {
  template <typename... InboxTags, typename Metavariables, typename ActionList,
            typename ParallelComponent, typename ArrayIndex>
  static void apply(
      db::DataBox<tmpl::list<CheckValueTag, CheckVectorTag>>& box,  // NOLINT
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/,
      const double residual_squared) noexcept {
    db::mutate<CheckValueTag>(
        make_not_null(&box), [residual_squared](const gsl::not_null<double*>
                                                    value_box) noexcept {
          *value_box = residual_squared;
        });
  }
};

struct MockOrthogonalizeOperand {
  template <typename... InboxTags, typename Metavariables, typename ActionList,
            typename ParallelComponent, typename ArrayIndex>
  static void apply(
      db::DataBox<tmpl::list<CheckValueTag, CheckVectorTag>>& box,  // NOLINT
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/,
      const double orthogonalization) noexcept {
    db::mutate<CheckValueTag>(
        make_not_null(&box), [orthogonalization](const gsl::not_null<double*>
                                                     value_box) noexcept {
          *value_box = orthogonalization;
        });
  }
};

struct MockNormalizeOperand {
  template <typename... InboxTags, typename Metavariables, typename ActionList,
            typename ParallelComponent, typename ArrayIndex>
  static void apply(
      db::DataBox<tmpl::list<CheckValueTag, CheckVectorTag>>& box,  // NOLINT
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/,
      const double normalization) noexcept {
    db::mutate<CheckValueTag>(
        make_not_null(&box), [normalization](const gsl::not_null<double*>
                                                 value_box) noexcept {
          *value_box = normalization;
        });
  }
};

struct MockUpdateFieldAndTerminate {
  template <typename... InboxTags, typename Metavariables, typename ActionList,
            typename ParallelComponent, typename ArrayIndex>
  static void apply(
      db::DataBox<tmpl::list<CheckValueTag, CheckVectorTag>>& box,  // NOLINT
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/,
      const DenseVector<double>& minres) noexcept {
    db::mutate<CheckVectorTag>(
        make_not_null(&box), [minres](const gsl::not_null<DenseVector<double>*>
                                          vector_box) noexcept {
          *vector_box = minres;
        });
  }
};

// This is used to receive action calls from the residual monitor
template <typename Metavariables>
struct MockElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tag_list = tmpl::list<>;
  using action_list = tmpl::list<>;
  using initial_databox =
      db::compute_databox_type<tmpl::list<CheckValueTag, CheckVectorTag>>;

  using replace_these_simple_actions =
      tmpl::list<LinearSolver::gmres_detail::NormalizeInitialOperand,
                 LinearSolver::gmres_detail::OrthogonalizeOperand,
                 LinearSolver::gmres_detail::NormalizeOperand,
                 LinearSolver::gmres_detail::UpdateFieldAndTerminate>;
  using with_these_simple_actions =
      tmpl::list<MockNormalizeInitialOperand, MockOrthogonalizeOperand,
                 MockNormalizeOperand, MockUpdateFieldAndTerminate>;
};

struct System {
  using fields_tag = VectorTag;
};

struct Metavariables {
  using component_list = tmpl::list<MockResidualMonitor<Metavariables>,
                                    MockElementArray<Metavariables>>;
  using system = System;
  using const_global_cache_tag_list = tmpl::list<>;
};

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearSolver.Gmres.ResidualMonitorActions",
                  "[Unit][NumericalAlgorithms][LinearSolver][Actions]") {
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  MockRuntimeSystem::TupleOfMockDistributedObjects dist_objects{};

  // Setup mock residual monitor
  using MockSingletonObjectsTag = MockRuntimeSystem::MockDistributedObjectsTag<
      MockResidualMonitor<Metavariables>>;
  const int singleton_id{0};
  tuples::get<MockSingletonObjectsTag>(dist_objects)
      .emplace(
          singleton_id,
          db::create<residual_monitor_tags<Metavariables>>(
              Verbosity::Verbose, std::numeric_limits<double>::signaling_NaN(),
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
               db::create<db::AddSimpleTags<CheckValueTag, CheckVectorTag>>(
                   std::numeric_limits<double>::signaling_NaN(),
                   DenseVector<double>{}));

  MockRuntimeSystem runner{{}, std::move(dist_objects)};

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
        .get_databox<db::compute_databox_type<
            db::AddSimpleTags<CheckValueTag, CheckVectorTag>>>();
  };

  SECTION("InitializeResidualMagnitude") {
    runner
        .simple_action<MockResidualMonitor<Metavariables>,
                       LinearSolver::gmres_detail::InitializeResidualMagnitude<
                           MockElementArray<Metavariables>>>(singleton_id, 2.);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    const auto& box = get_box();
    CHECK(db::get<residual_magnitude_tag>(box) == 2.);
    CHECK(db::get<LinearSolver::Tags::IterationId>(box).step_number == 0);
    const auto& mock_element_box = get_mock_element_box();
    CHECK(db::get<CheckValueTag>(mock_element_box) == 2.);
  }

  SECTION("InitializeSourceMagnitude") {
    runner.simple_action<MockResidualMonitor<Metavariables>,
                         LinearSolver::gmres_detail::InitializeSourceMagnitude>(
        singleton_id, 2.);
    const auto& box = get_box();
    CHECK(db::get<source_magnitude_tag>(box) == 2.);
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
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    runner.simple_action<MockResidualMonitor<Metavariables>,
                         LinearSolver::gmres_detail::InitializeSourceMagnitude>(
        singleton_id, 2.);
    runner.simple_action<MockResidualMonitor<Metavariables>,
                         LinearSolver::gmres_detail::StoreOrthogonalization<
                             MockElementArray<Metavariables>>>(singleton_id,
                                                               3.);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    const auto& intermediate_box = get_box();
    CHECK(db::get<residual_magnitude_tag>(intermediate_box) == 2.);
    CHECK(db::get<source_magnitude_tag>(intermediate_box) == 2.);
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
    const auto& box = get_box();
    // residual should be nonzero, so don't terminate
    CHECK(db::get<LinearSolver::Tags::IterationId>(box).step_number == 1);
    CHECK(db::get<orthogonalization_iteration_id_tag>(box).step_number == 0);
    CHECK(db::get<orthogonalization_history_tag>(box) ==
          DenseMatrix<double>({{3., 0.}, {2., 0.}, {0., 0.}}));
    const auto& mock_element_box = get_mock_element_box();
    CHECK(db::get<CheckValueTag>(mock_element_box) == 2.);
  }

  SECTION("StoreFinalOrthogonalizationAndTerminate") {
    runner
        .simple_action<MockResidualMonitor<Metavariables>,
                       LinearSolver::gmres_detail::InitializeResidualMagnitude<
                           MockElementArray<Metavariables>>>(singleton_id, 2.);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    runner.simple_action<MockResidualMonitor<Metavariables>,
                         LinearSolver::gmres_detail::InitializeSourceMagnitude>(
        singleton_id, 2.);
    runner.simple_action<MockResidualMonitor<Metavariables>,
                         LinearSolver::gmres_detail::StoreOrthogonalization<
                             MockElementArray<Metavariables>>>(singleton_id,
                                                               1.);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    runner.simple_action<
        MockResidualMonitor<Metavariables>,
        LinearSolver::gmres_detail::StoreFinalOrthogonalization<
            MockElementArray<Metavariables>>>(singleton_id, 0.);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    const auto& box = get_box();
    CHECK(db::get<orthogonalization_history_tag>(box) ==
          DenseMatrix<double>({{1.}, {0.}}));
    // residual should be zero, so terminate
    const auto& mock_element_box = get_mock_element_box();
    CHECK_ITERABLE_APPROX(db::get<CheckVectorTag>(mock_element_box),
                          DenseVector<double>({2.}));
  }
}
