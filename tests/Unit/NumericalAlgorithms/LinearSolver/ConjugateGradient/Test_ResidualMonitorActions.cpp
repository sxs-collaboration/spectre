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
#include "DataStructures/DenseVector.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/LinearSolver/ConjugateGradient/ResidualMonitor.hpp"
#include "NumericalAlgorithms/LinearSolver/ConjugateGradient/ResidualMonitorActions.hpp"  // IWYU pragma: keep
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
namespace cg_detail {
struct UpdateFieldValues;
struct UpdateOperand;
}  // namespace cg_detail
}  // namespace LinearSolver

namespace {

struct VectorTag : db::SimpleTag {
  using type = DenseVector<double>;
  static std::string name() noexcept { return "VectorTag"; }
};

struct CheckValueTag : db::SimpleTag {
  using type = double;
  static std::string name() noexcept { return "CheckValueTag"; }
};

struct CheckTerminateTag : db::SimpleTag {
  using type = bool;
  static std::string name() noexcept { return "CheckTerminateTag"; }
};

template <typename Metavariables>
using residual_monitor_tags =
    tmpl::append<typename LinearSolver::cg_detail::InitializeResidualMonitor<
                     Metavariables>::simple_tags,
                 typename LinearSolver::cg_detail::InitializeResidualMonitor<
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

struct MockUpdateFieldValues {
  template <typename... InboxTags, typename Metavariables, typename ActionList,
            typename ParallelComponent, typename ArrayIndex>
  static void apply(
      db::DataBox<tmpl::list<CheckValueTag, CheckTerminateTag>>& box,  // NOLINT
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/, const double alpha) noexcept {
    db::mutate<CheckValueTag>(
        make_not_null(&box), [alpha](const gsl::not_null<double*>
                                         value_box) noexcept {
          *value_box = alpha;
        });
  }
};

struct MockUpdateOperand {
  template <typename... InboxTags, typename Metavariables, typename ActionList,
            typename ParallelComponent, typename ArrayIndex>
  static void apply(
      db::DataBox<tmpl::list<CheckValueTag, CheckTerminateTag>>& box,  // NOLINT
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/, const double res_ratio,
      const bool terminate) noexcept {
    db::mutate<CheckValueTag, CheckTerminateTag>(
        make_not_null(&box),
        [res_ratio, terminate](
            const gsl::not_null<double*> value_box,
            const gsl::not_null<bool*> terminate_box) noexcept {
          *value_box = res_ratio;
          *terminate_box = terminate;
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
      db::compute_databox_type<tmpl::list<CheckValueTag, CheckTerminateTag>>;

  using replace_these_simple_actions =
      tmpl::list<LinearSolver::cg_detail::UpdateFieldValues,
                 LinearSolver::cg_detail::UpdateOperand>;
  using with_these_simple_actions =
      tmpl::list<MockUpdateFieldValues, MockUpdateOperand>;
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

SPECTRE_TEST_CASE(
    "Unit.Numerical.LinearSolver.ConjugateGradient.ResidualMonitorActions",
    "[Unit][NumericalAlgorithms][LinearSolver][Actions]") {
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  MockRuntimeSystem::TupleOfMockDistributedObjects dist_objects{};

  // Setup mock residual monitor
  using MockSingletonObjectsTag = MockRuntimeSystem::MockDistributedObjectsTag<
      MockResidualMonitor<Metavariables>>;
  const int singleton_id{0};
  tuples::get<MockSingletonObjectsTag>(dist_objects)
      .emplace(singleton_id,
               db::create<residual_monitor_tags<Metavariables>>(
                   Verbosity::Verbose, LinearSolver::IterationId{0},
                   std::numeric_limits<double>::signaling_NaN()));

  // Setup mock element array
  using MockDistributedObjectsTag =
      MockRuntimeSystem::MockDistributedObjectsTag<
          MockElementArray<Metavariables>>;
  const int element_id{0};
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(element_id,
               db::create<db::AddSimpleTags<CheckValueTag, CheckTerminateTag>>(
                   std::numeric_limits<double>::signaling_NaN(), false));

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
            db::AddSimpleTags<CheckValueTag, CheckTerminateTag>>>();
  };

  using residual_square_tag =
      LinearSolver::Tags::ResidualMagnitudeSquare<VectorTag>;

  SECTION("InitializeResidual") {
    runner.simple_action<MockResidualMonitor<Metavariables>,
                         LinearSolver::cg_detail::InitializeResidual>(
        singleton_id, 1.);
    const auto& box = get_box();
    CHECK(db::get<residual_square_tag>(box) == 1.);
    CHECK(db::get<LinearSolver::Tags::IterationId>(box).step_number == 0);
  }

  SECTION("ComputeAlpha") {
    runner.simple_action<MockResidualMonitor<Metavariables>,
                         LinearSolver::cg_detail::InitializeResidual>(
        singleton_id, 1.);
    runner.simple_action<
        MockResidualMonitor<Metavariables>,
        LinearSolver::cg_detail::ComputeAlpha<MockElementArray<Metavariables>>>(
        singleton_id, 2.);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    const auto& box = get_box();
    CHECK(db::get<residual_square_tag>(box) == 1.);
    CHECK(db::get<LinearSolver::Tags::IterationId>(box).step_number == 0);
    const auto& mock_element_box = get_mock_element_box();
    CHECK(db::get<CheckValueTag>(mock_element_box) == 0.5);
  }

  SECTION("UpdateResidual") {
    runner.simple_action<MockResidualMonitor<Metavariables>,
                         LinearSolver::cg_detail::InitializeResidual>(
        singleton_id, 1.);
    runner.simple_action<MockResidualMonitor<Metavariables>,
                         LinearSolver::cg_detail::UpdateResidual<
                             MockElementArray<Metavariables>>>(singleton_id,
                                                               10.);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    const auto& box = get_box();
    CHECK(db::get<residual_square_tag>(box) == 10.);
    CHECK(db::get<LinearSolver::Tags::IterationId>(box).step_number == 1);
    const auto& mock_element_box = get_mock_element_box();
    CHECK(db::get<CheckValueTag>(mock_element_box) == 10.);
    CHECK(db::get<CheckTerminateTag>(mock_element_box) == false);
  }

  SECTION("UpdateResidualAndTerminate") {
    runner.simple_action<MockResidualMonitor<Metavariables>,
                         LinearSolver::cg_detail::InitializeResidual>(
        singleton_id, 1.);
    runner.simple_action<MockResidualMonitor<Metavariables>,
                         LinearSolver::cg_detail::UpdateResidual<
                             MockElementArray<Metavariables>>>(singleton_id,
                                                               0.);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    const auto& box = get_box();
    CHECK(db::get<residual_square_tag>(box) == 0.);
    CHECK(db::get<LinearSolver::Tags::IterationId>(box).step_number == 1);
    const auto& mock_element_box = get_mock_element_box();
    CHECK(db::get<CheckValueTag>(mock_element_box) == 0.);
    CHECK(db::get<CheckTerminateTag>(mock_element_box) == true);
  }
}
