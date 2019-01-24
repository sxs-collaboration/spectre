// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>
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
#include "NumericalAlgorithms/LinearSolver/ConjugateGradient/ResidualMonitor.hpp"
#include "NumericalAlgorithms/LinearSolver/ConjugateGradient/ResidualMonitorActions.hpp"  // IWYU pragma: keep
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
namespace cg_detail {
struct InitializeResidualMagnitude;
struct UpdateFieldValues;
struct UpdateOperand;
}  // namespace cg_detail
}  // namespace LinearSolver

// NOLINTNEXTLINE(google-build-using-namespace)
using namespace ResidualMonitorActionsTestHelpers;

namespace {

struct VectorTag : db::SimpleTag {
  using type = DenseVector<double>;
  static std::string name() noexcept { return "VectorTag"; }
};

struct CheckValueTag : db::SimpleTag {
  using type = double;
  static std::string name() noexcept { return "CheckValueTag"; }
};

struct CheckConvergedTag : db::SimpleTag {
  using type = bool;
  static std::string name() noexcept { return "CheckConvergedTag"; }
};

template <typename Metavariables>
using residual_monitor_tags =
    tmpl::append<typename LinearSolver::cg_detail::InitializeResidualMonitor<
                     Metavariables>::simple_tags,
                 typename LinearSolver::cg_detail::InitializeResidualMonitor<
                     Metavariables>::compute_tags>;

template <typename Metavariables>
struct MockResidualMonitor {
  using component_being_mocked =
      LinearSolver::cg_detail::ResidualMonitor<Metavariables>;
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

struct MockInitializeResidualMagnitude {
  template <typename... InboxTags, typename Metavariables, typename ActionList,
            typename ParallelComponent, typename ArrayIndex>
  static void apply(
      db::DataBox<tmpl::list<CheckValueTag, CheckConvergedTag>>& box,  // NOLINT
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/,
      const double residual_magnitude) noexcept {
    db::mutate<CheckValueTag>(
        make_not_null(&box), [residual_magnitude](const gsl::not_null<double*>
                                                      value_box) noexcept {
          *value_box = residual_magnitude;
        });
  }
};

struct MockUpdateFieldValues {
  template <typename... InboxTags, typename Metavariables, typename ActionList,
            typename ParallelComponent, typename ArrayIndex>
  static void apply(
      db::DataBox<tmpl::list<CheckValueTag, CheckConvergedTag>>& box,  // NOLINT
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
      db::DataBox<tmpl::list<CheckValueTag, CheckConvergedTag>>& box,  // NOLINT
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/, const double res_ratio,
      const double residual_magnitude, const bool has_converged) noexcept {
    db::mutate<CheckValueTag, CheckConvergedTag>(
        make_not_null(&box),
        [ res_ratio, residual_magnitude, has_converged ](
            const gsl::not_null<double*> value_box,
            const gsl::not_null<bool*> converged_box) noexcept {
          *value_box = res_ratio + residual_magnitude;
          *converged_box = has_converged;
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
  using initial_databox =
      db::compute_databox_type<tmpl::list<CheckValueTag, CheckConvergedTag>>;

  using replace_these_simple_actions =
      tmpl::list<LinearSolver::cg_detail::InitializeResidualMagnitude,
                 LinearSolver::cg_detail::UpdateFieldValues,
                 LinearSolver::cg_detail::UpdateOperand>;
  using with_these_simple_actions =
      tmpl::list<MockInitializeResidualMagnitude, MockUpdateFieldValues,
                 MockUpdateOperand>;
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
               db::create<db::AddSimpleTags<CheckValueTag, CheckConvergedTag>>(
                   std::numeric_limits<double>::signaling_NaN(), false));

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
            db::AddSimpleTags<CheckValueTag, CheckConvergedTag>>>();
  };
  const auto get_mock_observer_writer_box =
      [&runner, &singleton_id]() -> decltype(auto) {
    return runner.algorithms<MockObserverWriter<Metavariables>>()
        .at(singleton_id)
        .get_databox<db::compute_databox_type<observer_writer_tags>>();
  };

  using residual_square_tag = db::add_tag_prefix<
      LinearSolver::Tags::MagnitudeSquare,
      db::add_tag_prefix<LinearSolver::Tags::Residual, VectorTag>>;

  SECTION("InitializeResidual") {
    runner.simple_action<MockResidualMonitor<Metavariables>,
                         LinearSolver::cg_detail::InitializeResidual<
                             MockElementArray<Metavariables>>>(singleton_id,
                                                               4.);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    const auto& box = get_box();
    CHECK(db::get<residual_square_tag>(box) == 4.);
    CHECK(db::get<LinearSolver::Tags::IterationId>(box).step_number == 0);
    const auto& mock_element_box = get_mock_element_box();
    CHECK(db::get<CheckValueTag>(mock_element_box) == 2.);
  }

  SECTION("ComputeAlpha") {
    runner.simple_action<MockResidualMonitor<Metavariables>,
                         LinearSolver::cg_detail::InitializeResidual<
                             MockElementArray<Metavariables>>>(singleton_id,
                                                               1.);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
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
                         LinearSolver::cg_detail::InitializeResidual<
                             MockElementArray<Metavariables>>>(singleton_id,
                                                               1.);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    runner.simple_action<MockResidualMonitor<Metavariables>,
                         LinearSolver::cg_detail::UpdateResidual<
                             MockElementArray<Metavariables>>>(singleton_id,
                                                               10.);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    runner.invoke_queued_threaded_action<MockObserverWriter<Metavariables>>(
        singleton_id);
    const auto& box = get_box();
    CHECK(db::get<residual_square_tag>(box) == 10.);
    CHECK(db::get<LinearSolver::Tags::IterationId>(box).step_number == 1);
    const auto& mock_element_box = get_mock_element_box();
    CHECK(db::get<CheckValueTag>(mock_element_box) == approx(10. + sqrt(10.)));
    CHECK(db::get<CheckConvergedTag>(mock_element_box) == false);
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
          approx(sqrt(10.)));
  }

  SECTION("Converge") {
    runner.simple_action<MockResidualMonitor<Metavariables>,
                         LinearSolver::cg_detail::InitializeResidual<
                             MockElementArray<Metavariables>>>(singleton_id,
                                                               1.);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    runner.simple_action<MockResidualMonitor<Metavariables>,
                         LinearSolver::cg_detail::UpdateResidual<
                             MockElementArray<Metavariables>>>(singleton_id,
                                                               0.);
    runner.invoke_queued_simple_action<MockElementArray<Metavariables>>(
        element_id);
    runner.invoke_queued_threaded_action<MockObserverWriter<Metavariables>>(
        singleton_id);
    const auto& box = get_box();
    CHECK(db::get<residual_square_tag>(box) == 0.);
    CHECK(db::get<LinearSolver::Tags::IterationId>(box).step_number == 1);
    const auto& mock_element_box = get_mock_element_box();
    CHECK(db::get<CheckValueTag>(mock_element_box) == 0.);
    CHECK(db::get<CheckConvergedTag>(mock_element_box) == true);
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
          approx(0.));
  }
}
