// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DynamicVector.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/LinearSolver/Actions/MakeIdentityIfSkipped.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct TestOptionsGroup {};

struct FieldsTag : db::SimpleTag {
  using type = blaze::DynamicVector<double>;
};

struct SourceTag : db::SimpleTag {
  using type = blaze::DynamicVector<double>;
};

struct CheckRunIfSkippedTag : db::SimpleTag {
  using type = bool;
};

struct TestLabel {};

struct TestLinearSolver {
  using options_group = TestOptionsGroup;
  using fields_tag = FieldsTag;
  using source_tag = SourceTag;
};

struct RunIfSkipped {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate<CheckRunIfSkippedTag>(
        [](const gsl::not_null<bool*> flag) { *flag = true; },
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

template <typename Metavariables, typename TestActions>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<
              tmpl::list<Convergence::Tags::HasConverged<TestOptionsGroup>,
                         FieldsTag, SourceTag, CheckRunIfSkippedTag>>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<TestActions, Parallel::Actions::TerminatePhase>>>;
};

template <typename TestActions>
struct Metavariables {
  using element_array = ElementArray<Metavariables, TestActions>;
  using component_list = tmpl::list<element_array>;
};

template <typename TestActions>
void test_make_identity_if_skipped(const bool skipped,
                                   const bool check_run_if_skipped = false) {
  CAPTURE(skipped);
  Convergence::HasConverged has_converged{skipped ? 0_st : 1_st, 0};
  using metavariables = Metavariables<TestActions>;
  using element_array = typename Metavariables<TestActions>::element_array;
  ActionTesting::MockRuntimeSystem<metavariables> runner{{}};
  ActionTesting::emplace_component_and_initialize<element_array>(
      make_not_null(&runner), 0,
      {has_converged, blaze::DynamicVector<double>{3, 1.},
       blaze::DynamicVector<double>{3, 2.}, false});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  while (not ActionTesting::get_terminate<element_array>(runner, 0)) {
    ActionTesting::next_action<element_array>(make_not_null(&runner), 0);
  }
  const auto get_tag = [&runner](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner, 0);
  };
  CHECK(get_tag(SourceTag{}) == blaze::DynamicVector<double>{3, 2.});
  CHECK(get_tag(FieldsTag{}) == (skipped
                                     ? blaze::DynamicVector<double>{3, 2.}
                                     : blaze::DynamicVector<double>{3, 1.}));
  if (check_run_if_skipped) {
    CHECK(get_tag(CheckRunIfSkippedTag{}) == skipped);
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelLinearSolver.Actions.MakeIdentityIfSkipped",
                  "[Unit][ParallelAlgorithms][LinearSolver][Actions]") {
  test_make_identity_if_skipped<
      LinearSolver::Actions::MakeIdentityIfSkipped<TestLinearSolver>>(false);
  test_make_identity_if_skipped<
      LinearSolver::Actions::MakeIdentityIfSkipped<TestLinearSolver>>(true);
  static_assert(
      std::is_same_v<
          LinearSolver::Actions::make_identity_if_skipped<TestLinearSolver>,
          LinearSolver::Actions::MakeIdentityIfSkipped<TestLinearSolver>>);
  test_make_identity_if_skipped<LinearSolver::Actions::make_identity_if_skipped<
      TestLinearSolver, RunIfSkipped>>(false, true);
  test_make_identity_if_skipped<LinearSolver::Actions::make_identity_if_skipped<
      TestLinearSolver, RunIfSkipped>>(true, true);
  test_make_identity_if_skipped<LinearSolver::Actions::make_identity_if_skipped<
      TestLinearSolver, RunIfSkipped, TestLabel>>(false, true);
  test_make_identity_if_skipped<LinearSolver::Actions::make_identity_if_skipped<
      TestLinearSolver, RunIfSkipped, TestLabel>>(true, true);
}
