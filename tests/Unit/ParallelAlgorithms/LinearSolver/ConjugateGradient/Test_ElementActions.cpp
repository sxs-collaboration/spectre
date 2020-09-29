// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DenseVector.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Actions/SetData.hpp"
#include "ParallelAlgorithms/LinearSolver/ConjugateGradient/ElementActions.hpp"
#include "ParallelAlgorithms/LinearSolver/ConjugateGradient/InitializeElement.hpp"
#include "ParallelAlgorithms/LinearSolver/ConjugateGradient/Tags/InboxTags.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

struct DummyOptionsGroup {};

struct VectorTag : db::SimpleTag {
  using type = DenseVector<double>;
};

using fields_tag = VectorTag;
using operator_applied_to_fields_tag =
    LinearSolver::Tags::OperatorAppliedTo<fields_tag>;
using operand_tag = LinearSolver::Tags::Operand<fields_tag>;
using operator_applied_to_operand_tag =
    LinearSolver::Tags::OperatorAppliedTo<operand_tag>;
using residual_tag = LinearSolver::Tags::Residual<fields_tag>;

template <typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<tmpl::list<VectorTag>>,
                     LinearSolver::cg::detail::InitializeElement<
                         fields_tag, DummyOptionsGroup>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<LinearSolver::cg::detail::InitializeHasConverged<
                         fields_tag, DummyOptionsGroup, DummyOptionsGroup>,
                     LinearSolver::cg::detail::UpdateOperand<
                         fields_tag, DummyOptionsGroup, DummyOptionsGroup>,
                     Parallel::Actions::TerminatePhase>>>;
};

struct Metavariables {
  using component_list = tmpl::list<ElementArray<Metavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.ParallelAlgorithms.LinearSolver.ConjugateGradient.ElementActions",
    "[Unit][ParallelAlgorithms][LinearSolver][Actions]") {
  using element_array = ElementArray<Metavariables>;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};

  // Setup mock element array
  ActionTesting::emplace_component_and_initialize<element_array>(
      make_not_null(&runner), 0, {DenseVector<double>(3, 0.)});
  ActionTesting::next_action<element_array>(make_not_null(&runner), 0);

  // DataBox shortcuts
  const auto get_tag = [&runner](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner, 0);
  };
  const auto tag_is_retrievable = [&runner](auto tag_v) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::tag_is_retrievable<element_array, tag>(runner, 0);
  };
  const auto set_tag = [&runner](auto tag_v, const auto& value) {
    using tag = std::decay_t<decltype(tag_v)>;
    ActionTesting::simple_action<element_array,
                                 ::Actions::SetData<tmpl::list<tag>>>(
        make_not_null(&runner), 0, value);
  };

  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);

  // Can't test the other element actions because reductions are not yet
  // supported. The full algorithm is tested in
  // `Test_ConjugateGradientAlgorithm.cpp` and
  // `Test_DistributedConjugateGradientAlgorithm.cpp`.

  {
    INFO("InitializeElement");
    CHECK(get_tag(Convergence::Tags::IterationId<DummyOptionsGroup>{}) ==
          std::numeric_limits<size_t>::max());
    tmpl::for_each<tmpl::list<operator_applied_to_fields_tag, operand_tag,
                              operator_applied_to_operand_tag, residual_tag>>(
        [&tag_is_retrievable](auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;
          CAPTURE(db::tag_name<tag>());
          CHECK(tag_is_retrievable(tag{}));
        });
    CHECK_FALSE(get_tag(Convergence::Tags::HasConverged<DummyOptionsGroup>{}));
  }

  const auto test_initialize_has_converged =
      [&runner, &get_tag,
       &set_tag](const Convergence::HasConverged& has_converged) {
        const size_t iteration_id = 0;
        set_tag(Convergence::Tags::IterationId<DummyOptionsGroup>{},
                iteration_id);
        REQUIRE_FALSE(ActionTesting::is_ready<element_array>(runner, 0));
        auto& inbox = ActionTesting::get_inbox_tag<
            element_array, LinearSolver::cg::detail::Tags::InitialHasConverged<
                               DummyOptionsGroup>>(make_not_null(&runner), 0);
        CAPTURE(has_converged);
        inbox[iteration_id] = has_converged;
        REQUIRE(ActionTesting::is_ready<element_array>(runner, 0));
        ActionTesting::next_action<element_array>(make_not_null(&runner), 0);
        CHECK(get_tag(Convergence::Tags::HasConverged<DummyOptionsGroup>{}) ==
              has_converged);
        CHECK(ActionTesting::get_next_action_index<element_array>(runner, 0) ==
              (has_converged ? 2 : 1));
      };
  SECTION("InitializeHasConverged (not yet converged: continue loop)") {
    test_initialize_has_converged(Convergence::HasConverged{1, 0});
  }
  SECTION("InitializeHasConverged (has converged: terminate loop)") {
    test_initialize_has_converged(Convergence::HasConverged{1, 1});
  }

  const auto test_update_operand = [&runner, &get_tag,
                                    &set_tag](const Convergence::HasConverged&
                                                  has_converged) {
    const size_t iteration_id = 0;
    set_tag(Convergence::Tags::IterationId<DummyOptionsGroup>{}, iteration_id);
    set_tag(operand_tag{}, DenseVector<double>(3, 2.));
    set_tag(residual_tag{}, DenseVector<double>(3, 1.));
    runner.template force_next_action_to_be<
        element_array, LinearSolver::cg::detail::UpdateOperand<
                           fields_tag, DummyOptionsGroup, DummyOptionsGroup>>(
        0);
    REQUIRE_FALSE(ActionTesting::is_ready<element_array>(runner, 0));
    auto& inbox = ActionTesting::get_inbox_tag<
        element_array, LinearSolver::cg::detail::Tags::
                           ResidualRatioAndHasConverged<DummyOptionsGroup>>(
        make_not_null(&runner), 0);
    const double res_ratio = 2.;
    CAPTURE(has_converged);
    inbox[iteration_id] = std::make_tuple(res_ratio, has_converged);
    REQUIRE(ActionTesting::is_ready<element_array>(runner, 0));
    ActionTesting::next_action<element_array>(make_not_null(&runner), 0);
    CHECK(get_tag(LinearSolver::Tags::Operand<VectorTag>{}) ==
          DenseVector<double>(3, 5.));
    CHECK(get_tag(Convergence::Tags::IterationId<DummyOptionsGroup>{}) == 1);
    CHECK(get_tag(Convergence::Tags::HasConverged<DummyOptionsGroup>{}) ==
          has_converged);
    CHECK(ActionTesting::get_next_action_index<element_array>(runner, 0) ==
          (has_converged ? 2 : 1));
  };
  SECTION("UpdateOperand (not yet converged: continue loop)") {
    test_update_operand(Convergence::HasConverged{1, 0});
  }
  SECTION("UpdateOperand (has converged: terminate loop)") {
    test_update_operand(Convergence::HasConverged{1, 1});
  }
}
