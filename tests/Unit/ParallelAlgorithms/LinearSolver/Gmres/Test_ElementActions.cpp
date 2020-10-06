// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DenseVector.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Actions/SetData.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/ElementActions.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/InitializeElement.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/Tags/InboxTags.hpp"
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
using initial_fields_tag = LinearSolver::Tags::Initial<fields_tag>;
using operand_tag = LinearSolver::Tags::Operand<fields_tag>;
using preconditioned_operand_tag =
    LinearSolver::Tags::Preconditioned<operand_tag>;
using operator_applied_to_operand_tag =
    LinearSolver::Tags::OperatorAppliedTo<operand_tag>;
using operator_applied_to_preconditioned_operand_tag =
    LinearSolver::Tags::OperatorAppliedTo<preconditioned_operand_tag>;
using orthogonalization_iteration_id_tag =
    LinearSolver::Tags::Orthogonalization<
        Convergence::Tags::IterationId<DummyOptionsGroup>>;
using basis_history_tag = LinearSolver::Tags::KrylovSubspaceBasis<operand_tag>;
using preconditioned_basis_history_tag =
    LinearSolver::Tags::KrylovSubspaceBasis<preconditioned_operand_tag>;

template <typename Metavariables, bool Preconditioned>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<tmpl::list<VectorTag>>,
                     LinearSolver::gmres::detail::InitializeElement<
                         fields_tag, DummyOptionsGroup, Preconditioned>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<
              LinearSolver::gmres::detail::NormalizeInitialOperand<
                  fields_tag, DummyOptionsGroup, Preconditioned,
                  DummyOptionsGroup>,
              LinearSolver::gmres::detail::PrepareStep<
                  fields_tag, DummyOptionsGroup, Preconditioned,
                  DummyOptionsGroup>,
              LinearSolver::gmres::detail::NormalizeOperandAndUpdateField<
                  fields_tag, DummyOptionsGroup, Preconditioned,
                  DummyOptionsGroup>,
              Parallel::Actions::TerminatePhase>>>;
};

template <bool Preconditioned>
struct Metavariables {
  using element_array = ElementArray<Metavariables, Preconditioned>;
  using component_list = tmpl::list<element_array>;
  enum class Phase { Initialization, Testing, Exit };
};

template <bool Preconditioned>
void test_element_actions() {
  CAPTURE(Preconditioned);
  using metavariables = Metavariables<Preconditioned>;
  using element_array = typename metavariables::element_array;

  ActionTesting::MockRuntimeSystem<metavariables> runner{{}};

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
                           metavariables::Phase::Testing);

  // Can't test the other element actions because reductions are not yet
  // supported. The full algorithm is tested in
  // `Test_GmresAlgorithm.cpp` and
  // `Test_DistributedGmresAlgorithm.cpp`.

  {
    INFO("InitializeElement");
    CHECK(get_tag(Convergence::Tags::IterationId<DummyOptionsGroup>{}) ==
          std::numeric_limits<size_t>::max());
    tmpl::for_each<tmpl::list<
        initial_fields_tag, operator_applied_to_fields_tag, operand_tag,
        std::conditional_t<Preconditioned,
                           operator_applied_to_preconditioned_operand_tag,
                           operator_applied_to_operand_tag>,
        orthogonalization_iteration_id_tag, basis_history_tag>>(
        [&tag_is_retrievable](auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;
          CAPTURE(db::tag_name<tag>());
          CHECK(tag_is_retrievable(tag{}));
        });
    CHECK(tag_is_retrievable(preconditioned_operand_tag{}) == Preconditioned);
    CHECK(tag_is_retrievable(preconditioned_basis_history_tag{}) ==
          Preconditioned);
    CHECK_FALSE(get_tag(LinearSolver::Tags::HasConverged<DummyOptionsGroup>{}));
  }

  const auto test_normalize_initial_operand =
      [&runner, &get_tag,
       &set_tag](const Convergence::HasConverged& has_converged) {
        const size_t iteration_id = 0;
        set_tag(Convergence::Tags::IterationId<DummyOptionsGroup>{},
                iteration_id);
        set_tag(operand_tag{}, DenseVector<double>(3, 2.));
        set_tag(basis_history_tag{},
                std::vector<DenseVector<double>>{DenseVector<double>(3, 0.5),
                                                 DenseVector<double>(3, 1.5)});
        REQUIRE_FALSE(ActionTesting::is_ready<element_array>(runner, 0));
        auto& inbox = ActionTesting::get_inbox_tag<
            element_array, LinearSolver::gmres::detail::Tags::
                               InitialOrthogonalization<DummyOptionsGroup>>(
            make_not_null(&runner), 0);
        const double residual_magnitude = 4.;
        CAPTURE(has_converged);
        inbox[iteration_id] =
            std::make_tuple(residual_magnitude, has_converged);
        REQUIRE(ActionTesting::is_ready<element_array>(runner, 0));
        ActionTesting::next_action<element_array>(make_not_null(&runner), 0);
        CHECK_ITERABLE_APPROX(get_tag(operand_tag{}),
                              DenseVector<double>(3, 0.5));
        CHECK(get_tag(basis_history_tag{}).size() == 3);
        CHECK(get_tag(basis_history_tag{})[2] == get_tag(operand_tag{}));
        CHECK(get_tag(LinearSolver::Tags::HasConverged<DummyOptionsGroup>{}) ==
              has_converged);
        CHECK(ActionTesting::get_next_action_index<element_array>(runner, 0) ==
              (has_converged ? 3 : 1));
      };
  SECTION("NormalizeInitialOperand (not yet converged: continue loop)") {
    test_normalize_initial_operand(Convergence::HasConverged{1, 0});
  }
  SECTION("NormalizeInitialOperand (has converged: terminate loop)") {
    test_normalize_initial_operand(Convergence::HasConverged{1, 1});
  }

  const auto test_normalize_operand_and_update_field =
      [&runner, &get_tag,
       &set_tag](const Convergence::HasConverged& has_converged) {
        const size_t iteration_id = 2;
        set_tag(Convergence::Tags::IterationId<DummyOptionsGroup>{},
                iteration_id);
        set_tag(initial_fields_tag{}, DenseVector<double>(3, -1.));
        set_tag(operand_tag{}, DenseVector<double>(3, 2.));
        set_tag(basis_history_tag{},
                std::vector<DenseVector<double>>{DenseVector<double>(3, 0.5),
                                                 DenseVector<double>(3, 1.5)});
        if constexpr (Preconditioned) {
          set_tag(preconditioned_basis_history_tag{},
                  get_tag(basis_history_tag{}));
        }
        runner.template force_next_action_to_be<
            element_array,
            LinearSolver::gmres::detail::NormalizeOperandAndUpdateField<
                fields_tag, DummyOptionsGroup, Preconditioned,
                DummyOptionsGroup>>(0);
        REQUIRE_FALSE(ActionTesting::is_ready<element_array>(runner, 0));
        auto& inbox = ActionTesting::get_inbox_tag<
            element_array, LinearSolver::gmres::detail::Tags::
                               FinalOrthogonalization<DummyOptionsGroup>>(
            make_not_null(&runner), 0);
        const double normalization = 4.;
        const DenseVector<double> minres{2., 4.};
        CAPTURE(has_converged);
        inbox[iteration_id] =
            std::make_tuple(normalization, minres, has_converged);
        ActionTesting::next_action<element_array>(make_not_null(&runner), 0);
        REQUIRE(ActionTesting::is_ready<element_array>(runner, 0));
        CHECK_ITERABLE_APPROX(get_tag(operand_tag{}),
                              DenseVector<double>(3, 0.5));
        CHECK(get_tag(basis_history_tag{}).size() == 3);
        CHECK(get_tag(basis_history_tag{})[2] == get_tag(operand_tag{}));
        // minres * basis_history - initial = 2 * 0.5 + 4 * 1.5 - 1 = 6
        CHECK_ITERABLE_APPROX(get_tag(VectorTag{}), DenseVector<double>(3, 6.));
        CHECK(get_tag(Convergence::Tags::IterationId<DummyOptionsGroup>{}) ==
              3);
        CHECK(get_tag(LinearSolver::Tags::HasConverged<DummyOptionsGroup>{}) ==
              has_converged);
        CHECK(ActionTesting::get_next_action_index<element_array>(runner, 0) ==
              (has_converged ? 3 : 1));
      };
  SECTION("NormalizeOperandAndUpdateField (not yet converged: continue loop)") {
    test_normalize_operand_and_update_field(Convergence::HasConverged{1, 0});
  }
  SECTION("NormalizeOperandAndUpdateField (has converged: terminate loop)") {
    test_normalize_operand_and_update_field(Convergence::HasConverged{1, 1});
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.LinearSolver.Gmres.ElementActions",
                  "[Unit][ParallelAlgorithms][LinearSolver][Actions]") {
  test_element_actions<true>();
  test_element_actions<false>();
}
