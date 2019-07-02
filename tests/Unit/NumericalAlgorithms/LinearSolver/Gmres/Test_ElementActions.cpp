// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DenseVector.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/LinearSolver/Gmres/ElementActions.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearSolver/Gmres/InitializeElement.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"  // IWYU pragma: keep
#include "Parallel/AddOptionsToDataBox.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_include <boost/variant/get.hpp>
// IWYU pragma: no_forward_declare db::DataBox

namespace {

struct VectorTag : db::SimpleTag {
  using type = DenseVector<double>;
  static std::string name() noexcept { return "VectorTag"; }
};

using initial_fields_tag =
    db::add_tag_prefix<LinearSolver::Tags::Initial, VectorTag>;
using operand_tag = db::add_tag_prefix<LinearSolver::Tags::Operand, VectorTag>;
using orthogonalization_iteration_id_tag =
    db::add_tag_prefix<LinearSolver::Tags::Orthogonalization,
                       LinearSolver::Tags::IterationId>;
using basis_history_tag = LinearSolver::Tags::KrylovSubspaceBasis<VectorTag>;

template <typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tag_list = tmpl::list<>;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<
          tmpl::list<VectorTag, operand_tag, LinearSolver::Tags::IterationId,
                     initial_fields_tag, orthogonalization_iteration_id_tag,
                     basis_history_tag, LinearSolver::Tags::HasConverged>>>>>;
};

struct System {
  using fields_tag = VectorTag;
};

struct Metavariables {
  using component_list = tmpl::list<ElementArray<Metavariables>>;
  using system = System;
  using const_global_cache_tag_list = tmpl::list<>;
  enum class Phase { Initialization, Testing, Exit };
};

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearSolver.Gmres.ElementActions",
                  "[Unit][NumericalAlgorithms][LinearSolver][Actions]") {
  using element_array = ElementArray<Metavariables>;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};

  // Setup mock element array
  ActionTesting::emplace_component_and_initialize<element_array>(
      make_not_null(&runner), 0,
      {DenseVector<double>(3, 0.), DenseVector<double>(3, 2.), 0_st,
       DenseVector<double>(3, -1.), 0_st,
       std::vector<DenseVector<double>>{DenseVector<double>(3, 0.5),
                                        DenseVector<double>(3, 1.5)},
       db::item_type<LinearSolver::Tags::HasConverged>{}});

  // DataBox shortcuts
  const auto get_tag = [&runner](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner, 0);
  };

  runner.set_phase(Metavariables::Phase::Testing);

  {
    CHECK(get_tag(LinearSolver::Tags::IterationId{}) == 0);
    CHECK(get_tag(initial_fields_tag{}) == DenseVector<double>(3, -1.));
    CHECK(get_tag(operand_tag{}) == DenseVector<double>(3, 2.));
    CHECK(get_tag(basis_history_tag{}).size() == 2);
  }

  // Can't test the other element actions because reductions are not yet
  // supported. The full algorithm is tested in
  // `Test_GmresAlgorithm.cpp` and
  // `Test_DistributedGmresAlgorithm.cpp`.

  SECTION("NormalizeInitialOperand") {
    ActionTesting::simple_action<
        element_array, LinearSolver::gmres_detail::NormalizeInitialOperand>(
        make_not_null(&runner), 0, 4.,
        db::item_type<LinearSolver::Tags::HasConverged>{
            {1, 0., 0.}, 1, 0., 0.});
    CHECK_ITERABLE_APPROX(get_tag(operand_tag{}), DenseVector<double>(3, 0.5));
    CHECK(get_tag(basis_history_tag{}).size() == 3);
    CHECK(get_tag(basis_history_tag{})[2] == get_tag(operand_tag{}));
    CHECK(get_tag(LinearSolver::Tags::HasConverged{}));
  }
  SECTION("NormalizeOperandAndUpdateField") {
    ActionTesting::simple_action<
        element_array,
        LinearSolver::gmres_detail::NormalizeOperandAndUpdateField>(
        make_not_null(&runner), 0, 4., DenseVector<double>{2., 4.},
        db::item_type<LinearSolver::Tags::HasConverged>{
            {1, 0., 0.}, 1, 0., 0.});
    CHECK(get_tag(LinearSolver::Tags::IterationId{}) == 1);
    CHECK(get_tag(orthogonalization_iteration_id_tag{}) == 0);
    CHECK_ITERABLE_APPROX(get_tag(operand_tag{}), DenseVector<double>(3, 0.5));
    CHECK(get_tag(basis_history_tag{}).size() == 3);
    CHECK(get_tag(basis_history_tag{})[2] == get_tag(operand_tag{}));
    // minres * basis_history - initial = 2 * 0.5 + 4 * 1.5 - 1 = 6
    CHECK_ITERABLE_APPROX(get_tag(VectorTag{}), DenseVector<double>(3, 6.));
    CHECK(get_tag(LinearSolver::Tags::HasConverged{}));
  }
}
