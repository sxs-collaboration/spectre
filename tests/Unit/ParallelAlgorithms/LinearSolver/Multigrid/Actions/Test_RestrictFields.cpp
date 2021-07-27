// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "IO/Logging/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Actions/RestrictFields.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Tags.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

struct DummyOptionsGroup {};

template <size_t N>
struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct MassiveTag : db::SimpleTag {
  using type = bool;
};

using fields_tag =
    ::Tags::Variables<tmpl::list<ScalarFieldTag<0>, ScalarFieldTag<1>>>;

template <size_t Dim, typename FieldsAreMassiveTag, typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Dim>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<tmpl::list<
              LinearSolver::multigrid::Tags::ParentId<Dim>,
              LinearSolver::multigrid::Tags::ChildIds<Dim>,
              domain::Tags::Mesh<Dim>,
              LinearSolver::multigrid::Tags::ParentMesh<Dim>,
              Convergence::Tags::IterationId<DummyOptionsGroup>, fields_tag>>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<
              LinearSolver::multigrid::Actions::SendFieldsToCoarserGrid<
                  tmpl::list<fields_tag>, DummyOptionsGroup,
                  FieldsAreMassiveTag>,
              LinearSolver::multigrid::Actions::ReceiveFieldsFromFinerGrid<
                  Dim, tmpl::list<fields_tag>, DummyOptionsGroup>,
              Parallel::Actions::TerminatePhase>>>;
};

template <size_t Dim, typename FieldsAreMassiveTag>
struct Metavariables {
  using element_array = ElementArray<Dim, FieldsAreMassiveTag, Metavariables>;
  using component_list = tmpl::list<element_array>;
  using const_global_cache_tags =
      tmpl::conditional_t<std::is_same_v<FieldsAreMassiveTag, void>,
                          tmpl::list<>, tmpl::list<FieldsAreMassiveTag>>;
  enum class Phase { Initialization, Testing, Exit };
};

template <typename FieldsAreMassiveTag>
void test_restrict_fields(const Mesh<1>& fine_mesh, const Mesh<1>& coarse_mesh,
                          const DataVector& fine_data_left,
                          const DataVector& fine_data_right,
                          const DataVector& expected_coarse_data,
                          const bool fields_are_massive = false) {
  constexpr size_t Dim = 1;
  using metavariables = Metavariables<Dim, FieldsAreMassiveTag>;
  using element_array = typename metavariables::element_array;

  const auto global_cache = [&fields_are_massive]() {
    if constexpr (std::is_same_v<FieldsAreMassiveTag, void>) {
      (void)fields_are_massive;
      return tuples::TaggedTuple<logging::Tags::Verbosity<DummyOptionsGroup>>{
          Verbosity::Verbose};
    } else {
      return tuples::TaggedTuple<logging::Tags::Verbosity<DummyOptionsGroup>,
                                 FieldsAreMassiveTag>{Verbosity::Verbose,
                                                      fields_are_massive};
    }
  }();

  ActionTesting::MockRuntimeSystem<metavariables> runner{
      std::move(global_cache)};

  // Setup element array
  const auto add_element =
      [&runner](const ElementId<Dim>& element_id,
                const std::optional<ElementId<Dim>>& parent_id,
                const std::unordered_set<ElementId<Dim>>& child_ids,
                const Mesh<Dim>& mesh,
                const std::optional<Mesh<Dim>>& parent_mesh,
                const std::optional<DataVector>& data) {
        typename fields_tag::type fields{};
        if (data.has_value()) {
          fields.initialize(mesh.number_of_grid_points());
          get(get<ScalarFieldTag<0>>(fields)) = *data;
          get(get<ScalarFieldTag<1>>(fields)) = *data;
        }
        ActionTesting::emplace_component_and_initialize<element_array>(
            make_not_null(&runner), element_id,
            {parent_id, child_ids, mesh, parent_mesh, size_t{0},
             std::move(fields)});
      };
  const ElementId<Dim> left_element_id{0, {{{1, 0}}}, 0};
  const ElementId<Dim> right_element_id{0, {{{1, 1}}}, 0};
  const ElementId<Dim> coarse_element_id{0, {{{0, 0}}}, 1};
  add_element(left_element_id, coarse_element_id, {}, fine_mesh, coarse_mesh,
              fine_data_left);
  add_element(right_element_id, coarse_element_id, {}, fine_mesh, coarse_mesh,
              fine_data_right);
  add_element(coarse_element_id, std::nullopt,
              {left_element_id, right_element_id}, coarse_mesh, std::nullopt,
              std::nullopt);

  ActionTesting::set_phase(make_not_null(&runner),
                           metavariables::Phase::Testing);

  // Skip over sending on coarse element
  ActionTesting::next_action<element_array>(make_not_null(&runner),
                                            coarse_element_id);
  REQUIRE_FALSE(ActionTesting::next_action_if_ready<element_array>(
      make_not_null(&runner), coarse_element_id));
  // Send from left element
  ActionTesting::next_action<element_array>(make_not_null(&runner),
                                            left_element_id);
  REQUIRE(ActionTesting::next_action_if_ready<element_array>(
      make_not_null(&runner), left_element_id));
  REQUIRE_FALSE(ActionTesting::next_action_if_ready<element_array>(
      make_not_null(&runner), coarse_element_id));
  // Send from right element
  ActionTesting::next_action<element_array>(make_not_null(&runner),
                                            right_element_id);
  REQUIRE(ActionTesting::next_action_if_ready<element_array>(
      make_not_null(&runner), right_element_id));
  // Receive on coarse element
  REQUIRE(ActionTesting::next_action_if_ready<element_array>(
      make_not_null(&runner), coarse_element_id));
  const auto& coarse_data =
      ActionTesting::get_databox_tag<element_array, fields_tag>(
          runner, coarse_element_id);
  CHECK_ITERABLE_APPROX(get(get<ScalarFieldTag<0>>(coarse_data)),
                        expected_coarse_data);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelMultigrid.Action.RestrictFields",
                  "[Unit][ParallelAlgorithms][LinearSolver][Actions]") {
  const Mesh<1> fine_mesh{4, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<1> coarse_mesh{3, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto};
  const DataVector fine_data_left{0., 1., 2., 3.};
  const DataVector fine_data_right{4., 5., 6., 7.};
  {
    const DataVector coarse_data =
        apply_matrices(
            make_array<1>(Spectral::projection_matrix_child_to_parent(
                fine_mesh, coarse_mesh, Spectral::ChildSize::LowerHalf)),
            fine_data_left, Index<1>{4}) +
        apply_matrices(
            make_array<1>(Spectral::projection_matrix_child_to_parent(
                fine_mesh, coarse_mesh, Spectral::ChildSize::UpperHalf)),
            fine_data_right, Index<1>{4});
    test_restrict_fields<void>(fine_mesh, coarse_mesh, fine_data_left,
                               fine_data_right, coarse_data);
    test_restrict_fields<MassiveTag>(fine_mesh, coarse_mesh, fine_data_left,
                                     fine_data_right, coarse_data, false);
  }
  {
    const DataVector coarse_data =
        apply_matrices(
            make_array<1>(Spectral::projection_matrix_child_to_parent(
                fine_mesh, coarse_mesh, Spectral::ChildSize::LowerHalf, true)),
            fine_data_left, Index<1>{4}) +
        apply_matrices(
            make_array<1>(Spectral::projection_matrix_child_to_parent(
                fine_mesh, coarse_mesh, Spectral::ChildSize::UpperHalf, true)),
            fine_data_right, Index<1>{4});
    test_restrict_fields<MassiveTag>(fine_mesh, coarse_mesh, fine_data_left,
                                     fine_data_right, coarse_data, true);
  }
}
