// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <unordered_set>
#include <utility>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "IO/Logging/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/SegmentSize.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Actions/RestrictFields.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Tags.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

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
          Parallel::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<tmpl::list<
              amr::Tags::ParentId<Dim>, amr::Tags::ChildIds<Dim>,
              domain::Tags::Mesh<Dim>, amr::Tags::ParentMesh<Dim>,
              Convergence::Tags::IterationId<DummyOptionsGroup>, fields_tag>>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
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
};

// Only dimension 0 is h-refined (for spherical shells this is the radial
// dimension; the angular dimensions cannot be h-refined). The two fine children
// are the lower and upper radial halves of the coarse element.
template <size_t Dim>
std::array<SegmentId, Dim> shell_segment_ids(const SegmentId& radial) {
  auto segment_ids = make_array<Dim>(SegmentId{0, 0});
  segment_ids[0] = radial;
  return segment_ids;
}

template <typename FieldsAreMassiveTag, size_t Dim>
void test_restrict_fields(const Mesh<Dim>& fine_mesh,
                          const Mesh<Dim>& coarse_mesh,
                          const DataVector& fine_data_left,
                          const DataVector& fine_data_right,
                          const DataVector& expected_coarse_data,
                          const bool fields_are_massive = false) {
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
  const ElementId<Dim> left_element_id{0, shell_segment_ids<Dim>({1, 0}), 0};
  const ElementId<Dim> right_element_id{0, shell_segment_ids<Dim>({1, 1}), 0};
  const ElementId<Dim> coarse_element_id{0, shell_segment_ids<Dim>({0, 0}), 1};
  add_element(left_element_id, coarse_element_id, {}, fine_mesh, coarse_mesh,
              fine_data_left);
  add_element(right_element_id, coarse_element_id, {}, fine_mesh, coarse_mesh,
              fine_data_right);
  add_element(coarse_element_id, std::nullopt,
              {left_element_id, right_element_id}, coarse_mesh, std::nullopt,
              std::nullopt);

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

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
                fine_mesh, coarse_mesh, Spectral::SegmentSize::LowerHalf)),
            fine_data_left, Index<1>{4}) +
        apply_matrices(
            make_array<1>(Spectral::projection_matrix_child_to_parent(
                fine_mesh, coarse_mesh, Spectral::SegmentSize::UpperHalf)),
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
                fine_mesh, coarse_mesh, Spectral::SegmentSize::LowerHalf,
                true)),
            fine_data_left, Index<1>{4}) +
        apply_matrices(
            make_array<1>(Spectral::projection_matrix_child_to_parent(
                fine_mesh, coarse_mesh, Spectral::SegmentSize::UpperHalf,
                true)),
            fine_data_right, Index<1>{4});
    test_restrict_fields<MassiveTag>(fine_mesh, coarse_mesh, fine_data_left,
                                     fine_data_right, coarse_data, true);
  }
  // Spherical shells: multigrid coarsens the radial dimension via h-refinement
  // (the angular dimensions cannot be h-refined) and may also lower the radial
  // or angular resolution
  {
    const auto shell_mesh = [](const size_t num_radial_points,
                               const size_t l_max) {
      return Mesh<3>{
          {{num_radial_points, l_max + 1, 2 * l_max + 1}},
          {{Spectral::Basis::Legendre, Spectral::Basis::SphericalHarmonic,
            Spectral::Basis::SphericalHarmonic}},
          {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss,
            Spectral::Quadrature::Equiangular}}};
    };
    const Mesh<3> fine_shell = shell_mesh(4, 6);
    const Mesh<3> coarse_shell = shell_mesh(3, 4);
    DataVector fine_data_left_shell(fine_shell.number_of_grid_points());
    DataVector fine_data_right_shell(fine_shell.number_of_grid_points());
    for (size_t i = 0; i < fine_data_left_shell.size(); ++i) {
      fine_data_left_shell[i] = 0.1 * static_cast<double>(i) - 1.0;
      fine_data_right_shell[i] = 1.0 - 0.05 * static_cast<double>(i);
    }
    const auto project_child = [&fine_shell, &coarse_shell](
                                   const DataVector& child_data,
                                   const Spectral::SegmentSize radial_size) {
      DataVector result{};
      Spectral::project(make_not_null(&result), child_data, fine_shell,
                        coarse_shell,
                        std::array{radial_size, Spectral::SegmentSize::Full,
                                   Spectral::SegmentSize::Full},
                        make_array<3>(Spectral::SegmentSize::Full), true);
      return result;
    };
    const DataVector coarse_data =
        project_child(fine_data_left_shell, Spectral::SegmentSize::LowerHalf) +
        project_child(fine_data_right_shell, Spectral::SegmentSize::UpperHalf);
    test_restrict_fields<MassiveTag>(fine_shell, coarse_shell,
                                     fine_data_left_shell,
                                     fine_data_right_shell, coarse_data, true);
  }
}
