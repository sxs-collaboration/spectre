// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Helpers/ParallelAlgorithms/LinearSolver/DistributedLinearSolverAlgorithmTestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Options.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/Tags/Section.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Tags.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace helpers_distributed = DistributedLinearSolverAlgorithmTestHelpers;

namespace TestHelpers::LinearSolver::multigrid {

namespace OptionTags {
struct LinearOperator {
  static constexpr Options::String help =
      "The linear operator A to invert. The outer list corresponds to "
      "multigrid levels, ordered from finest to coarsest grid. The inner list "
      "corresponds to the elements in the domain. The number of columns in "
      "each matrix corresponds to the number of grid points in the element, "
      "and the number of rows corresponds to the total number of grid points "
      "in the domain.";
  using type =
      std::vector<std::vector<DenseMatrix<double, blaze::columnMajor>>>;
};
struct OperatorIsMassive {
  static constexpr Options::String help =
      "Whether or not the linear operator includes a mass matrix. This is "
      "relevant for projection operations between grids.";
  using type = bool;
};
}  // namespace OptionTags

struct LinearOperator : db::SimpleTag {
  using type =
      std::vector<std::vector<DenseMatrix<double, blaze::columnMajor>>>;
  using option_tags = tmpl::list<OptionTags::LinearOperator>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& linear_operator) noexcept {
    return linear_operator;
  }
};

struct OperatorIsMassive : db::SimpleTag {
  using type = bool;
  using option_tags = tmpl::list<OptionTags::OperatorIsMassive>;
  static constexpr bool pass_metavariables = false;
  static bool create_from_options(const bool value) noexcept { return value; }
};

using fields_tag = helpers_distributed::fields_tag;
using sources_tag = helpers_distributed::sources_tag;

struct InitializeElement {
  using initialization_tags =
      tmpl::list<::domain::Tags::InitialExtents<1>,
                 ::domain::Tags::InitialRefinementLevels<1>>;
  using const_global_cache_tags =
      tmpl::list<helpers_distributed::Source, OperatorIsMassive>;
  using simple_tags =
      tmpl::list<::domain::Tags::Mesh<1>,
                 ::domain::Tags::Coordinates<1, Frame::Inertial>, fields_tag,
                 sources_tag>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<1>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    // Initialize geometry
    const auto& initial_extents =
        db::get<::domain::Tags::InitialExtents<1>>(box);
    const auto& domain = db::get<::domain::Tags::Domain<1>>(box);
    auto mesh = ::domain::Initialization::create_initial_mesh(
        initial_extents, element_id, Spectral::Quadrature::GaussLobatto);
    const auto logical_coords = logical_coordinates(mesh);
    const auto& block = domain.blocks()[element_id.block_id()];
    const ElementMap<1, Frame::Inertial> element_map{
        element_id, block.stationary_map().get_clone()};
    auto inertial_coords = element_map(logical_coords);
    // Initialize data
    const size_t element_index = helpers_distributed::get_index(element_id);
    const size_t multigrid_level = element_id.grid_index();
    const auto& source =
        multigrid_level == 0
            ? typename sources_tag::type(
                  gsl::at(get<helpers_distributed::Source>(box), element_index))
            : typename sources_tag::type{};
    const size_t num_points = mesh.number_of_grid_points();
    auto initial_fields = typename fields_tag::type{num_points, 0.};
    Initialization::mutate_assign<simple_tags>(
        make_not_null(&box), std::move(mesh), std::move(inertial_coords),
        std::move(initial_fields), std::move(source));
    return {std::move(box)};
  }
};

// The next two actions are responsible for applying the linear operator to the
// operand data. They apply a chunk of the full matrix on each element and
// perform a reduction to obtain the full matrix-vector product.

template <typename OperandTag, typename OperatorAppliedToOperandTag>
struct CollectOperatorAction;

template <typename OperandTag, typename OperatorAppliedToOperandTag>
struct ComputeOperatorAction {
  using const_global_cache_tags = tmpl::list<LinearOperator>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, Parallel::AlgorithmExecution>
  apply(db::DataBox<DbTagsList>& box,
        const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
        const Parallel::GlobalCache<Metavariables>& /*cache*/,
        const ElementId<1>& element_id, const ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) noexcept {
    const size_t multigrid_level = element_id.grid_index();
    const size_t element_index = helpers_distributed::get_index(element_id);
    const auto& operator_matrices = get<LinearOperator>(box)[multigrid_level];
    const size_t number_of_elements = operator_matrices.size();
    const auto& linear_operator = gsl::at(operator_matrices, element_index);
    const size_t number_of_grid_points = linear_operator.columns();
    const auto& operand = get<OperandTag>(box);

    typename OperandTag::type operator_applied_to_operand{
        number_of_grid_points * number_of_elements};
    // Could use apply_matrices here once it works with `DenseMatrix`, or
    // `Matrix` is option-creatable
    dgemv_('N', linear_operator.rows(), linear_operator.columns(), 1,
           linear_operator.data(), linear_operator.spacing(), operand.data(), 1,
           0, operator_applied_to_operand.data(), 1);

    auto& section = *db::get_mutable_reference<Parallel::Tags::Section<
        ParallelComponent, ::LinearSolver::multigrid::Tags::MultigridLevel>>(
        make_not_null(&box));
    Parallel::contribute_to_reduction<
        CollectOperatorAction<OperandTag, OperatorAppliedToOperandTag>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<typename OperandTag::type, funcl::Plus<>>,
            Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>>{
            operator_applied_to_operand, multigrid_level},
        section.proxy()[element_id], section.proxy(), make_not_null(&section));

    // Pause algorithm for now. The reduction will be broadcast to the next
    // action which is responsible for restarting the algorithm.
    return {std::move(box), Parallel::AlgorithmExecution::Pause};
  }
};

template <typename OperandTag, typename OperatorAppliedToOperandTag>
struct CollectOperatorAction {
  template <
      typename ParallelComponent, typename DbTagsList, typename Metavariables,
      Requires<tmpl::list_contains_v<DbTagsList, OperatorAppliedToOperandTag>> =
          nullptr>
  static void apply(
      db::DataBox<DbTagsList>& box, Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<1>& element_id,
      const typename OperandTag::type& operator_applied_to_operand_global_data,
      const size_t broadcasting_multigrid_level) noexcept {
    // We're receiving broadcasts also from reductions over other sections. See
    // issue: https://github.com/sxs-collaboration/spectre/issues/3220
    const size_t multigrid_level = element_id.grid_index();
    if (multigrid_level != broadcasting_multigrid_level) {
      return;
    }
    // Copy the slice of the global result corresponding to this element into
    // the DataBox
    const size_t element_index = helpers_distributed::get_index(element_id);
    const size_t number_of_grid_points =
        get<LinearOperator>(box)[multigrid_level][0].columns();
    db::mutate<OperatorAppliedToOperandTag>(
        make_not_null(&box),
        [&operator_applied_to_operand_global_data, &number_of_grid_points,
         &element_index](auto operator_applied_to_operand) noexcept {
          operator_applied_to_operand->initialize(number_of_grid_points);
          for (size_t i = 0; i < number_of_grid_points; ++i) {
            operator_applied_to_operand->data()[i] =
                operator_applied_to_operand_global_data
                    .data()[i + element_index * number_of_grid_points];
          }
        });
    // Proceed with algorithm
    Parallel::get_parallel_component<ParallelComponent>(cache)[element_id]
        .perform_algorithm(true);
  }
};

template <typename OptionsGroup>
struct TestResult {
  using const_global_cache_tags =
      tmpl::list<helpers_distributed::ExpectedResult>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<1>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    if (element_id.grid_index() > 0) {
      return {std::move(box), true};
    }
    const size_t element_index = helpers_distributed::get_index(element_id);
    const auto& expected_result =
        gsl::at(get<helpers_distributed::ExpectedResult>(box), element_index);
    const auto& result = get<helpers_distributed::ScalarFieldTag>(box).get();
    for (size_t i = 0; i < expected_result.size(); i++) {
      SPECTRE_PARALLEL_REQUIRE(result[i] == approx(expected_result[i]));
    }
    return {std::move(box), true};
  }
};

}  // namespace TestHelpers::LinearSolver::multigrid
