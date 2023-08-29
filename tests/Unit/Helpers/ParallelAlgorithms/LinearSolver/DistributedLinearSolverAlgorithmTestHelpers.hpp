// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DynamicMatrix.hpp"
#include "DataStructures/DynamicVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/InitialExtents.hpp"
#include "Domain/Creators/Tags/InitialRefinementLevels.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Helpers/ParallelAlgorithms/LinearSolver/LinearSolverAlgorithmTestHelpers.hpp"
#include "IO/Observer/Actions/RegisterWithObservers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "ParallelAlgorithms/LinearSolver/Actions/MakeIdentityIfSkipped.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

namespace helpers = LinearSolverAlgorithmTestHelpers;

/// Functionality to test parallel linear solvers on multiple elements
namespace DistributedLinearSolverAlgorithmTestHelpers {

namespace OptionTags {
// This option expects a list of N matrices that each have N*M rows and M
// columns, where N is the number of elements and M is a nonzero integer.
// Therefore, this option specifies a (N*M,N*M) matrix that has its columns
// split over all elements. In a context where the linear operator represents a
// DG discretization, M is the number of collocation points per element.
struct LinearOperator {
  static constexpr Options::String help = "The linear operator A to invert.";
  using type = std::vector<blaze::DynamicMatrix<double>>;
};
// Both of the following options expect a list of N vectors that have a size of
// M each, so that they constitute a vector of total size N*M (see above).
struct Source {
  static constexpr Options::String help = "The source b in the equation Ax=b.";
  using type = std::vector<blaze::DynamicVector<double>>;
};
struct ExpectedResult {
  static constexpr Options::String help = "The solution x in the equation Ax=b";
  using type = std::vector<blaze::DynamicVector<double>>;
};
}  // namespace OptionTags

struct LinearOperator : db::SimpleTag {
  using type = std::vector<blaze::DynamicMatrix<double>>;
  using option_tags = tmpl::list<OptionTags::LinearOperator>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& linear_operator) {
    return linear_operator;
  }
};

struct Source : db::SimpleTag {
  using type = std::vector<blaze::DynamicVector<double>>;
  using option_tags = tmpl::list<OptionTags::Source>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& source) { return source; }
};

struct ExpectedResult : db::SimpleTag {
  using type = std::vector<blaze::DynamicVector<double>>;
  using option_tags = tmpl::list<OptionTags::ExpectedResult>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& expected_result) {
    return expected_result;
  }
};

// The vector `x` we want to solve for
struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() { return "ScalarField"; }
};

using fields_tag = Tags::Variables<tmpl::list<ScalarFieldTag>>;
using sources_tag = db::add_tag_prefix<::Tags::FixedSource, fields_tag>;
using operator_applied_to_fields_tag =
    db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, fields_tag>;

// We are working on a one-dimension domain where each element corresponds to an
// entry of the vectors provided in the input file. This function translates the
// element to an index into these vectors. We assume the domain is composed of a
// single block.
inline size_t get_index(const ElementId<1>& element_id) {
  return element_id.segment_id(0).index();
}

// In the following `ComputeOperatorAction` and `CollectOperatorAction` actions
// we compute A(p)=sum_elements(A_element(p_element)) in a global reduction and
// then broadcast the global A(p) back to the elements so that they can extract
// their A_element(p). This is horribly inefficient parallelism but allows us to
// just provide a global matrix A (represented by the `LinearOperator` tag) in
// an input file.

// Forward declare to keep these actions in the order they are used
template <typename OperandTag>
struct CollectOperatorAction;

template <typename OperandTag>
struct ComputeOperatorAction {
  using const_global_cache_tags = tmpl::list<LinearOperator>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& cache,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ElementId<1>& element_id,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ActionList /*meta*/,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ParallelComponent* const /*meta*/) {
    const size_t element_index = get_index(element_id);
    const auto& operator_matrices = get<LinearOperator>(box);
    const auto number_of_elements = operator_matrices.size();
    const auto& linear_operator = gsl::at(operator_matrices, element_index);
    const auto number_of_grid_points = linear_operator.columns();
    const auto& operand = get<OperandTag>(box);

    typename OperandTag::type operator_applied_to_operand{
        number_of_grid_points * number_of_elements};
    dgemv_('N', linear_operator.rows(), linear_operator.columns(), 1,
           linear_operator.data(), linear_operator.spacing(), operand.data(), 1,
           0, operator_applied_to_operand.data(), 1);

    Parallel::contribute_to_reduction<CollectOperatorAction<OperandTag>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<typename OperandTag::type, funcl::Plus<>>>{
            operator_applied_to_operand},
        Parallel::get_parallel_component<ParallelComponent>(cache)[element_id],
        Parallel::get_parallel_component<ParallelComponent>(cache));

    // Terminate algorithm for now. The reduction will be broadcast to the
    // next action which is responsible for restarting the algorithm.
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

template <typename OperandTag>
struct CollectOperatorAction {
  using local_operator_applied_to_operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, OperandTag>;

  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ScalarFieldOperandTag,
            Requires<tmpl::list_contains_v<
                DbTagsList, local_operator_applied_to_operand_tag>> = nullptr>
  static void apply(
      db::DataBox<DbTagsList>& box,
      const Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<1>& element_id,
      const Variables<tmpl::list<ScalarFieldOperandTag>>& Ap_global_data) {
    const size_t element_index = get_index(element_id);
    // This could be generalized to work on the Variables instead of the
    // Scalar, but it's only for the purpose of this test.
    const auto number_of_grid_points = get<LinearOperator>(box)[0].columns();
    const auto& Ap_global = get<ScalarFieldOperandTag>(Ap_global_data).get();
    DataVector Ap_local{number_of_grid_points};
    std::copy(Ap_global.begin() +
                  static_cast<int>(element_index * number_of_grid_points),
              Ap_global.begin() +
                  static_cast<int>((element_index + 1) * number_of_grid_points),
              Ap_local.begin());
    db::mutate<local_operator_applied_to_operand_tag>(
        [&Ap_local, &number_of_grid_points](auto Ap) {
          *Ap = typename local_operator_applied_to_operand_tag::type{
              number_of_grid_points};
          get(get<LinearSolver::Tags::OperatorAppliedTo<ScalarFieldOperandTag>>(
              *Ap)) = Ap_local;
        },
        make_not_null(&box));
    // Proceed with algorithm
    Parallel::get_parallel_component<ParallelComponent>(cache)[element_id]
        .perform_algorithm(true);
  }
};

// Checks for the correct solution after the algorithm has terminated.
template <typename OptionsGroup>
struct TestResult {
  using const_global_cache_tags =
      tmpl::list<ExpectedResult, helpers::ExpectedConvergenceReason>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ElementId<1>& element_id,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ActionList /*meta*/,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ParallelComponent* const /*meta*/) {
    const size_t element_index = get_index(element_id);
    const auto& has_converged =
        get<Convergence::Tags::HasConverged<OptionsGroup>>(box);
    SPECTRE_PARALLEL_REQUIRE(has_converged);
    SPECTRE_PARALLEL_REQUIRE(has_converged.reason() ==
                             get<helpers::ExpectedConvergenceReason>(box));
    const auto& expected_result =
        gsl::at(get<ExpectedResult>(box), element_index);
    const auto& result = get<ScalarFieldTag>(box).get();
    for (size_t i = 0; i < expected_result.size(); i++) {
      SPECTRE_PARALLEL_REQUIRE(result[i] == approx(expected_result[i]));
    }
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

struct InitializeElement {
  using simple_tags_from_options =
      tmpl::list<domain::Tags::InitialExtents<1>,
                 domain::Tags::InitialRefinementLevels<1>>;
  using const_global_cache_tags = tmpl::list<Source>;
  using simple_tags =
      tmpl::list<domain::Tags::Mesh<1>, domain::Tags::Element<1>,
                 domain::Tags::Coordinates<1, Frame::ElementLogical>,
                 fields_tag, sources_tag>;
  using compute_tags = tmpl::list<>;
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<1>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    // Domain geometry
    const auto& domain = db::get<domain::Tags::Domain<1>>(box);
    const auto& initial_extents = db::get<domain::Tags::InitialExtents<1>>(box);
    const auto& initial_refinement =
        db::get<domain::Tags::InitialRefinementLevels<1>>(box);
    auto mesh = domain::Initialization::create_initial_mesh(
        initial_extents, element_id, Spectral::Quadrature::GaussLobatto);
    const auto& block = domain.blocks()[element_id.block_id()];
    auto element = domain::Initialization::create_initial_element(
        element_id, block, initial_refinement);
    auto logical_coords = logical_coordinates(mesh);
    // Element data
    const size_t element_index = get_index(element_id);
    const auto& source = gsl::at(get<Source>(box), element_index);
    const size_t num_points = source.size();
    ::Initialization::mutate_assign<simple_tags>(
        make_not_null(&box), std::move(mesh), std::move(element),
        std::move(logical_coords), typename fields_tag::type{num_points, 0.},
        typename sources_tag::type{source});
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

namespace detail {

template <typename Preconditioner>
struct run_preconditioner_impl {
  using type =
      tmpl::list<ComputeOperatorAction<typename Preconditioner::fields_tag>,
                 typename Preconditioner::template solve<ComputeOperatorAction<
                     typename Preconditioner::operand_tag>>,
                 LinearSolver::Actions::MakeIdentityIfSkipped<Preconditioner>>;
};

template <>
struct run_preconditioner_impl<void> {
  using type = tmpl::list<>;
};

template <typename Preconditioner>
using run_preconditioner =
    typename run_preconditioner_impl<Preconditioner>::type;

}  // namespace detail

template <typename Metavariables,
          typename LinearSolverType = typename Metavariables::linear_solver,
          typename PreconditionerType = typename Metavariables::preconditioner>
using initialization_actions =
    tmpl::list<InitializeElement, typename LinearSolverType::initialize_element,
               ComputeOperatorAction<fields_tag>,
               helpers::detail::init_preconditioner<PreconditionerType>,
               Parallel::Actions::TerminatePhase>;

template <typename Metavariables,
          typename LinearSolverType = typename Metavariables::linear_solver,
          typename PreconditionerType = typename Metavariables::preconditioner>
using register_actions =
    tmpl::list<typename LinearSolverType::register_element,
               helpers::detail::register_preconditioner<PreconditionerType>,
               Parallel::Actions::TerminatePhase>;

template <typename Metavariables,
          typename LinearSolverType = typename Metavariables::linear_solver,
          typename PreconditionerType = typename Metavariables::preconditioner>
using solve_actions = tmpl::list<
    typename LinearSolverType::template solve<tmpl::list<
        detail::run_preconditioner<PreconditionerType>,
        ComputeOperatorAction<typename LinearSolverType::operand_tag>>>,
    Parallel::Actions::TerminatePhase>;

template <typename Metavariables,
          typename LinearSolverType = typename Metavariables::linear_solver,
          typename PreconditionerType = typename Metavariables::preconditioner>
using test_actions =
    tmpl::list<TestResult<typename LinearSolverType::options_group>>;

template <typename Metavariables>
using ElementArray = elliptic::DgElementArray<
    Metavariables,
    tmpl::list<Parallel::PhaseActions<Parallel::Phase::Initialization,
                                      initialization_actions<Metavariables>>,
               Parallel::PhaseActions<Parallel::Phase::Register,
                                      register_actions<Metavariables>>,
               Parallel::PhaseActions<Parallel::Phase::Solve,
                                      solve_actions<Metavariables>>,
               Parallel::PhaseActions<Parallel::Phase::Testing,
                                      test_actions<Metavariables>>>>;

template <typename Metavariables>
using component_list =
    tmpl::push_back<tmpl::append<helpers::detail::get_component_list<
                                     typename Metavariables::linear_solver>,
                                 helpers::detail::get_component_list<
                                     typename Metavariables::preconditioner>>,
                    ElementArray<Metavariables>,
                    observers::Observer<Metavariables>,
                    observers::ObserverWriter<Metavariables>,
                    helpers::OutputCleaner<Metavariables>>;

}  // namespace DistributedLinearSolverAlgorithmTestHelpers
