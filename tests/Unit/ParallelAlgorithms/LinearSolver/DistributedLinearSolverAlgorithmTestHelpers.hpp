// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "AlgorithmArray.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/DenseVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "ErrorHandling/Error.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelAlgorithms/LinearSolver/Actions/TerminateIfConverged.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare db::DataBox

namespace DistributedLinearSolverAlgorithmTestHelpers {

namespace OptionTags {
struct NumberOfElements {
  static constexpr OptionString help =
      "The number of elements to distribute work on.";
  using type = size_t;
};
}  // namespace OptionTags

/// [array_allocation_tag]
namespace Initialization {
namespace Tags {
struct NumberOfElements : db::SimpleTag {
  using type = int;
  using option_tags = tmpl::list<OptionTags::NumberOfElements>;

  static constexpr bool pass_metavariables = false;
  static int create_from_options(const size_t number_of_elements) noexcept {
    return number_of_elements;
  }
};
}  // namespace Tags
}  // namespace Initialization
/// [array_allocation_tag]

namespace OptionTags {
// This option expects a list of N matrices that each have N*M rows and M
// columns, where N is the `NumberOfElements` and M is a nonzero integer.
// Therefore, this option specifies a (N*M,N*M) matrix that has its columns
// split over all elements. In a context where the linear operator represents a
// DG discretization, M is the number of collocation points per element.
struct LinearOperator {
  static constexpr OptionString help = "The linear operator A to invert.";
  using type = std::vector<DenseMatrix<double, blaze::columnMajor>>;
};
// Both of the following options expect a list of N vectors that have a size of
// M each, so that they constitute a vector of total size N*M (see above).
struct Source {
  static constexpr OptionString help = "The source b in the equation Ax=b.";
  using type = std::vector<DenseVector<double>>;
};
struct ExpectedResult {
  static constexpr OptionString help = "The solution x in the equation Ax=b";
  using type = std::vector<DenseVector<double>>;
};
}  // namespace OptionTags

struct LinearOperator : db::SimpleTag {
  using type = std::vector<DenseMatrix<double, blaze::columnMajor>>;
  using option_tags = tmpl::list<OptionTags::LinearOperator>;

  static constexpr bool pass_metavariables = false;
  static std::vector<DenseMatrix<double, blaze::columnMajor>>
  create_from_options(
      const std::vector<DenseMatrix<double, blaze::columnMajor>>&
          linear_operator) noexcept {
    return linear_operator;
  }
};

struct Source : db::SimpleTag {
  using type = std::vector<DenseVector<double>>;
  using option_tags = tmpl::list<OptionTags::Source>;

  static constexpr bool pass_metavariables = false;
  static std::vector<DenseVector<double>> create_from_options(
      const std::vector<DenseVector<double>>& source) noexcept {
    return source;
  }
};

struct ExpectedResult : db::SimpleTag {
  using type = std::vector<DenseVector<double>>;
  using option_tags = tmpl::list<OptionTags::ExpectedResult>;

  static constexpr bool pass_metavariables = false;
  static std::vector<DenseVector<double>> create_from_options(
      const std::vector<DenseVector<double>>& expected_result) noexcept {
    return expected_result;
  }
};

// The vector `x` we want to solve for
struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "ScalarField"; }
};

using fields_tag = Tags::Variables<tmpl::list<ScalarFieldTag>>;
using sources_tag = db::add_tag_prefix<::Tags::FixedSource, fields_tag>;
using operator_applied_to_fields_tag =
    db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, fields_tag>;
using operand_tag = db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
using operator_applied_to_operand_tag =
    db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, operand_tag>;

// In the following `ComputeOperatorAction` and `CollectOperatorAction` actions
// we compute A(p)=sum_elements(A_element(p_element)) in a global reduction and
// then broadcast the global A(p) back to the elements so that they can extract
// their A_element(p). This is horribly inefficient parallelism but allows us to
// just provide a global matrix A (represented by the `LinearOperator` tag) in
// an input file.

// Forward declare to keep these actions in the order they are used
struct CollectOperatorAction;

struct ComputeOperatorAction {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const int array_index,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ActionList /*meta*/,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& operator_matrices = get<LinearOperator>(cache);
    const auto number_of_elements = operator_matrices.size();
    const auto& linear_operator = gsl::at(operator_matrices, array_index);
    const auto number_of_grid_points = linear_operator.columns();
    const auto& operand = get<operand_tag>(box);

    db::item_type<fields_tag> operator_applied_to_operand{
        number_of_grid_points * number_of_elements};
    dgemv_('N', linear_operator.rows(), linear_operator.columns(), 1,
           linear_operator.data(), linear_operator.rows(), operand.data(), 1, 0,
           operator_applied_to_operand.data(), 1);

    Parallel::contribute_to_reduction<CollectOperatorAction>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<db::item_type<fields_tag>, funcl::Plus<>>>{
            operator_applied_to_operand},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<ParallelComponent>(cache));

    // Terminate algorithm for now. The reduction will be broadcast to the
    // next action which is responsible for restarting the algorithm.
    return {std::move(box), true};
  }
};

struct CollectOperatorAction {
  template <
      typename ParallelComponent, typename DbTagsList, typename Metavariables,
      Requires<tmpl::list_contains_v<DbTagsList, ScalarFieldTag>> = nullptr>
  static void apply(db::DataBox<DbTagsList>& box,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const int array_index,
                    const db::item_type<fields_tag>& Ap_global_data) noexcept {
    // This could be generalized to work on the Variables instead of the
    // Scalar, but it's only for the purpose of this test.
    const auto number_of_grid_points = get<LinearOperator>(cache)[0].columns();
    const auto& Ap_global = get<ScalarFieldTag>(Ap_global_data).get();
    DataVector Ap_local{number_of_grid_points};
    std::copy(Ap_global.begin() +
                  array_index * static_cast<int>(number_of_grid_points),
              Ap_global.begin() +
                  (array_index + 1) * static_cast<int>(number_of_grid_points),
              Ap_local.begin());
    db::mutate<LinearSolver::Tags::OperatorAppliedTo<
        LinearSolver::Tags::Operand<ScalarFieldTag>>>(
        make_not_null(&box), [&Ap_local](auto Ap) noexcept {
          *Ap = Scalar<DataVector>(Ap_local);
        });
    // Proceed with algorithm
    Parallel::get_parallel_component<ParallelComponent>(cache)[array_index]
        .perform_algorithm(true);
  }
};

// Checks for the correct solution after the algorithm has terminated.
struct TestResult {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const int array_index,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ActionList /*meta*/,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& has_converged = get<LinearSolver::Tags::HasConverged>(box);
    SPECTRE_PARALLEL_REQUIRE(has_converged);
    SPECTRE_PARALLEL_REQUIRE(has_converged.reason() ==
                             Convergence::Reason::AbsoluteResidual);
    const auto& expected_result =
        gsl::at(get<ExpectedResult>(cache), array_index);
    const auto& result = get<ScalarFieldTag>(box).get();
    for (size_t i = 0; i < expected_result.size(); i++) {
      SPECTRE_PARALLEL_REQUIRE(result[i] == approx(expected_result[i]));
    }
    return {std::move(box), true};
  }
};

struct InitializeElement {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const int array_index, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto& source = gsl::at(get<Source>(cache), array_index);
    const size_t num_points = source.size();

    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeElement,
            db::AddSimpleTags<fields_tag, sources_tag,
                              operator_applied_to_fields_tag, operand_tag,
                              operator_applied_to_operand_tag>>(
            std::move(box), db::item_type<fields_tag>{num_points, 0.},
            db::item_type<sources_tag>{source},
            db::item_type<operator_applied_to_fields_tag>{num_points, 0.},
            db::item_type<operand_tag>{
                num_points, std::numeric_limits<double>::signaling_NaN()},
            db::item_type<operator_applied_to_operand_tag>{
                num_points, std::numeric_limits<double>::signaling_NaN()}));
  }
};

template <typename Metavariables>
struct ElementArray {
  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<InitializeElement,
                     typename Metavariables::linear_solver::initialize_element,
                     Parallel::Actions::TerminatePhase>>,

      Parallel::PhaseActions<
          typename Metavariables::Phase,
          Metavariables::Phase::PerformLinearSolve,
          tmpl::list<LinearSolver::Actions::TerminateIfConverged,
                     typename Metavariables::linear_solver::prepare_step,
                     ComputeOperatorAction,
                     typename Metavariables::linear_solver::perform_step>>,

      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::TestResult,
                             tmpl::list<TestResult>>>;
  using array_allocation_tags =
      tmpl::list<Initialization::Tags::NumberOfElements>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>,
      array_allocation_tags>;
  using const_global_cache_tags =
      tmpl::list<LinearOperator, Source, ExpectedResult>;
  using array_index = int;

  static void allocate_array(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache,
      const tuples::tagged_tuple_from_typelist<initialization_tags>&
          initialization_items) noexcept {
    auto& array_proxy = Parallel::get_parallel_component<ElementArray>(
        *(global_cache.ckLocalBranch()));
    for (int i = 0, which_proc = 0,
             number_of_procs = Parallel::number_of_procs();
         i < get<Initialization::Tags::NumberOfElements>(initialization_items);
         i++) {
      array_proxy[i].insert(global_cache, initialization_items, which_proc);
      which_proc = which_proc + 1 == number_of_procs ? 0 : which_proc + 1;
    }
    array_proxy.doneInserting();
  }

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
    auto& local_component = Parallel::get_parallel_component<ElementArray>(
        *(global_cache.ckLocalBranch()));
    local_component.start_phase(next_phase);
  }
};

}  // namespace DistributedLinearSolverAlgorithmTestHelpers
