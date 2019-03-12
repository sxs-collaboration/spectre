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
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/DenseVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "ErrorHandling/Error.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/LinearSolver/Actions/TerminateIfConverged.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/Printf.hpp"
#include "Parallel/Reduction.hpp"
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

// The vector `x` we want to solve for
struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "ScalarField"; }
};

using fields_tag = Tags::Variables<tmpl::list<ScalarFieldTag>>;
using operand_tag = db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
using operator_tag =
    db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, operand_tag>;

// In the following `ComputeOperatorAction` and `CollectAp` actions we
// compute A(p)=sum_elements(A_element(p_element)) in a global reduction and
// then broadcast the global A(p) back to the elements so that they can extract
// their A_element(p). This is horribly inefficient parallelism but allows us to
// just provide a global matrix A (represented by the `LinearOperator` tag) in
// an input file.

// Forward declare to keep these actions in the order they are used
struct CollectAp;

struct ComputeOperatorAction {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const int array_index, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto& operator_matrices = get<LinearOperator>(cache);
    const auto number_of_elements = operator_matrices.size();
    const auto& A = gsl::at(operator_matrices, array_index);
    const auto number_of_grid_points = A.columns();
    const auto& p = get<operand_tag>(box);

    db::item_type<fields_tag> Ap{number_of_grid_points * number_of_elements};
    dgemv_('N', A.rows(), A.columns(), 1, A.data(), A.rows(), p.data(), 1, 0,
           Ap.data(), 1);

    Parallel::contribute_to_reduction<CollectAp>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<db::item_type<fields_tag>, funcl::Plus<>>>{
            Ap},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<ParallelComponent>(cache));

    // Terminate algorithm for now. The reduction will be broadcast to the
    // next action which is responsible for restarting the algorithm.
    return std::tuple<db::DataBox<DbTagsList>&&, bool>(std::move(box), true);
  }
};

struct CollectAp {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent,
            Requires<sizeof...(DbTags) != 0> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const int array_index, const ActionList /*meta*/,
                    const ParallelComponent* const /*component*/,
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
    // We use `ckLocal()` here since this is essentially retrieving "self",
    // which is guaranteed to be on the local processor. This ensures the calls
    // are evaluated in order.
    Parallel::get_parallel_component<ParallelComponent>(cache)[array_index]
        .ckLocal()
        ->set_terminate(false);
    Parallel::get_parallel_component<ParallelComponent>(cache)[array_index]
        .perform_algorithm();
  }
};

// Checks for the correct solution after the algorithm has terminated.
struct TestResult {
  template <
      typename... DbTags, typename... InboxTags, typename Metavariables,
      typename ActionList, typename ParallelComponent,
      Requires<tmpl2::flat_any_v<cpp17::is_same_v<fields_tag, DbTags>...>> =
          nullptr>
  static void apply(const db::DataBox<tmpl::list<DbTags...>>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const int array_index, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto& has_converged = get<LinearSolver::Tags::HasConverged>(box);
    SPECTRE_PARALLEL_REQUIRE(has_converged);
    SPECTRE_PARALLEL_REQUIRE(has_converged.reason() ==
                             Convergence::Reason::AbsoluteResidual);
    const auto& expected_result =
        gsl::at(get<ExpectedResult>(cache), array_index);
    const auto& result = get<ScalarFieldTag>(box).get();
    for (size_t i = 0; i < expected_result.size(); i++) {
      Parallel::printf("result=%f, expected=%f", result[i], expected_result[i]);
      SPECTRE_PARALLEL_REQUIRE(result[i] == approx(expected_result[i]));
    }
  }
};

struct InitializeElement {
  template <typename Metavariables>
  using return_tag_list =
      tmpl::append<tmpl::list<fields_tag, operand_tag, operator_tag>,
                   typename Metavariables::linear_solver::tags::simple_tags,
                   typename Metavariables::linear_solver::tags::compute_tags>;

  template <typename... InboxTags, typename Metavariables, typename ActionList,
            typename ParallelComponent>
  static auto apply(
      const db::DataBox<tmpl::list<>>& /*box*/,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const int array_index, const ActionList /*meta*/,
      const ParallelComponent* const parallel_component_meta) noexcept {
    const auto& source = gsl::at(get<Source>(cache), array_index);
    const size_t num_points = source.size();

    auto box = db::create<
        db::AddSimpleTags<tmpl::list<fields_tag, operand_tag, operator_tag>>>(
        db::item_type<fields_tag>{num_points, 0.},
        db::item_type<operand_tag>{
            num_points, std::numeric_limits<double>::signaling_NaN()},
        db::item_type<operator_tag>{
            num_points, std::numeric_limits<double>::signaling_NaN()});
    auto linear_solver_box = Metavariables::linear_solver::tags::initialize(
        std::move(box), cache, array_index, parallel_component_meta, source,
        db::item_type<db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo,
                                         fields_tag>>{num_points, 0.});
    return std::make_tuple(std::move(linear_solver_box));
  }
};

template <typename Metavariables>
struct ElementArray {
  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using action_list =
      tmpl::list<LinearSolver::Actions::TerminateIfConverged,
                 ComputeOperatorAction,
                 typename Metavariables::linear_solver::perform_step>;
  using initial_databox = db::compute_databox_type<
      typename InitializeElement::return_tag_list<Metavariables>>;
  using options = tmpl::list<OptionTags::NumberOfElements>;
  using const_global_cache_tag_list =
      tmpl::list<LinearOperator, Source, ExpectedResult>;
  using array_index = int;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache,
      const size_t number_of_elements) noexcept {
    auto& array_proxy = Parallel::get_parallel_component<ElementArray>(
        *(global_cache.ckLocalBranch()));

    for (int i = 0, which_proc = 0,
             number_of_procs = Parallel::number_of_procs();
         i < static_cast<int>(number_of_elements); i++) {
      array_proxy[i].insert(global_cache, which_proc);
      which_proc = which_proc + 1 == number_of_procs ? 0 : which_proc + 1;
    }
    array_proxy.doneInserting();

    Parallel::simple_action<InitializeElement>(array_proxy);
  }

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
    auto array_proxy = Parallel::get_parallel_component<ElementArray>(
        *(global_cache.ckLocalBranch()));
    switch (next_phase) {
      case Metavariables::Phase::PerformLinearSolve:
        array_proxy.perform_algorithm();
        break;
      case Metavariables::Phase::TestResult:
        Parallel::simple_action<TestResult>(array_proxy);
        break;
      case Metavariables::Phase::CleanOutput:
        break;
      default:
        ERROR(
            "The Metavariables is expected to have the following Phases: "
            "Initialization, PerformLinearSolve, TestResult, Exit");
    }
  }
};

struct System {
  using fields_tag = ::DistributedLinearSolverAlgorithmTestHelpers::fields_tag;
};

}  // namespace DistributedLinearSolverAlgorithmTestHelpers
