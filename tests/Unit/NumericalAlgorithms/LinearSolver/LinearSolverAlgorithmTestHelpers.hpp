// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "AlgorithmArray.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/DenseVector.hpp"
#include "ErrorHandling/Error.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/LinearSolver/Actions/TerminateIfConverged.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"  // IWYU pragma: keep
#include "Options/Options.hpp"
#include "Parallel/AddOptionsToDataBox.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Main.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

namespace LinearSolverAlgorithmTestHelpers {

struct LinearOperator {
  static constexpr OptionString help = "The linear operator A to invert.";
  using type = DenseMatrix<double>;
};
struct Source {
  static constexpr OptionString help = "The source b in the equation Ax=b.";
  using type = DenseVector<double>;
};
struct InitialGuess {
  static constexpr OptionString help = "The initial guess for the vector x.";
  using type = DenseVector<double>;
};
struct ExpectedResult {
  static constexpr OptionString help = "The solution x in the equation Ax=b";
  using type = DenseVector<double>;
};

// The vector `x` we want to solve for
struct VectorTag : db::SimpleTag {
  using type = DenseVector<double>;
  static std::string name() noexcept { return "VectorTag"; }
};

using operand_tag = LinearSolver::Tags::Operand<VectorTag>;
using operator_tag = LinearSolver::Tags::OperatorAppliedTo<operand_tag>;

struct ComputeOperatorAction {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const int /*array_index*/,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ActionList /*meta*/,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<operator_tag>(make_not_null(&box),
                             [](const auto Ap, const auto& A,
                                const auto& p) noexcept { *Ap = A * p; },
                             get<LinearOperator>(cache), get<operand_tag>(box));
    return {std::move(box), false};
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
      const int /*array_index*/,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ActionList /*meta*/,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& has_converged = get<LinearSolver::Tags::HasConverged>(box);
    SPECTRE_PARALLEL_REQUIRE(has_converged);
    SPECTRE_PARALLEL_REQUIRE(has_converged.reason() ==
                             Convergence::Reason::AbsoluteResidual);
    const auto& result = get<VectorTag>(box);
    const auto& expected_result = get<ExpectedResult>(cache);
    for (size_t i = 0; i < expected_result.size(); i++) {
      SPECTRE_PARALLEL_REQUIRE(result[i] == approx(expected_result[i]));
    }
    return {std::move(box), true};
  }
};

struct InitializeElement {
  template <
      typename DbTagsList, typename... InboxTags, typename Metavariables,
      typename ActionList, typename ParallelComponent,
      Requires<not tmpl::list_contains_v<DbTagsList, VectorTag>> = nullptr>
  static auto apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const int array_index, const ActionList /*meta*/,
      const ParallelComponent* const parallel_component_meta) noexcept {
    const auto& A = get<LinearOperator>(cache);
    const auto& b = get<Source>(cache);
    const auto& x0 = get<InitialGuess>(cache);

    auto vars_box = db::create_from<
        db::RemoveTags<>,
        db::AddSimpleTags<tmpl::list<VectorTag, operand_tag, operator_tag>>>(
        std::move(box), x0,
        make_with_value<db::item_type<operand_tag>>(
            x0, std::numeric_limits<double>::signaling_NaN()),
        make_with_value<db::item_type<operator_tag>>(
            x0, std::numeric_limits<double>::signaling_NaN()));
    auto linear_solver_box = Metavariables::linear_solver::tags::initialize(
        std::move(vars_box), cache, array_index, parallel_component_meta, b,
        A * x0);
    return std::make_tuple(std::move(linear_solver_box), true);
  }

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent,
            Requires<tmpl::list_contains_v<DbTagsList, VectorTag>> = nullptr>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      const db::DataBox<DbTagsList>& /*box*/,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const int /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    ERROR(
        "Re-initialization not supported. Did you forget to terminate the "
        "initialization phase?");
  }
};  // namespace

template <typename Metavariables>
struct ElementArray {
  using chare_type = Parallel::Algorithms::Array;
  using array_index = int;
  using metavariables = Metavariables;
  using options = tmpl::list<>;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  // In each step of the algorithm we must provide A(p). The linear solver then
  // takes care of updating x and p, as well as the internal variables r, its
  // magnitude and the iteration step number.
  /// [action_list]
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             tmpl::list<InitializeElement>>,

      Parallel::PhaseActions<
          typename Metavariables::Phase,
          Metavariables::Phase::PerformLinearSolve,
          tmpl::list<LinearSolver::Actions::TerminateIfConverged,
                     ComputeOperatorAction,
                     typename Metavariables::linear_solver::perform_step>>,

      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::TestResult,
                             tmpl::list<TestResult>>>;
  /// [action_list]
  using const_global_cache_tag_list =
      tmpl::list<LinearOperator, Source, InitialGuess, ExpectedResult>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
    auto& local_component = Parallel::get_parallel_component<ElementArray>(
        *(global_cache.ckLocalBranch()));
    local_component[0].insert(global_cache, tuples::TaggedTuple<>(), 0);
    local_component.doneInserting();
  }

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
    auto& local_component = Parallel::get_parallel_component<ElementArray>(
        *(global_cache.ckLocalBranch()));
    local_component.start_phase(next_phase);
  }
};

// After the algorithm completes we perform a cleanup phase that checks the
// expected output file was written and deletes it.
template <bool CheckExpectedOutput>
struct CleanOutput {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& reductions_file_name =
        get<observers::OptionTags::ReductionFileName>(cache) + ".h5";
    if (file_system::check_if_file_exists(reductions_file_name)) {
      file_system::rm(reductions_file_name, true);
    } else if (CheckExpectedOutput) {
      ERROR("Expected reductions file '" << reductions_file_name
                                         << "' does not exist");
    }
    return {std::move(box), true};
  }
};

template <typename Metavariables>
struct OutputCleaner {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<CleanOutput<false>>>,

                 Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::CleanOutput,
                                        tmpl::list<CleanOutput<true>>>>;
  using options = tmpl::list<>;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  using const_global_cache_tag_list = tmpl::list<>;

  static void initialize(Parallel::CProxy_ConstGlobalCache<
                         Metavariables>& /*global_cache*/) noexcept {}

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
    auto& local_component = Parallel::get_parallel_component<OutputCleaner>(
        *(global_cache.ckLocalBranch()));
    local_component.start_phase(next_phase);
  }
};

struct System {
  using fields_tag = VectorTag;
};

}  // namespace LinearSolverAlgorithmTestHelpers
