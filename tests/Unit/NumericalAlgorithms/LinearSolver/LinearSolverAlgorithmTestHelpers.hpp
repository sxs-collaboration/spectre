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
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const int /*array_index*/, const ActionList /*meta*/,
                    const ParallelComponent* const /*component*/) noexcept {
    db::mutate<operator_tag>(make_not_null(&box),
                             [](const auto Ap, const auto& A,
                                const auto& p) noexcept { *Ap = A * p; },
                             get<LinearOperator>(cache), get<operand_tag>(box));
    return std::forward_as_tuple(std::move(box));
  }
};

// Checks for the correct solution after the algorithm has terminated.
struct TestResult {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent,
            Requires<sizeof...(DbTags) != 0> = nullptr>
  static void apply(const db::DataBox<tmpl::list<DbTags...>>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const int /*array_index*/, const ActionList /*meta*/,
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
  }
};

struct InitializeElement {
  template <typename Metavariables>
  using return_tag_list =
      tmpl::append<tmpl::list<VectorTag, operand_tag, operator_tag>,
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
    const auto& A = get<LinearOperator>(cache);
    const auto& b = get<Source>(cache);
    const auto& x0 = get<InitialGuess>(cache);

    auto box = db::create<
        db::AddSimpleTags<tmpl::list<VectorTag, operand_tag, operator_tag>>>(
        x0,
        make_with_value<db::item_type<operand_tag>>(
            x0, std::numeric_limits<double>::signaling_NaN()),
        make_with_value<db::item_type<operator_tag>>(
            x0, std::numeric_limits<double>::signaling_NaN()));
    auto linear_solver_box = Metavariables::linear_solver::tags::initialize(
        std::move(box), cache, array_index, parallel_component_meta, b, A * x0);
    return std::make_tuple(std::move(linear_solver_box));
  }
};  // namespace

template <typename Metavariables>
struct ElementArray {
  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  // In each step of the algorithm we must provide A(p). The linear solver then
  // takes care of updating x and p, as well as the internal variables r, its
  // magnitude and the iteration step number.
  /// [action_list]
  using action_list =
      tmpl::list<LinearSolver::Actions::TerminateIfConverged,
                 ComputeOperatorAction,
                 typename Metavariables::linear_solver::perform_step>;
  /// [action_list]
  using initial_databox = db::compute_databox_type<
      typename InitializeElement::return_tag_list<Metavariables>>;
  using options = tmpl::list<>;
  using const_global_cache_tag_list =
      tmpl::list<LinearOperator, Source, InitialGuess, ExpectedResult>;
  using array_index = int;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
    auto& array_proxy = Parallel::get_parallel_component<ElementArray>(
        *(global_cache.ckLocalBranch()));
    array_proxy[0].insert(global_cache, 0);
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

// After the algorithm completes we perform a cleanup phase that checks the
// expected output file was written and deletes it.
struct CleanOutput {
  template <typename... InboxTags, typename Metavariables, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static void apply(const db::DataBox<tmpl::list<>>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const bool check_expected_output) noexcept {
    const auto& reductions_file_name =
        get<observers::OptionTags::ReductionFileName>(cache) + ".h5";
    if (file_system::check_if_file_exists(reductions_file_name)) {
      file_system::rm(reductions_file_name, true);
    } else if (check_expected_output) {
      ERROR("Expected reductions file '" << reductions_file_name
                                         << "' does not exist");
    }
  }
};

template <typename Metavariables>
struct OutputCleaner {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using action_list = tmpl::list<>;
  using initial_databox = db::compute_databox_type<tmpl::list<>>;
  using options = tmpl::list<>;
  using const_global_cache_tag_list = tmpl::list<>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
    Parallel::simple_action<CleanOutput>(
        Parallel::get_parallel_component<OutputCleaner>(
            *(global_cache.ckLocalBranch())), false);
  }

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
    if (next_phase == Metavariables::Phase::CleanOutput) {
      Parallel::simple_action<CleanOutput>(
          Parallel::get_parallel_component<OutputCleaner>(
              *(global_cache.ckLocalBranch())), true);
    }
  }
};

struct System {
  using fields_tag = VectorTag;
};

}  // namespace LinearSolverAlgorithmTestHelpers
