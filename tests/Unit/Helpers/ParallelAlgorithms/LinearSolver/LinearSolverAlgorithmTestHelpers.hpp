// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "AlgorithmArray.hpp"
#include "AlgorithmSingleton.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/DenseVector.hpp"
#include "ErrorHandling/Error.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "IO/Observer/Actions.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/RegisterObservers.hpp"
#include "IO/Observer/Tags.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/Convergence/Reason.hpp"
#include "Options/Options.hpp"
#include "Parallel/Actions/Goto.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelAlgorithms/LinearSolver/Actions/TerminateIfConverged.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace LinearSolverAlgorithmTestHelpers {

namespace OptionTags {
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
struct ExpectedConvergenceReason {
  static std::string name() noexcept { return "ConvergenceReason"; }
  static constexpr OptionString help = "The expected convergence reason";
  using type = Convergence::Reason;
};
}  // namespace OptionTags

struct LinearOperator : db::SimpleTag {
  using type = DenseMatrix<double>;
  using option_tags = tmpl::list<OptionTags::LinearOperator>;

  static constexpr bool pass_metavariables = false;
  static DenseMatrix<double> create_from_options(
      const DenseMatrix<double>& linear_operator) noexcept {
    return linear_operator;
  }
};

struct Source : db::SimpleTag {
  using type = DenseVector<double>;
  using option_tags = tmpl::list<OptionTags::Source>;

  static constexpr bool pass_metavariables = false;
  static DenseVector<double> create_from_options(
      const DenseVector<double>& source) noexcept {
    return source;
  }
};

struct InitialGuess : db::SimpleTag {
  using type = DenseVector<double>;
  using option_tags = tmpl::list<OptionTags::InitialGuess>;

  static constexpr bool pass_metavariables = false;
  static DenseVector<double> create_from_options(
      const DenseVector<double>& initial_guess) noexcept {
    return initial_guess;
  }
};

struct ExpectedResult : db::SimpleTag {
  using type = DenseVector<double>;
  using option_tags = tmpl::list<OptionTags::ExpectedResult>;

  static constexpr bool pass_metavariables = false;
  static DenseVector<double> create_from_options(
      const DenseVector<double>& expected_result) noexcept {
    return expected_result;
  }
};

struct ExpectedConvergenceReason : db::SimpleTag {
  using type = Convergence::Reason;
  using option_tags = tmpl::list<OptionTags::ExpectedConvergenceReason>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& option_value) noexcept {
    return option_value;
  }
};

// The vector `x` we want to solve for
struct VectorTag : db::SimpleTag {
  using type = DenseVector<double>;
};

using fields_tag = VectorTag;

template <typename OperandTag>
struct ComputeOperatorAction {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const int /*array_index*/,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ActionList /*meta*/,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<LinearSolver::Tags::OperatorAppliedTo<OperandTag>>(
        make_not_null(&box),
        [](const gsl::not_null<DenseVector<double>*>
               operator_applied_to_operand,
           const DenseMatrix<double>& linear_operator,
           const DenseVector<double>& operand) noexcept {
          *operator_applied_to_operand = linear_operator * operand;
        },
        get<LinearOperator>(box), get<OperandTag>(box));
    return {std::move(box), false};
  }
};

// Checks for the correct solution after the algorithm has terminated.
template <typename OptionsGroup>
struct TestResult {
  using const_global_cache_tags =
      tmpl::list<ExpectedResult, ExpectedConvergenceReason>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const int /*array_index*/,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ActionList /*meta*/,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& has_converged =
        get<LinearSolver::Tags::HasConverged<OptionsGroup>>(box);
    SPECTRE_PARALLEL_REQUIRE(has_converged);
    SPECTRE_PARALLEL_REQUIRE(has_converged.reason() ==
                             get<ExpectedConvergenceReason>(box));
    const auto& result = get<VectorTag>(box);
    const auto& expected_result = get<ExpectedResult>(box);
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
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const int /*array_index*/, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeElement,
            db::AddSimpleTags<VectorTag, ::Tags::FixedSource<VectorTag>>>(
            std::move(box), get<InitialGuess>(box), get<Source>(box)));
  }
};

namespace detail {

template <typename Preconditioner>
struct init_preconditioner_impl {
  using type = typename Preconditioner::initialize_element;
};

template <>
struct init_preconditioner_impl<void> {
  using type = tmpl::list<>;
};

template <typename Preconditioner>
using init_preconditioner =
    typename init_preconditioner_impl<Preconditioner>::type;

template <typename Preconditioner>
struct run_preconditioner_impl {
  using type =
      tmpl::list<ComputeOperatorAction<typename Preconditioner::fields_tag>,
                 typename Preconditioner::prepare_solve,
                 ::Actions::RepeatUntil<
                     LinearSolver::Tags::HasConverged<
                         typename Preconditioner::options_group>,
                     tmpl::list<typename Preconditioner::prepare_step,
                                ComputeOperatorAction<
                                    typename Preconditioner::operand_tag>,
                                typename Preconditioner::perform_step>>>;
};

template <>
struct run_preconditioner_impl<void> {
  using type = tmpl::list<>;
};

template <typename Preconditioner>
using run_preconditioner =
    typename run_preconditioner_impl<Preconditioner>::type;

}  // namespace detail

template <typename Metavariables>
struct ElementArray {
  using chare_type = Parallel::Algorithms::Array;
  using array_index = int;
  using metavariables = Metavariables;
  using linear_solver = typename Metavariables::linear_solver;
  using preconditioner = typename Metavariables::preconditioner;

  // In each step of the algorithm we must provide A(p). The linear solver then
  // takes care of updating x and p, as well as the internal variables r, its
  // magnitude and the iteration step number.
  /// [action_list]
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<InitializeElement,
                     typename linear_solver::initialize_element,
                     ComputeOperatorAction<fields_tag>,
                     detail::init_preconditioner<preconditioner>,
                     Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase,
          Metavariables::Phase::RegisterWithObserver,
          tmpl::list<typename linear_solver::register_element,
                     typename linear_solver::prepare_solve,
                     Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase,
          Metavariables::Phase::PerformLinearSolve,
          tmpl::list<LinearSolver::Actions::TerminateIfConverged<
                         typename linear_solver::options_group>,
                     typename linear_solver::prepare_step,
                     detail::run_preconditioner<preconditioner>,
                     ComputeOperatorAction<typename linear_solver::operand_tag>,
                     typename linear_solver::perform_step>>,

      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::TestResult,
          tmpl::list<TestResult<typename linear_solver::options_group>>>>;
  /// [action_list]
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;
  using const_global_cache_tags =
      tmpl::list<LinearOperator, Source, InitialGuess, ExpectedResult>;

  static void allocate_array(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache,
      const tuples::tagged_tuple_from_typelist<initialization_tags>&
          initialization_items) noexcept {
    auto& local_component = Parallel::get_parallel_component<ElementArray>(
        *(global_cache.ckLocalBranch()));
    local_component[0].insert(global_cache, initialization_items, 0);
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
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& reductions_file_name =
        get<observers::Tags::ReductionFileName>(box) + ".h5";
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
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
    auto& local_component = Parallel::get_parallel_component<OutputCleaner>(
        *(global_cache.ckLocalBranch()));
    local_component.start_phase(next_phase);
  }
};

enum class Phase {
  Initialization,
  RegisterWithObserver,
  PerformLinearSolve,
  TestResult,
  CleanOutput,
  Exit
};

template <typename Metavariables>
Phase determine_next_phase(const Phase& current_phase,
                           const Parallel::CProxy_ConstGlobalCache<
                               Metavariables>& /*cache_proxy*/) noexcept {
  switch (current_phase) {
    case Phase::Initialization:
      return Phase::RegisterWithObserver;
    case Phase::RegisterWithObserver:
      return Phase::PerformLinearSolve;
    case Phase::PerformLinearSolve:
      return Phase::TestResult;
    case Phase::TestResult:
      return Phase::CleanOutput;
    default:
      return Phase::Exit;
  }
}

namespace detail {

template <typename LinearSolver>
struct get_component_list_impl {
  using type = typename LinearSolver::component_list;
};

template <>
struct get_component_list_impl<void> {
  using type = tmpl::list<>;
};

template <typename LinearSolver>
using get_component_list = typename get_component_list_impl<LinearSolver>::type;

}  // namespace detail

template <typename Metavariables>
using component_list = tmpl::push_back<
    tmpl::append<
        detail::get_component_list<typename Metavariables::linear_solver>,
        detail::get_component_list<typename Metavariables::preconditioner>>,
    ElementArray<Metavariables>, observers::Observer<Metavariables>,
    observers::ObserverWriter<Metavariables>, OutputCleaner<Metavariables>>;

template <typename Metavariables>
using observed_reduction_data_tags =
    observers::collect_reduction_data_tags<tmpl::flatten<tmpl::list<
        typename Metavariables::linear_solver,
        tmpl::conditional_t<
            std::is_same_v<typename Metavariables::preconditioner, void>,
            tmpl::list<>, typename Metavariables::preconditioner>>>>;

}  // namespace LinearSolverAlgorithmTestHelpers
