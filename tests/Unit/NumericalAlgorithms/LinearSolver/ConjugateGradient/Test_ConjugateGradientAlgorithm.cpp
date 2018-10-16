// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

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
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "NumericalAlgorithms/LinearSolver/ConjugateGradient/ConjugateGradient.hpp"
#include "NumericalAlgorithms/LinearSolver/ConjugateGradient/ResidualMonitor.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"  // IWYU pragma: keep
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Main.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
// IWYU pragma: no_forward_declare db::DataBox

namespace {

// The symbols in this test are chosen to coincide with the pseudocode notation
// in the [Blaze documentation](https://bitbucket.org/blaze-lib/blaze/wiki/
// Getting%20Started#!a-complex-example)
const DenseMatrix<double> A{{4., 1.}, {1., 3.}};  // Must be symmetric.
const DenseVector<double> b{1., 2.};
const DenseVector<double> x0{2., 1.};
const DenseVector<double> expected_result{1. / 11., 7. / 11.};

// This is the vector we want to solve for. Corresponds to the symbol `x` in the
// notation referenced above.
struct VectorTag : db::SimpleTag {
  using type = DenseVector<double>;
  static std::string name() noexcept { return "VectorTag"; }
};

struct ComputeOperatorAction {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const int /*array_index*/, const ActionList /*meta*/,
                    const ParallelComponent* const /*component*/) noexcept {
    db::mutate<LinearSolver::Tags::OperatorAppliedTo<
        LinearSolver::Tags::Operand<VectorTag>>>(
        make_not_null(&box), [](auto Ap, auto p) noexcept { *Ap = A * p; },
        get<LinearSolver::Tags::Operand<VectorTag>>(box));

    return std::forward_as_tuple(std::move(box));
  }
};

// Checks for the correct solution after the algorithm has terminated.
struct TestResult {
  template <
      typename... DbTags, typename... InboxTags, typename Metavariables,
      typename ActionList, typename ParallelComponent,
      Requires<tmpl2::flat_any_v<cpp17::is_same_v<VectorTag, DbTags>...>> =
          nullptr>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const int /*array_index*/, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto& result = get<VectorTag>(box);
    for (size_t i = 0; i < expected_result.size(); i++) {
      SPECTRE_PARALLEL_REQUIRE(result[i] == approx(expected_result[i]));
    }
  }
};

struct InitializeElement {
  template <typename Metavariables>
  using return_tag_list =
      tmpl::append<tmpl::list<VectorTag>,
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
    auto box = db::create<db::AddSimpleTags<tmpl::list<VectorTag>>>(x0);

    auto linear_solver_box = Metavariables::linear_solver::tags::initialize(
        std::move(box), cache, array_index, parallel_component_meta, b, A * x0);

    return std::make_tuple(std::move(linear_solver_box));
  }
};  // namespace

template <typename Metavariables>
struct ArrayParallelComponent {
  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  // In each step of the algorithm we must provide A(p). The linear solver then
  // takes care of updating x and p, as well as the internal variables r, its
  // magnitude and the iteration step number.
  /// [action_list]
  using action_list =
      tmpl::list<ComputeOperatorAction,
                 typename Metavariables::linear_solver::perform_step>;
  /// [action_list]
  using initial_databox = db::compute_databox_type<
      typename InitializeElement::return_tag_list<Metavariables>>;
  using options = tmpl::list<>;
  using const_global_cache_tag_list = tmpl::list<>;
  using array_index = int;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
    auto& array_proxy =
        Parallel::get_parallel_component<ArrayParallelComponent>(
            *(global_cache.ckLocalBranch()));
    array_proxy[0].insert(global_cache, 0);
    array_proxy.doneInserting();

    Parallel::simple_action<InitializeElement>(array_proxy);
  }

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
    auto array_proxy = Parallel::get_parallel_component<ArrayParallelComponent>(
        *(global_cache.ckLocalBranch()));
    switch (next_phase) {
      case Metavariables::Phase::PerformConjugateGradient:
        array_proxy.perform_algorithm();
        break;
      case Metavariables::Phase::TestResult:
        Parallel::simple_action<TestResult>(array_proxy);
        break;
      default:
        break;
    }
  }
};

struct System {
  using fields_tag = VectorTag;
};

struct Metavariables {
  using system = System;

  using linear_solver = LinearSolver::ConjugateGradient<Metavariables>;

  using component_list =
      tmpl::append<tmpl::list<ArrayParallelComponent<Metavariables>>,
                   typename linear_solver::component_list>;
  using const_global_cache_tag_list = tmpl::list<>;

  static constexpr const char* const help{
      "Test conjugate gradient linear solver algorithm"};
  static constexpr bool ignore_unrecognized_command_line_options = false;

  enum class Phase {
    Initialization,
    PerformConjugateGradient,
    TestResult,
    Exit
  };

  static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_ConstGlobalCache<
          Metavariables>& /*cache_proxy*/) noexcept {
    switch (current_phase) {
      case Phase::Initialization:
        return Phase::PerformConjugateGradient;
      case Phase::PerformConjugateGradient:
        return Phase::TestResult;
      default:
        return Phase::Exit;
    }
  }
};

}  // namespace

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};

using charmxx_main_component = Parallel::Main<Metavariables>;

#include "Parallel/CharmMain.cpp"
