// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

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
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "NumericalAlgorithms/LinearSolver/ConjugateGradient/ConjugateGradient.hpp"
#include "NumericalAlgorithms/LinearSolver/ConjugateGradient/ResidualMonitor.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/Reduction.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
// IWYU pragma: no_forward_declare db::DataBox

namespace {

// This is a sample problem where the operator matrix represents a primal DG
// discretization of the 1D Poisson operator with an internal penalty flux. The
// source and solution are sinusoids on the interval [0, Pi].
constexpr size_t number_of_grid_points = 3;
constexpr size_t number_of_elements = 2;
const std::array<Matrix, number_of_elements> operator_matrices{
    {Matrix{{5.305164769729844, 0.8488263631567751, -0.7427230677621782},
            {0.8488263631567751, 3.395305452627100, -0.4244131815783875},
            {-0.7427230677621782, -0.4244131815783875, 3.395305452627100},
            {0.3183098861837906, -1.273239544735163, -1.909859317102744},
            {0., 0., -1.273239544735163},
            {0., 0., 0.3183098861837906}},
     Matrix{{0.3183098861837906, 0., 0.},
            {-1.273239544735163, 0., 0.},
            {-1.909859317102744, -1.273239544735163, 0.3183098861837906},
            {3.395305452627100, -0.4244131815783875, -0.7427230677621782},
            {-0.4244131815783875, 3.395305452627100, 0.8488263631567751},
            {-0.7427230677621782, 0.8488263631567751, 5.305164769729844}}}};
const std::array<DataVector, number_of_elements> sources{
    {DataVector{0., 0.740480489693061, 0.2617993877991494},
     DataVector{0.2617993877991494, 0.740480489693061, 0.}}};
const std::array<DataVector, number_of_elements> expected_results{
    {DataVector{-0.03634825103978584, 0.7235793356729763, 0.9928055333486299},
     DataVector{0.9928055333486298, 0.7235793356729763, -0.03634825103978584}}};

struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "ScalarField"; }
};

using VariablesTag = Tags::Variables<tmpl::list<ScalarFieldTag>>;
using VariablesType = db::item_type<VariablesTag>;

// Here we compute A(p)=sum_elements(A_element(p_element)) in a global reduction
// and then broadcast the global A(p) back to the elements so that they can
// extract their A_element(p).

struct CollectAp;

struct ComputeOperatorAction {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const int array_index, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto& A = gsl::at(operator_matrices, array_index);
    const auto& p =
        get<db::add_tag_prefix<LinearSolver::Tags::Operand, VariablesTag>>(box);

    VariablesType Ap{number_of_grid_points * number_of_elements};
    dgemv_('N', A.rows(), A.columns(), 1, A.data(), A.rows(), p.data(), 1, 0,
           Ap.data(), 1);

    Parallel::contribute_to_reduction<CollectAp>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<VariablesType, funcl::Plus<>>>{Ap},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<ParallelComponent>(cache));

    // Terminate algorithm for now. The reduction will be broadcasted to the
    // next action which is responsible for restarting the algorithm.
    return std::tuple<db::DataBox<DbTagsList>&&, bool>(std::move(box), true);
  }
};

struct CollectAp {
  template <
      typename... DbTags, typename... InboxTags, typename Metavariables,
      typename ActionList, typename ParallelComponent,
      Requires<tmpl2::flat_any_v<cpp17::is_same_v<VariablesTag, DbTags>...>> =
          nullptr>
  static auto apply(
      db::DataBox<tmpl::list<DbTags...>>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const int array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*component*/,
      const VariablesType& Ap_global_data) noexcept {
    // This could be generalized to work on the Variables instead of the
    // Scalar, but it's only for the purpose of this test.
    const auto& Ap_global = get<ScalarFieldTag>(Ap_global_data).get();
    DataVector Ap_local{number_of_grid_points};
    std::copy(Ap_global.begin() +
                  array_index * static_cast<int>(number_of_grid_points),
              Ap_global.begin() +
                  (array_index + 1) * static_cast<int>(number_of_grid_points),
              Ap_local.begin());
    db::mutate<db::add_tag_prefix<
        LinearSolver::Tags::OperatorAppliedTo,
        db::add_tag_prefix<LinearSolver::Tags::Operand, ScalarFieldTag>>>(
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
      Requires<tmpl2::flat_any_v<cpp17::is_same_v<VariablesTag, DbTags>...>> =
          nullptr>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const int array_index, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto& result = get<ScalarFieldTag>(box).get();
    for (size_t i = 0; i < number_of_grid_points; i++) {
      SPECTRE_PARALLEL_REQUIRE(
          result[i] == approx(gsl::at(expected_results, array_index)[i]));
    }
  }
};

struct InitializeElement {
  template <typename Metavariables>
  using return_tag_list =
      tmpl::append<tmpl::list<VariablesTag>,
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
    auto box = db::create<db::AddSimpleTags<tmpl::list<VariablesTag>>>(
        VariablesType{number_of_grid_points, 0.});

    auto linear_solver_box = Metavariables::linear_solver::tags::initialize(
        std::move(box), cache, array_index, parallel_component_meta,
        gsl::at(sources, array_index),
        db::item_type<db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo,
                                         VariablesTag>>{number_of_grid_points,
                                                        0.});

    return std::make_tuple(std::move(linear_solver_box));
  }
};

template <typename Metavariables>
struct ArrayParallelComponent {
  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using action_list =
      tmpl::list<ComputeOperatorAction,
                 typename Metavariables::linear_solver::perform_step>;
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

    for (int i = 0, which_proc = 0,
             number_of_procs = Parallel::number_of_procs();
         i < int(number_of_elements); i++) {
      array_proxy[i].insert(global_cache, which_proc);
      which_proc = which_proc + 1 == number_of_procs ? 0 : which_proc + 1;
    }
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
  using fields_tag = VariablesTag;
};

struct Metavariables {
  using system = System;

  using linear_solver = LinearSolver::ConjugateGradient<Metavariables>;

  using component_list =
      tmpl::append<tmpl::list<ArrayParallelComponent<Metavariables>>,
                   typename linear_solver::component_list>;
  using const_global_cache_tag_list = tmpl::list<>;

  static constexpr const char* const help{
      "Test conjugate gradient linear solver algorithm on multiple elements"};
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
