// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include <vector>

#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Helpers/ParallelAlgorithms/LinearSolver/LinearSolverAlgorithmTestHelpers.hpp"  // IWYU pragma: keep
#include "IO/Observer/Helpers.hpp"            // IWYU pragma: keep
#include "IO/Observer/ObserverComponent.hpp"  // IWYU pragma: keep
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Main.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/Gmres.hpp"
#include "Utilities/TMPL.hpp"

namespace helpers = LinearSolverAlgorithmTestHelpers;

namespace {

struct SerialGmres {
  static constexpr OptionString help =
      "Options for the iterative linear solver";
};

struct Metavariables {
  using linear_solver =
      LinearSolver::Gmres<Metavariables, helpers::fields_tag, SerialGmres>;

  using component_list =
      tmpl::append<tmpl::list<helpers::ElementArray<Metavariables>,
                              observers::ObserverWriter<Metavariables>,
                              helpers::OutputCleaner<Metavariables>>,
                   typename linear_solver::component_list>;

  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::list<linear_solver>>;

  static constexpr const char* const help{
      "Test the GMRES linear solver algorithm"};
  static constexpr bool ignore_unrecognized_command_line_options = false;

  enum class Phase {
    Initialization,
    RegisterWithObserver,
    PerformLinearSolve,
    TestResult,
    CleanOutput,
    Exit
  };

  static Phase determine_next_phase(
      const Phase& current_phase,
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
};

}  // namespace

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};

using charmxx_main_component = Parallel::Main<Metavariables>;

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
