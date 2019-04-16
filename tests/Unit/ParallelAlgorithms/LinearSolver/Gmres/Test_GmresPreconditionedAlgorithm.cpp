// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include <vector>

#include "Helpers/ParallelAlgorithms/LinearSolver/LinearSolverAlgorithmTestHelpers.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Main.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/Gmres.hpp"
#include "ParallelAlgorithms/LinearSolver/Richardson/Richardson.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace PUP {
class er;
}  // namespace PUP

namespace helpers = LinearSolverAlgorithmTestHelpers;

namespace {

struct SerialGmres {
  static constexpr Options::String help =
      "Options for the iterative linear solver";
};

struct Preconditioner {
  static constexpr Options::String help = "Options for the preconditioner";
};

struct Metavariables {
  static constexpr const char* const help{
      "Test the preconditioned GMRES linear solver algorithm"};

  using linear_solver =
      LinearSolver::gmres::Gmres<Metavariables, helpers::fields_tag,
                                 SerialGmres, true>;
  using preconditioner = LinearSolver::Richardson::Richardson<
      typename linear_solver::operand_tag, Preconditioner,
      typename linear_solver::preconditioner_source_tag>;

  using component_list = helpers::component_list<Metavariables>;
  using observed_reduction_data_tags =
      helpers::observed_reduction_data_tags<Metavariables>;
  static constexpr bool ignore_unrecognized_command_line_options = false;
  using Phase = helpers::Phase;
  static constexpr auto determine_next_phase =
      helpers::determine_next_phase<Metavariables>;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) noexcept {}
};

}  // namespace

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &setup_memory_allocation_failure_reporting};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};

using charmxx_main_component = Parallel::Main<Metavariables>;

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
