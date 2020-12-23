// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include <vector>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Helpers/ParallelAlgorithms/LinearSolver/DistributedLinearSolverAlgorithmTestHelpers.hpp"
#include "Helpers/ParallelAlgorithms/LinearSolver/LinearSolverAlgorithmTestHelpers.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Main.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/Gmres.hpp"
#include "ParallelAlgorithms/LinearSolver/Richardson/Richardson.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/TMPL.hpp"

namespace helpers = LinearSolverAlgorithmTestHelpers;
namespace helpers_distributed = DistributedLinearSolverAlgorithmTestHelpers;

namespace {

struct ParallelGmres {
  static constexpr Options::String help =
      "Options for the iterative linear solver";
};

struct Preconditioner {
  static constexpr Options::String help = "Options for the preconditioner";
};

struct Metavariables {
  static constexpr const char* const help{
      "Test the preconditioned GMRES linear solver algorithm on multiple "
      "elements"};
  static constexpr size_t volume_dim = 1;

  using linear_solver =
      LinearSolver::gmres::Gmres<Metavariables, helpers_distributed::fields_tag,
                                 ParallelGmres, true>;
  using preconditioner = LinearSolver::Richardson::Richardson<
      typename linear_solver::operand_tag, Preconditioner,
      typename linear_solver::preconditioner_source_tag>;

  using Phase = helpers::Phase;
  using component_list = helpers_distributed::component_list<Metavariables>;
  using observed_reduction_data_tags =
      helpers::observed_reduction_data_tags<Metavariables>;
  static constexpr bool ignore_unrecognized_command_line_options = false;
  static constexpr auto determine_next_phase =
      helpers::determine_next_phase<Metavariables>;
};

}  // namespace

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &domain::creators::register_derived_with_charm};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};

using charmxx_main_component = Parallel::Main<Metavariables>;

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
