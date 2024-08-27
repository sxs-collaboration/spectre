// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <vector>

#include "Helpers/ParallelAlgorithms/LinearSolver/LinearSolverAlgorithmTestHelpers.hpp"
#include "Parallel/CharmMain.tpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/Gmres.hpp"
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

struct Metavariables {
  static constexpr const char* const help{
      "Test the GMRES linear solver algorithm"};

  using linear_solver =
      LinearSolver::gmres::Gmres<Metavariables,
                                 helpers::fields_tag<std::complex<double>>,
                                 SerialGmres, false>;
  using preconditioner = void;

  using component_list = helpers::component_list<Metavariables>;
  using observed_reduction_data_tags =
      helpers::observed_reduction_data_tags<Metavariables>;
  static constexpr bool ignore_unrecognized_command_line_options = false;
  static constexpr auto default_phase_order = helpers::default_phase_order;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};

}  // namespace

extern "C" void CkRegisterMainModule() {
  Parallel::charmxx::register_main_module<Metavariables>();
  Parallel::charmxx::register_init_node_and_proc({}, {});
}
