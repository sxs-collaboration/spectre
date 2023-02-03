// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iosfwd>
#include <vector>

/// \cond
namespace Options {
class Option;
template <typename T>
struct create_from_yaml;
}  // namespace Options
/// \endcond

namespace Parallel {

/*!
 * \ingroup ParallelGroup
 * \brief The possible phases of an executable
 *
 * \details Spectre executables are split into distinct phases separated by
 * global synchronizations.  Each executable will start with phase
 * `Initialization` and end with phase `Exit`.  The metavariables of each
 * executable must define `default_phase_order`, an array of Parallel::Phase
 * that must contain at least `Initialization` as the first element and `Exit`
 * as the last element.  Usually the next phase is determined from the
 * `default_phase_order` provided by the metavariables.  If more complex
 * decision making is desired, use the PhaseControl infrastructure.
 *
 * \warning The phases are in alphabetical order.  Do not use the values of the
 * underlying integral type as they will change as new phases are added to the
 * list!
 *
 * \see PhaseChange and \ref dev_guide_parallelization_foundations
 * "Parallelization infrastructure" for details.
 *
 */
enum class Phase {
  // If you add a new phase: put it in the list alphabetically, add it to the
  // vector created by known_phases() below, and add it to the stream operator
  // If the new phase is the first or last in the list, update Test_Phase.

  ///  phase in which AMR adjusts the domain
  AdjustDomain,
  ///  phase in which sanity checks are done after AMR
  CheckDomain,
  ///  phase in which sanity checks on the history of the evolved variables are
  ///  done. Intended to be after InitializeTimeStepperHistory but before Evolve
  CheckTimeStepperHistory,
  ///  a cleanup phase
  Cleanup,
  ///  phase in which AMR criteria are evaluated
  EvaluateAmrCriteria,
  ///  phase in which time steps are taken for an evolution executable
  Evolve,
  ///  generic execution phase of an executable
  Execute,
  ///  final phase of an executable
  Exit,
  ///  phase in which initial data is imported from volume files
  ImportInitialData,
  ///  initial phase of an executable
  Initialization,
  ///  phase in which quantities dependent on imported initial data are
  ///  initialized
  InitializeInitialDataDependentQuantities,
  ///  phase in which the time stepper executes a self-start procedure
  InitializeTimeStepperHistory,
  ///  phase in which components are migrated
  LoadBalancing,
  ///  phase in which components know an error occurred and they need to do some
  ///  sort of cleanup, such as dumping data to disk.
  PostFailureCleanup,
  ///  phase in which components register with other components
  Register,
  ///  phase in which components register with the data importer components
  RegisterWithElementDataReader,
  ///  phase in which something is solved
  Solve,
  ///  phase in which something is tested
  Testing,
  ///  phase in which checkpoint files are written to disk
  WriteCheckpoint
};

std::vector<Phase> known_phases();

/// Output operator for a Phase.
std::ostream& operator<<(std::ostream& os, const Phase& phase);
}  // namespace Parallel

template <>
struct Options::create_from_yaml<Parallel::Phase> {
  template <typename Metavariables>
  static Parallel::Phase create(const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
Parallel::Phase Options::create_from_yaml<Parallel::Phase>::create<void>(
    const Options::Option& options);
