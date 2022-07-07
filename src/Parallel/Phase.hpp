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
 * executable must define a static member function with the signature:
 *
 * \code
 * static Parallel::Phase determine_next_phase(
 *    const gsl::not_null<
 *        tuples::TaggedTuple<Tags...>*> phase_change_decision_data,
 *    const Parallel::Phase& current_phase,
 *    const Parallel::CProxy_GlobalCache<Metavariables>& cache_proxy);
 * \endcode
 *
 * that is used to determine the next phase once the current phase has ended.
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

  ///  a cleanup phase
  Cleanup,
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
