// Distributed under the MIT License.
// See LICENSE.txt for details.

// This is the starter source for the Pi Monte Carlo developer-guide tutorial.
// Comments starting with "TUTORIAL" mark the code readers will add. The file
// already includes the necessary scaffolding and headers.
//
// The tutorial has readers save PiMonteCarlo.cpp as PiMonteCarloSolution.cpp,
// then copy this file over PiMonteCarlo.cpp so every checkpoint uses the
// regular PiMonteCarlo build target.
//
// If you get stuck, compare your work with PiMonteCarloSolution.cpp.

// Includes from the C++ standard library needed for this executable
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <random>
#include <unordered_set>

// Includes from SpECTRE libraries needed for this executable
#include "DataStructures/TaggedTuple.hpp"
#include "Options/String.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/CharmMain.tpp"
#include "Parallel/DistributedObject.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Parallel/Reduction.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// Forward declaration: promise that this way to access the global cache will
// be defined elsewhere.
namespace Parallel {
template <typename Metavariables>
class CProxy_GlobalCache;
}  // namespace Parallel

// Forward declarations: promise that the parallel components and actions will
// eventually be defined. This lets you refer to them before you actually
// define them.
// A Singleton component that computes pi from how many of the darts thrown
// at the unit square that hit the quarter unit circle.
template <typename Metavars>
struct PiEstimator;

// An array component, where each element throws some darts at the unit square
// and checks how many hit the unit quarter circle.
template <typename Metavars>
struct DartThrower;

// Forward declare the actions implemented in this file, so they can be
// referred to before being defined. There will be two actions:
//      1. ThrowDarts: an "iterable action" that can be called repeatedly.
//                     Throw some darts, check how many hit, then report
//                     the number of hits.
//      2. ProcessHitsAndThrows: a reduction action that sums how many darts
//                               hit and how many were thrown across all
//                               processors, and then uses that info to
//                               calculate pi.
namespace Actions {
struct ThrowDarts;
struct ProcessHitsAndThrows;
}  // namespace Actions

// TUTORIAL PART 0: Set up options to read from input (yaml) file
namespace OptionTags {
// TUTORIAL STEP 0.0: add structs for the two quantities the user will choose
// when running the executable: DartsPerIteration and AccuracyGoal.

}  // namespace OptionTags

// TUTORIAL PART 1: Set up quantities stored in DataBox
namespace Tags {
// TUTORIAL STEP 1.0: add structs to hold the two user-specified options in
// memory: DartsPerIteration and AccuracyGoal

// TUTORIAL STEP 1.1: add structs to hold two counters in memory: how many darts
// have been thrown on all processors so far (ThrowsAllProcs), and how many of
// those have hit the quarter unit circle (HitsAllProcs).
// Instead of initializing from a user-specified value, hard-code initializing
// them to zero.

}  // namespace Tags

// TUTORIAL PART 2: Complete the ThrowDarts and ProcessHitsAndThrows actions.
namespace Actions {
// In spectre, "iterable actions" (actions that can be done more than once) are
// made by creating a struct with a function apply with the following
// template parameters (compile-time parameters), parameters, and
// return type.
struct ThrowDarts {
  template <typename DbTags, typename... InboxTags, typename Metavars,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavars>& cache,
      const ArrayIndex& array_index, const ActionList& /*meta*/,
      const ParallelComponent* const /*meta*/
  ) {
    // TUTORIAL STEP 2.0: get how many darts to throw from the DataBox
    (void)box;  // Temporary: remove in step 2.0.

    // TUTORIAL STEP 2.1: throw N darts at the unit square, seeing how many
    // hit the quarter circle
    // Get a proxy (an object that might live on another compute node)
    // for each ParallelComponent. The PiEstimator Singleton component
    // will run the ProcessHitsAndThrows action to estimate pi.
    // The DartThrower parallel component calls ThrowDarts on multiple
    // processors.
    // TUTORIAL STEP 2.2: get the PiEstimator and DartThrower parallel
    // components.
    (void)cache;        // Temporary: remove in step 2.2.
    (void)array_index;  // Temporary: remove in step 2.2.

    // TUTORIAL STEP 2.3: contribute hits to reduction data

    // After this action completes, tell this element of the
    // DartThrower array parallel component to pause until further notice.
    // (That notice might come from the ProcessHitsAndThrows action, if it
    // decides that more darts should be thrown.)
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

// In spectre, "reduction actions" (actions that receive data from the
// elements of an array parallel component and then reduce them to a single
// result) are made by creating a struct with a function apply with the
// following template parameters (compile-time parameters), parameters, and
// return type.
struct ProcessHitsAndThrows {
  template <typename ParallelComponent, typename DbTags, typename Metavars,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavars>& cache,
                    const ArrayIndex& /*array_index*/, const size_t new_hits) {
    // TUTORIAL STEP 2.4: get number of processors from the cache
    (void)cache;  // Temporary: remove in step 2.4.

    // TUTORIAL STEP 2.5: get number of darts thrown each iteration
    // from the DataBox

    // TUTORIAL STEP 2.6: complete this lambda that updates quantities
    // in the DataBox:
    //  STEP 2.6.1: Add Tags::HitsAllProcs, Tags::ThrowsAllProcs to first
    //              tmpl::list<> in db::mutate_apply<>()
    //  STEP 2.6.2: Add a corresponding const gsl::not_null<size_t*>
    //              parameter corresponding to each of the two tags from
    //              step 2.6.1
    //  STEP 2.6.3: capture variables storing new_hits, darts_per_iteration,
    //              and number_of_processors to the capture list
    //  STEP 2.6.4: make the body increment the values pointed to by the
    //              pointers (e.g. *hits_all_procs += new_hits)
    (void)new_hits;  // Temporary: remove in step 2.6.
    db::mutate_apply<tmpl::list<>, tmpl::list<>>([]() {}, make_not_null(&box));

    // TUTORIAL STEP 2.7: estimate pi, compute the fractional accuracy, and
    // print the result using Parallel::printf

    // TUTORIAL STEP 2.8: if fractional accuracy is bigger than the accuracy
    // goal, tell each element of the DartThrower parallel component to unpause
    // (that is, throw some more darts).
  }
};
}  // namespace Actions

////////////////////////////////////////////////////////////////////////
// TUTORIAL STEP 3: Set up parallel components
////////////////////////////////////////////////////////////////////////

// TUTORIAL STEP 3.0: Create the PiEstimator parallel component struct.
template <typename Metavars>
struct PiEstimator {};

// TUTORIAL STEP 3.1: Define PiEstimator::execute_next_phase.
// This function is necessary boilerplate that tells
// SpECTRE to start the next phase when one phase ends.

// TUTORIAL STEP 3.2: Create the DartThrower parallel component struct.
template <typename Metavars>
struct DartThrower {};

// TUTORIAL STEP 3.3: Define DartThrower::execute_next_phase.
// This function is necessary boilerplate that tells
// SpECTRE to start the next phase when one phase ends.

// TUTORIAL STEP 3.4: Define DartThrower::allocate_array.
//
// This function assigns the array elements to
// specific cores (processors). The strategy is "round robin:" assign
// one per core until each core (except any the user wants to skip) has one,
// then repeat until each has two, etc.
//
// Note: since we choose here that there will be one DartThrower element
// per core, each core will get one element, unless the user asks to skip
// one or more cores.
// TUTORIAL STEP 4: Complete the Metavariables struct
struct Metavariables {
  // TUTORIAL STEP 4.1: Add the PiEstimator and DartThrower components
  // to the component list
  using component_list = tmpl::list<>;

  // TUTORIAL STEP 4.2: complete the help string for this executable.
  static constexpr Options::String help{"INSERT HELP TEXT HERE"};

  // Boilerplate defining phases.
  // All executables have an initialization and an exit phase.
  // Synchronization points occur at phase boundaries. Here, there's just one
  // phase in between where all the work gets done, called execute.
  static constexpr std::array<Parallel::Phase, 3> default_phase_order{
      {Parallel::Phase::Initialization, Parallel::Phase::Execute,
       Parallel::Phase::Exit}};

  // Boilerplate stating that this metavariables struct has no run-time content
  // that must be sent over the network when remote objects want the
  // metavariables. This is done by defining a pup (pack-unpack) function
  // that does nothing.
  void pup(PUP::er& /*p*/) {}
};

// Required boilerplate to make charm++ work
extern "C" void CkRegisterMainModule() {
  Parallel::charmxx::register_main_module<Metavariables>();
  Parallel::charmxx::register_init_node_and_proc({}, {});
}
