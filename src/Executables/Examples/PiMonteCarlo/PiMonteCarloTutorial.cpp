// Distributed under the MIT License.
// See LICENSE.txt for details.

// Notes:
// 1. This code is the starting point for a tutorial on how
//    spectre executables work. Comments starting with "TUTORIAL" indicate
//    steps to build a complete executable. This file already includes some
//    necessary boilerplate and all necessary includes.
//
// 2. Building this target as-is will generate unused-variable compiler
//    warnings. These warnings should vanish once the tutorial is successfully
//    completed.
//
// 3. The file PiMonteCarlo.cpp contains a solution for each tutorial step.
//    If you get stuck or run into trouble completing the tutorial, you can
//    always compare with the solution in PiMonteCarlo.cpp.

// Includes from the C++ standard library needed for this executable
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <unordered_set>
#include <vector>

// Includes from SpECTRE libraries needed for this executable
#include "DataStructures/DataVector.hpp"
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
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

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

};  // namespace OptionTags

// TUTORIAL PART 1: Set up quantities stored in DataBox
namespace Tags {
// TUTORIAL STEP 1.0: add structs to hold the two user-specified options in
// memory: DartsPerIteration and AccuracyGoal

// TUTORIAL STEP 1.1: add structs to hold two counters in memory: how many darts
// have been thrown on all processors so far (ThrowsAllProcs), and how many of
// those have hit the quarter unit circle (HitsAllProcs).
// Instead of initializing from a user-specified value, hard-code initializing
// them to zero.

}  // Namespace Tags

// TUTORIAL PART 2: Complete the Actions ThrowDarts and EstimatePi.
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

    // TUTORIAL STEP 2.1: throw N darts at the unit square, seeing how many
    // hit the quarter circle

    // Get a proxy (an object that might live on another compute node)
    // for each ParallelComponent. The PiEstimator Singleton component
    // will run the ProcessHitsAndThrows action to estimate pi.
    // The DartThrower parallel component calls ThrowDarts on a some processors.
    // TUTORIAL STEP 2.2: get the PiEstimator and DartThrower parallel
    // components.

    // Tutorial STEP 2.3: contribute hits to reduction data

    // After this action completes, tell this element of the
    // DartThrower array parallel component to pause until further notice.
    // (That notice might come frm the ProcessHitsAndThrows action, if it
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
    db::mutate_apply<tmpl::list<>, tmpl::list<>>([]() {}, make_not_null(&box));

    // TUTORIAL STEP 2.7: estiamte pi, compute the fractional accuracy, and
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
// After defining the type aliases, insert the following boilerplate
// declaration into the struct:
/*
  static void execute_next_phase(
      const Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavars>& global_cache);
*/
template <typename Metavars>
struct PiEstimator {};

// TUTORIAL STEP 3.1: After creating PiEstimator, uncomment this defintion.
// This function is necessary boilerplate that tells
// spectre when one phase ends, start the next one.
/*
template <typename Metavars>
void PiEstimator<Metavars>::execute_next_phase(
    const Parallel::Phase next_phase,
    const Parallel::CProxy_GlobalCache<Metavars>& global_cache) {
  auto& local_cache = *Parallel::local_branch(global_cache);
  Parallel::get_parallel_component<PiEstimator<Metavars>>(local_cache)
      .start_phase(next_phase);
}
*/

// TUTORIAL STEP 3.2: Create the DartThrower parallel component struct.
// After defining the type aliases, insert the following boilerplate
// static function declarations into the struct.
/*
  static void execute_next_phase(
      const Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavars>& global_cache);
  static void allocate_array(
      Parallel::CProxy_GlobalCache<Metavars>& global_cache,
      const tuples::tagged_tuple_from_typelist<simple_tags_from_options>&
          initialization_options,
      const tuples::tagged_tuple_from_typelist<array_allocation_tags>&
          array_allocation_options = {},
      const std::unordered_set<size_t>& procs_to_ignore = {});
*/
template <typename Metavars>
struct DartThrower {};

// TUTORIAL STEP 3.3: After creating DartThrower, uncomment this function
// definition.
// Then add it to the parallel component struct as a static function.
// This function is necessary boilerplate that tells
// spectre when one phase ends, start the next one.
/*
template <typename Metavars>
void DartThrower<Metavars>::execute_next_phase(
    const Parallel::Phase next_phase,
    const Parallel::CProxy_GlobalCache<Metavars>& global_cache) {
  auto& local_cache = *Parallel::local_branch(global_cache);
  Parallel::get_parallel_component<DartThrower<Metavars>>(local_cache)
      .start_phase(next_phase);
}
*/

// TUTORIAL STEP 3.4: After creating DartThrower, uncomment this function.
// Then add it to the parallel component struct as a static function.
//
// This function assigns the array elements to
// specific cores (processors). The strategy is "round robin:" assign
// one per core until each core (except any the user wants to skip) has one,
// then repeat until each has two, etc.
//
// Note: since we choose here that there will be one DartThrower element
// per core, each core will get one element, unless the user asks to skip
// one or more cores.
// template <typename Metavars>
// void DartThrower<Metavars>::allocate_array(
//     Parallel::CProxy_GlobalCache<Metavars>& global_cache,
//     const tuples::tagged_tuple_from_typelist<simple_tags_from_options>&
//         initialization_options,
//     const tuples::tagged_tuple_from_typelist<array_allocation_tags>&
//     /*array_allocation_options*/,
//     const std::unordered_set<size_t>& procs_to_ignore) {
//   auto& local_cache = *Parallel::local_branch(global_cache);
//   auto& array_proxy =
//       Parallel::get_parallel_component<DartThrower<Metavars>>(local_cache);

//   size_t which_proc = 0;
//   const size_t num_procs = Parallel::number_of_procs<size_t>(local_cache);
//   const size_t number_of_elements = num_procs;

//   for (size_t i = 0; i < number_of_elements; ++i) {
//     while (procs_to_ignore.find(which_proc) != procs_to_ignore.end()) {
//       which_proc = which_proc + 1 == num_procs ? 0 : which_proc + 1;
//     }
//     array_proxy[i].insert(global_cache, initialization_options, which_proc);
//     which_proc = which_proc + 1 == num_procs ? 0 : which_proc + 1;
//   }
//   array_proxy.doneInserting();
// }

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

  // Boilerplate stating that this metavariables truct has no run-time content
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
