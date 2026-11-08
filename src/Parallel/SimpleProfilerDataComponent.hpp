// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <atomic>

#include "DataStructures/DataBox/Protocols/Mutator.hpp"
#include "Parallel/Algorithms/AlgorithmNodegroup.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Tags/LastSimpleProfilerWrite.hpp"
#include "Parallel/Tags/SimpleProfilerData.hpp"
#include "ParallelAlgorithms/Actions/AddSimpleTags.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class CProxy_GlobalCache;
}  // namespace Parallel
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

namespace profiling {
/// DataBox mutator that sets
/// `Parallel::Tags::LastSimpleProfilerWrite` to negative infinity.
/// As profiler observation assumes the observation values are
/// increasing, this must be earlier than the first observation.
struct InitializeLastSimpleProfilerWrite
    : tt::ConformsTo<db::protocols::Mutator> {
  using return_tags = tmpl::list<Parallel::Tags::LastSimpleProfilerWrite>;
  using argument_tags = tmpl::list<>;

  static void apply(gsl::not_null<std::atomic<double>*> last_write);
};

/// Nodegroup component that stores per-thread profiling data.
template <typename Metavariables>
struct SimpleProfilerDataComponent {
  using chare_type = Parallel::Algorithms::Nodegroup;
  static constexpr bool checkpoint_data = true;
  using metavariables = Metavariables;

  using simple_tags_from_options =
      tmpl::list<Parallel::Tags::SimpleProfilerData>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<Initialization::Actions::AddSimpleTags<
                                            InitializeLastSimpleProfilerWrite>,
                                        Parallel::Actions::TerminatePhase>>>;

  using simple_tags = tmpl::list<>;

  static void execute_next_phase(
      const Parallel::Phase /*next_phase*/,
      Parallel::CProxy_GlobalCache<Metavariables>& /*global_cache*/) {}
};
}  // namespace profiling
