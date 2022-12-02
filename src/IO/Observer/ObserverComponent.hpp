// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Initialize.hpp"
#include "IO/Observer/Tags.hpp"
#include "Parallel/Algorithms/AlgorithmGroup.hpp"
#include "Parallel/Algorithms/AlgorithmNodegroup.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Tags/InputSource.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "Utilities/TMPL.hpp"

namespace observers {
/*!
 * \ingroup ObserversGroup
 * \brief The group parallel component that is responsible for reducing data
 * to be observed.
 *
 * Once the data from all elements on the processing element (usually a core)
 * has been collected, it is copied (not sent over the network) to the local
 * nodegroup parallel component, `ObserverWriter`, for writing to disk.
 */
template <class Metavariables>
struct Observer {
  using chare_type = Parallel::Algorithms::Group;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<Actions::Initialize<Metavariables>,
                                        Parallel::Actions::TerminatePhase>>>;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const Parallel::Phase /*next_phase*/,
      Parallel::CProxy_GlobalCache<Metavariables>& /*global_cache*/) {}
};

/*!
 * \ingroup ObserversGroup
 * \brief The nodegroup parallel component that is responsible for writing data
 * to disk.
 */
template <class Metavariables>
struct ObserverWriter {
  using chare_type = Parallel::Algorithms::Nodegroup;
  using const_global_cache_tags =
      tmpl::list<Tags::ReductionFileName, Tags::VolumeFileName,
                 ::Parallel::Tags::InputSource>;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<Actions::InitializeWriter<Metavariables>,
                 Parallel::Actions::TerminatePhase>>>;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const Parallel::Phase /*next_phase*/,
      Parallel::CProxy_GlobalCache<Metavariables>& /*global_cache*/) {}
};
}  // namespace observers
