// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Initialize.hpp"
#include "IO/Observer/Tags.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/Algorithms/AlgorithmGroup.hpp"
#include "Parallel/Algorithms/AlgorithmNodegroup.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
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
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename metavariables::Phase, metavariables::Phase::Initialization,
      tmpl::list<::Actions::SetupDataBox, Actions::Initialize<Metavariables>,
                 Initialization::Actions::RemoveOptionsAndTerminatePhase>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const typename Metavariables::Phase /*next_phase*/,
      Parallel::CProxy_GlobalCache<
          Metavariables>& /*global_cache*/) noexcept {}
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
      tmpl::list<Tags::ReductionFileName, Tags::VolumeFileName>;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename metavariables::Phase, metavariables::Phase::Initialization,
      tmpl::list<::Actions::SetupDataBox,
                 Actions::InitializeWriter<Metavariables>,
                 Initialization::Actions::RemoveOptionsAndTerminatePhase>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const typename Metavariables::Phase /*next_phase*/,
      Parallel::CProxy_GlobalCache<
          Metavariables>& /*global_cache*/) noexcept {}
};
}  // namespace observers
