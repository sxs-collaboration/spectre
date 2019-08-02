// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "AlgorithmGroup.hpp"
#include "AlgorithmNodegroup.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Initialize.hpp"
#include "IO/Observer/Tags.hpp"
#include "Parallel/AddOptionsToDataBox.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/PhaseDependentActionList.hpp"

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
  using const_global_cache_tag_list = tmpl::list<>;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename metavariables::Phase, metavariables::Phase::Initialization,
      tmpl::list<Actions::Initialize<Metavariables>>>>;

  using options = tmpl::list<>;

  static void execute_next_phase(
      const typename Metavariables::Phase /*next_phase*/,
      Parallel::CProxy_ConstGlobalCache<
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
  using const_global_cache_tag_list =
      tmpl::list<OptionTags::ReductionFileName, OptionTags::VolumeFileName>;
  using metavariables = Metavariables;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename metavariables::Phase, metavariables::Phase::Initialization,
      tmpl::list<Actions::InitializeWriter<Metavariables>>>>;

  using options = tmpl::list<>;

  static void execute_next_phase(
      const typename Metavariables::Phase /*next_phase*/,
      Parallel::CProxy_ConstGlobalCache<
          Metavariables>& /*global_cache*/) noexcept {}
};
}  // namespace observers
