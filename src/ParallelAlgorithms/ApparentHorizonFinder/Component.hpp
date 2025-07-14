// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>
#include <tuple>

#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Initialization.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Protocols/HorizonMetavars.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace ah {
/*!
 * \brief The singleton parallel component responsible for finding horizons.
 */
template <class Metavariables, typename HorizonMetavars>
struct Component {
  static_assert(tt::assert_conforms_to_v<HorizonMetavars,
                                         ah::protocols::HorizonMetavars>);
  using horizon_metavars = HorizonMetavars;
  using chare_type = Parallel::Algorithms::Singleton;
  static constexpr bool checkpoint_data = true;

  static std::string name() { return HorizonMetavars::name(); }

  using metavariables = Metavariables;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<Initialization::Actions::InitializeItems<
                     ::ah::Initialize<HorizonMetavars>>,
                 Parallel::Actions::TerminatePhase>>>;

  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<Component>(local_cache)
        .start_phase(next_phase);
  }
};
}  // namespace ah
