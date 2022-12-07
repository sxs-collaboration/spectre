// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Parallel/Algorithms/AlgorithmSingletonDeclarations.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/MemoryMonitor/Tags.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/Actions/AddSimpleTags.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/*!
 * \ingroup ParallelGroup
 * Holds the MemoryMonitor parallel component and all actions and tags related
 * to the memory monitor.
 */
namespace mem_monitor {}

namespace mem_monitor {
namespace detail {
struct InitializeMutator {
  using return_tags = tmpl::list<mem_monitor::Tags::MemoryHolder>;
  using argument_tags = tmpl::list<>;

  using tag_type = typename mem_monitor::Tags::MemoryHolder::type;

  static void apply(const gsl::not_null<tag_type*> /*holder*/) {}
};
}  // namespace detail

/*!
 * \brief Singleton parallel component used for monitoring memory usage of other
 * parallel components
 */
template <class Metavariables>
struct MemoryMonitor {
  using chare_type = Parallel::Algorithms::Singleton;

  using metavariables = Metavariables;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<Initialization::Actions::AddSimpleTags<
                     mem_monitor::detail::InitializeMutator>,
                 Parallel::Actions::TerminatePhase>>>;

  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<MemoryMonitor<Metavariables>>(local_cache)
        .start_phase(next_phase);
  }
};
}  // namespace mem_monitor
