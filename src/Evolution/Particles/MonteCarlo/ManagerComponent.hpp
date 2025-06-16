// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Particles/MonteCarlo/Tags.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "Utilities/TMPL.hpp"

namespace Particles::MonteCarlo {

namespace Actions {

template <typename Metavariables>
struct Initialize {
  using simple_tags_from_options = tmpl::list<>;
  using simple_tags =
      tmpl::list<Particles::MonteCarlo::Tags::MinimumPacketEnergyAtEmission<3>>;
  using mutable_global_cache_tags = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  using compute_tags = tmpl::list<>;
  using return_tags =
      tmpl::list<Particles::MonteCarlo::Tags::MinimumPacketEnergyAtEmission<3>>;
  using argument_tags =
      tmpl::list<Parallel::Tags::GlobalCache,
                 Particles::MonteCarlo::Tags::MonteCarloOptions<3>>;

  static void apply(
      const gsl::not_null<std::array<double, 3>*> minimum_packet_energy,
      const Parallel::GlobalCache<Metavariables>* const& /*cache*/,
      const Particles::MonteCarlo::MonteCarloOptions<3>& mc_options) {
    Parallel::printf("Initializing Monte Carlo Manager\n");
    (*minimum_packet_energy) = mc_options.get_initial_packet_energy();
  }
};
}  // namespace Actions

/*!
 * \brief The singleton parallel component responsible for managing
 * the Monte-Carlo evoltuion.
 *
 */
template <class Metavariables>
struct ManagerComponent {
  using chare_type = Parallel::Algorithms::Singleton;

  static std::string name() { return "MonteCarloManager"; }

  using metavariables = Metavariables;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<Initialization::Actions::InitializeItems<
                     Particles::MonteCarlo::Actions::Initialize<Metavariables>>,
                 Parallel::Actions::TerminatePhase>>>;

  using simple_tags_from_options = tmpl::list<>;

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<ManagerComponent<Metavariables>>(
        local_cache)
        .start_phase(next_phase);
  }
};

}  // namespace Particles::MonteCarlo
