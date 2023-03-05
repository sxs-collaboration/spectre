// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/InitializeElementFacesGridCoordinates.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/ReceiveElementData.hpp"
#include "IO/Observer/Actions/RegisterSingleton.hpp"
#include "Options/Options.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Tags/ResourceInfo.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "Utilities/System/ParallelInfo.hpp"

namespace CurvedScalarWave::Worldtube {

/*!
 * \brief The singleton component that represents the worldtube.
 *
 * \details The component receives from and sends data to the elements abutting
 * the worldtube. It holds and calculates a solution for the regular field
 * \f$\Psi^R\f$ which valid in a neighborhood of the scalar charge.
 */
template <class Metavariables>
struct WorldtubeSingleton {
  static constexpr size_t Dim = Metavariables::volume_dim;
  using chare_type = ::Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using evolved_vars = ::Tags::Variables<
      tmpl::list<CurvedScalarWave::Tags::Psi, CurvedScalarWave::Tags::Pi>>;

  using initialization_actions =
      tmpl::list<Initialization::InitializeElementFacesGridCoordinates<Dim>,
                 Parallel::Actions::TerminatePhase>;

  using step_actions = tmpl::list<Actions::ReceiveElementData>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             initialization_actions>,
      Parallel::PhaseActions<Parallel::Phase::Register,
                             tmpl::list<Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<Parallel::Phase::Evolve, step_actions>>;

  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const typename Parallel::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<WorldtubeSingleton<metavariables>>(
        local_cache)
        .start_phase(next_phase);
  }
};
}  // namespace CurvedScalarWave::Worldtube
