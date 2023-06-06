// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/ChangeSlabSize.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/InitializeElementFacesGridCoordinates.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/InitializeEvolvedVariables.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/InitializeSpacetimeTags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/ObserveWorldtubeSolution.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/ReceiveElementData.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/SendToElements.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/TimeDerivative.hpp"
#include "IO/Observer/Actions/RegisterSingleton.hpp"
#include "Options/String.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Tags/ResourceInfo.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/Actions/UpdateU.hpp"
#include "Time/SelfStart.hpp"
#include "Utilities/System/ParallelInfo.hpp"

namespace CurvedScalarWave::Worldtube {

struct Registration {
  template <typename ParallelComponent, typename DbTagsList,
            typename ArrayIndex>
  static std::pair<observers::TypeOfObservation, observers::ObservationKey>
  register_info(const db::DataBox<DbTagsList>& /*box*/,
                const ArrayIndex& /*array_index*/) {
    return {observers::TypeOfObservation::Reduction,
            observers::ObservationKey{"/Worldtube"}};
  }
};

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
  // not currently supported
  static constexpr bool local_time_stepping = false;
  using initialization_actions = tmpl::list<
      ::Initialization::Actions::InitializeItems<
          ::Initialization::TimeStepping<Metavariables, local_time_stepping>,
          Initialization::InitializeEvolvedVariables,
          Initialization::InitializeSpacetimeTags,
          Initialization::InitializeElementFacesGridCoordinates<Dim>>,
      Parallel::Actions::TerminatePhase>;

  struct worldtube_system {
    static constexpr size_t volume_dim = Dim;
    static constexpr bool has_primitive_and_conservative_vars = false;
    using variables_tag =
        ::Tags::Variables<tmpl::list<Tags::Psi0, Tags::dtPsi0>>;
  };
  using step_actions =
      tmpl::list<Actions::ChangeSlabSize, Actions::ReceiveElementData,
                 Actions::ComputeTimeDerivative,
                 ::Actions::RecordTimeStepperData<worldtube_system>,
                 ::Actions::UpdateU<worldtube_system>,
                 Actions::SendToElements<Metavariables>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             initialization_actions>,
      Parallel::PhaseActions<
          Parallel::Phase::InitializeTimeStepperHistory,
          SelfStart::self_start_procedure<step_actions, worldtube_system>>,
      Parallel::PhaseActions<
          Parallel::Phase::Register,
          tmpl::list<observers::Actions::RegisterSingletonWithObserverWriter<
                         Registration>,
                     Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<
          Parallel::Phase::Evolve,
          tmpl::list<step_actions, Actions::ObserveWorldtubeSolution,
                     ::Actions::AdvanceTime>>>;

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
