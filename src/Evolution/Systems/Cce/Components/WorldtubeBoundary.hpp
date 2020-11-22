// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/Cce/Actions/InitializeWorldtubeBoundary.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"

namespace Cce {

/*!
 * \brief Generic base class for components that supply CCE worldtube boundary
 * data. See class specializations for specific worldtube boundary components.
 */
template <typename WorldtubeComponent, typename Metavariables>
struct WorldtubeComponentBase {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using initialize_action_list =
      tmpl::list<::Actions::SetupDataBox,
                 Actions::InitializeWorldtubeBoundary<WorldtubeComponent>,
                 Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  using initialization_tags =
      Parallel::get_initialization_tags<initialize_action_list>;

  using worldtube_boundary_computation_steps = tmpl::list<>;

  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        initialize_action_list>,
                 Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Evolve,
                                        worldtube_boundary_computation_steps>>;

  using const_global_cache_tag_list =
      Parallel::detail::get_const_global_cache_tags_from_pdal<
          phase_dependent_action_list>;

  using options = tmpl::list<>;

  static void initialize(Parallel::CProxy_GlobalCache<
                         Metavariables>& /*global_cache*/) noexcept {}

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>&
          global_cache) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    if (next_phase == Metavariables::Phase::Evolve) {
      Parallel::get_parallel_component<WorldtubeComponent>(local_cache)
          .start_phase(next_phase);
    }
  }
};

/*!
 * \brief Component that supplies CCE worldtube boundary data.
 *
 * \details The \ref DataBoxGroup associated with the worldtube boundary
 * component contains a data manager (e.g. `WorldtubeDataManager`) linked to
 * an H5 file. The data manager handles buffering and interpolating to desired
 * target time points when requested via the simple action
 * `BoundaryComputeAndSendToEvolution`, at which point it will send the required
 * collection of boundary quantities to the identified 'CharacteristicEvolution'
 * component. It is assumed that the simple action
 * `BoundaryComputeAndSendToEvolution` will only be called during the
 * `Evolve` phase.
 *
 * Uses const global tags:
 * - `Tags::LMax`
 *
 * `Metavariables` must contain:
 * - the `enum` `Phase` with at least `Initialization` and `Evolve` phases.
 * - a type alias `cce_boundary_communication_tags` for the set of tags to send
 * from the worldtube to the evolution component. This will typically be
 * `Cce::Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>`.
 */
template <class Metavariables>
struct H5WorldtubeBoundary
    : public WorldtubeComponentBase<H5WorldtubeBoundary<Metavariables>,
                                    Metavariables> {
  using base_type =
      WorldtubeComponentBase<H5WorldtubeBoundary<Metavariables>, Metavariables>;
  using base_type::execute_next_phase;
  using base_type::initialize;
  using typename base_type::chare_type;
  using typename base_type::const_global_cache_tag_list;
  using typename base_type::initialization_tags;
  using typename base_type::metavariables;
  using typename base_type::options;
  using typename base_type::phase_dependent_action_list;
  using end_time_tag = Tags::EndTimeFromFile;
};
/*!
 * \brief Component that supplies CCE worldtube boundary data sourced from a
 * running GH system.
 *
 * \details The \ref DataBoxGroup associated with the worldtube boundary
 * component contains an interface manager (derived from
 * `Cce::GhWorldtubeInterfaceManager`) that stores and provides the data
 * received from the GH system. The data manager handles buffering
 * and interpolating to desired target time points when requested via the simple
 * action `Cce::Actions::BoundaryComputeAndSendToEvolution`, at which point it
 * will send the required collection of boundary quantities to the identified
 * `CharacteristicEvolution` component. It is assumed that the simple action
 * `Cce::Actions::BoundaryComputeAndSendToEvolution` will only be called during
 * the `Evolve` phase.
 *
 * Uses const global tags:
 * - `InitializationTags::LMax`
 * - `InitializationTags::ExtractionRadius`
 *
 * `Metavariables` must contain:
 * - the `enum` `Phase` with at least `Initialization` and `Evolve` phases.
 * - a type alias `cce_boundary_communication_tags` for the set of tags to send
 * from the worldtube to the evolution component. This will typically be
 * `Cce::Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>`.
 */
template <class Metavariables>
struct GhWorldtubeBoundary
    : public WorldtubeComponentBase<GhWorldtubeBoundary<Metavariables>,
                                    Metavariables> {
  using base_type =
      WorldtubeComponentBase<GhWorldtubeBoundary<Metavariables>, Metavariables>;
  using base_type::execute_next_phase;
  using base_type::initialize;
  using typename base_type::chare_type;
  using typename base_type::const_global_cache_tag_list;
  using typename base_type::initialization_tags;
  using typename base_type::metavariables;
  using typename base_type::options;
  using typename base_type::phase_dependent_action_list;
  using end_time_tag = Tags::NoEndTime;
};
}  // namespace Cce
