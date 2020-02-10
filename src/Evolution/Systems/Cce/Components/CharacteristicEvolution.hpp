// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Evolution/Systems/Cce/Actions/InitializeCharacteristicEvolutionScri.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeCharacteristicEvolutionTime.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeCharacteristicEvolutionVariables.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {

/*!
 * \brief The component for handling the CCE evolution and waveform output.
 *
 * \details The \ref DataBoxGroup associated with the CharacteristicEvolution
 * will contain many spin-weighted volume tags associated with the ongoing CCE
 * computation, as well as storage for the boundary values and quantities
 * related to managing the evolution.
 *
 * Metavariables requirements:
 * - Phases:
 *  - `Initialization`
 *  - `Evolve`
 * - Type aliases:
 *  - `evolved_coordinates_variables_tag`: A `Tags::Variables` with real-valued
 * tensors associated with coordinates that must be evolved.
 *  - `evolved_swsh_tag`: The spin-weighted quantity to be evolved (typically
 * `BondiJ`).
 *  - `evolved_swsh_dt_tag`: The spin-weighed quantity associated that is to act
 * as the time derivative to evolve `evolved_swsh_tag` (typically `BondiH`).
 *  - `cce_boundary_communication_tags`: A typelist of tags that will be
 * communicated between the worldtube boundary component and the extraction
 * component (typically
 * `Cce::Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>`)
 *  - `cce_gauge_boundary_tags`: A typelist of tags that will be derived via
 * `GaugeAdjustedBoundaryValue` and corresponding gauge utilities
 *  - `cce_integrand_tags`: A typelist of tags needed as inputs to the
 * linear solve. Obtainable from the metafunction
 * `Cce::integrand_terms_to_compute_for_bondi_variable`.
 *  - `cce_integration_independent_tags`: A typelist of tags that are to be
 * computed and stored for each hypersurface iteration, but have no dependencies
 * on the intermediate hypersurface steps (typically
 * `Cce::pre_computation_steps`).
 *  - `cce_temporary_equations_tags`: A typelist of temporary buffers maintained
 * for intermediate steps in the integrand equations. Obtainable from the
 * metafunction `Cce::integrand_terms_to_compute_for_bondi_variable`.
 *  - `cce_pre_swsh_derivatives_tags`: A typelist of inputs to spin-weighted
 * derivative calculations to compute and cache for intermediate steps of the
 * CCE calculation. (typically `Cce::all_pre_swsh_derivative_tags`)
 *  - `cce_swsh_derivative_tags`: A typelist of spin-weighted derivatives to
 * compute and cache for intermediate steps of the CCE calculation. (typically
 * `Cce::all_swsh_derivative_tags`)
 *  - `cce_transform_buffer_tags`: A typelist of spin-weighted spherical
 * harmonic transform modes used to compute the spin-weighted derivatives in the
 * modal representation. (typically `Cce::all_transform_buffer_tags`).
 *  - `cce_angular_coordinate_tags`: A typelist of real-valued angular
 * coordinates that are not evolved.
 *  - `cce_scri_tags` - the tags of quantities to compute at scri+
 */
template <class Metavariables>
struct CharacteristicEvolution {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;

  using initialize_action_list =
      tmpl::list<Actions::InitializeCharacteristicEvolutionVariables,
                 Actions::InitializeCharacteristicEvolutionTime,
                 Actions::InitializeCharacteristicEvolutionScri,
                 Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  using initialization_tags =
      Parallel::get_initialization_tags<initialize_action_list>;

  using extract_action_list = tmpl::list<::Actions::AdvanceTime>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             initialize_action_list>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Evolve,
                             extract_action_list>>;

  using const_global_cache_tag_list =
      Parallel::get_const_global_cache_tags_from_actions<
          phase_dependent_action_list>;

  static void initialize(Parallel::CProxy_ConstGlobalCache<
                         Metavariables>& /*global_cache*/) noexcept {}

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      const Parallel::CProxy_ConstGlobalCache<Metavariables>&
          global_cache) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    if (next_phase == Metavariables::Phase::RegisterWithObserver or
        next_phase == Metavariables::Phase::Evolve) {
      Parallel::get_parallel_component<CharacteristicEvolution<Metavariables>>(
          local_cache)
          .start_phase(next_phase);
    }
  }
};
}  // namespace Cce
