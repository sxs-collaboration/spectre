// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Evolution/Systems/Cce/Actions/InitializeKleinGordonFirstHypersurface.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeKleinGordonVariables.hpp"
#include "Evolution/Systems/Cce/Actions/PrecomputeKleinGordonSourceVariables.hpp"
#include "Evolution/Systems/Cce/Components/CharacteristicEvolution.hpp"
#include "Evolution/Systems/Cce/KleinGordonSource.hpp"
#include "Evolution/Systems/Cce/KleinGordonSystem.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Phase.hpp"
#include "Time/Actions/CleanHistory.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {

/*!
 * \brief The component for handling the CCE evolution for the Klein-Gordon
 * system coupled with General Relativity.
 *
 * \details The \ref DataBoxGroup associated with
 * KleinGordonCharacteristicEvolution will contain all the tags of
 * CharacteristicEvolution, with additional tags related to the scalar field.
 *
 * Metavariables requirements:
 * - Phases:
 *  - `Initialization`
 *  - `Evolve`
 * - Modified type aliases in comparison to CharacteristicEvolution:
 *  - `evolved_swsh_tags`: The spin-weighted quantities to be evolved (
 * `KleinGordonPsi` and `BondiJ`).
 *  - `evolved_swsh_dt_tags`: The spin-weighed quantities associated that are to
 * act as the time derivative to evolve `evolved_swsh_tags` (`KleinGordonPi` and
 * `BondiH`).
 * - Additional type aliases related to the scalar field:
 *  - `klein_gordon_boundary_communication_tags`:  A typelist of tags that will
 * be communicated between the worldtube boundary component and the extraction
 * component (`Cce::Tags::klein_gordon_worldtube_boundary_tags`).
 *  - `klein_gordon_gauge_boundary_tags`: A typelist of tags that will be
 * derived via `GaugeAdjustedBoundaryValue` and corresponding gauge utilities
 *  - `klein_gordon_scri_tags`: the tags of quantities to compute at scri+
 */
template <class Metavariables>
struct KleinGordonCharacteristicEvolution
    : CharacteristicEvolution<Metavariables> {
  using metavariables = Metavariables;
  static constexpr bool evolve_ccm = Metavariables::evolve_ccm;
  using cce_system = Cce::KleinGordonSystem<evolve_ccm>;

  using cce_base = CharacteristicEvolution<Metavariables>;
  using initialize_action_list = tmpl::append<
      tmpl::list<Actions::InitializeKleinGordonVariables<Metavariables>>,
      typename cce_base::initialize_action_list>;

  template <typename BondiTag>
  using hypersurface_computation =
      typename cce_base::template hypersurface_computation<BondiTag>;

  using klein_gordon_hypersurface_computation = tmpl::list<
      ::Actions::MutateApply<GaugeAdjustedBoundaryValue<Tags::KleinGordonPi>>,
      Actions::CalculateIntegrandInputsForTag<Tags::KleinGordonPi>,
      tmpl::transform<
          integrand_terms_to_compute_for_bondi_variable<Tags::KleinGordonPi>,
          tmpl::bind<::Actions::MutateApply,
                     tmpl::bind<ComputeBondiIntegrand, tmpl::_1>>>>;

  using simple_tags_from_options =
      Parallel::get_simple_tags_from_options<initialize_action_list>;

  using typename cce_base::compute_scri_quantities_and_observe;

  using self_start_extract_action_list = tmpl::list<
      Actions::RequestBoundaryData<
          typename Metavariables::cce_boundary_component,
          KleinGordonCharacteristicEvolution<Metavariables>>,
      Actions::ReceiveWorldtubeData<
          Metavariables,
          typename Metavariables::cce_boundary_communication_tags>,
      Actions::ReceiveWorldtubeData<
          Metavariables,
          typename Metavariables::klein_gordon_boundary_communication_tags>,
      // note that the initialization will only actually happen on the
      // iterations immediately following restarts
      Actions::InitializeFirstHypersurface<
          evolve_ccm, typename Metavariables::cce_boundary_component>,
      Actions::InitializeKleinGordonFirstHypersurface,
      tmpl::conditional_t<
          tt::is_a_v<AnalyticWorldtubeBoundary,
                     typename Metavariables::cce_boundary_component>,
          Actions::UpdateGauge<false>, Actions::UpdateGauge<evolve_ccm>>,
      Actions::PrecomputeGlobalCceDependencies,
      tmpl::conditional_t<evolve_ccm,
                          Actions::CalculatePsi0AndDerivAtInnerBoundary,
                          tmpl::list<>>,
      Actions::PrecomputeKleinGordonSourceVariables,
      tmpl::transform<
          bondi_hypersurface_step_tags,
          tmpl::bind<::Actions::MutateApply,
                     tmpl::bind<ComputeKleinGordonSource, tmpl::_1>>>,
      tmpl::transform<bondi_hypersurface_step_tags,
                      tmpl::bind<hypersurface_computation, tmpl::_1>>,
      klein_gordon_hypersurface_computation,
      Actions::FilterSwshVolumeQuantity<Tags::BondiH>,
      Actions::FilterSwshVolumeQuantity<Tags::KleinGordonPi>,
      ::Actions::MutateApply<
          CalculateScriPlusValue<::Tags::dt<Tags::InertialRetardedTime>>>,
      Actions::CalculateScriInputs,
      tmpl::transform<typename metavariables::cce_scri_tags,
                      tmpl::bind<::Actions::MutateApply,
                                 tmpl::bind<CalculateScriPlusValue, tmpl::_1>>>,
      ::Actions::RecordTimeStepperData<cce_system>,
      ::Actions::UpdateU<cce_system>>;

  using extract_action_list = tmpl::list<
      Actions::RequestBoundaryData<
          typename Metavariables::cce_boundary_component,
          KleinGordonCharacteristicEvolution<Metavariables>>,
      ::Actions::Label<CceEvolutionLabelTag>,
      tmpl::conditional_t<evolve_ccm, tmpl::list<>,
                          evolution::Actions::RunEventsAndTriggers<
                              Metavariables::local_time_stepping>>,
      Actions::ReceiveWorldtubeData<
          Metavariables,
          typename Metavariables::cce_boundary_communication_tags>,
      Actions::ReceiveWorldtubeData<
          Metavariables,
          typename Metavariables::klein_gordon_boundary_communication_tags>,
      Actions::InitializeFirstHypersurface<
          evolve_ccm, typename Metavariables::cce_boundary_component>,
      Actions::InitializeKleinGordonFirstHypersurface,
      tmpl::conditional_t<
          tt::is_a_v<AnalyticWorldtubeBoundary,
                     typename Metavariables::cce_boundary_component>,
          Actions::UpdateGauge<false>, Actions::UpdateGauge<evolve_ccm>>,
      Actions::PrecomputeGlobalCceDependencies,
      tmpl::conditional_t<evolve_ccm,
                          Actions::CalculatePsi0AndDerivAtInnerBoundary,
                          tmpl::list<>>,
      Actions::PrecomputeKleinGordonSourceVariables,
      tmpl::transform<
          bondi_hypersurface_step_tags,
          tmpl::bind<::Actions::MutateApply,
                     tmpl::bind<ComputeKleinGordonSource, tmpl::_1>>>,
      tmpl::transform<bondi_hypersurface_step_tags,
                      tmpl::bind<hypersurface_computation, tmpl::_1>>,
      klein_gordon_hypersurface_computation,
      Actions::FilterSwshVolumeQuantity<Tags::BondiH>,
      Actions::FilterSwshVolumeQuantity<Tags::KleinGordonPi>,
      compute_scri_quantities_and_observe,
      ::Actions::RecordTimeStepperData<cce_system>,
      ::Actions::UpdateU<cce_system>,
      ::Actions::ChangeStepSize<typename Metavariables::cce_step_choosers>,
      ::Actions::CleanHistory<cce_system, false>,
      // We cannot know our next step for certain until after we've performed
      // step size selection, as we may need to reject a step.
      Actions::RequestNextBoundaryData<
          typename Metavariables::cce_boundary_component,
          KleinGordonCharacteristicEvolution<Metavariables>>,
      ::Actions::AdvanceTime, Actions::ExitIfEndTimeReached,
      ::Actions::Goto<CceEvolutionLabelTag>>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             initialize_action_list>,
      Parallel::PhaseActions<Parallel::Phase::InitializeTimeStepperHistory,
                             SelfStart::self_start_procedure<
                                 self_start_extract_action_list, cce_system>>,
      Parallel::PhaseActions<Parallel::Phase::Evolve, extract_action_list>>;

  static void initialize(
      Parallel::CProxy_GlobalCache<Metavariables>& /*global_cache*/) {}

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<
        KleinGordonCharacteristicEvolution<Metavariables>>(local_cache)
        .start_phase(next_phase);
  }
};
}  // namespace Cce
