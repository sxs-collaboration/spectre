// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <vector>

#include "Evolution/Actions/RunEventsAndTriggers.hpp"
#include "Evolution/Executables/GeneralizedHarmonic/GeneralizedHarmonicBase.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/MemoryMonitor/MemoryMonitor.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "Parallel/Protocols/RegistrationMetavariables.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/GaugeWave.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ProtocolHelpers.hpp"

template <size_t VolumeDim>
struct EvolutionMetavars : public GeneralizedHarmonicTemplateBase<VolumeDim> {
  static constexpr size_t volume_dim = VolumeDim;
  using gh_base = GeneralizedHarmonicTemplateBase<volume_dim>;
  using typename gh_base::const_global_cache_tags;
  using typename gh_base::dg_registration_list;
  using initialization_actions =
      typename gh_base::template initialization_actions<EvolutionMetavars,
                                                        false>;
  using typename gh_base::initialize_initial_data_dependent_quantities_actions;
  using typename gh_base::observed_reduction_data_tags;
  using typename gh_base::system;

  using step_actions = typename gh_base::template step_actions<tmpl::list<>>;

  using gh_dg_element_array = DgElementArray<
      EvolutionMetavars,
      tmpl::flatten<tmpl::list<
          Parallel::PhaseActions<Parallel::Phase::Initialization,
                                 initialization_actions>,
          Parallel::PhaseActions<
              Parallel::Phase::RegisterWithElementDataReader,
              tmpl::list<importers::Actions::RegisterWithElementDataReader,
                         Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<
              Parallel::Phase::ImportInitialData,
              tmpl::list<
                  gh::Actions::SetInitialData,
                  tmpl::conditional_t<VolumeDim == 3,
                                      gh::Actions::ReceiveNumericInitialData,
                                      tmpl::list<>>,
                  Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<
              Parallel::Phase::InitializeInitialDataDependentQuantities,
              initialize_initial_data_dependent_quantities_actions>,
          Parallel::PhaseActions<
              Parallel::Phase::InitializeTimeStepperHistory,
              SelfStart::self_start_procedure<step_actions, system>>,
          Parallel::PhaseActions<Parallel::Phase::Register,
                                 tmpl::list<dg_registration_list,
                                            Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<
              Parallel::Phase::Evolve,
              tmpl::list<evolution::Actions::RunEventsAndTriggers,
                         Actions::ChangeSlabSize, step_actions,
                         Actions::AdvanceTime,
                         PhaseControl::Actions::ExecutePhaseChange>>>>>;

  struct registration
      : tt::ConformsTo<Parallel::protocols::RegistrationMetavariables> {
    using element_registrars =
        tmpl::map<tmpl::pair<gh_dg_element_array, dg_registration_list>>;
  };

  using component_list =
      tmpl::flatten<tmpl::list<observers::Observer<EvolutionMetavars>,
                               observers::ObserverWriter<EvolutionMetavars>,
                               mem_monitor::MemoryMonitor<EvolutionMetavars>,
                               importers::ElementDataReader<EvolutionMetavars>,
                               gh_dg_element_array>>;

  static constexpr Options::String help{
      "Evolve the Einstein field equations using the Generalized Harmonic "
      "formulation\n"};
};
