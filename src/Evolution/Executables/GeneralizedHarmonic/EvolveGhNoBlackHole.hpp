// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <vector>

#include "Evolution/Actions/RunEventsAndTriggers.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "Evolution/Executables/GeneralizedHarmonic/GeneralizedHarmonicBase.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/ArrayCollection/DgElementCollection.hpp"
#include "Parallel/MemoryMonitor/MemoryMonitor.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "Parallel/Protocols/RegistrationMetavariables.hpp"
#include "ParallelAlgorithms/Amr/Projectors/CopyFromCreatorOrLeaveAsIs.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/GaugeWave.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/ChangeSlabSize/Action.hpp"
#include "Time/ChangeSlabSize/Tags.hpp"
#include "Time/Tags/StepperErrors.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ProtocolHelpers.hpp"

template <size_t VolumeDim, bool UseLts>
struct EvolutionMetavars
    : public GeneralizedHarmonicTemplateBase<VolumeDim, UseLts> {
  static constexpr size_t volume_dim = VolumeDim;
  using gh_base = GeneralizedHarmonicTemplateBase<volume_dim, UseLts>;
  using typename gh_base::const_global_cache_tags;
  using typename gh_base::dg_registration_list;
  using initialization_actions =
      typename gh_base::template initialization_actions<EvolutionMetavars,
                                                        false>;
  using typename gh_base::initialize_initial_data_dependent_quantities_actions;
  using typename gh_base::observed_reduction_data_tags;
  using typename gh_base::system;
  static constexpr bool local_time_stepping = gh_base::local_time_stepping;
  static constexpr bool use_dg_element_collection =
      gh_base::use_dg_element_collection;

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
          Parallel::PhaseActions<Parallel::Phase::CheckDomain,
                                 tmpl::list<::amr::Actions::SendAmrDiagnostics,
                                            Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<
              Parallel::Phase::Evolve,
              tmpl::list<::evolution::Actions::RunEventsAndTriggers<
                             local_time_stepping>,
                         Actions::ChangeSlabSize, step_actions,
                         Actions::AdvanceTime,
                         PhaseControl::Actions::ExecutePhaseChange>>>>>;

  struct amr : tt::ConformsTo<::amr::protocols::AmrMetavariables> {
    using element_array = gh_dg_element_array;

    using projectors = tmpl::list<
        Initialization::ProjectTimeStepping<volume_dim>,
        evolution::dg::Initialization::ProjectDomain<volume_dim>,
        Initialization::ProjectTimeStepperHistory<EvolutionMetavars>,
        ::amr::projectors::ProjectVariables<volume_dim,
                                            typename system::variables_tag>,
        evolution::dg::Initialization::ProjectMortars<EvolutionMetavars>,
        evolution::Actions::ProjectRunEventsAndDenseTriggers,
        ::amr::projectors::DefaultInitialize<
            Initialization::Tags::InitialTimeDelta,
            Initialization::Tags::InitialSlabSize<local_time_stepping>,
            ::domain::Tags::InitialExtents<volume_dim>,
            ::domain::Tags::InitialRefinementLevels<volume_dim>,
            evolution::dg::Tags::Quadrature,
            Tags::StepperErrors<typename system::variables_tag>,
            SelfStart::Tags::InitialValue<typename system::variables_tag>,
            SelfStart::Tags::InitialValue<Tags::TimeStep>,
            SelfStart::Tags::InitialValue<Tags::Next<Tags::TimeStep>>,
            evolution::dg::Tags::BoundaryData<volume_dim>>,
        ::amr::projectors::CopyFromCreatorOrLeaveAsIs<
            Tags::ChangeSlabSize::NumberOfExpectedMessages,
            Tags::ChangeSlabSize::NewSlabSize>>;
  };

  struct registration
      : tt::ConformsTo<Parallel::protocols::RegistrationMetavariables> {
    using element_registrars =
        tmpl::map<tmpl::pair<gh_dg_element_array, dg_registration_list>>;
  };

  using component_list =
      tmpl::flatten<tmpl::list<::amr::Component<EvolutionMetavars>,
                               observers::Observer<EvolutionMetavars>,
                               observers::ObserverWriter<EvolutionMetavars>,
                               mem_monitor::MemoryMonitor<EvolutionMetavars>,
                               importers::ElementDataReader<EvolutionMetavars>,
                               gh_dg_element_array>>;

  static constexpr Options::String help{
      "Evolve the Einstein field equations using the Generalized Harmonic "
      "formulation\n"};
};
