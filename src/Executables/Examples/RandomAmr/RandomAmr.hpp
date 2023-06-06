// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Factory1D.hpp"
#include "Domain/Creators/Factory2D.hpp"
#include "Domain/Creators/Factory3D.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Executables/Examples/RandomAmr/InitializeDomain.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/CheckpointAndExitAfterWallclock.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Amr/Actions/CollectDataFromChildren.hpp"
#include "ParallelAlgorithms/Amr/Actions/Component.hpp"
#include "ParallelAlgorithms/Amr/Actions/CreateChild.hpp"
#include "ParallelAlgorithms/Amr/Actions/Initialize.hpp"
#include "ParallelAlgorithms/Amr/Actions/SendAmrDiagnostics.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/Amr/Criteria/DriveToTarget.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Random.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Tags/Criteria.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct DummySystem {};
}  // namespace

/// \page RandomAmrExecutablePage RandomAmr Executable
/// The RandomAmr executable is being used to develop the mechanics of
/// adaptive mesh refinement.
///
/// See RandomAmrMetavars for a description of the metavariables of this
/// executable.

/// \brief The metavariables for the RandomAmr executable
template <size_t Dim>
struct RandomAmrMetavars {
  static constexpr size_t volume_dim = Dim;
  using system = DummySystem;

  static constexpr Options::String help{
      "Test anisotropic refinement by randomly refining a grid.\n"};

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
        tmpl::pair<amr::Criterion,
                   tmpl::list<amr::Criteria::DriveToTarget<volume_dim>,
                              amr::Criteria::Random>>,
        tmpl::pair<
            PhaseChange,
            tmpl::list<
                PhaseControl::VisitAndReturn<
                    Parallel::Phase::EvaluateAmrCriteria>,
                PhaseControl::VisitAndReturn<Parallel::Phase::AdjustDomain>,
                PhaseControl::VisitAndReturn<Parallel::Phase::CheckDomain>,
                PhaseControl::CheckpointAndExitAfterWallclock>>,
        tmpl::pair<Trigger, tmpl::list<Triggers::Always>>>;
  };

  using const_global_cache_tags = tmpl::list<amr::Criteria::Tags::Criteria>;

  static constexpr auto default_phase_order =
      std::array{Parallel::Phase::Initialization, Parallel::Phase::CheckDomain,
                 Parallel::Phase::Evolve, Parallel::Phase::Exit};

  using dg_element_array = DgElementArray<
      RandomAmrMetavars,
      tmpl::list<
          Parallel::PhaseActions<
              Parallel::Phase::Initialization,
              tmpl::list<Initialization::Actions::InitializeItems<
                             amr::Initialization::Domain<volume_dim>,
                             amr::Initialization::Initialize<volume_dim>>,
                         Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<Parallel::Phase::CheckDomain,
                                 tmpl::list<::amr::Actions::SendAmrDiagnostics,
                                            Parallel::Actions::TerminatePhase>>,

          Parallel::PhaseActions<
              Parallel::Phase::Evolve,
              tmpl::list<PhaseControl::Actions::ExecutePhaseChange>>>>;

  using component_list =
      tmpl::list<amr::Component<RandomAmrMetavars>, dg_element_array>;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};

template <typename Component>
void register_amr_callbacks() {
  using ArrayIndex = typename Component::array_index;
  register_classes_with_charm(
      tmpl::list<
          Parallel::SimpleActionCallback<
              amr::Actions::CreateChild,
              CProxy_AlgorithmSingleton<amr::Component<metavariables>, int>,
              CProxy_AlgorithmArray<Component, ArrayIndex>, ArrayIndex,
              std::vector<ArrayIndex>, size_t>,
          Parallel::SimpleActionCallback<
              amr::Actions::SendDataToChildren,
              CProxyElement_AlgorithmArray<Component, ArrayIndex>,
              std::vector<ArrayIndex>>,
          Parallel::SimpleActionCallback<
              amr::Actions::CollectDataFromChildren,
              CProxyElement_AlgorithmArray<Component, ArrayIndex>, ArrayIndex,
              std::deque<ArrayIndex>>>{});
}

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &domain::creators::register_derived_with_charm,
    &register_factory_classes_with_charm<metavariables>,
    &register_amr_callbacks<typename metavariables::dg_element_array>};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions, &enable_segfault_handler};
