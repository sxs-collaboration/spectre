// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <vector>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Evolution/Executables/Cce/CharacteristicExtractBase.hpp"
#include "Evolution/Executables/GeneralizedHarmonic/GeneralizedHarmonicBase.hpp"
#include "Evolution/Systems/Cce/Actions/SendGhVarsToCce.hpp"
#include "Evolution/Systems/Cce/Callbacks/SendGhWorldtubeData.hpp"
#include "Evolution/Systems/Cce/Components/CharacteristicEvolution.hpp"
#include "Evolution/Systems/Cce/Components/WorldtubeBoundary.hpp"
#include "Evolution/Systems/Cce/Initialize/ConformalFactor.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Evolution/Systems/Cce/Initialize/InverseCubic.hpp"
#include "Evolution/Systems/Cce/Initialize/NoIncomingRadiation.hpp"
#include "Evolution/Systems/Cce/Initialize/RegisterInitializeJWithCharm.hpp"
#include "Evolution/Systems/Cce/Initialize/ZeroNonSmooth.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhInterfaceManager.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhLocalTimeStepping.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhLockstep.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/System.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/Cce/WorldtubeBufferUpdater.hpp"
#include "Evolution/Systems/Cce/WorldtubeDataManager.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/RegisterDerived.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/RegisterDerivedWithCharm.hpp"
#include "Evolution/Tags/Filter.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/CubicSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/LinearSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "Options/FactoryHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/MemoryMonitor/MemoryMonitor.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/CleanUpInterpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/ElementInitInterpPoints.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolationTargetReceiveVars.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorReceivePoints.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorReceiveVolumeData.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorRegisterElement.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/TryToInterpolate.hpp"
#include "ParallelAlgorithms/Interpolation/Events/Interpolate.hpp"
#include "ParallelAlgorithms/Interpolation/Events/InterpolateWithoutInterpComponent.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Interpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/KerrHorizon.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/Sphere.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

/// \cond
namespace Frame {
// IWYU pragma: no_forward_declare MathFunction
struct Inertial;
}  // namespace Frame
namespace Parallel {
template <typename Metavariables>
class CProxy_GlobalCache;
}  // namespace Parallel
/// \endcond

template <size_t VolumeDim, bool EvolveCcm>
struct EvolutionMetavars : public GeneralizedHarmonicTemplateBase<VolumeDim>,
                           public CharacteristicExtractDefaults<EvolveCcm> {
  static constexpr size_t volume_dim = VolumeDim;
  using cce_base = CharacteristicExtractDefaults<EvolveCcm>;
  using gh_base = GeneralizedHarmonicTemplateBase<volume_dim>;
  using typename gh_base::initialize_initial_data_dependent_quantities_actions;
  using cce_boundary_component = Cce::GhWorldtubeBoundary<EvolutionMetavars>;

  static constexpr bool local_time_stepping = gh_base::local_time_stepping;

  template <bool DuringSelfStart>
  struct CceWorldtubeTarget;

  using interpolator_source_vars =
      tmpl::list<::gr::Tags::SpacetimeMetric<DataVector, volume_dim>,
                 ::gh::Tags::Phi<DataVector, volume_dim>,
                 ::gh::Tags::Pi<DataVector, volume_dim>>;

  using dg_registration_list =
      tmpl::push_back<typename gh_base::dg_registration_list,
                      intrp::Actions::RegisterElementWithInterpolator>;

  using typename gh_base::system;

  using dg_step_choosers = tmpl::flatten<tmpl::list<
      StepChoosers::standard_step_choosers<system>,
      StepChoosers::standard_slab_choosers<system, local_time_stepping>>>;

  template <bool DuringSelfStart>
  using step_actions = tmpl::list<
      tmpl::conditional_t<
          DuringSelfStart,
          Cce::Actions::SendGhVarsToCce<CceWorldtubeTarget<true>>,
          Cce::Actions::SendGhVarsToCce<CceWorldtubeTarget<false>>>,
      evolution::dg::Actions::ComputeTimeDerivative<
          volume_dim, system, dg_step_choosers, local_time_stepping>,
      tmpl::conditional_t<
          local_time_stepping,
          tmpl::list<evolution::Actions::RunEventsAndDenseTriggers<
                         tmpl::list<evolution::dg::ApplyBoundaryCorrections<
                             local_time_stepping, system, volume_dim, true>>>,
                     evolution::dg::Actions::ApplyLtsBoundaryCorrections<
                         system, volume_dim, false>>,
          tmpl::list<
              evolution::dg::Actions::ApplyBoundaryCorrectionsToTimeDerivative<
                  system, volume_dim, false>,
              Actions::RecordTimeStepperData<system>,
              evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<>>,
              Actions::UpdateU<system>,
              dg::Actions::Filter<
                  Filters::Exponential<0>,
                  tmpl::list<gr::Tags::SpacetimeMetric<DataVector, volume_dim>,
                             gh::Tags::Pi<DataVector, volume_dim>,
                             gh::Tags::Phi<DataVector, volume_dim>>>>>>;

  using typename cce_base::cce_step_choosers;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = Options::add_factory_classes<
        typename gh_base::factory_creation::factory_classes,
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>, cce_step_choosers>>;
  };

  static constexpr bool override_functions_of_time = false;

  // initialization actions are the same as the default, with the single
  // addition of initializing the interpolation points (second-to-last action).
  using initialization_actions = tmpl::list<
      Initialization::Actions::InitializeItems<
          Initialization::TimeStepping<EvolutionMetavars, local_time_stepping>,
          evolution::dg::Initialization::Domain<volume_dim>,
          Initialization::TimeStepperHistory<EvolutionMetavars>>,
      Initialization::Actions::NonconservativeSystem<system>,
      Initialization::Actions::AddComputeTags<::Tags::DerivCompute<
          typename system::variables_tag,
          domain::Tags::InverseJacobian<volume_dim, Frame::ElementLogical,
                                        Frame::Inertial>,
          typename system::gradient_variables>>,
      gh::Actions::InitializeGhAnd3Plus1Variables<volume_dim>,
      Initialization::Actions::AddComputeTags<
          tmpl::push_back<StepChoosers::step_chooser_compute_tags<
              EvolutionMetavars, local_time_stepping>>>,
      ::evolution::dg::Initialization::Mortars<volume_dim, system>,
      evolution::Actions::InitializeRunEventsAndDenseTriggers,
      intrp::Actions::ElementInitInterpPoints<
          intrp::Tags::InterpPointInfo<EvolutionMetavars>>,
      Parallel::Actions::TerminatePhase>;

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
              tmpl::list<gh::Actions::SetInitialData,
                         gh::Actions::ReceiveNumericInitialData,
                         Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<
              Parallel::Phase::InitializeInitialDataDependentQuantities,
              initialize_initial_data_dependent_quantities_actions>,
          Parallel::PhaseActions<
              Parallel::Phase::InitializeTimeStepperHistory,
              SelfStart::self_start_procedure<step_actions<true>, system>>,
          Parallel::PhaseActions<Parallel::Phase::Register,
                                 tmpl::list<dg_registration_list,
                                            Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<
              Parallel::Phase::Evolve,
              tmpl::list<Actions::RunEventsAndTriggers, Actions::ChangeSlabSize,
                         step_actions<false>, Actions::AdvanceTime,
                         PhaseControl::Actions::ExecutePhaseChange>>>>>;

  template <bool DuringSelfStart>
  struct CceWorldtubeTarget
      : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::TimeStepId;

    static std::string name() {
      return DuringSelfStart ? "SelfStartCceWorldtubeTarget"
                             : "CceWorldtubeTarget";
    }
    using compute_items_on_source = tmpl::list<>;
    using compute_items_on_target = tmpl::list<>;
    using compute_target_points =
        intrp::TargetPoints::Sphere<CceWorldtubeTarget, ::Frame::Inertial>;
    using post_interpolation_callback = intrp::callbacks::SendGhWorldtubeData<
        Cce::CharacteristicEvolution<EvolutionMetavars>, CceWorldtubeTarget,
        DuringSelfStart, local_time_stepping>;
    using vars_to_interpolate_to_target = interpolator_source_vars;
    template <typename Metavariables>
    using interpolating_component = gh_dg_element_array;
  };

  using interpolation_target_tags =
      tmpl::list<CceWorldtubeTarget<false>, CceWorldtubeTarget<true>>;

  template <typename ParallelComponent>
  struct registration_list {
    using type = std::conditional_t<
        std::is_same_v<ParallelComponent, gh_dg_element_array>,
        dg_registration_list, tmpl::list<>>;
  };

  using component_list = tmpl::flatten<tmpl::list<
      observers::Observer<EvolutionMetavars>,
      observers::ObserverWriter<EvolutionMetavars>,
      mem_monitor::MemoryMonitor<EvolutionMetavars>,
      importers::ElementDataReader<EvolutionMetavars>, gh_dg_element_array,
      intrp::Interpolator<EvolutionMetavars>,
      tmpl::transform<interpolation_target_tags,
                      tmpl::bind<intrp::InterpolationTarget,
                                 tmpl::pin<EvolutionMetavars>, tmpl::_1>>,
      cce_boundary_component, Cce::CharacteristicEvolution<EvolutionMetavars>>>;

  static constexpr Options::String help{
      "Evolve the Einstein field equations using the Generalized Harmonic "
      "formulation\n"
      "with a coupled CCE evolution for asymptotic wave data output.\n"
      "The system shouldn't have black holes."};
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &setup_memory_allocation_failure_reporting,
    &disable_openblas_multithreading,
    &domain::creators::time_dependence::register_derived_with_charm,
    &domain::FunctionsOfTime::register_derived_with_charm,
    &domain::creators::register_derived_with_charm,
    &gh::BoundaryCorrections::register_derived_with_charm,
    &gh::ConstraintDamping::register_derived_with_charm,
    &Cce::register_initialize_j_with_charm<
        metavariables::evolve_ccm, metavariables::cce_boundary_component>,
    &register_derived_classes_with_charm<Cce::WorldtubeDataManager>,
    &register_derived_classes_with_charm<intrp::SpanInterpolator>,
    &register_factory_classes_with_charm<metavariables>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions, &enable_segfault_handler};
