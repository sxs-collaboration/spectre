// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "Domain/Creators/AlignedLattice.hpp"
#include "Domain/RadiallyCompressedCoordinates.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/RunEventsAndTriggers.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Elliptic/Executables/Solver.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/Actions/InitializeEffectiveSource.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/AmrCriteria/RefineAtPuncture.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/AnalyticData/CircularOrbit.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/BoundaryConditions/Angular.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/BoundaryConditions/Sommerfeld.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/Events/ObserveSelfForce.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/FirstOrderSystem.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/Tags.hpp"
#include "Elliptic/Triggers/Factory.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/ExecutePhaseChange.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Protocols/RegistrationMetavariables.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Amr/Actions/SendAmrDiagnostics.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Factory.hpp"
#include "ParallelAlgorithms/Amr/Protocols/AmrMetavariables.hpp"
#include "ParallelAlgorithms/Events/Completion.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/ElementsAllocator.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"
#include "PointwiseFunctions/MathFunctions/Factory.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP

struct Metavariables {
  static constexpr Options::String help{
      "Find the self-force for a scalar charge by solving a 2D elliptic "
      "problem."};

  static constexpr size_t volume_dim = 2;
  using system = ScalarSelfForce::FirstOrderSystem;
  using solver = elliptic::Solver<Metavariables, volume_dim, system>;

  using observe_fields = tmpl::append<
      typename system::primal_fields, typename system::background_fields,
      tmpl::list<ScalarSelfForce::Tags::SingularField,
                 ScalarSelfForce::Tags::BoyerLindquistRadius>,
      typename solver::observe_fields,
      tmpl::list<domain::Tags::Coordinates<volume_dim, Frame::Inertial>>>;
  using observer_compute_tags =
      tmpl::list<::Events::Tags::ObserverMeshCompute<volume_dim>,
                 ::Events::Tags::ObserverDetInvJacobianCompute<
                     Frame::ElementLogical, Frame::Inertial>>;

  // Collect all items to store in the cache.
  using const_global_cache_tags = tmpl::list<>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<DomainCreator<volume_dim>,
                   tmpl::list<domain::creators::AlignedLattice<2>>>,
        tmpl::pair<elliptic::analytic_data::Background,
                   tmpl::list<ScalarSelfForce::AnalyticData::CircularOrbit>>,
        tmpl::pair<elliptic::analytic_data::InitialGuess,
                   tmpl::list<ScalarSelfForce::AnalyticData::CircularOrbit>>,
        tmpl::pair<elliptic::analytic_data::AnalyticSolution, tmpl::list<>>,
        tmpl::pair<elliptic::BoundaryConditions::BoundaryCondition<volume_dim>,
                   tmpl::list<ScalarSelfForce::BoundaryConditions::Angular,
                              ScalarSelfForce::BoundaryConditions::Sommerfeld>>,
        tmpl::pair<
            ::amr::Criterion,
            tmpl::push_back<::amr::Criteria::standard_criteria<
                                volume_dim, typename system::primal_fields>,
                            ScalarSelfForce::AmrCriteria::RefineAtPuncture>>,
        tmpl::pair<Event,
                   tmpl::flatten<tmpl::list<
                       Events::Completion,
                       ScalarSelfForce::Events::ObserveSelfForce<
                           amr::Tags::IsFinestGrid>,
                       dg::Events::field_observations<
                           volume_dim, observe_fields, observer_compute_tags,
                           amr::Tags::IsFinestGrid>>>>,
        tmpl::pair<Trigger, elliptic::Triggers::all_triggers<
                                ::amr::OptionTags::AmrGroup>>,
        tmpl::pair<
            PhaseChange,
            tmpl::list<
                // Phases for AMR
                PhaseControl::VisitAndReturn<
                    Parallel::Phase::EvaluateAmrCriteria>,
                PhaseControl::VisitAndReturn<Parallel::Phase::AdjustDomain>,
                PhaseControl::VisitAndReturn<Parallel::Phase::UpdateSections>,
                PhaseControl::VisitAndReturn<Parallel::Phase::CheckDomain>>>>;
  };

  // Collect all reduction tags for observers
  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::push_back<
          tmpl::at<typename factory_creation::factory_classes, Event>, solver>>;

  using initialization_actions =
      tmpl::push_back<typename solver::initialization_actions,
                      Parallel::Actions::TerminatePhase>;

  using register_actions =
      tmpl::push_back<typename solver::register_actions,
                      observers::Actions::RegisterEventsWithObservers>;

  using solve_actions = typename solver::template solve_actions<tmpl::list<>>;

  using dg_element_array = elliptic::DgElementArray<
      Metavariables,
      tmpl::list<
          Parallel::PhaseActions<
              Parallel::Phase::Initialization,
              tmpl::replace<initialization_actions,
                            elliptic::Actions::InitializeFixedSources<
                                system, typename solver::background_tag>,
                            ScalarSelfForce::Actions::InitializeEffectiveSource<
                                system, typename solver::background_tag>>>,
          Parallel::PhaseActions<
              Parallel::Phase::Register,
              tmpl::push_back<register_actions,
                              Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<
              Parallel::Phase::Restart,
              tmpl::push_back<register_actions,
                              Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<Parallel::Phase::Solve, solve_actions>,
          Parallel::PhaseActions<
              Parallel::Phase::CheckDomain,
              tmpl::list<::amr::Actions::SendAmrDiagnostics,
                         Parallel::Actions::TerminatePhase>>>,
      LinearSolver::multigrid::ElementsAllocator<
          volume_dim, typename solver::multigrid::options_group>>;

  struct amr : tt::ConformsTo<::amr::protocols::AmrMetavariables> {
    using element_array = dg_element_array;
    using projectors =
        tmpl::replace<typename solver::amr_projectors,
                      elliptic::Actions::InitializeFixedSources<
                          system, typename solver::background_tag>,
                      ScalarSelfForce::Actions::InitializeEffectiveSource<
                          system, typename solver::background_tag>>;
    static constexpr bool keep_coarse_grids = true;
    static constexpr bool p_refine_only_in_event = false;
  };

  struct registration
      : tt::ConformsTo<Parallel::protocols::RegistrationMetavariables> {
    using element_registrars =
        tmpl::map<tmpl::pair<dg_element_array, register_actions>>;
  };

  // Specify all parallel components that will execute actions at some point.
  using component_list = tmpl::flatten<
      tmpl::list<dg_element_array, typename solver::component_list,
                 observers::Observer<Metavariables>,
                 observers::ObserverWriter<Metavariables>>>;

  static constexpr std::array<Parallel::Phase, 6> default_phase_order{
      {Parallel::Phase::Initialization, Parallel::Phase::Register,
       Parallel::Phase::UpdateSections, Parallel::Phase::CheckDomain,
       Parallel::Phase::Solve, Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};
/// \endcond
