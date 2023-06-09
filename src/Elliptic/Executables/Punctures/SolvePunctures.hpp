// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "Domain/Creators/Factory3D.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/RadiallyCompressedCoordinates.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Elliptic/Executables/NonlinearEllipticSolver.hpp"
#include "Elliptic/SubdomainPreconditioners/RegisterDerived.hpp"
#include "Elliptic/Systems/Punctures/BoundaryConditions/Flatness.hpp"
#include "Elliptic/Systems/Punctures/FirstOrderSystem.hpp"
#include "Elliptic/Triggers/Factory.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/Events/ObserveVolumeIntegrals.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/ElementsAllocator.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Tags.hpp"
#include "PointwiseFunctions/AnalyticData/Punctures/MultiplePunctures.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Punctures/Flatness.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"
#include "PointwiseFunctions/Punctures/AdmIntegrals.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
struct Metavariables {
  static constexpr Options::String help{"Solve for puncture initial data"};

  static constexpr size_t volume_dim = 3;
  using system = Punctures::FirstOrderSystem;
  using solver = elliptic::nonlinear_solver::Solver<Metavariables>;

  using observe_integral_fields =
      tmpl::list<Punctures::Tags::AdmMassIntegrandCompute>;
  using observe_fields = tmpl::append<
      typename system::primal_fields, typename system::background_fields,
      observe_integral_fields,
      tmpl::list<domain::Tags::Coordinates<volume_dim, Frame::Inertial>,
                 domain::Tags::RadiallyCompressedCoordinatesCompute<
                     volume_dim, Frame::Inertial>>>;
  using observer_compute_tags =
      tmpl::list<::Events::Tags::ObserverMeshCompute<volume_dim>,
                 ::Events::Tags::ObserverDetInvJacobianCompute<
                     Frame::ElementLogical, Frame::Inertial>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
        tmpl::pair<elliptic::analytic_data::Background,
                   tmpl::list<Punctures::AnalyticData::MultiplePunctures>>,
        tmpl::pair<elliptic::analytic_data::InitialGuess,
                   tmpl::list<Punctures::Solutions::Flatness>>,
        tmpl::pair<elliptic::analytic_data::AnalyticSolution, tmpl::list<>>,
        tmpl::pair<elliptic::BoundaryConditions::BoundaryCondition<volume_dim>,
                   tmpl::list<Punctures::BoundaryConditions::Flatness>>,
        tmpl::pair<
            Event,
            tmpl::flatten<tmpl::list<
                Events::Completion,
                dg::Events::field_observations<
                    volume_dim, typename solver::nonlinear_solver_iteration_id,
                    observe_fields, observer_compute_tags,
                    LinearSolver::multigrid::Tags::IsFinestGrid>,
                dg::Events::ObserveVolumeIntegrals<
                    volume_dim, typename solver::nonlinear_solver_iteration_id,
                    observe_integral_fields, observer_compute_tags,
                    LinearSolver::multigrid::Tags::IsFinestGrid>>>>,
        tmpl::pair<Trigger,
                   elliptic::Triggers::all_triggers<
                       typename solver::nonlinear_solver::options_group>>>;
  };

  // Additional items to store in the global cache
  using const_global_cache_tags =
      tmpl::list<domain::Tags::RadiallyCompressedCoordinatesOptions>;

  // Collect all reduction tags for observers
  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::push_back<
          tmpl::at<factory_creation::factory_classes, Event>, solver>>;

  using initialization_actions =
      tmpl::push_back<typename solver::initialization_actions,
                      Parallel::Actions::TerminatePhase>;

  using register_actions =
      tmpl::push_back<typename solver::register_actions,
                      observers::Actions::RegisterEventsWithObservers,
                      Parallel::Actions::TerminatePhase>;

  using step_actions = tmpl::list<Actions::RunEventsAndTriggers>;

  using solve_actions =
      tmpl::push_back<typename solver::template solve_actions<step_actions>,
                      Parallel::Actions::TerminatePhase>;

  using dg_element_array = elliptic::DgElementArray<
      Metavariables,
      tmpl::list<
          Parallel::PhaseActions<Parallel::Phase::Initialization,
                                 initialization_actions>,
          Parallel::PhaseActions<Parallel::Phase::Register, register_actions>,
          Parallel::PhaseActions<Parallel::Phase::Solve, solve_actions>>,
      LinearSolver::multigrid::ElementsAllocator<
          volume_dim, typename solver::multigrid::options_group>>;

  // Specify all parallel components that will execute actions at some point.
  using component_list = tmpl::flatten<
      tmpl::list<dg_element_array, typename solver::component_list,
                 observers::Observer<Metavariables>,
                 observers::ObserverWriter<Metavariables>>>;

  static constexpr std::array<Parallel::Phase, 4> default_phase_order{
      {Parallel::Phase::Initialization, Parallel::Phase::Register,
       Parallel::Phase::Solve, Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &setup_memory_allocation_failure_reporting,
    &disable_openblas_multithreading,
    &domain::creators::register_derived_with_charm,
    &register_derived_classes_with_charm<
        metavariables::solver::schwarz_smoother::subdomain_solver>,
    &elliptic::subdomain_preconditioners::register_derived_with_charm,
    &register_factory_classes_with_charm<metavariables>};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions, &enable_segfault_handler};
/// \endcond
