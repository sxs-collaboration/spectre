// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "Domain/Creators/Factory3D.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/InitializeAnalyticSolution.hpp"
#include "Elliptic/Actions/InitializeFields.hpp"
#include "Elliptic/Actions/InitializeFixedSources.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/Tags/BoundaryFields.hpp"
#include "Elliptic/DiscontinuousGalerkin/Actions/ApplyOperator.hpp"
#include "Elliptic/DiscontinuousGalerkin/Actions/InitializeDomain.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/InitializeSubdomain.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/SubdomainOperator.hpp"
#include "Elliptic/SubdomainPreconditioners/MinusLaplacian.hpp"
#include "Elliptic/SubdomainPreconditioners/RegisterDerived.hpp"
#include "Elliptic/Systems/Xcts/BoundaryConditions/Factory.hpp"
#include "Elliptic/Systems/Xcts/FirstOrderSystem.hpp"
#include "Elliptic/Tags.hpp"
#include "Elliptic/Triggers/Factory.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "Options/Options.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/Events/ObserveNorms.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "ParallelAlgorithms/LinearSolver/Actions/MakeIdentityIfSkipped.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/Gmres.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/ElementsAllocator.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Multigrid.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Actions/ResetSubdomainSolver.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Schwarz.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "ParallelAlgorithms/NonlinearSolver/NewtonRaphson/NewtonRaphson.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/Binary.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Factory.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"
#include "PointwiseFunctions/Xcts/SpacetimeQuantities.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace SolveXcts::OptionTags {
struct NonlinearSolverGroup {
  static std::string name() { return "NonlinearSolver"; }
  static constexpr Options::String help = "The iterative nonlinear solver";
};
struct NewtonRaphsonGroup {
  static std::string name() { return "NewtonRaphson"; }
  static constexpr Options::String help =
      "Options for the Newton-Raphson nonlinear solver";
  using group = NonlinearSolverGroup;
};
struct LinearSolverGroup {
  static std::string name() { return "LinearSolver"; }
  static constexpr Options::String help =
      "The iterative Krylov-subspace linear solver";
};
struct GmresGroup {
  static std::string name() { return "Gmres"; }
  static constexpr Options::String help = "Options for the GMRES linear solver";
  using group = LinearSolverGroup;
};
struct SchwarzSmootherGroup {
  static std::string name() { return "SchwarzSmoother"; }
  static constexpr Options::String help = "Options for the Schwarz smoother";
  using group = LinearSolverGroup;
};
struct MultigridGroup {
  static std::string name() { return "Multigrid"; }
  static constexpr Options::String help = "Options for the multigrid";
  using group = LinearSolverGroup;
};
}  // namespace SolveXcts::OptionTags

/// \cond
struct Metavariables {
  static constexpr size_t volume_dim = 3;
  static constexpr int conformal_matter_scale = 0;
  using system =
      Xcts::FirstOrderSystem<Xcts::Equations::HamiltonianLapseAndShift,
                             Xcts::Geometry::Curved, conformal_matter_scale>;

  using background_tag =
      elliptic::Tags::Background<elliptic::analytic_data::Background>;
  using initial_guess_tag =
      elliptic::Tags::InitialGuess<elliptic::analytic_data::InitialGuess>;

  static constexpr Options::String help{
      "Find the solution to an XCTS problem."};

  // These are the fields we solve for
  using fields_tag = ::Tags::Variables<typename system::primal_fields>;
  // These are the fluxes corresponding to the fields, i.e. essentially their
  // first derivatives. These are background fields for the linearized sources.
  using fluxes_tag = ::Tags::Variables<typename system::primal_fluxes>;
  // These are the fixed sources, i.e. the RHS of the equations
  using fixed_sources_tag = db::add_tag_prefix<::Tags::FixedSource, fields_tag>;
  using operator_applied_to_fields_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::OperatorAppliedTo, fields_tag>;

  using nonlinear_solver = NonlinearSolver::newton_raphson::NewtonRaphson<
      Metavariables, fields_tag, SolveXcts::OptionTags::NewtonRaphsonGroup,
      fixed_sources_tag, LinearSolver::multigrid::Tags::IsFinestGrid>;
  using nonlinear_solver_iteration_id =
      Convergence::Tags::IterationId<typename nonlinear_solver::options_group>;

  // The linear solver algorithm. We must use GMRES since the operator is
  // not guaranteed to be symmetric.
  using linear_solver = LinearSolver::gmres::Gmres<
      Metavariables, typename nonlinear_solver::linear_solver_fields_tag,
      SolveXcts::OptionTags::GmresGroup, true,
      typename nonlinear_solver::linear_solver_source_tag,
      LinearSolver::multigrid::Tags::IsFinestGrid>;
  using linear_solver_iteration_id =
      Convergence::Tags::IterationId<typename linear_solver::options_group>;
  // Precondition each linear solver iteration with a multigrid V-cycle
  using multigrid = LinearSolver::multigrid::Multigrid<
      volume_dim, typename linear_solver::operand_tag,
      SolveXcts::OptionTags::MultigridGroup, elliptic::dg::Tags::Massive,
      typename linear_solver::preconditioner_source_tag>;
  // Smooth each multigrid level with a number of Schwarz smoothing steps
  using subdomain_operator =
      elliptic::dg::subdomain_operator::SubdomainOperator<
          system, SolveXcts::OptionTags::SchwarzSmootherGroup>;
  using subdomain_preconditioners = tmpl::list<
      elliptic::subdomain_preconditioners::Registrars::MinusLaplacian<
          volume_dim, SolveXcts::OptionTags::SchwarzSmootherGroup>>;
  // This data needs to be communicated on subdomain overlap regions
  using communicated_overlap_tags = tmpl::list<
      // For linearized sources
      fields_tag, fluxes_tag,
      // For linearized boundary conditions
      domain::Tags::Faces<volume_dim, Xcts::Tags::ConformalFactor<DataVector>>,
      domain::Tags::Faces<volume_dim,
                          Xcts::Tags::LapseTimesConformalFactor<DataVector>>,
      domain::Tags::Faces<volume_dim,
                          ::Tags::NormalDotFlux<Xcts::Tags::ShiftExcess<
                              DataVector, volume_dim, Frame::Inertial>>>>;
  using schwarz_smoother = LinearSolver::Schwarz::Schwarz<
      typename multigrid::smooth_fields_tag,
      SolveXcts::OptionTags::SchwarzSmootherGroup, subdomain_operator,
      subdomain_preconditioners, typename multigrid::smooth_source_tag,
      LinearSolver::multigrid::Tags::MultigridLevel>;
  // For the GMRES linear solver we need to apply the DG operator to its
  // internal "operand" in every iteration of the algorithm.
  using correction_vars_tag = typename linear_solver::operand_tag;
  using operator_applied_to_correction_vars_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo,
                         correction_vars_tag>;
  // The correction fluxes can be stored in an arbitrary tag. We don't need to
  // access them anywhere, they're just a memory buffer for the linearized
  // operator.
  using correction_fluxes_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Correction, fluxes_tag>;

  using analytic_solution_fields = tmpl::append<typename system::primal_fields,
                                                typename system::primal_fluxes>;
  using spacetime_quantities_compute = Xcts::Tags::SpacetimeQuantitiesCompute<
      tmpl::list<gr::Tags::HamiltonianConstraint<DataVector>,
                 gr::Tags::MomentumConstraint<3, Frame::Inertial, DataVector>,
                 gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                 gr::Tags::Lapse<DataVector>,
                 gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                 gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>>>;
  using observe_fields =
      tmpl::append<analytic_solution_fields, typename system::background_fields,
                   typename spacetime_quantities_compute::tags_list>;

  // Collect all items to store in the cache.
  using const_global_cache_tags =
      tmpl::list<background_tag, initial_guess_tag, Tags::EventsAndTriggers>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using analytic_solutions_and_data = tmpl::push_back<
        Xcts::Solutions::all_analytic_solutions,
        Xcts::AnalyticData::Binary<elliptic::analytic_data::AnalyticSolution,
                                   Xcts::Solutions::all_analytic_solutions>>;

    using factory_classes = tmpl::map<
        tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
        tmpl::pair<elliptic::analytic_data::Background,
                   analytic_solutions_and_data>,
        tmpl::pair<elliptic::analytic_data::InitialGuess,
                   analytic_solutions_and_data>,
        tmpl::pair<elliptic::analytic_data::AnalyticSolution,
                   Xcts::Solutions::all_analytic_solutions>,
        tmpl::pair<
            elliptic::BoundaryConditions::BoundaryCondition<volume_dim>,
            Xcts::BoundaryConditions::standard_boundary_conditions<system>>,
        tmpl::pair<Event,
                   tmpl::flatten<tmpl::list<
                       Events::Completion,
                       dg::Events::field_observations<
                           volume_dim, nonlinear_solver_iteration_id,
                           observe_fields, analytic_solution_fields,
                           tmpl::list<spacetime_quantities_compute>,
                           LinearSolver::multigrid::Tags::IsFinestGrid>,
                       Events::ObserveNorms<
                           nonlinear_solver_iteration_id, observe_fields,
                           tmpl::list<spacetime_quantities_compute>,
                           LinearSolver::multigrid::Tags::IsFinestGrid>>>>,
        tmpl::pair<Trigger, elliptic::Triggers::all_triggers<
                                typename nonlinear_solver::options_group>>>;
  };

  // Collect all reduction tags for observers
  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::flatten<tmpl::list<
          tmpl::at<factory_creation::factory_classes, Event>, nonlinear_solver,
          linear_solver, multigrid, schwarz_smoother>>>;

  // Specify all global synchronization points.
  enum class Phase { Initialization, RegisterWithObserver, Solve, Exit };

  using initialization_actions = tmpl::list<
      Actions::SetupDataBox,
      elliptic::dg::Actions::InitializeDomain<volume_dim>,
      typename nonlinear_solver::initialize_element,
      typename linear_solver::initialize_element,
      typename multigrid::initialize_element,
      typename schwarz_smoother::initialize_element,
      elliptic::Actions::InitializeFields<system, initial_guess_tag>,
      elliptic::Actions::InitializeFixedSources<system, background_tag>,
      elliptic::Actions::InitializeOptionalAnalyticSolution<
          background_tag, analytic_solution_fields,
          elliptic::analytic_data::AnalyticSolution>,
      elliptic::dg::Actions::initialize_operator<system, background_tag>,
      elliptic::dg::subdomain_operator::Actions::InitializeSubdomain<
          system, background_tag, typename schwarz_smoother::options_group>,
      ::Initialization::Actions::AddComputeTags<tmpl::list<
          // For linearized boundary conditions
          elliptic::Tags::BoundaryFieldsCompute<volume_dim, fields_tag>,
          elliptic::Tags::BoundaryFluxesCompute<volume_dim, fields_tag,
                                                fluxes_tag>>>,
      Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  template <bool Linearized>
  using build_operator_actions = elliptic::dg::Actions::apply_operator<
      system, Linearized,
      tmpl::conditional_t<Linearized, linear_solver_iteration_id,
                          nonlinear_solver_iteration_id>,
      tmpl::conditional_t<Linearized, correction_vars_tag, fields_tag>,
      tmpl::conditional_t<Linearized, correction_fluxes_tag, fluxes_tag>,
      tmpl::conditional_t<Linearized, operator_applied_to_correction_vars_tag,
                          operator_applied_to_fields_tag>>;

  using register_actions =
      tmpl::list<observers::Actions::RegisterEventsWithObservers,
                 typename nonlinear_solver::register_element,
                 typename multigrid::register_element,
                 typename schwarz_smoother::register_element,
                 Parallel::Actions::TerminatePhase>;

  template <typename Label>
  using smooth_actions =
      typename schwarz_smoother::template solve<build_operator_actions<true>,
                                                Label>;

  using solve_actions = tmpl::list<
      typename nonlinear_solver::template solve<
          build_operator_actions<false>,
          tmpl::list<
              LinearSolver::multigrid::Actions::ReceiveFieldsFromFinerGrid<
                  volume_dim, tmpl::list<fields_tag, fluxes_tag>,
                  typename multigrid::options_group>,
              LinearSolver::multigrid::Actions::SendFieldsToCoarserGrid<
                  tmpl::list<fields_tag, fluxes_tag>,
                  typename multigrid::options_group, void>,
              LinearSolver::Schwarz::Actions::SendOverlapFields<
                  communicated_overlap_tags,
                  typename schwarz_smoother::options_group, false>,
              LinearSolver::Schwarz::Actions::ReceiveOverlapFields<
                  volume_dim, communicated_overlap_tags,
                  typename schwarz_smoother::options_group>,
              LinearSolver::Schwarz::Actions::ResetSubdomainSolver<
                  typename schwarz_smoother::options_group>,
              typename linear_solver::template solve<tmpl::list<
                  typename multigrid::template solve<
                      build_operator_actions<true>,
                      smooth_actions<LinearSolver::multigrid::VcycleDownLabel>,
                      smooth_actions<LinearSolver::multigrid::VcycleUpLabel>>,
                  ::LinearSolver::Actions::make_identity_if_skipped<
                      multigrid, build_operator_actions<true>>>>>,
          Actions::RunEventsAndTriggers>,
      Parallel::Actions::TerminatePhase>;

  using dg_element_array = elliptic::DgElementArray<
      Metavariables,
      tmpl::list<Parallel::PhaseActions<Phase, Phase::Initialization,
                                        initialization_actions>,
                 Parallel::PhaseActions<Phase, Phase::RegisterWithObserver,
                                        register_actions>,
                 Parallel::PhaseActions<Phase, Phase::Solve, solve_actions>>,
      LinearSolver::multigrid::ElementsAllocator<
          volume_dim, typename multigrid::options_group>>;

  // Specify all parallel components that will execute actions at some point.
  using component_list = tmpl::flatten<
      tmpl::list<dg_element_array, typename nonlinear_solver::component_list,
                 typename linear_solver::component_list,
                 typename multigrid::component_list,
                 typename schwarz_smoother::component_list,
                 observers::Observer<Metavariables>,
                 observers::ObserverWriter<Metavariables>>>;

  // Specify the transitions between phases.
  template <typename... Tags>
  static Phase determine_next_phase(
      const gsl::not_null<
          tuples::TaggedTuple<Tags...>*> /*phase_change_decision_data*/,
      const Phase& current_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& /*cache_proxy*/) {
    switch (current_phase) {
      case Phase::Initialization:
        return Phase::RegisterWithObserver;
      case Phase::RegisterWithObserver:
        return Phase::Solve;
      case Phase::Solve:
        return Phase::Exit;
      case Phase::Exit:
        ERROR(
            "Should never call determine_next_phase with the current phase "
            "being 'Exit'");
      default:
        ERROR(
            "Unknown type of phase. Did you static_cast<Phase> an integral "
            "value?");
    }
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &setup_memory_allocation_failure_reporting,
    &disable_openblas_multithreading,
    &domain::creators::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<
        metavariables::schwarz_smoother::subdomain_solver>,
    &elliptic::subdomain_preconditioners::register_derived_with_charm,
    &Parallel::register_factory_classes_with_charm<metavariables>};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
/// \endcond
