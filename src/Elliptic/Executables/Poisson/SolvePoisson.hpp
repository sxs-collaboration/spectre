// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "Domain/Creators/Factory1D.hpp"
#include "Domain/Creators/Factory2D.hpp"
#include "Domain/Creators/Factory3D.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/RadiallyCompressedCoordinates.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/InitializeAnalyticSolution.hpp"
#include "Elliptic/Actions/InitializeFields.hpp"
#include "Elliptic/Actions/InitializeFixedSources.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/DiscontinuousGalerkin/Actions/ApplyOperator.hpp"
#include "Elliptic/DiscontinuousGalerkin/Actions/InitializeDomain.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/InitializeSubdomain.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/SubdomainOperator.hpp"
#include "Elliptic/Systems/Poisson/BoundaryConditions/Factory.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Tags.hpp"
#include "Elliptic/Triggers/Factory.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Actions/RandomizeVariables.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "ParallelAlgorithms/LinearSolver/Actions/MakeIdentityIfSkipped.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/Gmres.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/ElementsAllocator.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Multigrid.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Schwarz.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/Factory.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"
#include "PointwiseFunctions/MathFunctions/Factory.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace SolvePoisson::OptionTags {
struct LinearSolverGroup {
  static std::string name() { return "LinearSolver"; }
  static constexpr Options::String help =
      "The iterative Krylov-subspace linear solver";
};
struct GmresGroup {
  static std::string name() { return "GMRES"; }
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
}  // namespace SolvePoisson::OptionTags

/// \cond
template <size_t Dim>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  using system =
      Poisson::FirstOrderSystem<Dim, Poisson::Geometry::FlatCartesian>;

  using background_tag =
      elliptic::Tags::Background<elliptic::analytic_data::Background>;
  using initial_guess_tag =
      elliptic::Tags::InitialGuess<elliptic::analytic_data::InitialGuess>;

  static constexpr Options::String help{
      "Find the solution to a Poisson problem."};

  // These are the fields we solve for
  using fields_tag = ::Tags::Variables<typename system::primal_fields>;
  // These are the fixed sources, i.e. the RHS of the equations
  using fixed_sources_tag = db::add_tag_prefix<::Tags::FixedSource, fields_tag>;
  // This is the linear operator applied to the fields. We'll only use it to
  // apply the operator to the initial guess, so an optimization would be to
  // re-use the `operator_applied_to_vars_tag` below. This optimization needs a
  // few minor changes to the parallel linear solver algorithm.
  using operator_applied_to_fields_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, fields_tag>;

  // The linear solver algorithm. We must use GMRES since the operator is
  // not guaranteed to be symmetric. It can be made symmetric by multiplying by
  // the DG mass matrix.
  using linear_solver = LinearSolver::gmres::Gmres<
      Metavariables, fields_tag, SolvePoisson::OptionTags::LinearSolverGroup,
      true, fixed_sources_tag, LinearSolver::multigrid::Tags::IsFinestGrid>;
  using linear_solver_iteration_id =
      Convergence::Tags::IterationId<typename linear_solver::options_group>;
  // Precondition each linear solver iteration with a multigrid V-cycle
  using multigrid = LinearSolver::multigrid::Multigrid<
      volume_dim, typename linear_solver::operand_tag,
      SolvePoisson::OptionTags::MultigridGroup, elliptic::dg::Tags::Massive,
      typename linear_solver::preconditioner_source_tag>;
  // Smooth each multigrid level with a number of Schwarz smoothing steps
  using subdomain_operator =
      elliptic::dg::subdomain_operator::SubdomainOperator<
          system, SolvePoisson::OptionTags::SchwarzSmootherGroup>;
  using schwarz_smoother = LinearSolver::Schwarz::Schwarz<
      typename multigrid::smooth_fields_tag,
      SolvePoisson::OptionTags::SchwarzSmootherGroup, subdomain_operator,
      tmpl::list<>, typename multigrid::smooth_source_tag,
      LinearSolver::multigrid::Tags::MultigridLevel>;
  // For the GMRES linear solver we need to apply the DG operator to its
  // internal "operand" in every iteration of the algorithm.
  using vars_tag = typename linear_solver::operand_tag;
  using operator_applied_to_vars_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, vars_tag>;
  // We'll buffer the corresponding fluxes in this tag, but won't actually need
  // to access them outside applying the operator
  using fluxes_vars_tag =
      ::Tags::Variables<db::wrap_tags_in<LinearSolver::Tags::Operand,
                                         typename system::primal_fluxes>>;

  using analytic_solution_fields = typename system::primal_fields;
  using error_compute = ::Tags::ErrorsCompute<analytic_solution_fields>;
  using error_tags = db::wrap_tags_in<Tags::Error, analytic_solution_fields>;
  using observe_fields = tmpl::append<
      analytic_solution_fields, error_tags,
      tmpl::list<domain::Tags::Coordinates<volume_dim, Frame::Inertial>,
                 domain::Tags::RadiallyCompressedCoordinatesCompute<
                     volume_dim, Frame::Inertial>>>;
  using observer_compute_tags =
      tmpl::list<::Events::Tags::ObserverMeshCompute<volume_dim>,
                 error_compute>;

  // Collect all items to store in the cache.
  using const_global_cache_tags =
      tmpl::list<background_tag, initial_guess_tag,
                 domain::Tags::RadiallyCompressedCoordinatesOptions>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
        tmpl::pair<elliptic::analytic_data::Background,
                   Poisson::Solutions::all_analytic_solutions<volume_dim>>,
        tmpl::pair<elliptic::analytic_data::InitialGuess,
                   Poisson::Solutions::all_analytic_solutions<volume_dim>>,
        tmpl::pair<elliptic::analytic_data::AnalyticSolution,
                   Poisson::Solutions::all_analytic_solutions<volume_dim>>,
        tmpl::pair<
            ::MathFunction<volume_dim, Frame::Inertial>,
            MathFunctions::all_math_functions<volume_dim, Frame::Inertial>>,
        tmpl::pair<
            elliptic::BoundaryConditions::BoundaryCondition<volume_dim>,
            Poisson::BoundaryConditions::standard_boundary_conditions<system>>,
        tmpl::pair<Event,
                   tmpl::flatten<tmpl::list<
                       Events::Completion,
                       dg::Events::field_observations<
                           volume_dim, linear_solver_iteration_id,
                           observe_fields, observer_compute_tags,
                           LinearSolver::multigrid::Tags::IsFinestGrid>>>>,
        tmpl::pair<Trigger, elliptic::Triggers::all_triggers<
                                typename linear_solver::options_group>>>;
  };

  // Collect all reduction tags for observers
  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::flatten<tmpl::list<
          tmpl::at<typename factory_creation::factory_classes, Event>,
          linear_solver, multigrid, schwarz_smoother>>>;

  // For labeling the yaml option for RandomizeVariables
  struct RandomizeInitialGuess {};

  using initialization_actions = tmpl::list<
      elliptic::dg::Actions::InitializeDomain<volume_dim>,
      typename linear_solver::initialize_element,
      typename multigrid::initialize_element,
      typename schwarz_smoother::initialize_element,
      elliptic::Actions::InitializeFields<system, initial_guess_tag>,
      Actions::RandomizeVariables<
          ::Tags::Variables<typename system::primal_fields>,
          RandomizeInitialGuess>,
      elliptic::Actions::InitializeFixedSources<system, background_tag>,
      elliptic::Actions::InitializeOptionalAnalyticSolution<
          background_tag,
          tmpl::append<typename system::primal_fields,
                       typename system::primal_fluxes>,
          elliptic::analytic_data::AnalyticSolution>,
      elliptic::dg::Actions::initialize_operator<system>,
      elliptic::dg::subdomain_operator::Actions::InitializeSubdomain<
          system, background_tag, typename schwarz_smoother::options_group>,
      elliptic::dg::Actions::ImposeInhomogeneousBoundaryConditionsOnSource<
          system, fixed_sources_tag>,
      // Apply the DG operator to the initial guess
      elliptic::dg::Actions::apply_operator<
          system, true, linear_solver_iteration_id, fields_tag, fluxes_vars_tag,
          operator_applied_to_fields_tag, vars_tag, fluxes_vars_tag>,
      Parallel::Actions::TerminatePhase>;

  using build_linear_operator_actions = elliptic::dg::Actions::apply_operator<
      system, true, linear_solver_iteration_id, vars_tag, fluxes_vars_tag,
      operator_applied_to_vars_tag>;

  using register_actions =
      tmpl::list<observers::Actions::RegisterEventsWithObservers,
                 typename schwarz_smoother::register_element,
                 typename multigrid::register_element,
                 Parallel::Actions::TerminatePhase>;

  template <typename Label>
  using smooth_actions =
      typename schwarz_smoother::template solve<build_linear_operator_actions,
                                                Label>;

  using solve_actions = tmpl::list<
      typename linear_solver::template solve<
          tmpl::list<
              typename multigrid::template solve<
                  build_linear_operator_actions,
                  smooth_actions<LinearSolver::multigrid::VcycleDownLabel>,
                  smooth_actions<LinearSolver::multigrid::VcycleUpLabel>>,
              ::LinearSolver::Actions::make_identity_if_skipped<
                  multigrid, build_linear_operator_actions>>,
          Actions::RunEventsAndTriggers>,
      Parallel::Actions::TerminatePhase>;

  using dg_element_array = elliptic::DgElementArray<
      Metavariables,
      tmpl::list<
          Parallel::PhaseActions<Parallel::Phase::Initialization,
                                 initialization_actions>,
          Parallel::PhaseActions<Parallel::Phase::Register, register_actions>,
          Parallel::PhaseActions<Parallel::Phase::Solve, solve_actions>>,
      LinearSolver::multigrid::ElementsAllocator<
          volume_dim, typename multigrid::options_group>>;

  // Specify all parallel components that will execute actions at some point.
  using component_list = tmpl::flatten<
      tmpl::list<dg_element_array, typename linear_solver::component_list,
                 typename multigrid::component_list,
                 typename schwarz_smoother::component_list,
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
        metavariables::schwarz_smoother::subdomain_solver>,
    &register_factory_classes_with_charm<metavariables>};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions, &enable_segfault_handler};
/// \endcond
