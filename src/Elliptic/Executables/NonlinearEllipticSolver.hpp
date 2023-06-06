// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags/Faces.hpp"
#include "Elliptic/Actions/InitializeAnalyticSolution.hpp"
#include "Elliptic/Actions/InitializeFields.hpp"
#include "Elliptic/Actions/InitializeFixedSources.hpp"
#include "Elliptic/BoundaryConditions/Tags/BoundaryFields.hpp"
#include "Elliptic/DiscontinuousGalerkin/Actions/ApplyOperator.hpp"
#include "Elliptic/DiscontinuousGalerkin/Actions/InitializeDomain.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/InitializeSubdomain.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/SubdomainOperator.hpp"
#include "Elliptic/DiscontinuousGalerkin/Tags.hpp"
#include "Elliptic/Protocols/FirstOrderSystem.hpp"
#include "Elliptic/SubdomainPreconditioners/MinusLaplacian.hpp"
#include "Elliptic/Tags.hpp"
#include "IO/Observer/Helpers.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/LinearSolver/Actions/MakeIdentityIfSkipped.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/Gmres.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Actions/RestrictFields.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Multigrid.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Actions/CommunicateOverlapFields.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Actions/ResetSubdomainSolver.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Schwarz.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "ParallelAlgorithms/NonlinearSolver/NewtonRaphson/NewtonRaphson.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// Items related to composing nonlinear elliptic solver executables
namespace elliptic::nonlinear_solver {

/// Option tags for nonlinear elliptic solver executables
namespace OptionTags {

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

}  // namespace OptionTags

/*!
 * \brief A complete nonlinear elliptic solver stack. Use to compose an
 * executable.
 *
 * Uses `Metavariables::volume_dim` and `Metavariables::system`. Also uses
 * `Metavariables` to instantiate parallel components.
 */
template <typename Metavariables>
struct Solver {
  static constexpr size_t volume_dim = Metavariables::volume_dim;
  using system = typename Metavariables::system;
  static_assert(
      tt::assert_conforms_to_v<system, elliptic::protocols::FirstOrderSystem>);

  using background_tag =
      elliptic::Tags::Background<elliptic::analytic_data::Background>;
  using initial_guess_tag =
      elliptic::Tags::InitialGuess<elliptic::analytic_data::InitialGuess>;

  /// These are the fields we solve for
  using fields_tag = ::Tags::Variables<typename system::primal_fields>;
  /// These are the fluxes corresponding to the fields, i.e. essentially their
  /// first derivatives. These are background fields for the linearized sources.
  using fluxes_tag = ::Tags::Variables<typename system::primal_fluxes>;
  /// These are the fixed sources, i.e. the RHS of the equations
  using fixed_sources_tag = db::add_tag_prefix<::Tags::FixedSource, fields_tag>;
  using operator_applied_to_fields_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::OperatorAppliedTo, fields_tag>;

  /// The nonlinear solver algorithm
  using nonlinear_solver = NonlinearSolver::newton_raphson::NewtonRaphson<
      Metavariables, fields_tag, OptionTags::NewtonRaphsonGroup,
      fixed_sources_tag, LinearSolver::multigrid::Tags::IsFinestGrid>;
  using nonlinear_solver_iteration_id =
      Convergence::Tags::IterationId<typename nonlinear_solver::options_group>;

  /// The linear solver algorithm. We use GMRES since the operator is not
  /// necessarily to be symmetric. Using CG here would be an optimization for
  /// symmetric problems.
  using linear_solver = LinearSolver::gmres::Gmres<
      Metavariables, typename nonlinear_solver::linear_solver_fields_tag,
      OptionTags::GmresGroup, true,
      typename nonlinear_solver::linear_solver_source_tag,
      LinearSolver::multigrid::Tags::IsFinestGrid>;
  using linear_solver_iteration_id =
      Convergence::Tags::IterationId<typename linear_solver::options_group>;

  /// Precondition each linear solver iteration with a multigrid V-cycle
  using multigrid = LinearSolver::multigrid::Multigrid<
      volume_dim, typename linear_solver::operand_tag,
      OptionTags::MultigridGroup, elliptic::dg::Tags::Massive,
      typename linear_solver::preconditioner_source_tag>;

  /// Smooth each multigrid level with a number of Schwarz smoothing steps
  using subdomain_operator =
      elliptic::dg::subdomain_operator::SubdomainOperator<
          system, OptionTags::SchwarzSmootherGroup>;
  using subdomain_preconditioners = tmpl::list<
      elliptic::subdomain_preconditioners::Registrars::MinusLaplacian<
          volume_dim, OptionTags::SchwarzSmootherGroup>>;
  using schwarz_smoother = LinearSolver::Schwarz::Schwarz<
      typename multigrid::smooth_fields_tag, OptionTags::SchwarzSmootherGroup,
      subdomain_operator, subdomain_preconditioners,
      typename multigrid::smooth_source_tag,
      LinearSolver::multigrid::Tags::MultigridLevel>;

  /// For the GMRES linear solver we need to apply the DG operator to its
  /// internal "operand" in every iteration of the algorithm.
  using correction_vars_tag = typename linear_solver::operand_tag;
  using operator_applied_to_correction_vars_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo,
                         correction_vars_tag>;
  /// The correction fluxes can be stored in an arbitrary tag. We don't need to
  /// access them anywhere, they're just a memory buffer for the linearized
  /// operator.
  using correction_fluxes_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Correction, fluxes_tag>;

  /// Collect all reduction tags for observers
  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::list<nonlinear_solver, linear_solver, multigrid, schwarz_smoother>>;

  using initialization_actions = tmpl::list<
      elliptic::dg::Actions::InitializeDomain<volume_dim>,
      typename nonlinear_solver::initialize_element,
      typename linear_solver::initialize_element,
      typename multigrid::initialize_element,
      typename schwarz_smoother::initialize_element,
      elliptic::Actions::InitializeFields<system, initial_guess_tag>,
      elliptic::Actions::InitializeFixedSources<system, background_tag>,
      elliptic::Actions::InitializeOptionalAnalyticSolution<
          background_tag,
          tmpl::append<typename system::primal_fields,
                       typename system::primal_fluxes>,
          elliptic::analytic_data::AnalyticSolution>,
      elliptic::dg::Actions::initialize_operator<system, background_tag>,
      elliptic::dg::subdomain_operator::Actions::InitializeSubdomain<
          system, background_tag, typename schwarz_smoother::options_group>,
      ::Initialization::Actions::AddComputeTags<tmpl::list<
          // For linearized boundary conditions
          elliptic::Tags::BoundaryFieldsCompute<volume_dim, fields_tag>,
          elliptic::Tags::BoundaryFluxesCompute<volume_dim, fields_tag,
                                                fluxes_tag>>>>;

  using register_actions =
      tmpl::list<typename nonlinear_solver::register_element,
                 typename multigrid::register_element,
                 typename schwarz_smoother::register_element>;

  template <bool Linearized>
  using build_operator_actions = elliptic::dg::Actions::apply_operator<
      system, Linearized,
      tmpl::conditional_t<Linearized, linear_solver_iteration_id,
                          nonlinear_solver_iteration_id>,
      tmpl::conditional_t<Linearized, correction_vars_tag, fields_tag>,
      tmpl::conditional_t<Linearized, correction_fluxes_tag, fluxes_tag>,
      tmpl::conditional_t<Linearized, operator_applied_to_correction_vars_tag,
                          operator_applied_to_fields_tag>>;

  template <typename Label>
  using smooth_actions =
      typename schwarz_smoother::template solve<build_operator_actions<true>,
                                                Label>;

  /// This data needs to be communicated on subdomain overlap regions
  using communicated_overlap_tags = tmpl::flatten<tmpl::list<
      // For linearized sources
      fields_tag, fluxes_tag,
      // For linearized boundary conditions
      domain::make_faces_tags<volume_dim, typename system::primal_fields>,
      domain::make_faces_tags<
          volume_dim, db::wrap_tags_in<::Tags::NormalDotFlux,
                                       typename system::primal_fields>>>>;

  template <typename StepActions>
  using solve_actions = typename nonlinear_solver::template solve<
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
      StepActions>;

  using component_list = tmpl::list<typename nonlinear_solver::component_list,
                                    typename linear_solver::component_list,
                                    typename multigrid::component_list,
                                    typename schwarz_smoother::component_list>;
};

}  // namespace elliptic::nonlinear_solver
