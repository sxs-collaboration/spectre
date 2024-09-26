// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "Domain/Creators/Factory1D.hpp"
#include "Domain/Creators/Factory2D.hpp"
#include "Domain/Creators/Factory3D.hpp"
#include "Domain/RadiallyCompressedCoordinates.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/RunEventsAndTriggers.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Elliptic/Executables/Solver.hpp"
#include "Elliptic/Systems/Poisson/BoundaryConditions/Factory.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Triggers/Factory.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Protocols/RegistrationMetavariables.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Amr/Actions/SendAmrDiagnostics.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Factory.hpp"
#include "ParallelAlgorithms/Amr/Protocols/AmrMetavariables.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/ElementsAllocator.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/Factory.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"
#include "PointwiseFunctions/InitialDataUtilities/NumericData.hpp"
#include "PointwiseFunctions/MathFunctions/Factory.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP

template <size_t Dim>
struct Metavariables {
  static constexpr Options::String help{
      "Find the solution to a Poisson problem."};

  static constexpr size_t volume_dim = Dim;
  using system =
      Poisson::FirstOrderSystem<Dim, Poisson::Geometry::FlatCartesian>;
  using solver = elliptic::Solver<Metavariables>;

  using analytic_solution_fields = typename system::primal_fields;
  using error_compute = ::Tags::ErrorsCompute<analytic_solution_fields>;
  using error_tags = db::wrap_tags_in<Tags::Error, analytic_solution_fields>;
  using observe_fields = tmpl::append<
      analytic_solution_fields, error_tags, typename solver::observe_fields,
      tmpl::list<domain::Tags::Coordinates<volume_dim, Frame::Inertial>,
                 ::Events::Tags::ObserverDetInvJacobianCompute<
                     Frame::ElementLogical, Frame::Inertial>,
                 domain::Tags::RadiallyCompressedCoordinatesCompute<
                     volume_dim, Frame::Inertial>>>;
  using observer_compute_tags =
      tmpl::list<::Events::Tags::ObserverMeshCompute<volume_dim>,
                 error_compute>;

  // Collect all items to store in the cache.
  using const_global_cache_tags =
      tmpl::list<domain::Tags::RadiallyCompressedCoordinatesOptions>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
        tmpl::pair<elliptic::analytic_data::Background,
                   tmpl::push_back<
                       Poisson::Solutions::all_analytic_solutions<volume_dim>,
                       elliptic::analytic_data::NumericData>>,
        tmpl::pair<elliptic::analytic_data::InitialGuess,
                   tmpl::push_back<
                       Poisson::Solutions::all_analytic_solutions<volume_dim>,
                       elliptic::analytic_data::NumericData>>,
        tmpl::pair<elliptic::analytic_data::AnalyticSolution,
                   Poisson::Solutions::all_analytic_solutions<volume_dim>>,
        tmpl::pair<
            ::MathFunction<volume_dim, Frame::Inertial>,
            MathFunctions::all_math_functions<volume_dim, Frame::Inertial>>,
        tmpl::pair<
            elliptic::BoundaryConditions::BoundaryCondition<volume_dim>,
            Poisson::BoundaryConditions::standard_boundary_conditions<system>>,
        tmpl::pair<
            ::amr::Criterion,
            ::amr::Criteria::standard_criteria<
                volume_dim, tmpl::list<Poisson::Tags::Field<DataVector>>>>,
        tmpl::pair<Event,
                   tmpl::flatten<tmpl::list<
                       Events::Completion,
                       dg::Events::field_observations<
                           volume_dim, observe_fields, observer_compute_tags,
                           LinearSolver::multigrid::Tags::IsFinestGrid>>>>,
        tmpl::pair<Trigger, elliptic::Triggers::all_triggers<
                                ::amr::OptionTags::AmrGroup>>,
        tmpl::pair<
            PhaseChange,
            tmpl::list<
                // Phase for building a matrix representation of the operator
                PhaseControl::VisitAndReturn<Parallel::Phase::BuildMatrix>,
                // Phases for AMR
                PhaseControl::VisitAndReturn<
                    Parallel::Phase::EvaluateAmrCriteria>,
                PhaseControl::VisitAndReturn<Parallel::Phase::AdjustDomain>,
                PhaseControl::VisitAndReturn<Parallel::Phase::CheckDomain>>>>;
  };

  // Collect all reduction tags for observers
  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::flatten<tmpl::list<
          tmpl::at<typename factory_creation::factory_classes, Event>,
          solver>>>;

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
          Parallel::PhaseActions<Parallel::Phase::Initialization,
                                 initialization_actions>,
          Parallel::PhaseActions<
              Parallel::Phase::Register,
              tmpl::push_back<register_actions,
                              Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<Parallel::Phase::Solve, solve_actions>,
          Parallel::PhaseActions<Parallel::Phase::CheckDomain,
                                 tmpl::list<::amr::Actions::SendAmrDiagnostics,
                                            Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<
              Parallel::Phase::BuildMatrix,
              tmpl::push_back<typename solver::build_matrix_actions,
                              Parallel::Actions::TerminatePhase>>>,
      LinearSolver::multigrid::ElementsAllocator<
          volume_dim, typename solver::multigrid::options_group>>;

  struct amr : tt::ConformsTo<::amr::protocols::AmrMetavariables> {
    using element_array = dg_element_array;
    using projectors = typename solver::amr_projectors;
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

  static constexpr std::array<Parallel::Phase, 4> default_phase_order{
      {Parallel::Phase::Initialization, Parallel::Phase::Register,
       Parallel::Phase::Solve, Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};
/// \endcond
