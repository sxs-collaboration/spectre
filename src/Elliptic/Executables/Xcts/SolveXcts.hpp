// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Domain/Creators/Factory3D.hpp"
#include "Domain/RadiallyCompressedCoordinates.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/RunEventsAndTriggers.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Elliptic/Executables/Solver.hpp"
#include "Elliptic/Systems/Xcts/BoundaryConditions/Factory.hpp"
#include "Elliptic/Systems/Xcts/Events/ObserveAdmIntegrals.hpp"
#include "Elliptic/Systems/Xcts/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Xcts/HydroQuantities.hpp"
#include "Elliptic/Triggers/Factory.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Protocols/RegistrationMetavariables.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Amr/Actions/SendAmrDiagnostics.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Factory.hpp"
#include "ParallelAlgorithms/Amr/Protocols/AmrMetavariables.hpp"
#include "ParallelAlgorithms/Amr/Tags.hpp"
#include "ParallelAlgorithms/Events/Factory.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/ElementsAllocator.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Tags.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/Binary.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/BinaryWithGravitationalWaves.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Factory.hpp"
#include "PointwiseFunctions/Hydro/LowerSpatialFourVelocity.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"
#include "PointwiseFunctions/Xcts/SpacetimeQuantities.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
struct Metavariables {
  static constexpr size_t volume_dim = 3;
  static constexpr int conformal_matter_scale = 0;
  using system =
      Xcts::FirstOrderSystem<Xcts::Equations::HamiltonianLapseAndShift,
                             Xcts::Geometry::Curved, conformal_matter_scale>;
  using solver = elliptic::Solver<Metavariables>;

  static constexpr Options::String help{
      "Find the solution to an XCTS problem."};

  using analytic_solution_fields = tmpl::append<typename system::primal_fields>;
  using spacetime_quantities_compute = Xcts::Tags::SpacetimeQuantitiesCompute<
      tmpl::list<Xcts::Tags::ConformalFactor<DataVector>,
                 Xcts::Tags::LapseTimesConformalFactor<DataVector>,
                 ::Tags::deriv<Xcts::Tags::ConformalFactor<DataVector>,
                               tmpl::size_t<3>, Frame::Inertial>,
                 gr::Tags::HamiltonianConstraint<DataVector>,
                 gr::Tags::MomentumConstraint<DataVector, 3>,
                 gr::Tags::SpatialMetric<DataVector, 3>,
                 gr::Tags::InverseSpatialMetric<DataVector, 3>,
                 gr::Tags::SpatialChristoffelSecondKind<DataVector, 3>,
                 gr::Tags::SpatialRicci<DataVector, 3>,
                 gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
                 gr::Tags::ExtrinsicCurvature<DataVector, 3>>>;
  using hydro_quantities_compute = Xcts::Tags::HydroQuantitiesCompute<
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpecificEnthalpy<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 hydro::Tags::MagneticField<DataVector, 3>>>;
  using error_compute = ::Tags::ErrorsCompute<analytic_solution_fields>;
  using error_tags = db::wrap_tags_in<Tags::Error, analytic_solution_fields>;
  using observe_fields = tmpl::append<
      analytic_solution_fields, typename system::background_fields,
      typename spacetime_quantities_compute::tags_list,
      typename hydro_quantities_compute::tags_list, error_tags,
      typename solver::observe_fields,
      tmpl::list<domain::Tags::Coordinates<volume_dim, Frame::Inertial>,
                 domain::Tags::RadiallyCompressedCoordinatesCompute<
                     volume_dim, Frame::Inertial>,
                 ::Tags::NonEuclideanMagnitude<
                     Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>,
                     gr::Tags::SpatialMetric<DataVector, 3>>,
                 hydro::Tags::LowerSpatialFourVelocityCompute>>;
  using observer_compute_tags = tmpl::list<
      ::Events::Tags::ObserverMeshCompute<volume_dim>,
      ::Events::Tags::ObserverDetInvJacobianCompute<Frame::ElementLogical,
                                                    Frame::Inertial>,
      spacetime_quantities_compute, hydro_quantities_compute, error_compute>;

  // Collect all items to store in the cache.
  using const_global_cache_tags =
      tmpl::list<domain::Tags::RadiallyCompressedCoordinatesOptions>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using analytic_solutions_and_data = tmpl::push_back<
        Xcts::Solutions::all_analytic_solutions,
        Xcts::AnalyticData::Binary<elliptic::analytic_data::AnalyticSolution,
                                   Xcts::Solutions::all_analytic_solutions>,
        Xcts::AnalyticData::BinaryWithGravitationalWaves>;

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
        tmpl::pair<::amr::Criterion,
                   ::amr::Criteria::standard_criteria<
                       volume_dim, typename system::primal_fields>>,
        tmpl::pair<Event,
                   tmpl::flatten<tmpl::list<
                       Events::Completion,
                       Events::ObserveAdmIntegrals<
                           LinearSolver::multigrid::Tags::IsFinestGrid>,
                       dg::Events::field_observations<
                           volume_dim, observe_fields, observer_compute_tags,
                           LinearSolver::multigrid::Tags::IsFinestGrid>>>>,
        tmpl::pair<Trigger, elliptic::Triggers::all_triggers<
                                ::amr::OptionTags::AmrGroup>>,
        tmpl::pair<
            PhaseChange,
            tmpl::list<
                PhaseControl::VisitAndReturn<
                    Parallel::Phase::EvaluateAmrCriteria>,
                PhaseControl::VisitAndReturn<Parallel::Phase::AdjustDomain>,
                PhaseControl::VisitAndReturn<Parallel::Phase::CheckDomain>>>,
        tmpl::pair<
            grmhd::AnalyticData::InitialMagneticFields::InitialMagneticField,
            grmhd::AnalyticData::InitialMagneticFields::
                initial_magnetic_fields>>;
  };

  // Collect all reduction tags for observers
  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::flatten<tmpl::list<
          tmpl::at<factory_creation::factory_classes, Event>, solver>>>;

  using initialization_actions =
      tmpl::push_back<typename solver::initialization_actions,
                      Parallel::Actions::TerminatePhase>;

  using register_actions =
      tmpl::push_back<typename solver::register_actions,
                      observers::Actions::RegisterEventsWithObservers>;

  using solve_actions = typename solver::template solve_actions<tmpl::list<>>;

  using dg_element_array = elliptic::DgElementArray<
      Metavariables,
      tmpl::list<Parallel::PhaseActions<Parallel::Phase::Initialization,
                                        initialization_actions>,
                 Parallel::PhaseActions<
                     Parallel::Phase::Register,
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
