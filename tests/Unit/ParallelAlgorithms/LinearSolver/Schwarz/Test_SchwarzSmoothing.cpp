// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include <cstddef>
#include <random>
#include <vector>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/InitializeAnalyticSolution.hpp"
#include "Elliptic/Actions/InitializeFixedSources.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Elliptic/DiscontinuousGalerkin/ImposeBoundaryConditions.hpp"
#include "Elliptic/DiscontinuousGalerkin/ImposeInhomogeneousBoundaryConditionsOnSource.hpp"
#include "Elliptic/DiscontinuousGalerkin/InitializeFirstOrderOperator.hpp"
#include "Elliptic/DiscontinuousGalerkin/NumericalFluxes/FirstOrderInternalPenalty.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/InitializeSubdomain.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/SubdomainOperator.hpp"
#include "Elliptic/FirstOrderOperator.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Poisson/Geometry.hpp"
#include "Elliptic/Tags.hpp"
#include "Elliptic/Triggers/EveryNIterations.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/FirstOrderScheme.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/CollectDataForFluxes.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/FluxCommunication.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeInterfaces.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeMortars.hpp"
#include "ParallelAlgorithms/Events/ObserveErrorNorms.hpp"
#include "ParallelAlgorithms/Events/ObserveFields.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "ParallelAlgorithms/LinearSolver/Actions/TerminateIfConverged.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Schwarz.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/SubdomainPreconditioners/ExplicitInverse.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/ProductOfSinusoids.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/TMPL.hpp"

#include "Framework/TestingFramework.hpp"
#include "Parallel/Printf.hpp"
#include "ParallelAlgorithms/LinearSolver/Richardson/Richardson.hpp"

namespace {

struct SchwarzSmoother {
  static constexpr Options::String help =
      "Options for the iterative Schwarz smoother";
};

namespace OptionTags {
struct ErrorTolerance {
  using type = double;
  static constexpr Options::String help =
      "Pointwise tolerance for the error to the analytic solution";
};
}  // namespace OptionTags
namespace Tags {
struct ErrorTolerance : db::SimpleTag {
  using type = double;

  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<OptionTags::ErrorTolerance>;
  static type create_from_options(const type& option_value) noexcept {
    return option_value;
  }
};
}  // namespace Tags

struct InitializeRandomInitialData {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ElementId<Dim>& element_id, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    using fields_tag = typename system::fields_tag;
    using analytic_fields_tag =
        db::add_tag_prefix<::Tags::Analytic, fields_tag>;

    // Make random initial data distributed around the solution
    auto initial_fields = get<analytic_fields_tag>(box);
    std::mt19937 generator(std::hash<ElementId<Dim>>{}(element_id));
    std::uniform_real_distribution<> dist(-1., 1.);
    initial_fields += make_with_random_values<db::item_type<fields_tag>>(
        make_not_null(&generator), make_not_null(&dist), initial_fields);

    // auto initial_fields = db::item_type<fields_tag>{
    //     get<domain::Tags::Mesh<Dim>>(box).number_of_grid_points(), 0.};

    return std::make_tuple(
        ::Initialization::merge_into_databox<InitializeRandomInitialData,
                                             db::AddSimpleTags<fields_tag>>(
            std::move(box), std::move(initial_fields)));
  }
};

struct TestResult {
  using const_global_cache_tags = tmpl::list<Tags::ErrorTolerance>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    using fields_tag = typename system::fields_tag;
    using all_fields_tags = typename fields_tag::tags_list;

    const double tolerance = get<Tags::ErrorTolerance>(box);

    tmpl::for_each<all_fields_tags>([&box, &tolerance,
                                     &element_id](auto tag_v) {
      using tag = tmpl::type_from<decltype(tag_v)>;
      const auto& tensor = get<tag>(box);
      const auto& analytic_solution = get<::Tags::Analytic<tag>>(box);
      //   Parallel::printf(
      //       "Result for " + db::tag_name<tag>() + " on element %s:\n%s\n",
      //       element_id, tensor);
      for (size_t i = 0; i < tensor.size(); ++i) {
        const DataVector error = abs(tensor[i] - analytic_solution[i]);
        for (size_t j = 0; j < error.size(); ++j) {
          if (error[j] > tolerance) {
            Parallel::printf(db::tag_name<tag>() +
                                 " on element %s exceeds tolerance %e: %e\n",
                             element_id, tolerance, error[j]);
          }
          SPECTRE_PARALLEL_REQUIRE(error[j] <= tolerance);
        }
      }
    });

    return {std::move(box)};
  }
};

template <size_t Dim>
struct Metavariables {
  static constexpr const char* const help{
      "Test the Schwarz linear solver algorithm"};

  static constexpr size_t volume_dim = Dim;

  static constexpr bool massive_operator = true;

  // Choose a first-order Poisson system
  using system = Poisson::FirstOrderSystem<Dim, Poisson::Geometry::Euclidean>;
  using fields_tag = typename system::fields_tag;
  using primal_fields = typename system::primal_fields;
  using auxiliary_fields = typename system::auxiliary_fields;
  using fluxes_computer_tag =
      elliptic::Tags::FluxesComputer<typename system::fluxes>;

  // Choose a simple analytic solution
  using analytic_solution = Poisson::Solutions::ProductOfSinusoids<Dim>;
  using analytic_solution_tag = ::Tags::AnalyticSolution<analytic_solution>;

  // Use the analytic solution to impose boundary conditions
  using boundary_conditions = analytic_solution;

  // Choose a numerical flux
  using normal_dot_numerical_flux = ::Tags::NumericalFlux<
      elliptic::dg::NumericalFluxes::FirstOrderInternalPenalty<
          volume_dim, fluxes_computer_tag, primal_fields, auxiliary_fields>>;

  // Choose a strong first-order DG scheme
  using linear_solver_iteration_id =
      LinearSolver::Tags::IterationId<SchwarzSmoother>;
  using boundary_scheme = dg::FirstOrderScheme::FirstOrderScheme<
      Dim, fields_tag, normal_dot_numerical_flux, linear_solver_iteration_id,
      massive_operator>;

  // Set up the Schwarz smoother
  using subdomain_operator = elliptic::dg::SubdomainOperator<
      volume_dim, primal_fields, auxiliary_fields, fluxes_computer_tag,
      tmpl::list<>, typename system::sources, tmpl::list<>,
      normal_dot_numerical_flux, SchwarzSmoother, tmpl::list<>,
      massive_operator>;
  using subdomain_preconditioner =
      LinearSolver::Schwarz::subdomain_preconditioners::ExplicitInverse<
          volume_dim>;
  using linear_solver =
      LinearSolver::Schwarz::Schwarz<Metavariables, fields_tag, SchwarzSmoother,
                                     subdomain_operator,
                                     subdomain_preconditioner>;
  using preconditioner = void;

  // Set up observations
  using system_fields = typename fields_tag::tags_list;
  using observe_fields = tmpl::append<
      system_fields, db::wrap_tags_in<::Tags::FixedSource, system_fields>,
      db::wrap_tags_in<LinearSolver::Tags::Residual, system_fields>,
      tmpl::list<LinearSolver::Schwarz::Tags::Weight<SchwarzSmoother>,
                 LinearSolver::Schwarz::Tags::SummedIntrudingOverlapWeights<
                     Dim, SchwarzSmoother>>>;
  using analytic_solution_fields = system_fields;
  struct element_observation_type {};
  using events =
      tmpl::list<dg::Events::Registrars::ObserveFields<
                     volume_dim, linear_solver_iteration_id, observe_fields,
                     analytic_solution_fields>,
                 dg::Events::Registrars::ObserveErrorNorms<
                     linear_solver_iteration_id, analytic_solution_fields>>;
  using triggers = tmpl::list<elliptic::Triggers::Registrars::EveryNIterations<
      linear_solver_iteration_id>>;
  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::flatten<tmpl::list<
          typename Event<events>::creatable_classes, linear_solver>>>;

  // Collect all items to store in the cache.
  using const_global_cache_tags =
      tmpl::list<analytic_solution_tag, fluxes_computer_tag,
                 normal_dot_numerical_flux,
                 ::Tags::EventsAndTriggers<events, triggers>>;

  // Define the phases of the executable
  enum class Phase {
    Initialization,
    RegisterWithObserver,
    Smooth,
    TestResult,
    Exit
  };

  using initialization_actions = tmpl::list<
      // Domain geometry
      dg::Actions::InitializeDomain<Dim>,
      dg::Actions::InitializeInterfaces<
          system, dg::Initialization::slice_tags_to_face<>,
          dg::Initialization::slice_tags_to_exterior<>,
          dg::Initialization::face_compute_tags<>,
          dg::Initialization::exterior_compute_tags<>, false, false>,
      // Analytic solution
      elliptic::Actions::InitializeAnalyticSolution<analytic_solution_tag,
                                                    analytic_solution_fields>,
      // Initial data
      InitializeRandomInitialData,
      // Fixed sources
      elliptic::Actions::InitializeFixedSources,
      // Boundary conditions
      elliptic::dg::Actions::ImposeInhomogeneousBoundaryConditionsOnSource<
          Metavariables>,
      // DG operator
      dg::Actions::InitializeMortars<boundary_scheme>,
      elliptic::dg::Actions::InitializeFirstOrderOperator<
          volume_dim, typename system::fluxes, typename system::sources,
          fields_tag, primal_fields, auxiliary_fields>,
      // Linear solver
      typename linear_solver::initialize_element,
      // Subdomain geometry
      elliptic::dg::Actions::InitializeSubdomain<Dim, SchwarzSmoother>,
      // Diagnostics
      ::Initialization::Actions::AddComputeTags<
          tmpl::list<LinearSolver::Schwarz::Tags::SummedIntrudingOverlapWeights<
              Dim, SchwarzSmoother>>>,
      ::Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  using build_linear_operator_actions = tmpl::list<
      dg::Actions::CollectDataForFluxes<
          boundary_scheme, domain::Tags::InternalDirections<volume_dim>>,
      dg::Actions::SendDataForFluxes<boundary_scheme>,
      Actions::MutateApply<elliptic::FirstOrderOperator<
          volume_dim, LinearSolver::Tags::OperatorAppliedTo, fields_tag,
          massive_operator>>,
      elliptic::dg::Actions::ImposeHomogeneousDirichletBoundaryConditions<
          fields_tag, primal_fields>,
      dg::Actions::CollectDataForFluxes<
          boundary_scheme,
          domain::Tags::BoundaryDirectionsInterior<volume_dim>>,
      dg::Actions::ReceiveDataForFluxes<boundary_scheme>,
      Actions::MutateApply<boundary_scheme>>;

  using register_actions =
      tmpl::list<observers::Actions::RegisterEventsWithObservers,
                 typename linear_solver::register_element,
                 Parallel::Actions::TerminatePhase>;

  using smooth_actions = tmpl::list<
      build_linear_operator_actions,
      typename linear_solver::template solve<tmpl::list<
          Actions::RunEventsAndTriggers, build_linear_operator_actions>>,
      Parallel::Actions::TerminatePhase>;

  using test_actions =
      tmpl::list<TestResult, Parallel::Actions::TerminatePhase>;

  using dg_element_array = elliptic::DgElementArray<
      Metavariables,
      tmpl::list<
          Parallel::PhaseActions<Phase, Phase::Initialization,
                                 initialization_actions>,
          Parallel::PhaseActions<Phase, Phase::RegisterWithObserver,
                                 register_actions>,
          Parallel::PhaseActions<Phase, Phase::Smooth, smooth_actions>,
          Parallel::PhaseActions<Phase, Phase::TestResult, test_actions>>>;

  using component_list = tmpl::flatten<
      tmpl::list<dg_element_array, typename linear_solver::component_list,
                 observers::Observer<Metavariables>,
                 observers::ObserverWriter<Metavariables>>>;

  // Specify the transitions between phases.
  static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_GlobalCache<
          Metavariables>& /*cache_proxy*/) noexcept {
    switch (current_phase) {
      case Phase::Initialization:
        return Phase::RegisterWithObserver;
      case Phase::RegisterWithObserver:
        return Phase::Smooth;
      case Phase::Smooth:
        return Phase::TestResult;
      case Phase::TestResult:
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

  static constexpr bool ignore_unrecognized_command_line_options = false;
};

}  // namespace

using metavariables = Metavariables<2>;

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &domain::creators::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<
        Event<metavariables::events>>,
    &Parallel::register_derived_classes_with_charm<
        Trigger<metavariables::triggers>>};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};

using charmxx_main_component = Parallel::Main<metavariables>;

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
