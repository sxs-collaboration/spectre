// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Domain/Creators/AlignedLattice.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Creators/Tags/InitialExtents.hpp"
#include "Domain/Creators/Tags/InitialRefinementLevels.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Elliptic/Actions/InitializeAnalyticSolution.hpp"
#include "Elliptic/Actions/InitializeFixedSources.hpp"
#include "Elliptic/BoundaryConditions/AnalyticSolution.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/DiscontinuousGalerkin/Actions/ApplyOperator.hpp"
#include "Elliptic/DiscontinuousGalerkin/Actions/InitializeDomain.hpp"
#include "Elliptic/DiscontinuousGalerkin/Tags.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/Actions/Goto.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "ParallelAlgorithms/Actions/SetData.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Amr/Actions/Component.hpp"
#include "ParallelAlgorithms/Amr/Actions/EvaluateRefinementCriteria.hpp"
#include "ParallelAlgorithms/Amr/Actions/Initialize.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Random.hpp"
#include "ParallelAlgorithms/Amr/Policies/Isotropy.hpp"
#include "ParallelAlgorithms/Amr/Policies/Limits.hpp"
#include "ParallelAlgorithms/Amr/Policies/Policies.hpp"
#include "ParallelAlgorithms/Amr/Policies/Tags.hpp"
#include "ParallelAlgorithms/Amr/Protocols/AmrMetavariables.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/Lorentzian.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/ProductOfSinusoids.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/CartesianProduct.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace {

template <typename Tag>
struct DgOperatorAppliedTo : db::SimpleTag, db::PrefixTag {
  using type = typename Tag::type;
  using tag = Tag;
};

template <typename Tag>
struct Var : db::SimpleTag, db::PrefixTag {
  using type = typename Tag::type;
  using tag = Tag;
};

struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct AuxFieldTag : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim>;
};

struct TemporalIdTag : db::SimpleTag {
  using type = size_t;
};

struct IncrementTemporalId {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate<TemporalIdTag>([](const auto temporal_id) { (*temporal_id)++; },
                              make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

// Label to indicate the start of the apply-operator actions
struct ApplyOperatorStart {};

template <typename System, bool Linearized, typename Metavariables>
struct ElementArray {
  static constexpr size_t Dim = System::volume_dim;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Dim>;
  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<Dim>>;

  // Wrap fields in a prefix to make sure this works
  using primal_vars = db::wrap_tags_in<Var, typename System::primal_fields>;
  using primal_fluxes_vars =
      db::wrap_tags_in<Var, typename System::primal_fluxes>;
  using vars_tag = ::Tags::Variables<primal_vars>;
  using primal_fluxes_vars_tag = ::Tags::Variables<primal_fluxes_vars>;
  using operator_applied_to_vars_tag =
      ::Tags::Variables<db::wrap_tags_in<DgOperatorAppliedTo, primal_vars>>;
  using dg_operator =
      ::elliptic::dg::Actions::DgOperator<System, Linearized, TemporalIdTag,
                                          vars_tag, primal_fluxes_vars_tag,
                                          operator_applied_to_vars_tag>;
  // Don't wrap the fixed sources in the `Var` prefix because typically we want
  // to impose inhomogeneous boundary conditions on the un-prefixed vars, i.e.
  // not necessarily the vars we apply the linearized operator to
  using fixed_sources_tag = ::Tags::Variables<
      db::wrap_tags_in<::Tags::FixedSource, typename System::primal_fields>>;
  using analytic_solution_tag =
      ::Tags::AnalyticSolution<typename metavariables::analytic_solution>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<
                         tmpl::list<domain::Tags::InitialRefinementLevels<Dim>,
                                    domain::Tags::InitialExtents<Dim>>>,
                     ::elliptic::dg::Actions::InitializeDomain<Dim>,
                     ::elliptic::Actions::InitializeOptionalAnalyticSolution<
                         Dim, analytic_solution_tag,
                         tmpl::append<typename System::primal_fields,
                                      typename System::primal_fluxes>,
                         typename metavariables::analytic_solution>,
                     ::elliptic::Actions::InitializeFixedSources<
                         System, analytic_solution_tag>,
                     ::elliptic::dg::Actions::initialize_operator<System>,
                     Initialization::Actions::InitializeItems<
                         ::amr::Initialization::Initialize<Dim>>,
                     Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<::elliptic::dg::Actions::
                         ImposeInhomogeneousBoundaryConditionsOnSource<
                             System, fixed_sources_tag>,
                     ::Actions::Label<ApplyOperatorStart>,
                     typename dg_operator::apply_actions, IncrementTemporalId,
                     Parallel::Actions::TerminatePhase>>>;
};

template <typename Metavariables>
struct AmrComponent {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockSingletonChare;
  using array_index = int;
  using component_being_mocked = ::amr::Component<Metavariables>;
  using const_global_cache_tags =
      tmpl::list<::amr::Criteria::Tags::Criteria, ::amr::Tags::Policies,
                 logging::Tags::Verbosity<amr::OptionTags::AmrGroup>>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<tmpl::list<>>>>>;
};

template <typename Metavariables>
struct ProjectTemporalId : tt::ConformsTo<::amr::protocols::Projector> {
  using return_tags =
      tmpl::list<TemporalIdTag,
                 // Work around a segfault because this tag isn't handled
                 // correctly by the testing framework
                 Parallel::Tags::GlobalCacheImpl<Metavariables>>;
  using argument_tags = tmpl::list<>;
  // p-refinement
  template <size_t Dim>
  static void apply(
      const gsl::not_null<size_t*> /*temporal_id*/,
      const gsl::not_null<Parallel::GlobalCache<Metavariables>**> /*cache*/,
      const std::pair<Mesh<Dim>, Element<Dim>>& /*old_mesh_and_element*/) {}
  // h-refinement
  template <typename... ParentTags>
  static void apply(
      const gsl::not_null<size_t*> temporal_id,
      const gsl::not_null<Parallel::GlobalCache<Metavariables>**> /*cache*/,
      const tuples::TaggedTuple<ParentTags...>& parent_items) {
    *temporal_id = get<TemporalIdTag>(parent_items);
  }
  // h-coarsening
  template <size_t Dim, typename... ChildTags>
  static void apply(
      const gsl::not_null<size_t*> temporal_id,
      const gsl::not_null<Parallel::GlobalCache<Metavariables>**> /*cache*/,
      const std::unordered_map<
          ElementId<Dim>, tuples::TaggedTuple<ChildTags...>>& children_items) {
    *temporal_id = get<TemporalIdTag>(children_items.begin()->second);
  }
};

template <typename System, bool Linearized, typename AnalyticSolution>
struct Metavariables {
  static constexpr size_t volume_dim = System::volume_dim;
  using analytic_solution = AnalyticSolution;
  using element_array = ElementArray<System, Linearized, Metavariables>;
  using amr_component = AmrComponent<Metavariables>;
  using component_list = tmpl::list<element_array, amr_component>;
  using const_global_cache_tags =
      tmpl::list<::Tags::AnalyticSolution<AnalyticSolution>>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<
            elliptic::BoundaryConditions::BoundaryCondition<System::volume_dim>,
            tmpl::list<elliptic::BoundaryConditions::AnalyticSolution<System>>>,
        tmpl::pair<elliptic::analytic_data::AnalyticSolution,
                   tmpl::list<AnalyticSolution>>,
        tmpl::pair<AnalyticSolution, tmpl::list<AnalyticSolution>>,
        tmpl::pair<::amr::Criterion, tmpl::list<::amr::Criteria::Random>>>;
  };
  struct amr : tt::ConformsTo<::amr::protocols::AmrMetavariables> {
    using projectors = tmpl::flatten<tmpl::list<
        ProjectTemporalId<Metavariables>,
        ::amr::projectors::DefaultInitialize<
            tmpl::list<domain::Tags::InitialExtents<volume_dim>,
                       domain::Tags::InitialRefinementLevels<volume_dim>,
                       typename element_array::vars_tag>>,
        elliptic::dg::ProjectGeometry<volume_dim>,
        elliptic::Actions::InitializeFixedSources<
            System, typename element_array::analytic_solution_tag>,
        ::elliptic::Actions::InitializeOptionalAnalyticSolution<
            volume_dim, typename element_array::analytic_solution_tag,
            tmpl::append<typename System::primal_fields,
                         typename System::primal_fluxes>,
            analytic_solution>,
        elliptic::dg::Actions::amr_projectors<
            System, typename element_array::analytic_solution_tag>,
        typename element_array::dg_operator::amr_projectors>>;
  };

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};

template <
    typename System, bool Linearized, typename AnalyticSolution,
    size_t Dim = System::volume_dim,
    typename Metavars = Metavariables<System, Linearized, AnalyticSolution>,
    typename ElementArray = typename Metavars::element_array>
void test_dg_operator(
    const DomainCreator<Dim>& domain_creator, const double penalty_parameter,
    const bool use_massive_dg_operator, const Spectral::Quadrature quadrature,
    const AnalyticSolution& analytic_solution,
    // NOLINTNEXTLINE(google-runtime-references)
    Approx& analytic_solution_aux_approx,
    // NOLINTNEXTLINE(google-runtime-references)
    Approx& analytic_solution_operator_approx,
    const std::vector<std::tuple<
        std::unordered_map<ElementId<Dim>,
                           typename ElementArray::vars_tag::type>,
        std::unordered_map<ElementId<Dim>,
                           typename ElementArray::primal_fluxes_vars_tag::type>,
        std::unordered_map<
            ElementId<Dim>,
            typename ElementArray::operator_applied_to_vars_tag::type>>>&
        tests_data,
    const bool test_amr = false) {
  CAPTURE(penalty_parameter);
  CAPTURE(use_massive_dg_operator);

  using element_array = ElementArray;
  using vars_tag = typename element_array::vars_tag;
  using primal_fluxes_vars_tag = typename element_array::primal_fluxes_vars_tag;
  using operator_applied_to_vars_tag =
      typename element_array::operator_applied_to_vars_tag;
  using fixed_sources_tag = typename element_array::fixed_sources_tag;
  using Vars = typename vars_tag::type;
  using PrimalFluxesVars = typename primal_fluxes_vars_tag::type;
  using OperatorAppliedToVars = typename operator_applied_to_vars_tag::type;

  register_factory_classes_with_charm<Metavars>();

  // Get a list of all elements in the domain
  auto domain = domain_creator.create_domain();
  auto boundary_conditions = domain_creator.external_boundary_conditions();
  const auto initial_ref_levs = domain_creator.initial_refinement_levels();
  const auto initial_extents = domain_creator.initial_extents();
  std::unordered_set<ElementId<Dim>> all_element_ids{};
  for (const auto& block : domain.blocks()) {
    auto block_element_ids =
        initial_element_ids(block.id(), initial_ref_levs[block.id()]);
    for (auto& element_id : block_element_ids) {
      all_element_ids.insert(std::move(element_id));
    }
  }

  // Configure AMR criteria so we _increase_ (never decrease) resolution
  std::vector<std::unique_ptr<::amr::Criterion>> amr_criteria{};
  amr_criteria.push_back(std::make_unique<::amr::Criteria::Random>(
      std::unordered_map<::amr::Flag, size_t>{
          // h-refinement is not supported yet in the action testing framework
          {::amr::Flag::Split, 0},
          {::amr::Flag::IncreaseResolution, 1},
          {::amr::Flag::DoNothing, 1}}));

  ActionTesting::MockRuntimeSystem<Metavars> runner{tuples::TaggedTuple<
      domain::Tags::Domain<Dim>, domain::Tags::FunctionsOfTimeInitialize,
      domain::Tags::ExternalBoundaryConditions<Dim>,
      ::elliptic::dg::Tags::PenaltyParameter, ::elliptic::dg::Tags::Massive,
      ::elliptic::dg::Tags::Quadrature, ::elliptic::dg::Tags::Formulation,
      ::Tags::AnalyticSolution<AnalyticSolution>,
      ::amr::Criteria::Tags::Criteria, ::amr::Tags::Policies,
      logging::Tags::Verbosity<::amr::OptionTags::AmrGroup>>{
      std::move(domain), domain_creator.functions_of_time(),
      std::move(boundary_conditions), penalty_parameter,
      use_massive_dg_operator, quadrature, ::dg::Formulation::StrongInertial,
      analytic_solution, std::move(amr_criteria),
      ::amr::Policies{::amr::Isotropy::Anisotropic, ::amr::Limits{}},
      ::Verbosity::Debug}};

  // DataBox shortcuts
  const auto get_tag =
      [&runner](auto tag_v,
                const ElementId<Dim>& local_element_id) -> const auto& {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                              local_element_id);
  };
  const auto set_tag = [&runner](auto tag_v, auto value,
                                 const ElementId<Dim>& local_element_id) {
    using tag = std::decay_t<decltype(tag_v)>;
    ActionTesting::simple_action<element_array,
                                 ::Actions::SetData<tmpl::list<tag>>>(
        make_not_null(&runner), local_element_id, std::move(value));
  };

  // Initialize all elements in the domain
  for (const auto& element_id : all_element_ids) {
    ActionTesting::emplace_component_and_initialize<element_array>(
        &runner, element_id, {initial_ref_levs, initial_extents});
    while (
        not ActionTesting::get_terminate<element_array>(runner, element_id)) {
      ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                element_id);
    }
    set_tag(TemporalIdTag{}, 0_st, element_id);
  }
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  if constexpr (Linearized) {
    INFO(
        "Test imposing inhomogeneous boundary conditions on source, i.e. b -= "
        "A(x=0)");
    for (const auto& element_id : all_element_ids) {
      ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                element_id);
    }
    // The analytic-solution test below uses the fixed sources that were
    // modified here.
  }

  const auto apply_operator_and_check_result =
      [&runner, &all_element_ids, &get_tag, &set_tag](
          const std::unordered_map<ElementId<Dim>, Vars>& all_vars,
          const std::unordered_map<ElementId<Dim>, PrimalFluxesVars>&
              all_expected_primal_fluxes_vars,
          const std::unordered_map<ElementId<Dim>, OperatorAppliedToVars>&
              all_expected_operator_applied_to_vars,
          Approx& custom_aux_approx = approx,
          Approx& custom_operator_approx = approx) {
        // Set variables data on central element and on its neighbors
        for (const auto& element_id : all_element_ids) {
          const auto vars = all_vars.find(element_id);
          if (vars == all_vars.end()) {
            const size_t num_points =
                get_tag(domain::Tags::Mesh<Dim>{}, element_id)
                    .number_of_grid_points();
            set_tag(vars_tag{}, Vars{num_points, 0.}, element_id);
          } else {
            set_tag(vars_tag{}, vars->second, element_id);
          }
        }

        // Apply DG operator
        // 1. Prepare elements and send data
        for (const auto& element_id : all_element_ids) {
          runner.template force_next_action_to_be<
              element_array, ::Actions::Label<ApplyOperatorStart>>(element_id);
          runner.template mock_distributed_objects<element_array>()
              .at(element_id)
              .set_terminate(false);
          while (not ActionTesting::get_terminate<element_array>(runner,
                                                                 element_id) and
                 ActionTesting::next_action_if_ready<element_array>(
                     make_not_null(&runner), element_id)) {
          }
        }
        // 2. Receive data and apply operator
        for (const auto& element_id : all_element_ids) {
          while (not ActionTesting::get_terminate<element_array>(runner,
                                                                 element_id)) {
            ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                      element_id);
          }
        }

        // Check result
        {
          INFO("Auxiliary variables");
          for (const auto& [element_id, expected_primal_fluxes_vars] :
               all_expected_primal_fluxes_vars) {
            CAPTURE(element_id);
            const auto& inertial_coords = get_tag(
                domain::Tags::Coordinates<Dim, Frame::Inertial>{}, element_id);
            CAPTURE(inertial_coords);
            const auto& vars = get_tag(vars_tag{}, element_id);
            CAPTURE(vars);
            const auto& primal_fluxes_vars =
                get_tag(primal_fluxes_vars_tag{}, element_id);
            // Working around a bug in GCC 7.3 that fails to handle the
            // structured bindings here
            const auto& local_expected_primal_fluxes_vars =
                expected_primal_fluxes_vars;
            CHECK_VARIABLES_CUSTOM_APPROX(primal_fluxes_vars,
                                          local_expected_primal_fluxes_vars,
                                          custom_aux_approx);
          }
        }
        {
          INFO("Operator applied to variables");
          for (const auto& [element_id, expected_operator_applied_to_vars] :
               all_expected_operator_applied_to_vars) {
            CAPTURE(element_id);
            const auto& inertial_coords = get_tag(
                domain::Tags::Coordinates<Dim, Frame::Inertial>{}, element_id);
            CAPTURE(inertial_coords);
            const auto& vars = get_tag(vars_tag{}, element_id);
            CAPTURE(vars);
            const auto& operator_applied_to_vars =
                get_tag(operator_applied_to_vars_tag{}, element_id);
            // Working around a bug in GCC 7.3 that fails to handle the
            // structured bindings here
            const auto& local_expected_operator_applied_to_vars =
                expected_operator_applied_to_vars;
            CHECK_VARIABLES_CUSTOM_APPROX(
                operator_applied_to_vars,
                local_expected_operator_applied_to_vars,
                custom_operator_approx);
          }
        }
      };
  {
    INFO("Test that A(0) = 0");
    std::unordered_map<ElementId<Dim>, PrimalFluxesVars>
        all_zero_primal_fluxes{};
    std::unordered_map<ElementId<Dim>, OperatorAppliedToVars>
        all_zero_operator_vars{};
    for (const auto& element_id : all_element_ids) {
      const size_t num_points = get_tag(domain::Tags::Mesh<Dim>{}, element_id)
                                    .number_of_grid_points();
      all_zero_primal_fluxes[element_id] = PrimalFluxesVars{num_points, 0.};
      all_zero_operator_vars[element_id] =
          OperatorAppliedToVars{num_points, 0.};
    }
    apply_operator_and_check_result({}, all_zero_primal_fluxes,
                                    all_zero_operator_vars);
  }
  const auto test_analytic_solution =
      [&analytic_solution, &all_element_ids, &get_tag,
       &apply_operator_and_check_result, &analytic_solution_aux_approx,
       &analytic_solution_operator_approx]() {
        INFO("Test A(x) = b with analytic solution");
        std::unordered_map<ElementId<Dim>, Vars> analytic_primal_vars{};
        std::unordered_map<ElementId<Dim>, PrimalFluxesVars>
            analytic_primal_fluxes{};
        std::unordered_map<ElementId<Dim>, OperatorAppliedToVars>
            analytic_fixed_source_with_inhom_bc{};
        for (const auto& element_id : all_element_ids) {
          const auto& inertial_coords = get_tag(
              domain::Tags::Coordinates<Dim, Frame::Inertial>{}, element_id);
          analytic_primal_vars[element_id] =
              variables_from_tagged_tuple(analytic_solution.variables(
                  inertial_coords, typename System::primal_fields{}));
          analytic_primal_fluxes[element_id] =
              variables_from_tagged_tuple(analytic_solution.variables(
                  inertial_coords, typename System::primal_fluxes{}));
          analytic_fixed_source_with_inhom_bc[element_id] =
              get_tag(fixed_sources_tag{}, element_id);
        }
        apply_operator_and_check_result(
            analytic_primal_vars, analytic_primal_fluxes,
            analytic_fixed_source_with_inhom_bc, analytic_solution_aux_approx,
            analytic_solution_operator_approx);
      };
  test_analytic_solution();
  {
    INFO("Test A(x) = b with custom x and b");
    for (const auto& test_data : tests_data) {
      std::apply(apply_operator_and_check_result, test_data);
    }
  }

  // Stop here if we're not testing AMR
  if (not test_amr) {
    return;
  }

  INFO("Run AMR!");
  const auto invoke_all_simple_actions = [&runner, &all_element_ids]() {
    bool quiescence = false;
    while (not quiescence) {
      quiescence = true;
      for (const auto& element_id : all_element_ids) {
        while (not ActionTesting::is_simple_action_queue_empty<element_array>(
            runner, element_id)) {
          ActionTesting::invoke_queued_simple_action<element_array>(
              make_not_null(&runner), element_id);
          quiescence = false;
        }
      }
    }
  };
  using amr_component = typename Metavars::amr_component;
  ActionTesting::emplace_singleton_component<amr_component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0});
  auto& cache = ActionTesting::cache<amr_component>(runner, 0);
  Parallel::simple_action<::amr::Actions::EvaluateRefinementCriteria>(
      Parallel::get_parallel_component<element_array>(cache));
  invoke_all_simple_actions();
  Parallel::simple_action<::amr::Actions::AdjustDomain>(
      Parallel::get_parallel_component<element_array>(cache));
  invoke_all_simple_actions();
  // h-refinement is not supported yet in the action testing framework
  // // Invoke `CreateChild` on AMR component
  // while (not ActionTesting::is_simple_action_queue_empty<amr_component>(
  //              runner, 0)) {
  //   ActionTesting::invoke_queued_simple_action<amr_component>(
  //       make_not_null(&runner), 0);
  // }
  // // Update list of element IDs
  // all_element_ids.clear();
  // for (const auto& [element_id, element] :
  //      runner.template mock_distributed_objects<element_array>()) {
  //   (void)element;
  //   all_element_ids.insert(element_id);
  // }
  // CAPTURE(all_element_ids);
  // Go to the start of the testing phase to run the
  // `ImposeInhomogeneousBoundaryConditionsOnSource` action again
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  if constexpr (Linearized) {
    for (const auto& element_id : all_element_ids) {
      ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                element_id);
    }
  }

  // Test analytic solution again after AMR. It should still pass because we
  // only refined, never coarsened.
  test_analytic_solution();
}

}  // namespace

// [[TimeOut, 10]]
SPECTRE_TEST_CASE("Unit.Elliptic.DG.Operator", "[Unit][Elliptic]") {
  // Needed for Brick
  using VariantType = std::variant<
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>,
      domain::creators::Brick::LowerUpperBoundaryCondition<
          domain::BoundaryConditions::BoundaryCondition>>;

  domain::creators::register_derived_with_charm();
  // This is what the tests below check:
  //
  // - The DG operator passes some basic consistency checks, e.g. A(0) = 0.
  // - It is a numerical approximation of the system equation, i.e. A(x) = b for
  //   an analytic solution x and corresponding fixed source b.
  // - It produces an expected output from a random (but hard-coded) set of
  //   variables. This is an important regression test, i.e. it ensures the
  //   output does not change with optimizations etc.
  // - It runs on various domain geometries. Currently only a few rectilinear
  //   domains are tested, but this can be expanded.
  // - The free functions, actions and initialization actions are compatible.
  //
  // Notes:
  //
  // - Relative tolerances for analytic-solution tests include a
  //   problem-dependent scale. The scale is set by the analytic solution and
  //   has an additional penalty factor C * p^2 / h for the DG operator, where C
  //   is the penalty parameter, p is the number of grid points and h is the
  //   size of an element (in the least-resolved dimension).
  // - Hard-coded numbers in regression tests are always compared with default
  //   precision (1.e-14 or so).
  const double penalty_parameter = 1.5;
  {
    INFO("1D rectilinear");
    using system =
        Poisson::FirstOrderSystem<1, Poisson::Geometry::FlatCartesian>;
    Poisson::Solutions::ProductOfSinusoids<1> analytic_solution{{{M_PI}}};
    {
      INFO("Regression tests");
      // Domain decomposition:
      // [ | | | ]-> xi
      const domain::creators::Interval domain_creator{
          {{-0.5}},
          {{1.5}},
          {{2}},
          {{4}},
          std::make_unique<
              elliptic::BoundaryConditions::AnalyticSolution<system>>(
              analytic_solution.get_clone(),
              elliptic::BoundaryConditionType::Dirichlet),
          std::make_unique<
              elliptic::BoundaryConditions::AnalyticSolution<system>>(
              analytic_solution.get_clone(),
              elliptic::BoundaryConditionType::Neumann)};
      const ElementId<1> left_id{0, {{{2, 0}}}};
      const ElementId<1> midleft_id{0, {{{2, 1}}}};
      const ElementId<1> midright_id{0, {{{2, 2}}}};
      const ElementId<1> right_id{0, {{{2, 3}}}};
      using Vars = Variables<tmpl::list<Var<Poisson::Tags::Field>>>;
      using PrimalFluxes = Variables<tmpl::list<Var<::Tags::Flux<
          Poisson::Tags::Field, tmpl::size_t<1>, Frame::Inertial>>>>;
      using OperatorVars =
          Variables<tmpl::list<DgOperatorAppliedTo<Var<Poisson::Tags::Field>>>>;
      Vars vars_rnd_left{4};
      get(get<Var<Poisson::Tags::Field>>(vars_rnd_left)) =
          DataVector{0.6964691855978616, 0.28613933495037946,
                     0.2268514535642031, 0.5513147690828912};
      Vars vars_rnd_midleft{4};
      get(get<Var<Poisson::Tags::Field>>(vars_rnd_midleft)) =
          DataVector{0.7194689697855631, 0.42310646012446096,
                     0.9807641983846155, 0.6848297385848633};
      Vars vars_rnd_midright{4};
      get(get<Var<Poisson::Tags::Field>>(vars_rnd_midright)) =
          DataVector{0.48093190148436094, 0.3921175181941505,
                     0.3431780161508694, 0.7290497073840416};
      Vars vars_rnd_right{4};
      get(get<Var<Poisson::Tags::Field>>(vars_rnd_right)) =
          DataVector{0.4385722446796244, 0.05967789660956835,
                     0.3980442553304314, 0.7379954057320357};
      PrimalFluxes expected_primal_fluxes_rnd_left{4};
      get<0>(get<Var<::Tags::Flux<Poisson::Tags::Field, tmpl::size_t<1>,
                                  Frame::Inertial>>>(
          expected_primal_fluxes_rnd_left)) =
          DataVector{-4.027188081328469, -1.9207736184862605,
                     1.3653213194134493, 3.3207435803332324};
      PrimalFluxes expected_primal_fluxes_rnd_right{4};
      get<0>(get<Var<::Tags::Flux<Poisson::Tags::Field, tmpl::size_t<1>,
                                  Frame::Inertial>>>(
          expected_primal_fluxes_rnd_right)) =
          DataVector{-5.281316261986065, -0.5513540594510129,
                     2.6634207403536934, 1.9071387227305365};
      OperatorVars expected_operator_vars_rnd_left{4};
      get(get<DgOperatorAppliedTo<Var<Poisson::Tags::Field>>>(
          expected_operator_vars_rnd_left)) =
          DataVector{1384.5953530154895, 41.55242721423977, -41.54933376905333,
                     -32.40787768105913};
      OperatorVars expected_operator_vars_rnd_right{4};
      get(get<DgOperatorAppliedTo<Var<Poisson::Tags::Field>>>(
          expected_operator_vars_rnd_right)) =
          DataVector{-215.34463496424186, -37.924582769790135,
                     2.1993037260176997, 51.85418137198471};
      // Large tolerances for the comparison to the analytic solution because
      // this regression test runs at very low resolution. Below is another
      // analytic-solution test at higher resolution. The hard-coded numbers
      // are compared at higher (default) precision.
      Approx analytic_solution_aux_approx =
          Approx::custom().epsilon(3.e-2).scale(M_PI);
      Approx analytic_solution_operator_approx =
          Approx::custom().epsilon(3.e-2).scale(M_PI * penalty_parameter *
                                                square(4) / 0.5);
      test_dg_operator<system, true>(
          domain_creator, penalty_parameter, false,
          Spectral::Quadrature::GaussLobatto, analytic_solution,
          analytic_solution_aux_approx, analytic_solution_operator_approx,
          {{{{left_id, std::move(vars_rnd_left)},
             {midleft_id, std::move(vars_rnd_midleft)},
             {midright_id, std::move(vars_rnd_midright)},
             {right_id, std::move(vars_rnd_right)}},
            {{left_id, std::move(expected_primal_fluxes_rnd_left)},
             {right_id, std::move(expected_primal_fluxes_rnd_right)}},
            {{left_id, std::move(expected_operator_vars_rnd_left)},
             {right_id, std::move(expected_operator_vars_rnd_right)}}}});
    }
    {
      INFO("Higher-resolution analytic-solution tests");
      const domain::creators::Interval domain_creator{
          {{-0.5}},
          {{1.5}},
          {{1}},
          {{12}},
          std::make_unique<
              elliptic::BoundaryConditions::AnalyticSolution<system>>(
              analytic_solution.get_clone(),
              elliptic::BoundaryConditionType::Dirichlet),
          std::make_unique<
              elliptic::BoundaryConditions::AnalyticSolution<system>>(
              analytic_solution.get_clone(),
              elliptic::BoundaryConditionType::Neumann)};
      Approx analytic_solution_aux_approx =
          Approx::custom().epsilon(1.e-8).scale(M_PI);
      Approx analytic_solution_operator_approx =
          Approx::custom().epsilon(1.e-8).scale(M_PI * penalty_parameter *
                                                square(12));
      for (const auto& [use_massive_dg_operator, quadrature] :
           cartesian_product(make_array(true, false),
                             make_array(Spectral::Quadrature::GaussLobatto,
                                        Spectral::Quadrature::Gauss))) {
        test_dg_operator<system, true>(
            domain_creator, penalty_parameter, use_massive_dg_operator,
            quadrature, analytic_solution, analytic_solution_aux_approx,
            analytic_solution_operator_approx, {}, true);
      }
    }
  }
  {
    INFO("2D rectilinear");
    using system =
        Poisson::FirstOrderSystem<2, Poisson::Geometry::FlatCartesian>;
    Poisson::Solutions::ProductOfSinusoids<2> analytic_solution{{{M_PI, M_PI}}};
    {
      INFO("Regression tests");
      // Domain decomposition:
      // ^ eta
      // +-+-+> xi
      // | | |
      // +-+-+
      // | | |
      // +-+-+
      const domain::creators::Rectangle domain_creator{
          {{-0.5, 0.}},
          {{1.5, 1.}},
          {{1, 1}},
          {{3, 2}},
          std::make_unique<
              elliptic::BoundaryConditions::AnalyticSolution<system>>(
              analytic_solution.get_clone(),
              elliptic::BoundaryConditionType::Dirichlet),
          nullptr};
      const ElementId<2> northwest_id{0, {{{1, 0}, {1, 1}}}};
      const ElementId<2> southwest_id{0, {{{1, 0}, {1, 0}}}};
      const ElementId<2> northeast_id{0, {{{1, 1}, {1, 1}}}};
      using Vars = Variables<tmpl::list<Var<Poisson::Tags::Field>>>;
      using PrimalFluxes = Variables<tmpl::list<Var<::Tags::Flux<
          Poisson::Tags::Field, tmpl::size_t<2>, Frame::Inertial>>>>;
      using OperatorVars =
          Variables<tmpl::list<DgOperatorAppliedTo<Var<Poisson::Tags::Field>>>>;
      Vars vars_rnd_northwest{6};
      get(get<Var<Poisson::Tags::Field>>(vars_rnd_northwest)) = DataVector{
          0.9807641983846155, 0.6848297385848633, 0.48093190148436094,
          0.3921175181941505, 0.3431780161508694, 0.7290497073840416};
      Vars vars_rnd_southwest{6};
      get(get<Var<Poisson::Tags::Field>>(vars_rnd_southwest)) = DataVector{
          0.6964691855978616, 0.28613933495037946, 0.2268514535642031,
          0.5513147690828912, 0.7194689697855631,  0.42310646012446096};
      Vars vars_rnd_northeast{6};
      get(get<Var<Poisson::Tags::Field>>(vars_rnd_northeast)) = DataVector{
          0.5315513738418384, 0.5318275870968661, 0.6344009585513211,
          0.8494317940777896, 0.7244553248606352, 0.6110235106775829};
      PrimalFluxes expected_primal_fluxes_rnd{6};
      get<0>(get<Var<::Tags::Flux<Poisson::Tags::Field, tmpl::size_t<2>,
                                  Frame::Inertial>>>(
          expected_primal_fluxes_rnd)) = DataVector{
          -0.683905542298754,  -0.49983229690025466, -0.3157590515017552,
          -0.5326901973630155, 0.3369321891898911,   1.2065545757427976};
      get<1>(get<Var<::Tags::Flux<Poisson::Tags::Field, tmpl::size_t<2>,
                                  Frame::Inertial>>>(
          expected_primal_fluxes_rnd)) = DataVector{
          -1.1772933603809301, -0.6833034448679879, 0.4962356117993614,
          -1.1772933603809301, -0.6833034448679879, 0.4962356117993614};
      OperatorVars expected_operator_vars_rnd{6};
      get(get<DgOperatorAppliedTo<Var<Poisson::Tags::Field>>>(
          expected_operator_vars_rnd)) =
          DataVector{164.82044058319110, 9.68580366789113,  -2.370110768729976,
                     91.310963425225239, 30.31342257245155, 56.03237239864831};
      // Large tolerances for the comparison to the analytic solution because
      // this regression test runs at very low resolution. Below is another
      // analytic-solution test at higher resolution. The hard-coded numbers
      // are compared at higher (default) precision.
      Approx analytic_solution_aux_approx =
          Approx::custom().epsilon(7.e-1).scale(M_PI);
      Approx analytic_solution_operator_approx =
          Approx::custom().epsilon(7.e-1).scale(M_PI * penalty_parameter *
                                                square(3));
      test_dg_operator<system, true>(
          domain_creator, penalty_parameter, false,
          Spectral::Quadrature::GaussLobatto, analytic_solution,
          analytic_solution_aux_approx, analytic_solution_operator_approx,
          {{{{northwest_id, std::move(vars_rnd_northwest)},
             {southwest_id, std::move(vars_rnd_southwest)},
             {northeast_id, std::move(vars_rnd_northeast)}},
            {{northwest_id, std::move(expected_primal_fluxes_rnd)}},
            {{northwest_id, std::move(expected_operator_vars_rnd)}}}});
    }
    {
      INFO("Higher-resolution analytic-solution tests");
      const domain::creators::Rectangle domain_creator{
          {{-0.5, 0.}},
          {{1.5, 1.}},
          {{1, 1}},
          {{12, 12}},
          std::make_unique<
              elliptic::BoundaryConditions::AnalyticSolution<system>>(
              analytic_solution.get_clone(),
              elliptic::BoundaryConditionType::Dirichlet),
          nullptr};
      Approx analytic_solution_aux_approx =
          Approx::custom().epsilon(1.e-8).scale(M_PI);
      Approx analytic_solution_operator_approx =
          Approx::custom().epsilon(1.e-8).scale(M_PI * penalty_parameter *
                                                square(12));
      for (const auto& [use_massive_dg_operator, quadrature] :
           cartesian_product(make_array(true, false),
                             make_array(Spectral::Quadrature::GaussLobatto,
                                        Spectral::Quadrature::Gauss))) {
        test_dg_operator<system, true>(
            domain_creator, penalty_parameter, use_massive_dg_operator,
            quadrature, analytic_solution, analytic_solution_aux_approx,
            analytic_solution_operator_approx, {}, true);
      }
    }
  }
  {
    INFO("3D rectilinear");
    using system =
        Poisson::FirstOrderSystem<3, Poisson::Geometry::FlatCartesian>;
    Poisson::Solutions::ProductOfSinusoids<3> analytic_solution{
        {{M_PI, M_PI, M_PI}}};
    {
      INFO("Regression tests");
      const domain::creators::Brick domain_creator{
          {{-0.5, 0., -1.}},
          {{1.5, 1., 3.}},
          {{1, 1, 1}},
          {{2, 3, 4}},
          VariantType{std::make_unique<
              elliptic::BoundaryConditions::AnalyticSolution<system>>(
              analytic_solution.get_clone(),
              elliptic::BoundaryConditionType::Dirichlet)},
          VariantType{std::make_unique<
              elliptic::BoundaryConditions::AnalyticSolution<system>>(
              analytic_solution.get_clone(),
              elliptic::BoundaryConditionType::Dirichlet)},
          VariantType{std::make_unique<
              elliptic::BoundaryConditions::AnalyticSolution<system>>(
              analytic_solution.get_clone(),
              elliptic::BoundaryConditionType::Dirichlet)},
          nullptr};
      const ElementId<3> self_id{0, {{{1, 0}, {1, 0}, {1, 0}}}};
      const ElementId<3> neighbor_id_xi{0, {{{1, 1}, {1, 0}, {1, 0}}}};
      const ElementId<3> neighbor_id_eta{0, {{{1, 0}, {1, 1}, {1, 0}}}};
      const ElementId<3> neighbor_id_zeta{0, {{{1, 0}, {1, 0}, {1, 1}}}};
      using Vars = Variables<tmpl::list<Var<Poisson::Tags::Field>>>;
      using OperatorVars =
          Variables<tmpl::list<DgOperatorAppliedTo<Var<Poisson::Tags::Field>>>>;
      Vars vars_rnd_self{24};
      get(get<Var<Poisson::Tags::Field>>(vars_rnd_self)) = DataVector{
          0.6964691855978616, 0.28613933495037946, 0.2268514535642031,
          0.5513147690828912, 0.7194689697855631,  0.42310646012446096,
          0.9807641983846155, 0.6848297385848633,  0.48093190148436094,
          0.3921175181941505, 0.3431780161508694,  0.7290497073840416,
          0.4385722446796244, 0.05967789660956835, 0.3980442553304314,
          0.7379954057320357, 0.18249173045349998, 0.17545175614749253,
          0.5315513738418384, 0.5318275870968661,  0.6344009585513211,
          0.8494317940777896, 0.7244553248606352,  0.6110235106775829};
      Vars vars_rnd_neighbor_xi{24};
      get(get<Var<Poisson::Tags::Field>>(vars_rnd_neighbor_xi)) = DataVector{
          0.15112745234808023, 0.39887629272615654,  0.24085589772362448,
          0.34345601404832493, 0.5131281541990022,   0.6666245501640716,
          0.10590848505681383, 0.13089495066408074,  0.32198060646830806,
          0.6615643366662437,  0.8465062252707221,   0.5532573447967134,
          0.8544524875245048,  0.3848378112757611,   0.31678789711837974,
          0.3542646755916785,  0.17108182920509907,  0.8291126345018904,
          0.3386708459143266,  0.5523700752940731,   0.578551468108833,
          0.5215330593973323,  0.002688064574320692, 0.98834541928282};
      Vars vars_rnd_neighbor_eta{24};
      get(get<Var<Poisson::Tags::Field>>(vars_rnd_neighbor_eta)) = DataVector{
          0.5194851192598093, 0.6128945257629677,  0.12062866599032374,
          0.8263408005068332, 0.6030601284109274,  0.5450680064664649,
          0.3427638337743084, 0.3041207890271841,  0.4170222110247016,
          0.6813007657927966, 0.8754568417951749,  0.5104223374780111,
          0.6693137829622723, 0.5859365525622129,  0.6249035020955999,
          0.6746890509878248, 0.8423424376202573,  0.08319498833243877,
          0.7636828414433382, 0.243666374536874,   0.19422296057877086,
          0.5724569574914731, 0.09571251661238711, 0.8853268262751396};
      Vars vars_rnd_neighbor_zeta{24};
      get(get<Var<Poisson::Tags::Field>>(vars_rnd_neighbor_zeta)) = DataVector{
          0.7224433825702216,  0.3229589138531782,  0.3617886556223141,
          0.22826323087895561, 0.29371404638882936, 0.6309761238544878,
          0.09210493994507518, 0.43370117267952824, 0.4308627633296438,
          0.4936850976503062,  0.425830290295828,   0.3122612229724653,
          0.4263513069628082,  0.8933891631171348,  0.9441600182038796,
          0.5018366758843366,  0.6239529517921112,  0.11561839507929572,
          0.3172854818203209,  0.4148262119536318,  0.8663091578833659,
          0.2504553653965067,  0.48303426426270435, 0.985559785610705};
      OperatorVars expected_operator_vars_rnd{24};
      get(get<DgOperatorAppliedTo<Var<Poisson::Tags::Field>>>(
          expected_operator_vars_rnd)) = DataVector{
          490.4639148694344,  218.0462676038626,   40.25819450876480,
          83.251427119123945, 176.2577516255026,   -15.624072102187602,
          586.8311752841304,  391.1570577346553,   30.26559434437542,
          13.73062841146597,  -15.356914932182377, 100.25724641399877,
          266.84155600862254, 48.507637035283153,  15.93132587805462,
          23.036282532293448, -149.02371403160321, -142.12061361346408,
          314.95846885876898, 329.52673640830761,  46.245499085300622,
          70.51403818198222,  45.264231133848220,  90.061981328724073};
      // Large tolerances for the comparison to the analytic solution because
      // this regression test runs at very low resolution. Below is another
      // analytic-solution test at higher resolution. The hard-coded numbers
      // are compared at higher (default) precision.
      Approx analytic_solution_aux_approx =
          Approx::custom().epsilon(8.e-1).scale(M_PI);
      Approx analytic_solution_operator_approx =
          Approx::custom().epsilon(8.e-1).scale(M_PI * penalty_parameter *
                                                square(4) / 2.);
      test_dg_operator<system, true>(
          domain_creator, penalty_parameter, false,
          Spectral::Quadrature::GaussLobatto, analytic_solution,
          analytic_solution_aux_approx, analytic_solution_operator_approx,
          {{{{self_id, std::move(vars_rnd_self)},
             {neighbor_id_xi, std::move(vars_rnd_neighbor_xi)},
             {neighbor_id_eta, std::move(vars_rnd_neighbor_eta)},
             {neighbor_id_zeta, std::move(vars_rnd_neighbor_zeta)}},
            {},
            {{self_id, std::move(expected_operator_vars_rnd)}}}});
    }
    {
      INFO("Higher-resolution analytic-solution tests");
      const domain::creators::Brick domain_creator{
          {{-0.5, 0., -1.}},
          {{1.5, 1., 3.}},
          {{1, 1, 1}},
          {{12, 12, 12}},
          VariantType{std::make_unique<
              elliptic::BoundaryConditions::AnalyticSolution<system>>(
              analytic_solution.get_clone(),
              elliptic::BoundaryConditionType::Dirichlet)},
          VariantType{std::make_unique<
              elliptic::BoundaryConditions::AnalyticSolution<system>>(
              analytic_solution.get_clone(),
              elliptic::BoundaryConditionType::Dirichlet)},
          VariantType{std::make_unique<
              elliptic::BoundaryConditions::AnalyticSolution<system>>(
              analytic_solution.get_clone(),
              elliptic::BoundaryConditionType::Dirichlet)},
          nullptr};
      Approx analytic_solution_aux_approx =
          Approx::custom().epsilon(1.e-4).scale(M_PI);
      Approx analytic_solution_operator_approx =
          Approx::custom().epsilon(1.e-4).scale(M_PI * penalty_parameter *
                                                square(12) / 2.);
      for (const auto& [use_massive_dg_operator, quadrature] :
           cartesian_product(make_array(true, false),
                             make_array(Spectral::Quadrature::GaussLobatto,
                                        Spectral::Quadrature::Gauss))) {
        test_dg_operator<system, true>(
            domain_creator, penalty_parameter, use_massive_dg_operator,
            quadrature, analytic_solution, analytic_solution_aux_approx,
            analytic_solution_operator_approx, {}, true);
      }
    }
  }
  {
    INFO("2D with h-nonconforming boundary");
    using system =
        Poisson::FirstOrderSystem<2, Poisson::Geometry::FlatCartesian>;
    Poisson::Solutions::ProductOfSinusoids<2> analytic_solution{{{M_PI, M_PI}}};
    {
      INFO("Regression tests");
      // Domain decomposition:
      // ^ eta
      // +-+-+> xi
      // | | |
      // +-+ |
      // | | |
      // +-+-+
      const domain::creators::AlignedLattice<2> domain_creator{
          // Start with 2 unrefined blocks
          {{{0., 0.25, 0.5}, {0., 0.5}}},
          {{0, 0}},
          {{3, 2}},
          // Refine once in eta in left block in sketch above
          {{{{0, 0}}, {{1, 1}}, {{0, 1}}}},
          {},
          {},
          std::make_unique<
              elliptic::BoundaryConditions::AnalyticSolution<system>>(
              analytic_solution.get_clone(),
              elliptic::BoundaryConditionType::Dirichlet)};
      const ElementId<2> lowerleft_id{0, {{{0, 0}, {1, 0}}}};
      const ElementId<2> upperleft_id{0, {{{0, 0}, {1, 1}}}};
      const ElementId<2> right_id{1, {{{0, 0}, {0, 0}}}};
      using Vars = Variables<tmpl::list<Var<Poisson::Tags::Field>>>;
      using PrimalFluxes = Variables<tmpl::list<Var<::Tags::Flux<
          Poisson::Tags::Field, tmpl::size_t<2>, Frame::Inertial>>>>;
      using OperatorVars =
          Variables<tmpl::list<DgOperatorAppliedTo<Var<Poisson::Tags::Field>>>>;
      Vars vars_rnd_lowerleft{6};
      get(get<Var<Poisson::Tags::Field>>(vars_rnd_lowerleft)) = DataVector{
          0.9807641983846155, 0.6848297385848633, 0.48093190148436094,
          0.3921175181941505, 0.3431780161508694, 0.7290497073840416};
      Vars vars_rnd_upperleft{6};
      get(get<Var<Poisson::Tags::Field>>(vars_rnd_upperleft)) = DataVector{
          0.6964691855978616, 0.28613933495037946, 0.2268514535642031,
          0.5513147690828912, 0.7194689697855631,  0.42310646012446096};
      Vars vars_rnd_right{6};
      get(get<Var<Poisson::Tags::Field>>(vars_rnd_right)) = DataVector{
          0.5315513738418384, 0.5318275870968661, 0.6344009585513211,
          0.8494317940777896, 0.7244553248606352, 0.6110235106775829};
      PrimalFluxes expected_primal_fluxes_rnd_lowerleft{6};
      get<0>(get<Var<::Tags::Flux<Poisson::Tags::Field, tmpl::size_t<2>,
                                  Frame::Inertial>>>(
          expected_primal_fluxes_rnd_lowerleft)) = DataVector{
          -2.73562216919501644, -1.99932918760101819, -1.26303620600701905,
          -2.13076078945206238, 1.34772875675956438,  4.82621830297119114};
      get<1>(get<Var<::Tags::Flux<Poisson::Tags::Field, tmpl::size_t<2>,
                                  Frame::Inertial>>>(
          expected_primal_fluxes_rnd_lowerleft)) = DataVector{
          -2.35458672076185982, -1.36660688973597555, 0.99247122359872275,
          -2.35458672076185982, -1.36660688973597555, 0.99247122359872275};
      PrimalFluxes expected_primal_fluxes_rnd_right{6};
      get<0>(get<Var<::Tags::Flux<Poisson::Tags::Field, tmpl::size_t<2>,
                                  Frame::Inertial>>>(
          expected_primal_fluxes_rnd_right)) = DataVector{
          -0.40697892675748815, 0.41139833883793075, 1.22977560443334877,
          -1.04599037387364335, -0.9536331336008268, -0.86127589332801024};
      get<1>(get<Var<::Tags::Flux<Poisson::Tags::Field, tmpl::size_t<2>,
                                  Frame::Inertial>>>(
          expected_primal_fluxes_rnd_right)) = DataVector{
          0.6357608404719024, 0.38525547552753836, -0.04675489574747638,
          0.6357608404719024, 0.38525547552753836, -0.04675489574747638};
      OperatorVars expected_operator_vars_rnd_lowerleft{6};
      get(get<DgOperatorAppliedTo<Var<Poisson::Tags::Field>>>(
          expected_operator_vars_rnd_lowerleft)) = DataVector{
          13.52385143255982491, 6.82929107083524833, 0.04527695301629676,
          4.39834760568600736,  0.65041277314590373, 0.78222246779709226};
      OperatorVars expected_operator_vars_rnd_right{6};
      get(get<DgOperatorAppliedTo<Var<Poisson::Tags::Field>>>(
          expected_operator_vars_rnd_right)) = DataVector{
          1.60618450837350557, 4.71949148249203443, 15.72408761869980509,
          4.06376669069456398, 5.88578668691303086, 15.11012286654655945};
      // Large tolerances for the comparison to the analytic solution because
      // this regression test runs at very low resolution. Below is another
      // analytic-solution test at higher resolution. The hard-coded numbers
      // are compared at higher (default) precision.
      Approx analytic_solution_aux_approx =
          Approx::custom().epsilon(7.e-1).scale(M_PI);
      Approx analytic_solution_operator_approx =
          Approx::custom().epsilon(7.e-1).scale(M_PI * penalty_parameter *
                                                square(3));
      test_dg_operator<system, true>(
          domain_creator, penalty_parameter, true,
          Spectral::Quadrature::GaussLobatto, analytic_solution,
          analytic_solution_aux_approx, analytic_solution_operator_approx,
          {{{{lowerleft_id, std::move(vars_rnd_lowerleft)},
             {upperleft_id, std::move(vars_rnd_upperleft)},
             {right_id, std::move(vars_rnd_right)}},
            {{lowerleft_id, std::move(expected_primal_fluxes_rnd_lowerleft)},
             {right_id, std::move(expected_primal_fluxes_rnd_right)}},
            {{lowerleft_id, std::move(expected_operator_vars_rnd_lowerleft)},
             {right_id, std::move(expected_operator_vars_rnd_right)}}}});
    }
    {
      INFO("Higher-resolution analytic-solution tests");
      const domain::creators::AlignedLattice<2> domain_creator{
          {{{0., 0.25, 0.5}, {0., 0.5}}},
          {{1, 1}},
          {{12, 12}},
          {{{{0, 0}}, {{1, 1}}, {{1, 2}}}},
          {},
          {},
          std::make_unique<
              elliptic::BoundaryConditions::AnalyticSolution<system>>(
              analytic_solution.get_clone(),
              elliptic::BoundaryConditionType::Dirichlet)};
      Approx analytic_solution_aux_approx =
          Approx::custom().epsilon(1.e-11).scale(M_PI);
      Approx analytic_solution_operator_approx =
          Approx::custom().epsilon(1.e-11).scale(M_PI * penalty_parameter *
                                                 square(12));
      for (const auto& [massive, quadrature] :
           cartesian_product(make_array(true, false),
                             make_array(Spectral::Quadrature::GaussLobatto,
                                        Spectral::Quadrature::Gauss))) {
        test_dg_operator<system, true>(
            domain_creator, penalty_parameter, massive, quadrature,
            analytic_solution, analytic_solution_aux_approx,
            analytic_solution_operator_approx, {}, true);
      }
    }
  }
  {
    INFO("3D sphere");
    using system =
        Poisson::FirstOrderSystem<3, Poisson::Geometry::FlatCartesian>;
    Poisson::Solutions::Lorentzian<3> analytic_solution{0.};
    {
      INFO("Regression tests");
      const domain::creators::Sphere domain_creator{
          1.,
          3.,
          domain::creators::Sphere::InnerCube{0.},
          0_st,
          3_st,
          true,
          {},
          {},
          domain::CoordinateMaps::Distribution::Linear,
          ShellWedges::All,
          std::nullopt,
          std::make_unique<
              elliptic::BoundaryConditions::AnalyticSolution<system>>(
              analytic_solution.get_clone(),
              elliptic::BoundaryConditionType::Dirichlet)};
      const ElementId<3> center_id{6};
      const ElementId<3> wedge_id{0};
      using Vars = Variables<tmpl::list<Var<Poisson::Tags::Field>>>;
      using OperatorVars =
          Variables<tmpl::list<DgOperatorAppliedTo<Var<Poisson::Tags::Field>>>>;
      Vars vars_rnd_center{27};
      get(get<Var<Poisson::Tags::Field>>(vars_rnd_center)) = DataVector{
          0.09649087905837017, 0.8726189695670388,  0.2550997406970058,
          0.9371182951233519,  0.6633191884394125,  0.7910467053506253,
          0.8976248825657923,  0.8253597552079079,  0.2614810810491418,
          0.37703802268313935, 0.18811315681728424, 0.9286960528710335,
          0.7799863844018025,  0.29158508058745125, 0.3078187493178257,
          0.11582259489772695, 0.9504180903537383,  0.6290805984444998,
          0.4156900249686264,  0.6322967693109368,  0.17729543326351738,
          0.30252861325858027, 0.0700623995180415,  0.7014453661500705,
          0.09301082210249478, 0.2507901753830596,  0.7325836921946847};
      Vars vars_rnd_wedge{27};
      get(get<Var<Poisson::Tags::Field>>(vars_rnd_wedge)) = DataVector{
          0.33924615234165656, 0.8449024556165292,  0.4541579237278579,
          0.8205705845059463,  0.35547255892487073, 0.015009371066981636,
          0.9211805795375337,  0.4002407787315444,  0.0024764405702156767,
          0.8731417034578796,  0.5033095307249483,  0.3680070578003968,
          0.176441382310489,   0.6746744554291877,  0.6668359104611603,
          0.8208960531956108,  0.6243130189945111,  0.19954959587642307,
          0.7370176442442482,  0.4477495010295536,  0.8282274072733592,
          0.9199607746918792,  0.9635732092068038,  0.3504092315426307,
          0.11524312515866864, 0.17417975746930436, 0.5870281973363631};
      OperatorVars expected_operator_vars_rnd_center{27, 0.};
      get(get<DgOperatorAppliedTo<Var<Poisson::Tags::Field>>>(
          expected_operator_vars_rnd_center)) =
          DataVector{0.13900947487894668719,   2.8012452842448594126,
                     0.70606635684663743291,   3.0110762308109988439,
                     2.2653131088752034294,    2.4389044697636719228,
                     2.8069304514918704818,    2.5816683614488340481,
                     0.63609994355853216597,   1.0496348030246254179,
                     0.060338632695005833817,  2.9890473588327037824,
                     2.5568233674986675652,    0.39929198048901121121,
                     0.71163724792443083800,   0.023791492919156290164,
                     3.3146326096778735426,    1.9211926341751801584,
                     0.77620065471560839576,   0.91371751068503725968,
                     -0.033856352764628408480, -0.22189988085469286583,
                     -0.98838426761034614554,  2.0433388956068792019,
                     -0.76756810709326805942,  -0.037296848003322793930,
                     2.2419306171966684182};
      OperatorVars expected_operator_vars_rnd_wedge{27, 0.};
      get(get<DgOperatorAppliedTo<Var<Poisson::Tags::Field>>>(
          expected_operator_vars_rnd_wedge)) =
          DataVector{0.16499268635017833029, 1.8644717241195962742,
                     0.50779733947485694578, 1.8863114030787757613,
                     -2.7740067251377729107, -1.4130776603409314074,
                     1.9928746337497433849,  0.63283835581422620553,
                     -1.0343599185085030623, 5.8966821182729969308,
                     5.5504203240642899786,  2.6011964191689305181,
                     3.3040790531837145316,  20.091810674690414373,
                     8.0212451646648474934,  5.1144424758479178905,
                     6.8301727856302010267,  1.0869453410405538474,
                     5.2284593589347760911,  9.3539358392378026963,
                     5.7609110245159351749,  19.038783549101548687,
                     95.952139513786036673,  8.0688375547259152398,
                     0.69551590983732591855, 4.0994207069118830944,
                     4.2071397513902919485};
      // Large tolerances for the comparison to the analytic solution because
      // this regression test runs at very low resolution. Below is another
      // analytic-solution test at higher resolution. The hard-coded numbers
      // are compared at higher (default) precision.
      Approx analytic_solution_aux_approx =
          Approx::custom().epsilon(0.3).scale(1.);
      Approx analytic_solution_operator_approx =
          Approx::custom().epsilon(0.3).scale(1.);
      test_dg_operator<system, true>(
          domain_creator, penalty_parameter, true,
          Spectral::Quadrature::GaussLobatto, analytic_solution,
          analytic_solution_aux_approx, analytic_solution_operator_approx,
          {{{{center_id, std::move(vars_rnd_center)},
             {wedge_id, std::move(vars_rnd_wedge)}},
            {},
            {{center_id, std::move(expected_operator_vars_rnd_center)},
             {wedge_id, std::move(expected_operator_vars_rnd_wedge)}}}});
    }
    {
      INFO("Higher-resolution analytic-solution tests");
      const domain::creators::Sphere domain_creator{
          1.,
          3.,
          domain::creators::Sphere::InnerCube{0.},
          0_st,
          12_st,
          true,
          {},
          {},
          domain::CoordinateMaps::Distribution::Linear,
          ShellWedges::All,
          std::nullopt,
          std::make_unique<
              elliptic::BoundaryConditions::AnalyticSolution<system>>(
              analytic_solution.get_clone(),
              elliptic::BoundaryConditionType::Dirichlet)};
      Approx analytic_solution_aux_approx =
          Approx::custom().epsilon(1.e-4).scale(1.);
      Approx analytic_solution_operator_approx =
          Approx::custom().epsilon(1.e-4).scale(1.);
      for (const auto quadrature :
           {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}) {
        test_dg_operator<system, true>(
            domain_creator, penalty_parameter, true, quadrature,
            analytic_solution, analytic_solution_aux_approx,
            analytic_solution_operator_approx, {}, true);
      }
    }
  }
}
