// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/DynamicVector.h>
#include <cstddef>
#include <memory>
#include <optional>
#include <random>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/Creators/AlignedLattice.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Cylinder.hpp"
#include "Domain/Creators/Disk.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/ExpandOverBlocks.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/RotatedBricks.hpp"
#include "Domain/Creators/RotatedIntervals.hpp"
#include "Domain/Creators/RotatedRectangles.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Creators/Tags/InitialExtents.hpp"
#include "Domain/Creators/Tags/InitialRefinementLevels.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/InitializeBackgroundFields.hpp"
#include "Elliptic/BoundaryConditions/AnalyticSolution.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/DiscontinuousGalerkin/Actions/ApplyOperator.hpp"
#include "Elliptic/DiscontinuousGalerkin/Actions/InitializeDomain.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/InitializeSubdomain.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/SubdomainOperator.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/Tags.hpp"
#include "Elliptic/Systems/Elasticity/Actions/InitializeConstitutiveRelation.hpp"
#include "Elliptic/Systems/Elasticity/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Poisson/Geometry.hpp"
#include "Elliptic/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/LinearSolver/BuildMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
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
#include "ParallelAlgorithms/Amr/Projectors/CopyFromCreatorOrLeaveAsIs.hpp"
#include "ParallelAlgorithms/Amr/Protocols/AmrMetavariables.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Actions/CommunicateOverlapFields.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/ElementCenteredSubdomainData.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "Utilities/CartesianProduct.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

// The tests in this file check that the subdomain operator is equivalent to
// applying the full DG-operator to a domain where all points outside the
// subdomain are zero. This should hold for every element in the domain, at any
// refinement level (h and p) and for any number of overlap points.
//
// We use the action-testing framework for these tests because then we can have
// the InitializeSubdomain action do the tedious job of constructing the
// geometry. This has the added benefit that we test the subdomain operator is
// consistent with the InitializeSubdomain action.

struct DummyOptionsGroup {};

template <typename SubdomainOperatorType>
struct SubdomainOperatorTag : db::SimpleTag {
  using type = SubdomainOperatorType;
};

struct TemporalIdTag : db::SimpleTag {
  using type = size_t;
};

template <typename Tag>
struct DgOperatorAppliedTo : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

template <typename Tag>
struct Operand : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

template <size_t Dim, typename Fields>
struct SubdomainDataTag : db::SimpleTag {
  using type = LinearSolver::Schwarz::ElementCenteredSubdomainData<Dim, Fields>;
};

template <size_t Dim, typename Fields>
struct SubdomainOperatorAppliedToDataTag : db::SimpleTag {
  using type = LinearSolver::Schwarz::ElementCenteredSubdomainData<
      Dim, db::wrap_tags_in<DgOperatorAppliedTo, Fields>>;
};

struct OverrideBoundaryConditionsTag : db::SimpleTag {
  using type = bool;
};

template <typename System>
std::unique_ptr<typename System::boundary_conditions_base>
make_boundary_condition(
    const elliptic::BoundaryConditionType boundary_condition_type) {
  return std::make_unique<
      elliptic::BoundaryConditions::AnalyticSolution<System>>(
      nullptr, boundary_condition_type);
}

// Generate some random element-centered subdomain data on each element
template <typename SubdomainOperator, typename Fields>
struct InitializeRandomSubdomainData {
  using simple_tags =
      tmpl::list<SubdomainDataTag<SubdomainOperator::volume_dim, Fields>>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    using SubdomainData = typename SubdomainDataTag<Dim, Fields>::type;

    db::mutate<SubdomainDataTag<Dim, Fields>>(
        [](const auto subdomain_data, const auto& mesh, const auto& element,
           const auto& all_overlap_meshes, const auto& all_overlap_extents) {
          MAKE_GENERATOR(gen);
          UniformCustomDistribution<double> dist{-1., 1.};
          subdomain_data->element_data =
              make_with_random_values<typename SubdomainData::ElementData>(
                  make_not_null(&gen), make_not_null(&dist),
                  mesh.number_of_grid_points());
          for (const auto& [direction, neighbors] : element.neighbors()) {
            const auto& orientation = neighbors.orientation();
            for (const auto& neighbor_id : neighbors) {
              const auto overlap_id =
                  DirectionalId<Dim>{direction, neighbor_id};
              const size_t overlap_extent = all_overlap_extents.at(overlap_id);
              if (overlap_extent == 0) {
                continue;
              }
              subdomain_data->overlap_data.emplace(
                  overlap_id,
                  make_with_random_values<typename SubdomainData::OverlapData>(
                      make_not_null(&gen), make_not_null(&dist),
                      LinearSolver::Schwarz::overlap_num_points(
                          all_overlap_meshes.at(overlap_id).extents(),
                          overlap_extent, orientation(direction).dimension())));
            }
          }
        },
        make_not_null(&box), db::get<domain::Tags::Mesh<Dim>>(box),
        db::get<domain::Tags::Element<Dim>>(box),
        db::get<LinearSolver::Schwarz::Tags::Overlaps<domain::Tags::Mesh<Dim>,
                                                      Dim, DummyOptionsGroup>>(
            box),
        db::get<LinearSolver::Schwarz::Tags::Overlaps<
            elliptic::dg::subdomain_operator::Tags::ExtrudingExtent, Dim,
            DummyOptionsGroup>>(box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

// Generate random data for background fields. Not using a RNG because
// evaluating the background on the same coordinates should return the same
// data. Instead, we just pick some arbitrary combination of the coordinate
// values.
template <size_t Dim>
struct RandomBackground : elliptic::analytic_data::Background {
  RandomBackground() = default;
  explicit RandomBackground(CkMigrateMessage* m)
      : elliptic::analytic_data::Background(m) {}
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(RandomBackground);  // NOLINT
#pragma GCC diagnostic pop

  // NOLINTBEGIN(readability-convert-member-functions-to-static)
  template <typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(  // NOLINT
      const tnsr::I<DataVector, Dim>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    return {variables(x, RequestedTags{})...};
  }
  // [background_vars_fct_derivs]
  template <typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(  // NOLINT
      const tnsr::I<DataVector, Dim>& x, const Mesh<Dim>& /*mesh*/,
      const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            Frame::Inertial>&
      /*inv_jacobian*/,
      tmpl::list<RequestedTags...> /*meta*/) const {
    // [background_vars_fct_derivs]
    return variables(x, tmpl::list<RequestedTags...>{});
  }
  // NOLINTEND(readability-convert-member-functions-to-static)
  static tnsr::II<DataVector, Dim> variables(
      const tnsr::I<DataVector, Dim>& x,
      gr::Tags::InverseSpatialMetric<DataVector, Dim> /*meta*/) {
    tnsr::II<DataVector, Dim> inv_metric{x.begin()->size()};
    const DataVector r = get(magnitude(x));
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = 0; j < Dim; ++j) {
        inv_metric.get(i, j) = x.get(i) * x.get(j) / (1. + square(r));
      }
      inv_metric.get(i, i) += 1.;
    }
    return inv_metric;
  }
  static tnsr::i<DataVector, Dim> variables(
      const tnsr::I<DataVector, Dim>& x,
      gr::Tags::SpatialChristoffelSecondKindContracted<DataVector,
                                                       Dim> /*meta*/) {
    tnsr::i<DataVector, Dim> result{x.begin()->size()};
    const DataVector r = get(magnitude(x));
    for (size_t i = 0; i < Dim; ++i) {
      result.get(i) = x.get(i) / (1. + r);
    }
    return result;
  }
};

template <size_t Dim>
PUP::able::PUP_ID RandomBackground<Dim>::my_PUP_ID = 0;  // NOLINT

template <typename SubdomainOperator, typename Fields>
struct ApplySubdomainOperator {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& subdomain_data = db::get<SubdomainDataTag<Dim, Fields>>(box);

    // Override boundary conditions
    using system = typename SubdomainOperator::system;
    using BoundaryConditionsBase = typename system::boundary_conditions_base;
    std::unique_ptr<BoundaryConditionsBase> bc_override{};
    std::unordered_map<std::pair<size_t, Direction<Dim>>,
                       const BoundaryConditionsBase&,
                       boost::hash<std::pair<size_t, Direction<Dim>>>>
        override_boundary_conditions{};
    if (db::get<OverrideBoundaryConditionsTag>(box)) {
      bc_override = make_boundary_condition<system>(
          elliptic::BoundaryConditionType::Dirichlet);
      for (const auto& block :
           db::get<domain::Tags::Domain<Dim>>(box).blocks()) {
        for (const auto& direction : block.external_boundaries()) {
          override_boundary_conditions.emplace(
              std::make_pair(block.id(), direction), *bc_override);
        }
      }
    }

    // Apply the subdomain operator
    const auto& subdomain_operator =
        db::get<SubdomainOperatorTag<SubdomainOperator>>(box);
    auto subdomain_result = make_with_value<
        typename SubdomainOperatorAppliedToDataTag<Dim, Fields>::type>(
        subdomain_data, 0.);
    subdomain_operator(make_not_null(&subdomain_result), subdomain_data, box,
                       override_boundary_conditions);

    // Store result in the DataBox for checks
    db::mutate<SubdomainOperatorAppliedToDataTag<Dim, Fields>>(
        [&subdomain_result](const auto subdomain_operator_applied_to_data) {
          *subdomain_operator_applied_to_data = std::move(subdomain_result);
        },
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

template <typename SubdomainOperator, typename Fields>
struct TestSubdomainOperatorMatrix {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& subdomain_data = db::get<SubdomainDataTag<Dim, Fields>>(box);

    // Build the explicit matrix representation of the subdomain operator
    // Note: This would be an interesting unit of work to benchmark.
    const auto& subdomain_operator =
        db::get<SubdomainOperatorTag<SubdomainOperator>>(box);
    const size_t operator_size = subdomain_data.size();
    blaze::DynamicMatrix<double, blaze::columnMajor> operator_matrix{
        operator_size, operator_size};
    auto operand_buffer =
        make_with_value<std::decay_t<decltype(subdomain_data)>>(subdomain_data,
                                                                0.);
    auto result_buffer = make_with_value<
        typename SubdomainOperatorAppliedToDataTag<Dim, Fields>::type>(
        subdomain_data, 0.);
    ::LinearSolver::Serial::build_matrix(
        make_not_null(&operator_matrix), make_not_null(&operand_buffer),
        make_not_null(&result_buffer), subdomain_operator,
        std::forward_as_tuple(box));

    // Check the matrix is equivalent to the operator by applying it to the
    // data. We need to do the matrix multiplication on a contiguous buffer
    // because the `ElementCenteredSubdomainData` is not contiguous.
    blaze::DynamicVector<double> contiguous_operand(operator_size);
    std::copy(subdomain_data.begin(), subdomain_data.end(),
              contiguous_operand.begin());
    const blaze::DynamicVector<double> contiguous_result =
        operator_matrix * contiguous_operand;
    std::copy(contiguous_result.begin(), contiguous_result.end(),
              result_buffer.begin());
    const auto& expected_operator_applied_to_data =
        db::get<SubdomainOperatorAppliedToDataTag<Dim, Fields>>(box);
    Approx custom_approx = Approx::custom().epsilon(1.e-12).scale(1.0);
    CHECK_ITERABLE_CUSTOM_APPROX(
        result_buffer, expected_operator_applied_to_data, custom_approx);
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

template <typename Metavariables, typename System, typename SubdomainOperator,
          typename ExtraInitActions>
struct ElementArray {
  static constexpr size_t Dim = SubdomainOperator::volume_dim;
  static constexpr bool has_background_fields =
      not std::is_same_v<typename System::background_fields, tmpl::list<>>;

  // We prefix the system fields with an "operand" tag to make sure the
  // subdomain operator works with prefixed variables
  using fields_tag = ::Tags::Variables<
      db::wrap_tags_in<Operand, typename System::primal_fields>>;
  using fluxes_tag = ::Tags::Variables<
      db::wrap_tags_in<Operand, typename System::primal_fluxes>>;
  using operator_applied_to_fields_tag =
      db::add_tag_prefix<DgOperatorAppliedTo, fields_tag>;
  using subdomain_operator_applied_to_fields_tag =
      SubdomainOperatorAppliedToDataTag<Dim, typename fields_tag::tags_list>;

  using dg_operator =
      ::elliptic::dg::Actions::DgOperator<System, true, TemporalIdTag,
                                          fields_tag, fluxes_tag,
                                          operator_applied_to_fields_tag>;

  using background_tag =
      elliptic::Tags::Background<elliptic::analytic_data::Background>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Dim>;
  using const_global_cache_tags =
      tmpl::list<domain::Tags::Domain<Dim>, background_tag>;

  // These tags are communicated on subdomain overlaps to initialize the
  // subdomain geometry. AMR updates these tags, so we have to communicate them
  // after each AMR step.
  using subdomain_init_tags =
      tmpl::list<domain::Tags::Mesh<Dim>, domain::Tags::Element<Dim>,
                 domain::Tags::NeighborMesh<Dim>>;
  using init_subdomain_action =
      ::elliptic::dg::subdomain_operator::Actions::InitializeSubdomain<
          System, background_tag, DummyOptionsGroup, false>;
  using init_random_subdomain_data_action =
      InitializeRandomSubdomainData<SubdomainOperator,
                                    typename fields_tag::tags_list>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<
                         tmpl::list<domain::Tags::InitialRefinementLevels<Dim>,
                                    domain::Tags::InitialExtents<Dim>,
                                    SubdomainOperatorTag<SubdomainOperator>,
                                    subdomain_operator_applied_to_fields_tag,
                                    OverrideBoundaryConditionsTag>>,
                     ::elliptic::dg::Actions::InitializeDomain<Dim>,
                     ::elliptic::dg::Actions::initialize_operator<
                         System, background_tag>,
                     Initialization::Actions::InitializeItems<
                         ::amr::Initialization::Initialize<Dim>>,
                     ExtraInitActions, Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<
              // Init subdomain
              LinearSolver::Schwarz::Actions::SendOverlapFields<
                  subdomain_init_tags, DummyOptionsGroup, false, TemporalIdTag>,
              LinearSolver::Schwarz::Actions::ReceiveOverlapFields<
                  Dim, subdomain_init_tags, DummyOptionsGroup, false,
                  TemporalIdTag>,
              init_subdomain_action, init_random_subdomain_data_action,
              Parallel::Actions::TerminatePhase,
              // Full DG operator
              typename dg_operator::apply_actions,
              Parallel::Actions::TerminatePhase,
              // Subdomain operator
              ApplySubdomainOperator<SubdomainOperator,
                                     typename fields_tag::tags_list>,
              TestSubdomainOperatorMatrix<SubdomainOperator,
                                          typename fields_tag::tags_list>,
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
                 logging::Tags::Verbosity<::amr::OptionTags::AmrGroup>>;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<tmpl::list<>>>>>;
};

template <typename System, typename SubdomainOperator,
          typename ExtraInitActions>
struct Metavariables {
  using element_array =
      ElementArray<Metavariables, System, SubdomainOperator, ExtraInitActions>;
  using amr_component = AmrComponent<Metavariables>;
  using component_list = tmpl::list<element_array, amr_component>;
  using const_global_cache_tags = tmpl::list<>;
  static constexpr size_t volume_dim = System::volume_dim;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<
            elliptic::BoundaryConditions::BoundaryCondition<volume_dim>,
            tmpl::list<elliptic::BoundaryConditions::AnalyticSolution<System>>>,
        tmpl::pair<elliptic::analytic_data::Background,
                   tmpl::list<RandomBackground<volume_dim>>>,
        tmpl::pair<::amr::Criterion, tmpl::list<::amr::Criteria::Random>>>;
  };
  template <typename Tag>
  using overlaps_tag =
      LinearSolver::Schwarz::Tags::Overlaps<Tag, volume_dim, DummyOptionsGroup>;
  struct amr : tt::ConformsTo<::amr::protocols::AmrMetavariables> {
    using projectors = tmpl::flatten<tmpl::list<
        ::amr::projectors::DefaultInitialize<tmpl::append<
            tmpl::list<domain::Tags::InitialExtents<volume_dim>,
                       domain::Tags::InitialRefinementLevels<volume_dim>,
                       SubdomainOperatorTag<SubdomainOperator>,
                       typename element_array::
                           subdomain_operator_applied_to_fields_tag,
                       typename element_array::fields_tag>,
            // Tags communicated on subdomain overlaps. No need to project
            // these during AMR because they will be communicated.
            db::wrap_tags_in<overlaps_tag,
                             typename element_array::subdomain_init_tags>,
            // Tags initialized on subdomains. No need to project these during
            // AMR because they will get re-initialized after communication.
            typename element_array::init_subdomain_action::simple_tags,
            typename element_array::init_random_subdomain_data_action::
                simple_tags>>,
        ::amr::projectors::CopyFromCreatorOrLeaveAsIs<
            OverrideBoundaryConditionsTag, TemporalIdTag,
            // Work around a segfault because this tag isn't handled
            // correctly by the testing framework
            Parallel::Tags::GlobalCacheImpl<Metavariables>>,
        elliptic::dg::ProjectGeometry<volume_dim>,
        elliptic::dg::Actions::amr_projectors<
            System, typename element_array::background_tag>,
        typename element_array::dg_operator::amr_projectors, ExtraInitActions>>;
  };

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};

// The test should work for any elliptic system. For systems with fluxes or
// sources that take arguments out of the DataBox this test can insert actions
// that initialize those arguments.
template <typename System, typename ExtraInitActions = tmpl::list<>,
          size_t Dim = System::volume_dim>
void test_subdomain_operator(
    const DomainCreator<Dim>& domain_creator,
    const bool use_massive_dg_operator = true,
    const Spectral::Quadrature quadrature = Spectral::Quadrature::GaussLobatto,
    const bool override_boundary_conditions = false,
    // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
    const size_t max_overlap = 3, const double penalty_parameter = 1.2) {
  CAPTURE(Dim);
  CAPTURE(penalty_parameter);
  CAPTURE(use_massive_dg_operator);

  using SubdomainOperator =
      elliptic::dg::subdomain_operator::SubdomainOperator<System,
                                                          DummyOptionsGroup>;

  using metavariables =
      Metavariables<System, SubdomainOperator, ExtraInitActions>;
  using element_array = typename metavariables::element_array;

  using fields_tag = typename element_array::fields_tag;
  using operator_applied_to_fields_tag =
      typename element_array::operator_applied_to_fields_tag;
  using subdomain_data_tag =
      SubdomainDataTag<Dim, typename fields_tag::tags_list>;
  using subdomain_operator_applied_to_fields_tag =
      typename element_array::subdomain_operator_applied_to_fields_tag;

  register_factory_classes_with_charm<metavariables>();

  // Select randomly which iteration of the loops below perform expensive tests.
  MAKE_GENERATOR(gen);
  std::uniform_int_distribution<size_t> dist_select_overlap(0, max_overlap);
  const size_t rnd_overlap = dist_select_overlap(gen);

  // The test should hold for any number of overlap points
  for (size_t overlap = 0; overlap <= max_overlap; overlap++) {
    CAPTURE(overlap);

    // Re-create the domain in every iteration of this loop because it's not
    // copyable
    auto domain = domain_creator.create_domain();
    auto boundary_conditions = domain_creator.external_boundary_conditions();
    const auto initial_ref_levs = domain_creator.initial_refinement_levels();
    const auto initial_extents = domain_creator.initial_extents();
    const auto element_ids = ::initial_element_ids(initial_ref_levs);
    CAPTURE(element_ids.size());

    // Configure AMR criteria
    std::vector<std::unique_ptr<::amr::Criterion>> amr_criteria{};
    amr_criteria.push_back(std::make_unique<::amr::Criteria::Random>(
        std::unordered_map<::amr::Flag, size_t>{
            // h-refinement is not supported yet in the action testing framework
            {::amr::Flag::IncreaseResolution, 1},
            {::amr::Flag::DoNothing, 1}}));

    ActionTesting::MockRuntimeSystem<metavariables> runner{tuples::TaggedTuple<
        domain::Tags::Domain<Dim>, domain::Tags::FunctionsOfTimeInitialize,
        domain::Tags::ExternalBoundaryConditions<Dim>,
        elliptic::Tags::Background<elliptic::analytic_data::Background>,
        LinearSolver::Schwarz::Tags::MaxOverlap<DummyOptionsGroup>,
        logging::Tags::Verbosity<DummyOptionsGroup>,
        elliptic::dg::Tags::PenaltyParameter, elliptic::dg::Tags::Massive,
        elliptic::dg::Tags::Quadrature, elliptic::dg::Tags::Formulation,
        ::amr::Criteria::Tags::Criteria, ::amr::Tags::Policies,
        logging::Tags::Verbosity<::amr::OptionTags::AmrGroup>>{
        std::move(domain), domain_creator.functions_of_time(),
        std::move(boundary_conditions),
        std::make_unique<RandomBackground<Dim>>(), overlap,
        ::Verbosity::Verbose, penalty_parameter, use_massive_dg_operator,
        quadrature, ::dg::Formulation::StrongInertial, std::move(amr_criteria),
        ::amr::Policies{::amr::Isotropy::Anisotropic, ::amr::Limits{}},
        ::Verbosity::Debug}};

    // Initialize all elements, generating random subdomain data
    for (const auto& element_id : element_ids) {
      CAPTURE(element_id);
      ActionTesting::emplace_component_and_initialize<element_array>(
          &runner, element_id,
          {initial_ref_levs, initial_extents, SubdomainOperator{},
           typename subdomain_operator_applied_to_fields_tag::type{},
           override_boundary_conditions});
      while (
          not ActionTesting::get_terminate<element_array>(runner, element_id)) {
        ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                  element_id);
      }
    }
    // DataBox shortcuts
    const auto get_tag = [&runner](const ElementId<Dim>& local_element_id,
                                   auto tag_v) -> decltype(auto) {
      using tag = std::decay_t<decltype(tag_v)>;
      return ActionTesting::get_databox_tag<element_array, tag>(
          runner, local_element_id);
    };
    const auto set_tag = [&runner](const ElementId<Dim>& local_element_id,
                                   auto tag_v, const auto& value) {
      using tag = std::decay_t<decltype(tag_v)>;
      ActionTesting::simple_action<element_array,
                                   ::Actions::SetData<tmpl::list<tag>>>(
          make_not_null(&runner), local_element_id, value);
    };

    const auto test_subdomain_operator_equals_dg_operator = [&runner,
                                                             &element_ids,
                                                             &overlap,
                                                             &rnd_overlap, &gen,
                                                             &get_tag,
                                                             &set_tag]() {
      ActionTesting::set_phase(make_not_null(&runner),
                               Parallel::Phase::Testing);
      for (const auto& element_id : element_ids) {
        ActionTesting::next_action_if_ready<element_array>(
            make_not_null(&runner), element_id);
      }
      for (const auto& element_id : element_ids) {
        while (not ActionTesting::get_terminate<element_array>(runner,
                                                               element_id) and
               ActionTesting::next_action_if_ready<element_array>(
                   make_not_null(&runner), element_id)) {
        }
      }

      // For selection of expensive tests
      std::uniform_int_distribution<size_t> dist_select_subdomain_center(
          0, element_ids.size() - 1);
      const auto& rnd_subdomain_center = dist_select_subdomain_center(gen);
      size_t subdomain_center_id = 0;

      // Take each element as the subdomain-center in turn
      for (const auto& subdomain_center : element_ids) {
        CAPTURE(subdomain_center);

        // First, reset the data on all elements to zero
        for (const auto& element_id : element_ids) {
          set_tag(element_id, fields_tag{},
                  typename fields_tag::type{
                      get_tag(element_id, domain::Tags::Mesh<Dim>{})
                          .number_of_grid_points(),
                      0.});
        }

        // Set data on the central element and its neighbors to the subdomain
        // data
        const auto& subdomain_data =
            get_tag(subdomain_center, subdomain_data_tag{});
        const auto& all_overlap_extents =
            get_tag(subdomain_center,
                    LinearSolver::Schwarz::Tags::Overlaps<
                        elliptic::dg::subdomain_operator::Tags::ExtrudingExtent,
                        Dim, DummyOptionsGroup>{});
        const auto& central_element =
            get_tag(subdomain_center, domain::Tags::Element<Dim>{});
        set_tag(subdomain_center, fields_tag{}, subdomain_data.element_data);
        for (const auto& [overlap_id, overlap_data] :
             subdomain_data.overlap_data) {
          const auto& direction = overlap_id.direction();
          const auto& neighbor_id = overlap_id.id();
          const auto direction_from_neighbor =
              central_element.neighbors().at(direction).orientation()(
                  direction.opposite());
          set_tag(
              neighbor_id, fields_tag{},
              LinearSolver::Schwarz::extended_overlap_data(
                  overlap_data,
                  get_tag(neighbor_id, domain::Tags::Mesh<Dim>{}).extents(),
                  all_overlap_extents.at(overlap_id), direction_from_neighbor));
        }

        // Run actions to compute the full DG-operator
        for (const auto& element_id : element_ids) {
          CAPTURE(element_id);
          runner.template mock_distributed_objects<element_array>()
              .at(element_id)
              .force_next_action_to_be(5);
          runner.template mock_distributed_objects<element_array>()
              .at(element_id)
              .set_terminate(false);
          while (not ActionTesting::get_terminate<element_array>(runner,
                                                                 element_id) and
                 ActionTesting::next_action_if_ready<element_array>(
                     make_not_null(&runner), element_id)) {
          }
        }
        // Break here so all elements have sent mortar data before receiving it
        for (const auto& element_id : element_ids) {
          CAPTURE(element_id);
          while (not ActionTesting::get_terminate<element_array>(runner,
                                                                 element_id) and
                 ActionTesting::next_action_if_ready<element_array>(
                     make_not_null(&runner), element_id)) {
          }
        }

        // Invoke ApplySubdomainOperator action only on the subdomain center
        ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                  subdomain_center);

        // Test that the subdomain operator and the full DG-operator computed
        // the same result within the subdomain
        const auto& subdomain_result = get_tag(
            subdomain_center, subdomain_operator_applied_to_fields_tag{});
        Approx custom_approx = Approx::custom().epsilon(1.e-12).scale(1.0);
        CHECK_VARIABLES_CUSTOM_APPROX(
            subdomain_result.element_data,
            get_tag(subdomain_center, operator_applied_to_fields_tag{}),
            custom_approx);
        REQUIRE(subdomain_result.overlap_data.size() ==
                subdomain_data.overlap_data.size());
        for (const auto& [overlap_id, overlap_result] :
             subdomain_result.overlap_data) {
          CAPTURE(overlap_id);
          const auto& direction = overlap_id.direction();
          const auto& neighbor_id = overlap_id.id();
          const auto direction_from_neighbor =
              central_element.neighbors().at(direction).orientation()(
                  direction.opposite());
          const auto expected_overlap_result =
              LinearSolver::Schwarz::data_on_overlap(
                  get_tag(neighbor_id, operator_applied_to_fields_tag{}),
                  get_tag(neighbor_id, domain::Tags::Mesh<Dim>{}).extents(),
                  all_overlap_extents.at(overlap_id), direction_from_neighbor);
          CHECK_VARIABLES_CUSTOM_APPROX(overlap_result, expected_overlap_result,
                                        custom_approx);
        }

        // Now build the matrix representation of the subdomain operator
        // explicitly, and apply it to the data to make sure the matrix is
        // equivalent to the matrix-free operator. This is important to test
        // because the subdomain operator includes optimizations for when it is
        // invoked on sparse data, i.e. data that is mostly zero, which is the
        // case when building it explicitly column-by-column.
        if (overlap == rnd_overlap and
            subdomain_center_id == rnd_subdomain_center) {
          ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                    subdomain_center);
        }
        ++subdomain_center_id;
      }  // loop over subdomain centers
    };

    test_subdomain_operator_equals_dg_operator();

    INFO("Run AMR!");
    const auto invoke_all_simple_actions = [&runner, &element_ids]() {
      bool quiescence = false;
      while (not quiescence) {
        quiescence = true;
        for (const auto& element_id : element_ids) {
          while (not ActionTesting::is_simple_action_queue_empty<element_array>(
              runner, element_id)) {
            ActionTesting::invoke_queued_simple_action<element_array>(
                make_not_null(&runner), element_id);
            quiescence = false;
          }
        }
      }
    };
    using amr_component = typename metavariables::amr_component;
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

    // Test again after AMR
    test_subdomain_operator_equals_dg_operator();
  }  // loop over overlaps
}

// Add a constitutive relation for elasticity systems to the DataBox
template <size_t Dim>
struct InitializeConstitutiveRelation
    : tt::ConformsTo<::amr::protocols::Projector> {
 public:  // Iterative action
  using simple_tags =
      tmpl::list<Elasticity::Tags::ConstitutiveRelationPerBlock<Dim>>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate_apply<InitializeConstitutiveRelation>(make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }

 public:  // amr::protocols::Projector
  using return_tags = simple_tags;
  using argument_tags = tmpl::list<domain::Tags::Domain<Dim>>;
  template <typename... AmrData>
  static void apply(
      const gsl::not_null<std::vector<std::unique_ptr<
          Elasticity::ConstitutiveRelations::ConstitutiveRelation<Dim>>>*>
          constitutive_relations,
      const Domain<Dim>& domain, const AmrData&... /*amr_data*/) {
    const domain::ExpandOverBlocks<std::unique_ptr<
        Elasticity::ConstitutiveRelations::ConstitutiveRelation<Dim>>>
        expand_over_blocks{domain.blocks().size()};
    *constitutive_relations = expand_over_blocks(
        std::make_unique<
            Elasticity::ConstitutiveRelations::IsotropicHomogeneous<Dim>>(1.,
                                                                          2.));
  }
};

}  // namespace

// This test constructs a selection of domains and tests the subdomain operator
// for _every_ element in those domains and for a range of overlaps. We increase
// the timeout for the test because going over so many elements is relatively
// expensive but also very important to ensure that the subdomain operator
// handles all of these geometries correctly.
// [[TimeOut, 25]]
SPECTRE_TEST_CASE("Unit.Elliptic.DG.SubdomainOperator", "[Unit][Elliptic]") {
  // Needed for Brick
  using VariantType = std::variant<
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>,
      domain::creators::Brick::LowerUpperBoundaryCondition<
          domain::BoundaryConditions::BoundaryCondition>>;

  domain::creators::register_derived_with_charm();
  {
    INFO("Rectilinear and aligned");
    {
      INFO("1D");
      using system = Poisson::FirstOrderSystem<1, Poisson::Geometry::Curved>;
      const domain::creators::Interval domain_creator{
          {{-2.}},
          {{2.}},
          {{1}},
          {{3}},
          make_boundary_condition<system>(
              elliptic::BoundaryConditionType::Dirichlet),
          make_boundary_condition<system>(
              elliptic::BoundaryConditionType::Neumann)};
      for (const auto& [use_massive_dg_operator, quadrature] :
           cartesian_product(make_array(false, true),
                             make_array(Spectral::Quadrature::GaussLobatto,
                                        Spectral::Quadrature::Gauss))) {
        test_subdomain_operator<system>(domain_creator, use_massive_dg_operator,
                                        quadrature);
      }
    }
    {
      INFO("2D");
      using system = Poisson::FirstOrderSystem<2, Poisson::Geometry::Curved>;
      const domain::creators::Rectangle domain_creator{
          {{-2., 0.}},
          {{2., 1.}},
          {{1, 1}},
          {{3, 3}},
          make_boundary_condition<system>(
              elliptic::BoundaryConditionType::Dirichlet),
          nullptr};
      test_subdomain_operator<system>(domain_creator);
      test_subdomain_operator<system>(domain_creator, true,
                                      Spectral::Quadrature::GaussLobatto, true);
    }
    {
      INFO("3D");
      using system = Poisson::FirstOrderSystem<3, Poisson::Geometry::Curved>;
      const domain::creators::Brick domain_creator{
          {{-2., 0., -1.}},
          {{2., 1., 1.}},
          {{1, 0, 1}},
          {{3, 3, 3}},
          VariantType{make_boundary_condition<system>(
              elliptic::BoundaryConditionType::Dirichlet)},
          VariantType{make_boundary_condition<system>(
              elliptic::BoundaryConditionType::Dirichlet)},
          VariantType{make_boundary_condition<system>(
              elliptic::BoundaryConditionType::Dirichlet)},
          nullptr};
      test_subdomain_operator<system>(domain_creator);
    }
  }
  {
    INFO("Rotated");
    {
      INFO("1D");
      using system = Poisson::FirstOrderSystem<1, Poisson::Geometry::Curved>;
      const domain::creators::RotatedIntervals domain_creator{
          {{-2.}},
          {{0.}},
          {{2.}},
          {{1}},
          {{{{3, 3}}}},
          make_boundary_condition<system>(
              elliptic::BoundaryConditionType::Dirichlet),
          make_boundary_condition<system>(
              elliptic::BoundaryConditionType::Neumann),
          nullptr};
      test_subdomain_operator<system>(domain_creator);
    }
    {
      INFO("2D");
      using system = Poisson::FirstOrderSystem<2, Poisson::Geometry::Curved>;
      const domain::creators::RotatedRectangles domain_creator{
          {{-2., 0.}},
          {{0., 0.5}},
          {{2., 1.}},
          {{1, 1}},
          {{{{3, 3}}, {{3, 3}}}},
          make_boundary_condition<system>(
              elliptic::BoundaryConditionType::Dirichlet)};
      test_subdomain_operator<system>(domain_creator);
    }
    {
      INFO("2D flat-cartesian");
      using system =
          Poisson::FirstOrderSystem<2, Poisson::Geometry::FlatCartesian>;
      const domain::creators::RotatedRectangles domain_creator{
          {{-2., 0.}},
          {{0., 0.5}},
          {{2., 1.}},
          {{1, 1}},
          {{{{3, 3}}, {{3, 3}}}},
          make_boundary_condition<system>(
              elliptic::BoundaryConditionType::Dirichlet)};
      test_subdomain_operator<system>(domain_creator);
    }
    {
      INFO("3D");
      using system = Poisson::FirstOrderSystem<3, Poisson::Geometry::Curved>;
      const domain::creators::RotatedBricks domain_creator{
          {{-2., 0., -1.}},
          {{0., 0.5, 0.}},
          {{2., 1., 1.}},
          {{1, 0, 0}},
          {{{{3, 3}}, {{3, 3}}, {{3, 3}}}},
          make_boundary_condition<system>(
              elliptic::BoundaryConditionType::Dirichlet)};
      test_subdomain_operator<system>(domain_creator);
    }
  }
  {
    INFO("Refined");
    {
      INFO("1D");
      using system = Poisson::FirstOrderSystem<1, Poisson::Geometry::Curved>;
      //  |-B0-|--B1---|
      //  [oooo|ooo|ooo]-> xi
      //  ^    ^   ^   ^
      // -2    0   1   2
      const domain::creators::AlignedLattice<1> domain_creator{
          {{{-2., 0., 2.}}},
          {{0}},
          {{3}},
          {{{{1}}, {{2}}, {{1}}}},  // Refine once in block 1
          {{{{0}}, {{1}}, {{4}}}},  // Increase num points in block 0
          {},
          make_boundary_condition<system>(
              elliptic::BoundaryConditionType::Dirichlet)};
      test_subdomain_operator<system>(domain_creator);
    }
    {
      INFO("2D");
      using system = Poisson::FirstOrderSystem<2, Poisson::Geometry::Curved>;
      //   -2    0   2
      // -2 +----+---+> xi
      //    |oooo|ooo|
      //    |    |ooo|
      //    |    |ooo|
      // -1 |oooo+---+
      //    |    |ooo|
      //    |    |ooo|
      //    |oooo|ooo|
      //  0 +----+---+
      //    |ooo |ooo|
      //    |ooo |ooo|
      //    |ooo |ooo|
      //  2 +----+---+
      //    v eta
      const domain::creators::AlignedLattice<2> domain_creator{
          // Start with 4 unrefined blocks
          {{{-2., 0., 2.}, {-2., 0., 2.}}},
          {{0, 0}},
          {{3, 3}},
          // Refine once in eta in upper-right block in sketch above
          {{{{1, 0}}, {{2, 1}}, {{0, 1}}}},
          // Increase num points in xi in upper-left block in sketch above
          {{{{0, 0}}, {{1, 1}}, {{4, 3}}}},
          {},
          make_boundary_condition<system>(
              elliptic::BoundaryConditionType::Dirichlet)};
      test_subdomain_operator<system>(domain_creator);
    }
    {
      INFO("3D");
      using system = Poisson::FirstOrderSystem<3, Poisson::Geometry::Curved>;
      const domain::creators::AlignedLattice<3> domain_creator{
          {{{-2., 0., 2.}, {-2., 0., 2.}, {-2., 0., 2.}}},
          {{0, 0, 0}},
          {{3, 3, 3}},
          {{{{1, 0, 0}}, {{2, 1, 1}}, {{0, 1, 1}}}},
          {{{{0, 0, 0}}, {{1, 1, 1}}, {{4, 3, 2}}}},
          {},
          make_boundary_condition<system>(
              elliptic::BoundaryConditionType::Dirichlet)};
      test_subdomain_operator<system>(domain_creator);
    }
  }
  {
    INFO("Curved mesh");
    {
      INFO("2D");
      using system = Poisson::FirstOrderSystem<2, Poisson::Geometry::Curved>;
      const domain::creators::Disk domain_creator{
          0.5,
          2.,
          1,
          {{3, 4}},
          false,
          make_boundary_condition<system>(
              elliptic::BoundaryConditionType::Dirichlet)};
      test_subdomain_operator<system>(domain_creator);
    }
    {
      INFO("3D");
      using system = Poisson::FirstOrderSystem<3, Poisson::Geometry::Curved>;
      const domain::creators::Cylinder domain_creator{
          0.5,
          2.,
          0.,
          2.,
          make_boundary_condition<system>(
              elliptic::BoundaryConditionType::Dirichlet),
          make_boundary_condition<system>(
              elliptic::BoundaryConditionType::Dirichlet),
          make_boundary_condition<system>(
              elliptic::BoundaryConditionType::Dirichlet),
          size_t{0},
          std::array<size_t, 3>{{3, 4, 2}},
          false};
      for (const auto& [use_massive_dg_operator, quadrature] :
           cartesian_product(make_array(false, true),
                             make_array(Spectral::Quadrature::GaussLobatto,
                                        Spectral::Quadrature::Gauss))) {
        test_subdomain_operator<system>(domain_creator, use_massive_dg_operator,
                                        quadrature);
      }
    }
  }
  {
    INFO("System with fluxes args");
    using system = Elasticity::FirstOrderSystem<3>;
    const domain::creators::Brick domain_creator{
        {{-2., 0., -1.}},
        {{2., 1., 1.}},
        {{1, 1, 1}},
        {{3, 3, 3}},
        VariantType{make_boundary_condition<system>(
            elliptic::BoundaryConditionType::Dirichlet)},
        VariantType{make_boundary_condition<system>(
            elliptic::BoundaryConditionType::Dirichlet)},
        VariantType{make_boundary_condition<system>(
            elliptic::BoundaryConditionType::Dirichlet)},
        nullptr};
    test_subdomain_operator<system,
                            tmpl::list<InitializeConstitutiveRelation<3>>>(
        domain_creator);
  }
}
