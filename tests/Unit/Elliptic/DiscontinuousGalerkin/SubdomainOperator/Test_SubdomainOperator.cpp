// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
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
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/RotatedBricks.hpp"
#include "Domain/Creators/RotatedIntervals.hpp"
#include "Domain/Creators/RotatedRectangles.hpp"
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
#include "Elliptic/Systems/Elasticity/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Poisson/Geometry.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Actions/SetData.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/ElementCenteredSubdomainData.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
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

// Generate some random element-centered subdomain data on each element
template <typename SubdomainOperator, typename Fields>
struct InitializeRandomSubdomainData {
  using simple_tags =
      tmpl::list<SubdomainDataTag<SubdomainOperator::volume_dim, Fields>>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    using SubdomainData = typename SubdomainDataTag<Dim, Fields>::type;

    db::mutate<SubdomainDataTag<Dim, Fields>>(
        make_not_null(&box),
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
              const auto overlap_id = std::make_pair(direction, neighbor_id);
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
        db::get<domain::Tags::Mesh<Dim>>(box),
        db::get<domain::Tags::Element<Dim>>(box),
        db::get<LinearSolver::Schwarz::Tags::Overlaps<domain::Tags::Mesh<Dim>,
                                                      Dim, DummyOptionsGroup>>(
            box),
        db::get<LinearSolver::Schwarz::Tags::Overlaps<
            elliptic::dg::subdomain_operator::Tags::ExtrudingExtent, Dim,
            DummyOptionsGroup>>(box));
    return {std::move(box)};
  }
};

// Generate random data for background fields. Not using a RNG because
// evaluating the background on the same coordinates should return the same
// data. Instead, we just pick some arbitrary combination of the coordinate
// values.
template <size_t Dim>
struct RandomBackground {
  template <typename... RequestedTags>
  static tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataVector, Dim>& x,
      tmpl::list<RequestedTags...> /*meta*/) noexcept {
    return {variables(x, RequestedTags{})...};
  }
  template <typename... RequestedTags>
  static tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataVector, Dim>& x, const Mesh<Dim>& /*mesh*/,
      const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
      /*inv_jacobian*/,
      tmpl::list<RequestedTags...> /*meta*/) noexcept {
    return variables(x, tmpl::list<RequestedTags...>{});
  }
  static tnsr::II<DataVector, Dim> variables(
      const tnsr::I<DataVector, Dim>& x,
      gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial,
                                     DataVector> /*meta*/) noexcept {
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
      gr::Tags::SpatialChristoffelSecondKindContracted<
          Dim, Frame::Inertial, DataVector> /*meta*/) noexcept {
    tnsr::i<DataVector, Dim> result{x.begin()->size()};
    const DataVector r = get(magnitude(x));
    for (size_t i = 0; i < Dim; ++i) {
      result.get(i) = x.get(i) / (1. + r);
    }
    return result;
  }
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) noexcept {}
};

template <size_t Dim>
struct RandomBackgroundTag : db::SimpleTag {
  using type = RandomBackground<Dim>;
};

template <typename SubdomainOperator, typename Fields>
struct ApplySubdomainOperator {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& subdomain_data = db::get<SubdomainDataTag<Dim, Fields>>(box);

    // Apply the subdomain operator
    SubdomainOperator subdomain_operator{};
    auto subdomain_result = make_with_value<
        typename SubdomainOperatorAppliedToDataTag<Dim, Fields>::type>(
        subdomain_data, 0.);
    subdomain_operator(make_not_null(&subdomain_result), subdomain_data, box);

    // Store result in the DataBox for checks
    db::mutate<SubdomainOperatorAppliedToDataTag<Dim, Fields>>(
        make_not_null(&box),
        [&subdomain_result](const auto subdomain_operator_applied_to_data) {
          *subdomain_operator_applied_to_data = std::move(subdomain_result);
        });
    return {std::move(box)};
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

  using apply_full_dg_operator_actions =
      ::elliptic::dg::Actions::apply_operator<System, true, TemporalIdTag,
                                              fields_tag, fluxes_tag,
                                              operator_applied_to_fields_tag>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Dim>;
  using const_global_cache_tags =
      tmpl::list<domain::Tags::Domain<Dim>, RandomBackgroundTag<Dim>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<
                  tmpl::list<domain::Tags::InitialRefinementLevels<Dim>,
                             domain::Tags::InitialExtents<Dim>,
                             subdomain_operator_applied_to_fields_tag>>,
              Actions::SetupDataBox,
              ::elliptic::dg::Actions::InitializeDomain<Dim>,
              ::elliptic::dg::Actions::initialize_operator<
                  System, RandomBackgroundTag<Dim>>,
              ::elliptic::dg::subdomain_operator::Actions::InitializeSubdomain<
                  System,
                  tmpl::conditional_t<has_background_fields,
                                      RandomBackgroundTag<Dim>, void>,
                  DummyOptionsGroup>,
              InitializeRandomSubdomainData<SubdomainOperator,
                                            typename fields_tag::tags_list>,
              ExtraInitActions, Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<apply_full_dg_operator_actions,
                     // Break here so it's easy to apply the subdomain operator
                     // only on a particular element
                     Parallel::Actions::TerminatePhase,
                     ApplySubdomainOperator<SubdomainOperator,
                                            typename fields_tag::tags_list>,
                     Parallel::Actions::TerminatePhase>>>;
};

template <typename System, typename SubdomainOperator,
          typename ExtraInitActions>
struct Metavariables {
  using element_array =
      ElementArray<Metavariables, System, SubdomainOperator, ExtraInitActions>;
  using component_list = tmpl::list<element_array>;
  using const_global_cache_tags = tmpl::list<>;
  enum class Phase { Initialization, Testing, Exit };
};

// The test should work for any elliptic system. For systems with fluxes or
// sources that take arguments out of the DataBox this test can insert actions
// that initialize those arguments.
template <typename System, typename ExtraInitActions = tmpl::list<>,
          typename ArgsTagsFromCenter = tmpl::list<>,
          size_t Dim = System::volume_dim>
void test_subdomain_operator(
    const DomainCreator<Dim>& domain_creator,
    const bool use_massive_dg_operator = true,
    // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
    const size_t max_overlap = 3, const double penalty_parameter = 1.2) {
  CAPTURE(Dim);
  CAPTURE(penalty_parameter);
  CAPTURE(use_massive_dg_operator);

  using SubdomainOperator = elliptic::dg::subdomain_operator::SubdomainOperator<
      System, DummyOptionsGroup, ArgsTagsFromCenter>;

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

  // The test should hold for any number of overlap points
  for (size_t overlap = 0; overlap <= max_overlap; overlap++) {
    CAPTURE(overlap);

    // Re-create the domain in every iteration of this loop because it's not
    // copyable
    auto domain = domain_creator.create_domain();
    const auto initial_ref_levs = domain_creator.initial_refinement_levels();
    const auto initial_extents = domain_creator.initial_extents();
    const auto element_ids = ::initial_element_ids(initial_ref_levs);
    CAPTURE(element_ids.size());

    ActionTesting::MockRuntimeSystem<metavariables> runner{tuples::TaggedTuple<
        domain::Tags::Domain<Dim>, RandomBackgroundTag<Dim>,
        LinearSolver::Schwarz::Tags::MaxOverlap<DummyOptionsGroup>,
        elliptic::dg::Tags::PenaltyParameter, elliptic::dg::Tags::Massive>{
        std::move(domain), RandomBackground<Dim>{}, overlap, penalty_parameter,
        use_massive_dg_operator}};

    // Initialize all elements, generating random subdomain data
    for (const auto& element_id : element_ids) {
      CAPTURE(element_id);
      ActionTesting::emplace_component_and_initialize<element_array>(
          &runner, element_id,
          {initial_ref_levs, initial_extents,
           typename subdomain_operator_applied_to_fields_tag::type{}});
      while (
          not ActionTesting::get_terminate<element_array>(runner, element_id)) {
        ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                  element_id);
      }
    }
    ActionTesting::set_phase(make_not_null(&runner),
                             metavariables::Phase::Testing);
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

      // Set data on the central element and its neighbors to the subdomain data
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
        const auto& [direction, neighbor_id] = overlap_id;
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
            .force_next_action_to_be(0);
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

      // Test that the subdomain operator and the full DG-operator computed the
      // same result within the subdomain
      const auto& subdomain_result =
          get_tag(subdomain_center, subdomain_operator_applied_to_fields_tag{});
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
        const auto& [direction, neighbor_id] = overlap_id;
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
    }  // loop over subdomain centers
  }    // loop over overlaps
}

// Add a constitutive relation for elasticity systems to the DataBox
template <size_t Dim>
struct InitializeConstitutiveRelation {
  using simple_tags = tmpl::list<Elasticity::Tags::ConstitutiveRelation<Dim>>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    ::Initialization::mutate_assign<simple_tags>(
        make_not_null(&box),
        std::make_unique<
            Elasticity::ConstitutiveRelations::IsotropicHomogeneous<Dim>>(1.,
                                                                          2.));
    return {std::move(box)};
  }
};

template <typename System>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
make_boundary_condition(
    const elliptic::BoundaryConditionType boundary_condition_type) {
  using boundary_condition_registrars =
      typename System::boundary_conditions_base::registrars;
  using AnalyticSolutionBoundaryCondition =
      typename elliptic::BoundaryConditions::Registrars::AnalyticSolution<
          System>::template f<boundary_condition_registrars>;
  Parallel::register_derived_classes_with_charm<
      typename System::boundary_conditions_base>();
  return std::make_unique<AnalyticSolutionBoundaryCondition>(
      boundary_condition_type);
}

}  // namespace

// This test constructs a selection of domains and tests the subdomain operator
// for _every_ element in those domains and for a range of overlaps. We increase
// the timeout for the test because going over so many elements is relatively
// expensive but also very important to ensure that the subdomain operator
// handles all of these geometries correctly.
// [[TimeOut, 25]]
SPECTRE_TEST_CASE("Unit.Elliptic.DG.SubdomainOperator", "[Unit][Elliptic]") {
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
              elliptic::BoundaryConditionType::Neumann),
          nullptr};
      for (const bool use_massive_dg_operator : {false, true}) {
        test_subdomain_operator<system>(domain_creator,
                                        use_massive_dg_operator);
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
    }
    {
      INFO("3D");
      using system = Poisson::FirstOrderSystem<3, Poisson::Geometry::Curved>;
      const domain::creators::Brick domain_creator{
          {{-2., 0., -1.}},
          {{2., 1., 1.}},
          {{1, 0, 1}},
          {{3, 3, 3}},
          make_boundary_condition<system>(
              elliptic::BoundaryConditionType::Dirichlet),
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
      for (const bool use_massive_dg_operator : {false, true}) {
        test_subdomain_operator<system>(domain_creator,
                                        use_massive_dg_operator);
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
        make_boundary_condition<system>(
            elliptic::BoundaryConditionType::Dirichlet),
        nullptr};
    test_subdomain_operator<
        system, tmpl::list<InitializeConstitutiveRelation<3>>,
        tmpl::list<::Elasticity::Tags::ConstitutiveRelation<3>>>(
        domain_creator);
  }
}
