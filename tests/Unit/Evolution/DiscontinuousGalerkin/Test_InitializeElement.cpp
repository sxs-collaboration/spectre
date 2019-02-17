// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"  // IWYU pragma: keep
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/InitializeElement.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"  // IWYU pragma: keep
#include "Time/StepControllers/SplitRemaining.hpp"
#include "Time/Tags.hpp"  // IWYU pragma: keep
#include "Time/TimeSteppers/AdamsBashforthN.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_include <boost/variant/get.hpp>
// IWYU pragma: no_include <pup.h>

// IWYU pragma: no_forward_declare ElementIndex
class TimeStepper;
// IWYU pragma: no_forward_declare Variables
namespace PUP {
class er;  // IWYU pragma: keep
}  // namespace PUP
namespace Tags {
template <typename Tag, size_t VolumeDim, typename Fr>
struct ComputeNormalDotFlux;
template <size_t Dim>
struct MortarSize;
template <typename Tag, size_t VolumeDim>
struct Mortars;
}  // namespace Tags
// IWYU pragma: no_forward_declare db::DataBox

namespace {
struct Var : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "Var"; }
};

struct SystemAnalyticSolution {
  template <size_t Dim>
  tuples::TaggedTuple<Var> variables(const tnsr::I<DataVector, Dim>& x,
                                     double t, tmpl::list<Var> /*meta*/) const
      noexcept {
    tuples::TaggedTuple<Var> vars(x.get(0) + t);
    for (size_t d = 1; d < Dim; ++d) {
      get(get<Var>(vars)) += x.get(d) + t;
    }
    return vars;
  }

  template <size_t Dim>
  tuples::TaggedTuple<Tags::dt<Var>> variables(
      const tnsr::I<DataVector, Dim>& x, double t,
      tmpl::list<Tags::dt<Var>> /*meta*/) const noexcept {
    tuples::TaggedTuple<Tags::dt<Var>> vars{
        Scalar<DataVector>(2.0 * x.get(0) + t)};
    for (size_t d = 1; d < Dim; ++d) {
      get(get<Tags::dt<Var>>(vars)) += 2.0 * x.get(d) + t;
    }
    return vars;
  }

  // clang-tidy: do not use references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

template <size_t Dim, bool IsInFluxConservativeForm>
struct System {
  static constexpr bool is_in_flux_conservative_form = IsInFluxConservativeForm;
  static constexpr bool has_primitive_and_conservative_vars = false;
  static constexpr size_t volume_dim = Dim;
  using variables_tag = Tags::Variables<tmpl::list<Var>>;
  using gradients_tags = tmpl::list<Var>;
  template <typename Tag>
  using magnitude_tag = Tags::EuclideanMagnitude<Tag>;
};

struct NormalDotNumericalFluxTag {
  using type = struct { using package_tags = tmpl::list<Var>; };
};

template <size_t Dim, typename Metavariables>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIndex<Dim>;
  using const_global_cache_tag_list =
      tmpl::list<OptionTags::TypedTimeStepper<TimeStepper>,
                 OptionTags::AnalyticSolution<SystemAnalyticSolution>>;
  using action_list = tmpl::list<>;
  using initial_databox =
      db::compute_databox_type<typename dg::Actions::InitializeElement<
          Dim>::template return_tag_list<Metavariables>>;
};

template <size_t Dim, bool IsConservative, bool LocalTimeStepping,
          typename ConstGlobalCacheTagList>
struct Metavariables {
  using component_list = tmpl::list<component<Dim, Metavariables>>;
  using system = System<Dim, IsConservative>;
  using temporal_id = Tags::TimeId;
  static constexpr bool local_time_stepping = LocalTimeStepping;
  using normal_dot_numerical_flux = NormalDotNumericalFluxTag;
  using const_global_cache_tag_list = ConstGlobalCacheTagList;
};

template <typename Tag, typename Box>
bool box_contains(const Box& /*box*/) noexcept {
  return tmpl::list_contains_v<typename Box::tags_list, Tag>;
}

template <typename Tag, typename Box, typename = cpp17::void_t<>>
struct tag_is_retrievable : std::false_type {};

template <typename Tag, typename Box>
struct tag_is_retrievable<Tag, Box,
                          cpp17::void_t<decltype(db::get<Tag>(Box{}))>>
    : std::true_type {};

template <typename Tag, typename Box>
constexpr bool tag_is_retrievable_v = tag_is_retrievable<Tag, Box>::value;

template <bool IsConservative>
struct TestConservativeOrNonconservativeParts {
  template <typename Metavariables, typename DbTags>
  static void apply(const gsl::not_null<db::DataBox<DbTags>*> box) noexcept {
    using system = typename Metavariables::system;
    constexpr size_t dim = system::volume_dim;

    CHECK(box_contains<Tags::DerivCompute<
              typename system::variables_tag,
              Tags::InverseJacobian<Tags::ElementMap<dim>,
                                    Tags::LogicalCoordinates<dim>>,
              typename system::gradients_tags>>(*box));
  }
};

template <>
struct TestConservativeOrNonconservativeParts<true> {
  template <typename Metavariables, typename DbTags>
  static void apply(const gsl::not_null<db::DataBox<DbTags>*> box) noexcept {
    using system = typename Metavariables::system;
    constexpr size_t dim = system::volume_dim;
    using variables_tag = typename system::variables_tag;

    const size_t number_of_grid_points =
        get<Tags::Mesh<dim>>(*box).number_of_grid_points();

    CHECK(
        db::get<db::add_tag_prefix<Tags::Flux, Tags::Variables<tmpl::list<Var>>,
                                   tmpl::size_t<dim>, Frame::Inertial>>(*box)
            .number_of_grid_points() == number_of_grid_points);
    CHECK(db::get<db::add_tag_prefix<Tags::Source, variables_tag>>(*box)
              .number_of_grid_points() == number_of_grid_points);

    CHECK(box_contains<Tags::DivCompute<
              db::add_tag_prefix<Tags::Flux, variables_tag, tmpl::size_t<dim>,
                                 Frame::Inertial>,
              Tags::InverseJacobian<Tags::ElementMap<dim>,
                                    Tags::LogicalCoordinates<dim>>>>(*box));

    CHECK(tag_is_retrievable_v<
          Tags::Interface<
              Tags::InternalDirections<dim>,
              Tags::ComputeNormalDotFlux<variables_tag, dim, Frame::Inertial>>,
          std::decay_t<decltype(*box)>>);
  }
};

template <typename DirectionsTag, typename System, typename DataBox_t>
void test_interface_tags() {
  const auto dim = System::volume_dim;
  CHECK(tag_is_retrievable_v<
        Tags::Interface<DirectionsTag, Tags::UnnormalizedFaceNormal<dim>>,
        DataBox_t>);
  using magnitude_tag =
      Tags::EuclideanMagnitude<Tags::UnnormalizedFaceNormal<dim>>;
  CHECK(tag_is_retrievable_v<Tags::Interface<DirectionsTag, magnitude_tag>,
                             DataBox_t>);
  CHECK(tag_is_retrievable_v<
        Tags::Interface<DirectionsTag,
                        Tags::Normalized<Tags::UnnormalizedFaceNormal<dim>>>,
        DataBox_t>);
  CHECK(tag_is_retrievable_v<
        Tags::Interface<DirectionsTag, typename System::variables_tag>,
        DataBox_t>);
}

template <typename Metavariables, typename DomainCreatorType,
          typename CacheTuple>
void test_initialize_element(
    CacheTuple cache_tuple,
    const ElementId<Metavariables::system::volume_dim>& element_id,
    const double start_time, const double dt, const double slab_size,
    const DomainCreatorType& domain_creator) noexcept {
  using system = typename Metavariables::system;
  constexpr size_t dim = system::volume_dim;

  const auto domain = domain_creator.create_domain();

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  using my_component = component<dim, Metavariables>;
  using MockDistributedObjectsTag =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          my_component>;
  typename MockRuntimeSystem::TupleOfMockDistributedObjects dist_objects{};
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(element_id,
               ActionTesting::MockDistributedObject<my_component>{});

  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      std::move(cache_tuple), std::move(dist_objects)};

  runner.template simple_action<my_component,
                                dg::Actions::InitializeElement<dim>>(
      element_id, domain_creator.initial_extents(),
      domain_creator.create_domain(), start_time, dt, slab_size);
  auto& box =
      runner.template algorithms<my_component>()
          .at(element_id)
          .template get_databox<typename my_component::initial_databox>();

  const auto& stepper = Parallel::get<OptionTags::TimeStepper>(runner.cache());

  CHECK(db::get<Tags::TimeStep>(box).value() == dt);
  CHECK(db::get<Tags::Next<Tags::TimeId>>(box).time_runs_forward());
  CHECK(db::get<Tags::Next<Tags::TimeId>>(box).slab_number() ==
        -static_cast<int64_t>(stepper.number_of_past_steps()));
  CHECK(db::get<Tags::Next<Tags::TimeId>>(box).time().value() == start_time);
  CHECK(
      db::get<Tags::Next<Tags::TimeId>>(box).time().slab().duration().value() ==
      slab_size);
  // The TimeId is uninitialized and is updated immediately by the
  // algorithm loop.
  CHECK(box_contains<Tags::TimeId>(box));
  CHECK(box_contains<Tags::Time>(box));

  const auto& my_block = domain.blocks()[element_id.block_id()];
  ElementMap<dim, Frame::Inertial> map{element_id,
                                       my_block.coordinate_map().get_clone()};
  Element<dim> element = create_initial_element(element_id, my_block);
  Mesh<dim> mesh{domain_creator.initial_extents()[element_id.block_id()],
                 Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};
  auto logical_coords = logical_coordinates(mesh);
  auto inertial_coords = map(logical_coords);
  CHECK(db::get<Tags::LogicalCoordinates<dim>>(box) == logical_coords);
  CHECK(db::get<Tags::Mesh<dim>>(box) == mesh);
  CHECK(db::get<Tags::Element<dim>>(box) == element);
  CHECK(box_contains<Tags::ElementMap<dim>>(box));
  CHECK(db::get<Var>(box) == ([&inertial_coords, &start_time]() {
          Scalar<DataVector> var{inertial_coords.get(0) + start_time};
          for (size_t d = 1; d < dim; ++d) {
            get(var) += inertial_coords.get(d) + start_time;
          }
          return var;
        }()));
  {
    const auto& history = db::get<Tags::HistoryEvolvedVariables<
        typename system::variables_tag,
        db::add_tag_prefix<Tags::dt, typename system::variables_tag>>>(
        box);
    CHECK(history.size() == 0);
    const SystemAnalyticSolution solution{};
    double past_t = start_time;
    for (size_t i = history.size(); i > 0; --i) {
      const auto entry =
          history.begin() +
          static_cast<
              typename std::decay_t<decltype(history)>::difference_type>(i - 1);
      past_t -= dt;

      CHECK(entry->value() == past_t);
      CHECK(get<Var>(entry.value()) ==
            get<Var>(solution.variables(inertial_coords, past_t,
                                        tmpl::list<Var>{})));
      CHECK(get<Tags::dt<Var>>(entry.derivative()) ==
            get<Tags::dt<Var>>(solution.variables(
                inertial_coords, past_t, tmpl::list<Tags::dt<Var>>{})));
    }
  }
  CHECK((db::get<Tags::MappedCoordinates<Tags::ElementMap<dim>,
                                         Tags::LogicalCoordinates<dim>>>(
            box)) == inertial_coords);
  CHECK((db::get<Tags::InverseJacobian<Tags::ElementMap<dim>,
                                       Tags::LogicalCoordinates<dim>>>(box)) ==
        map.inv_jacobian(logical_coords));
  CHECK(db::get<
            db::add_tag_prefix<Tags::dt, typename system::variables_tag>>(
            box)
            .size() == mesh.number_of_grid_points());

  if (Metavariables::local_time_stepping) {
    CHECK(box_contains<typename dg::FluxCommunicationTypes<
              Metavariables>::local_time_stepping_mortar_data_tag>(box));
  } else {
    CHECK(box_contains<typename dg::FluxCommunicationTypes<
              Metavariables>::simple_mortar_data_tag>(box));
  }
  CHECK(db::get<Tags::VariablesBoundaryData>(box).size() ==
        2 * dim);
  CHECK(db::get<Tags::Mortars<Tags::Next<Tags::TimeId>, dim>>(box).size() ==
        2 * dim);
  CHECK(db::get<Tags::Mortars<Tags::Mesh<dim - 1>, dim>>(box).size() ==
        2 * dim);
  CHECK(db::get<Tags::Mortars<Tags::MortarSize<dim - 1>, dim>>(box).size() ==
        2 * dim);

  using databox_t = std::decay_t<decltype(box)>;
  test_interface_tags<Tags::InternalDirections<dim>, system, databox_t>();
  test_interface_tags<Tags::BoundaryDirectionsInterior<dim>, system,
                      databox_t>();
  test_interface_tags<Tags::BoundaryDirectionsExterior<dim>, system,
                      databox_t>();

  TestConservativeOrNonconservativeParts<system::is_in_flux_conservative_form>::
      template apply<Metavariables>(make_not_null(&box));
}

void test_mortar_orientation() noexcept {
  using metavariables = Metavariables<3, false, false, tmpl::list<>>;
  ElementId<3> element_id(0);

  // This is the domain from the OrientationMap and corner numbering
  // tutorial.
  Domain<3, Frame::Inertial> domain(
      domain::make_vector_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          domain::CoordinateMaps::Identity<3>{},
          domain::CoordinateMaps::Identity<3>{}),
      {{{0, 1, 3, 4, 6, 7, 9, 10}}, {{1, 4, 7, 10, 2, 5, 8, 11}}});
  const auto neighbor_direction = Direction<3>::upper_xi();
  const auto mortar_id = std::make_pair(neighbor_direction, ElementId<3>(1));
  const std::vector<std::array<size_t, 3>> extents{{{2, 2, 2}}, {{3, 4, 5}}};

  using my_component = component<3, metavariables>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavariables>;
  using MockDistributedObjectsTag =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          my_component>;
  typename MockRuntimeSystem::TupleOfMockDistributedObjects dist_objects{};
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(ElementIndex<3>{element_id},
               ActionTesting::MockDistributedObject<my_component>{});

  ActionTesting::MockRuntimeSystem<metavariables> runner{
      {std::make_unique<TimeSteppers::AdamsBashforthN>(4),
       SystemAnalyticSolution{}},
      std::move(dist_objects)};

  runner.simple_action<my_component, dg::Actions::InitializeElement<3>>(
      element_id, extents, std::move(domain), 0., 1., 1.);
  const auto& box =
      runner.template algorithms<my_component>()
          .at(element_id)
          .template get_databox<typename my_component::initial_databox>();

  CHECK(db::get<Tags::Mortars<Tags::Mesh<2>, 3>>(box).at(mortar_id).extents() ==
        Index<2>{{{3, 4}}});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.dG.InitializeElement",
                  "[Unit][Evolution][Actions]") {
  test_initialize_element<Metavariables<1, false, false, tmpl::list<>>>(
      ActionTesting::MockRuntimeSystem<
          Metavariables<1, false, false, tmpl::list<>>>::CacheTuple{
          std::make_unique<TimeSteppers::AdamsBashforthN>(4),
          SystemAnalyticSolution{}},
      ElementId<1>{0, {{SegmentId{2, 1}}}}, 3., 1., 1.,
      domain::creators::Interval<Frame::Inertial>{
          {{-0.5}}, {{1.5}}, {{false}}, {{2}}, {{4}}});

  test_initialize_element<Metavariables<1, false, false, tmpl::list<>>>(
      ActionTesting::MockRuntimeSystem<
          Metavariables<1, false, false, tmpl::list<>>>::CacheTuple{
          std::make_unique<TimeSteppers::AdamsBashforthN>(4),
          SystemAnalyticSolution{}},
      ElementId<1>{0, {{SegmentId{2, 0}}}}, 3., 1., 1.,
      domain::creators::Interval<Frame::Inertial>{
          {{-0.5}}, {{1.5}}, {{false}}, {{2}}, {{4}}});

  test_initialize_element<Metavariables<2, false, false, tmpl::list<>>>(
      ActionTesting::MockRuntimeSystem<
          Metavariables<2, false, false, tmpl::list<>>>::CacheTuple{
          std::make_unique<TimeSteppers::AdamsBashforthN>(4),
          SystemAnalyticSolution{}},
      ElementId<2>{0, {{SegmentId{2, 0}, SegmentId{3, 2}}}}, 3., 1., 1.,
      domain::creators::Rectangle<Frame::Inertial>{
          {{-0.5, -0.75}}, {{1.5, 2.4}}, {{false, false}}, {{2, 3}}, {{4, 5}}});

  test_initialize_element<Metavariables<2, false, false, tmpl::list<>>>(
      ActionTesting::MockRuntimeSystem<
          Metavariables<2, false, false, tmpl::list<>>>::CacheTuple{
          std::make_unique<TimeSteppers::AdamsBashforthN>(4),
          SystemAnalyticSolution{}},
      ElementId<2>{0, {{SegmentId{2, 0}, SegmentId{3, 7}}}}, 3., 1., 1.,
      domain::creators::Rectangle<Frame::Inertial>{
          {{-0.5, -0.75}}, {{1.5, 2.4}}, {{false, false}}, {{2, 3}}, {{4, 5}}});

  test_initialize_element<Metavariables<3, false, false, tmpl::list<>>>(
      ActionTesting::MockRuntimeSystem<
          Metavariables<3, false, false, tmpl::list<>>>::CacheTuple{
          std::make_unique<TimeSteppers::AdamsBashforthN>(4),
          SystemAnalyticSolution{}},
      ElementId<3>{0, {{SegmentId{2, 1}, SegmentId{3, 2}, SegmentId{1, 0}}}},
      3., 1., 1.,
      domain::creators::Brick<Frame::Inertial>{{{-0.5, -0.75, -1.2}},
                                               {{1.5, 2.4, 1.2}},
                                               {{false, false, true}},
                                               {{2, 3, 1}},
                                               {{4, 5, 3}}});

  test_initialize_element<Metavariables<3, false, false, tmpl::list<>>>(
      ActionTesting::MockRuntimeSystem<
          Metavariables<3, false, false, tmpl::list<>>>::CacheTuple{
          std::make_unique<TimeSteppers::AdamsBashforthN>(4),
          SystemAnalyticSolution{}},

      ElementId<3>{0, {{SegmentId{0, 0}, SegmentId{0, 0}, SegmentId{1, 1}}}},
      3., 1., 1.,
      domain::creators::Brick<Frame::Inertial>{{{-0.5, -0.75, -1.2}},
                                               {{1.5, 2.4, 1.2}},
                                               {{false, false, false}},
                                               {{0, 0, 1}},
                                               {{4, 5, 3}}});

  test_initialize_element<Metavariables<2, true, false, tmpl::list<>>>(
      ActionTesting::MockRuntimeSystem<
          Metavariables<2, true, false, tmpl::list<>>>::CacheTuple{
          std::make_unique<TimeSteppers::AdamsBashforthN>(4),
          SystemAnalyticSolution{}},
      ElementId<2>{0, {{SegmentId{2, 1}, SegmentId{3, 2}}}}, 3., 1., 1.,
      domain::creators::Rectangle<Frame::Inertial>{
          {{-0.5, -0.75}}, {{1.5, 2.4}}, {{false, false}}, {{2, 3}}, {{4, 5}}});

  // local time-stepping
  test_initialize_element<
      Metavariables<2, false, true, tmpl::list<OptionTags::StepController>>>(
      ActionTesting::MockRuntimeSystem<Metavariables<
          2, false, true, tmpl::list<OptionTags::StepController>>>::CacheTuple{
          std::make_unique<StepControllers::SplitRemaining>(),
          std::make_unique<TimeSteppers::AdamsBashforthN>(4),
          SystemAnalyticSolution{}},
      ElementId<2>{0, {{SegmentId{2, 1}, SegmentId{3, 2}}}}, 1.5, 0.25, 0.5,
      domain::creators::Rectangle<Frame::Inertial>{
          {{-0.5, -0.75}}, {{1.5, 2.4}}, {{false, false}}, {{2, 3}}, {{4, 5}}});

  test_mortar_orientation();
}
