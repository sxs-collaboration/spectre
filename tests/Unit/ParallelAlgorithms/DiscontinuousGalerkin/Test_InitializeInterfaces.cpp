// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeInterfaces.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct OtherDataTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <typename VarsTag>
struct SomeComputeTag : db::SimpleTag {
  using type = size_t;
};

template <typename VarsTag>
struct SomeComputeTagCompute : SomeComputeTag<VarsTag>, db::ComputeTag {
  using base = SomeComputeTag<VarsTag>;
  using return_type = size_t;
  static void function(gsl::not_null<size_t*> result,
                       const typename VarsTag::type& vars) {
    *result = vars.number_of_grid_points();
  }
  using argument_tags = tmpl::list<VarsTag>;
};

using vars_tag = Tags::Variables<tmpl::list<ScalarFieldTag>>;
using other_vars_tag = Tags::Variables<tmpl::list<OtherDataTag>>;

template <size_t Dim>
struct System {
  static constexpr size_t volume_dim = Dim;
  using variables_tag = vars_tag;
  template <typename Tag>
  using magnitude_tag = ::Tags::EuclideanMagnitude<Tag>;
};

template <size_t Dim, typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  static constexpr bool use_moving_mesh = Metavariables::use_moving_mesh;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Dim>;
  using const_global_cache_tags = tmpl::append<
      tmpl::list<domain::Tags::Domain<Dim>>,
      tmpl::conditional_t<use_moving_mesh,
                          tmpl::list<Tags::TimeStepper<TimeStepper>>,
                          tmpl::list<>>>;
  using mutable_global_cache_tags = tmpl::conditional_t<
      use_moving_mesh, tmpl::list<domain::Tags::FunctionsOfTime>, tmpl::list<>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::push_front<
              tmpl::conditional_t<
                  use_moving_mesh,
                  tmpl::list<
                      Actions::SetupDataBox,
                      Initialization::Actions::TimeAndTimeStep<Metavariables>,
                      evolution::dg::Initialization::Domain<Dim>>,
                  tmpl::list<Actions::SetupDataBox,
                             dg::Actions::InitializeDomain<Dim>>>,
              ActionTesting::InitializeDataBox<tmpl::append<
                  tmpl::list<domain::Tags::InitialRefinementLevels<Dim>,
                             domain::Tags::InitialExtents<Dim>,
                             evolution::dg::Tags::Quadrature, vars_tag,
                             other_vars_tag>,
                  tmpl::conditional_t<
                      use_moving_mesh,
                      tmpl::list<Initialization::Tags::InitialTime,
                                 Initialization::Tags::InitialTimeDelta,
                                 Initialization::Tags::InitialSlabSize<
                                     Metavariables::local_time_stepping>>,
                      tmpl::list<>>>>>>,

      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<
              dg::Actions::InitializeInterfaces<
                  System<Dim>, dg::Initialization::slice_tags_to_face<vars_tag>,
                  dg::Initialization::slice_tags_to_exterior<other_vars_tag>,
                  dg::Initialization::face_compute_tags<
                      SomeComputeTagCompute<vars_tag>>,
                  dg::Initialization::exterior_compute_tags<
                      SomeComputeTagCompute<other_vars_tag>>,
                  true, use_moving_mesh>>>>;
};

template <size_t Dim, bool UseMovingMesh>
struct Metavariables {
  static constexpr size_t dim = Dim;
  static constexpr bool use_moving_mesh = UseMovingMesh;
  static constexpr bool local_time_stepping = false;
  using system = System<Dim>;
  using element_array = ElementArray<Dim, Metavariables>;
  using component_list = tmpl::list<element_array>;
  enum class Phase { Initialization, Testing, Exit };
};

template <size_t Dim, bool UseMovingMesh>
void check_compute_items(const ActionTesting::MockRuntimeSystem<
                             Metavariables<Dim, UseMovingMesh>>& runner,
                         const ElementId<Dim>& element_id) noexcept {
  // The compute items themselves are tested elsewhere, so just check if they
  // were indeed added by the initializer
  const auto tag_is_retrievable = [&runner,
                                   &element_id](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::tag_is_retrievable<
        ElementArray<Dim, Metavariables<Dim, UseMovingMesh>>, tag>(runner,
                                                                   element_id);
  };
  CHECK(tag_is_retrievable(domain::Tags::InternalDirections<Dim>{}));
  CHECK(tag_is_retrievable(domain::Tags::BoundaryDirectionsInterior<Dim>{}));
  CHECK(tag_is_retrievable(domain::Tags::BoundaryDirectionsExterior<Dim>{}));
  CHECK(tag_is_retrievable(domain::Tags::Interface<
                           domain::Tags::BoundaryDirectionsExterior<Dim>,
                           domain::Tags::Coordinates<Dim, Frame::Inertial>>{}));
  CHECK(tag_is_retrievable(
      domain::Tags::Interface<domain::Tags::InternalDirections<Dim>,
                              vars_tag>{}));
  CHECK(tag_is_retrievable(
      domain::Tags::Interface<domain::Tags::BoundaryDirectionsInterior<Dim>,
                              vars_tag>{}));
  CHECK(tag_is_retrievable(
      domain::Tags::Interface<domain::Tags::BoundaryDirectionsExterior<Dim>,
                              other_vars_tag>{}));
  CHECK(tag_is_retrievable(
      domain::Tags::Interface<domain::Tags::InternalDirections<Dim>,
                              SomeComputeTag<vars_tag>>{}));
  CHECK(tag_is_retrievable(
      domain::Tags::Interface<domain::Tags::BoundaryDirectionsInterior<Dim>,
                              SomeComputeTag<vars_tag>>{}));
  CHECK(tag_is_retrievable(
      domain::Tags::Interface<domain::Tags::BoundaryDirectionsExterior<Dim>,
                              SomeComputeTag<other_vars_tag>>{}));
  CHECK(tag_is_retrievable(
      domain::Tags::Interface<
          domain::Tags::InternalDirections<Dim>,
          ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>{}));
  CHECK(tag_is_retrievable(
      domain::Tags::Interface<
          domain::Tags::BoundaryDirectionsInterior<Dim>,
          ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>{}));
  CHECK(tag_is_retrievable(
      domain::Tags::Interface<
          domain::Tags::BoundaryDirectionsExterior<Dim>,
          ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>{}));
}

template <typename Metavariables,
          Requires<Metavariables::use_moving_mesh> = nullptr>
void create_runner_and_run_tests(
    const DomainCreator<Metavariables::dim>& domain_creator,
    const ElementId<Metavariables::dim>& element_id) {
  using element_array = typename Metavariables::element_array;
  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {domain_creator.create_domain(),
       std::make_unique<TimeSteppers::RungeKutta3>()},
      {domain_creator.functions_of_time()}};
  constexpr size_t num_points =
      Metavariables::dim == 1 ? 4 : Metavariables::dim == 2 ? 12 : 24;
  typename vars_tag::type vars{num_points, 0.};
  typename other_vars_tag::type other_vars{num_points, 0.};

  ActionTesting::emplace_component_and_initialize<
      typename Metavariables::element_array>(
      &runner, element_id,
      {domain_creator.initial_refinement_levels(),
       domain_creator.initial_extents(), Spectral::Quadrature::GaussLobatto,
       vars, other_vars, 0.0, 0.1, 0.1});

  ActionTesting::next_action<element_array>(make_not_null(&runner), element_id);
  ActionTesting::next_action<element_array>(make_not_null(&runner), element_id);
  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
  }

  check_compute_items(runner, element_id);
}

template <typename Metavariables,
          Requires<not Metavariables::use_moving_mesh> = nullptr>
void create_runner_and_run_tests(
    const DomainCreator<Metavariables::dim>& domain_creator,
    const ElementId<Metavariables::dim>& element_id) {
  using element_array = typename Metavariables::element_array;
  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {domain_creator.create_domain()}};
  constexpr size_t num_points =
      Metavariables::dim == 1 ? 4 : Metavariables::dim == 2 ? 12 : 24;
  typename vars_tag::type vars{num_points, 0.};
  typename other_vars_tag::type other_vars{num_points, 0.};

  ActionTesting::emplace_component_and_initialize<
      typename Metavariables::element_array>(
      &runner, element_id,
      {domain_creator.initial_refinement_levels(),
       domain_creator.initial_extents(), Spectral::Quadrature::GaussLobatto,
       vars, other_vars});

  ActionTesting::next_action<element_array>(make_not_null(&runner), element_id);
  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);
  ActionTesting::next_action<element_array>(make_not_null(&runner), element_id);

  check_compute_items(runner, element_id);
}

template <bool UseMovingMesh>
void test_1d() {
  INFO("1D");
  // Reference element:
  // [ |X| | ]-> xi
  const ElementId<1> element_id{0, {{SegmentId{2, 1}}}};
  const domain::creators::Interval domain_creator{{{-0.5}}, {{1.5}},   {{2}},
                                                  {{4}},    {{false}}, nullptr};
  // Register the coordinate map for serialization
  PUPable_reg(
      SINGLE_ARG(domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                                       domain::CoordinateMaps::Affine>));

  create_runner_and_run_tests<Metavariables<1, UseMovingMesh>>(domain_creator,
                                                               element_id);
}

template <bool UseMovingMesh>
void test_2d() {
  INFO("2D");
  // Reference element:
  // ^ eta
  // +-+-+-+-+> xi
  // | |X| | |
  // +-+-+-+-+
  const ElementId<2> element_id{0, {{SegmentId{2, 1}, SegmentId{0, 0}}}};
  const domain::creators::Rectangle domain_creator{
      {{-0.5, 0.}}, {{1.5, 2.}}, {{false, false}}, {{2, 0}}, {{4, 3}}};
  // Register the coordinate map for serialization
  PUPable_reg(
      SINGLE_ARG(domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                                       domain::CoordinateMaps::ProductOf2Maps<
                                           domain::CoordinateMaps::Affine,
                                           domain::CoordinateMaps::Affine>>));

  create_runner_and_run_tests<Metavariables<2, UseMovingMesh>>(domain_creator,
                                                               element_id);
}

template <bool UseMovingMesh>
void test_3d() {
  INFO("3D");
  const ElementId<3> element_id{
      0, {{SegmentId{2, 1}, SegmentId{0, 0}, SegmentId{1, 1}}}};
  const domain::creators::Brick domain_creator{{{-0.5, 0., -1.}},
                                               {{1.5, 2., 3.}},
                                               {{false, false, false}},
                                               {{2, 0, 1}},
                                               {{4, 3, 2}}};
  // Register the coordinate map for serialization
  PUPable_reg(SINGLE_ARG(
      domain::CoordinateMap<
          Frame::Logical, Frame::Inertial,
          domain::CoordinateMaps::ProductOf3Maps<
              domain::CoordinateMaps::Affine, domain::CoordinateMaps::Affine,
              domain::CoordinateMaps::Affine>>));

  create_runner_and_run_tests<Metavariables<3, UseMovingMesh>>(domain_creator,
                                                               element_id);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelDG.InitializeInterfaces", "[Unit][Actions]") {
  domain::creators::register_derived_with_charm();
  Parallel::register_derived_classes_with_charm<TimeStepper>();

  test_1d<false>();
  test_2d<false>();
  test_3d<false>();

  test_1d<true>();
  test_2d<true>();
  test_3d<true>();
}
