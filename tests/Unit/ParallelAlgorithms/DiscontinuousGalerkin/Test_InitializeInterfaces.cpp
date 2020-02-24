// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
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
#include "Domain/Domain.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeInterfaces.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {

struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct OtherDataTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <typename VarsTag>
struct SomeComputeTag : db::ComputeTag {
  static std::string name() noexcept { return "SomeComputeTag"; }
  static size_t function(const db::const_item_type<VarsTag>& vars) {
    return vars.number_of_grid_points();
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
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIndex<Dim>;
  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<Dim>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<tmpl::list<
                  domain::Tags::InitialExtents<Dim>, vars_tag, other_vars_tag>>,
              dg::Actions::InitializeDomain<Dim>>>,

      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<dg::Actions::InitializeInterfaces<
              System<Dim>, dg::Initialization::slice_tags_to_face<vars_tag>,
              dg::Initialization::slice_tags_to_exterior<other_vars_tag>,
              dg::Initialization::face_compute_tags<SomeComputeTag<vars_tag>>,
              dg::Initialization::exterior_compute_tags<
                  SomeComputeTag<other_vars_tag>>>>>>;
};

template <size_t Dim>
struct Metavariables {
  using system = System<Dim>;
  using component_list = tmpl::list<ElementArray<Dim, Metavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};

template <size_t Dim>
void check_compute_items(
    const ActionTesting::MockRuntimeSystem<Metavariables<Dim>>& runner,
    const ElementId<Dim>& element_id) {
  // The compute items themselves are tested elsewhere, so just check if they
  // were indeed added by the initializer
  const auto tag_is_retrievable = [&runner,
                                   &element_id](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::tag_is_retrievable<
        ElementArray<Dim, Metavariables<Dim>>, tag>(runner, element_id);
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

}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelDG.InitializeInterfaces", "[Unit][Actions]") {
  {
    INFO("1D");
    // Reference element:
    // [ |X| | ]-> xi
    const ElementId<1> element_id{0, {{SegmentId{2, 1}}}};
    const domain::creators::Interval domain_creator{
        {{-0.5}}, {{1.5}}, {{false}}, {{2}}, {{4}}};
    // Register the coordinate map for serialization
    PUPable_reg(
        SINGLE_ARG(domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                                         domain::CoordinateMaps::Affine>));

    db::item_type<vars_tag> vars{4, 0.};
    db::item_type<other_vars_tag> other_vars{4, 0.};

    using metavariables = Metavariables<1>;
    using element_array = ElementArray<1, metavariables>;
    ActionTesting::MockRuntimeSystem<metavariables> runner{
        {domain_creator.create_domain()}};
    ActionTesting::emplace_component_and_initialize<element_array>(
        &runner, element_id,
        {domain_creator.initial_extents(), vars, other_vars});
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    runner.set_phase(metavariables::Phase::Testing);
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);

    check_compute_items(runner, element_id);
  }
  {
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

    db::item_type<vars_tag> vars{12, 0.};
    db::item_type<other_vars_tag> other_vars{12, 0.};

    using metavariables = Metavariables<2>;
    using element_array = ElementArray<2, metavariables>;
    ActionTesting::MockRuntimeSystem<metavariables> runner{
        {domain_creator.create_domain()}};
    ActionTesting::emplace_component_and_initialize<element_array>(
        &runner, element_id,
        {domain_creator.initial_extents(), vars, other_vars});
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    runner.set_phase(metavariables::Phase::Testing);
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);

    check_compute_items(runner, element_id);
  }
  {
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

    db::item_type<vars_tag> vars{24, 0.};
    db::item_type<other_vars_tag> other_vars{24, 0.};

    using metavariables = Metavariables<3>;
    using element_array = ElementArray<3, metavariables>;
    ActionTesting::MockRuntimeSystem<metavariables> runner{
        {domain_creator.create_domain()}};
    ActionTesting::emplace_component_and_initialize<element_array>(
        &runner, element_id,
        {domain_creator.initial_extents(), vars, other_vars});
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    runner.set_phase(metavariables::Phase::Testing);
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);

    check_compute_items(runner, element_id);
  }
}
