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
#include "Domain/Actions/InitializeDomain.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/InitializeInterfaces.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/InitializeMortars.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/AddOptionsToDataBox.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

namespace {

struct TemporalId : db::SimpleTag {
  static std::string name() noexcept { return "TemporalId"; };
  using type = int;
};

struct ScalarFieldTag : db::SimpleTag {
  static std::string name() noexcept { return "ScalarFieldTag"; };
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct System {
  static constexpr size_t volume_dim = Dim;
  using variables_tag = Tags::Variables<tmpl::list<ScalarFieldTag>>;
  template <typename Tag>
  using magnitude_tag = Tags::EuclideanMagnitude<Tag>;
};

template <size_t Dim, typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIndex<Dim>;
  using const_global_cache_tag_list = tmpl::list<>;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<
                         tmpl::list<::Tags::Domain<Dim, Frame::Inertial>,
                                    ::Tags::InitialExtents<Dim>, TemporalId>>,
                     domain::Actions::InitializeDomain<Dim>,
                     dg::Actions::InitializeInterfaces<
                         System<Dim>, dg::Initialization::slice_tags_to_face<>,
                         dg::Initialization::slice_tags_to_exterior<>>>>,

      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<dg::Actions::InitializeMortars<
              metavariables, true, Initialization::MergePolicy::Overwrite>>>>;
};

template <size_t Dim>
struct Metavariables {
  using system = System<Dim>;
  using component_list = tmpl::list<ElementArray<Dim, Metavariables>>;
  using temporal_id = TemporalId;
  static constexpr bool local_time_stepping = false;
  struct normal_dot_numerical_flux {
    struct type {
      using package_tags = tmpl::list<ScalarFieldTag>;
    };
  };
  using const_global_cache_tag_list = tmpl::list<>;
  enum class Phase { Initialization, Testing, Exit };
};

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Actions.InitializeMortars",
                  "[Unit][Elliptic][Actions]") {
  {
    INFO("1D");
    // Reference element:
    // [X| | | ] -xi->
    const ElementId<1> element_id{0, {{{2, 0}}}};
    const domain::creators::Interval<Frame::Inertial> domain_creator{
        {{-0.5}}, {{1.5}}, {{false}}, {{2}}, {{4}}};
    // Register the coordinate map for serialization
    PUPable_reg(
        SINGLE_ARG(domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                                         domain::CoordinateMaps::Affine>));

    using metavariables = Metavariables<1>;
    using element_array = ElementArray<1, metavariables>;
    ActionTesting::MockRuntimeSystem<metavariables> runner{{}};
    ActionTesting::emplace_component_and_initialize<element_array>(
        &runner, element_id,
        {domain_creator.create_domain(), domain_creator.initial_extents(), 0});
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    runner.set_phase(metavariables::Phase::Testing);
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    const auto get_tag = [&runner, &element_id](auto tag_v) -> decltype(auto) {
      using tag = std::decay_t<decltype(tag_v)>;
      return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                                element_id);
    };

    // We are working with 2 mortars here: a domain boundary at lower xi and an
    // interface at upper xi.
    const auto boundary_mortar_id = std::make_pair(
        Direction<1>::lower_xi(), ElementId<1>::external_boundary_id());
    const auto interface_mortar_id = std::make_pair(
        Direction<1>::upper_xi(), ElementId<1>(0, {{SegmentId{2, 1}}}));
    const auto& mortar_next_temporal_ids =
        get_tag(Tags::Mortars<Tags::Next<TemporalId>, 1>{});
    CHECK(mortar_next_temporal_ids.at(boundary_mortar_id) == 0);
    CHECK(mortar_next_temporal_ids.at(interface_mortar_id) == 0);
    const auto& mortar_meshes = get_tag(Tags::Mortars<Tags::Mesh<0>, 1>{});
    CHECK(mortar_meshes.at(boundary_mortar_id) == Mesh<0>());
    CHECK(mortar_meshes.at(interface_mortar_id) == Mesh<0>());
    const auto& mortar_sizes = get_tag(Tags::Mortars<Tags::MortarSize<0>, 1>{});
    CHECK(mortar_sizes.at(boundary_mortar_id).empty());
    CHECK(mortar_sizes.at(interface_mortar_id).empty());
    const auto& mortar_data = get_tag(Tags::VariablesBoundaryData{});
    // Just make sure this exists, it is not expected to hold any data
    mortar_data.at(boundary_mortar_id);
    mortar_data.at(interface_mortar_id);
  }
  {
    INFO("2D");
    // Reference element:
    // ^ eta
    // +-+-+> xi
    // |X| |
    // +-+-+
    // | | |
    // +-+-+
    const ElementId<2> element_id{0, {{{1, 0}, {1, 1}}}};
    const domain::creators::Rectangle<Frame::Inertial> domain_creator{
        {{-0.5, 0.}}, {{1.5, 1.}}, {{false, false}}, {{1, 1}}, {{3, 2}}};
    // Register the coordinate map for serialization
    PUPable_reg(
        SINGLE_ARG(domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                                         domain::CoordinateMaps::ProductOf2Maps<
                                             domain::CoordinateMaps::Affine,
                                             domain::CoordinateMaps::Affine>>));

    using metavariables = Metavariables<2>;
    using element_array = ElementArray<2, metavariables>;
    ActionTesting::MockRuntimeSystem<metavariables> runner{{}};
    ActionTesting::emplace_component_and_initialize<element_array>(
        &runner, element_id,
        {domain_creator.create_domain(), domain_creator.initial_extents(), 0});
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    runner.set_phase(metavariables::Phase::Testing);
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    const auto get_tag = [&runner, &element_id](auto tag_v) -> decltype(auto) {
      using tag = std::decay_t<decltype(tag_v)>;
      return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                                element_id);
    };

    // We are working with 4 mortars here: the domain boundary west and north,
    // and interfaces south and east.
    const auto boundary_mortar_id_west = std::make_pair(
        Direction<2>::lower_xi(), ElementId<2>::external_boundary_id());
    const auto boundary_mortar_id_north = std::make_pair(
        Direction<2>::upper_eta(), ElementId<2>::external_boundary_id());
    const auto interface_mortar_id_east =
        std::make_pair(Direction<2>::upper_xi(),
                       ElementId<2>(0, {{SegmentId{1, 1}, SegmentId{1, 1}}}));
    const auto interface_mortar_id_south =
        std::make_pair(Direction<2>::lower_eta(),
                       ElementId<2>(0, {{SegmentId{1, 0}, SegmentId{1, 0}}}));
    const auto& mortar_next_temporal_ids =
        get_tag(Tags::Mortars<Tags::Next<TemporalId>, 2>{});
    CHECK(mortar_next_temporal_ids.at(boundary_mortar_id_west) == 0);
    CHECK(mortar_next_temporal_ids.at(boundary_mortar_id_north) == 0);
    CHECK(mortar_next_temporal_ids.at(interface_mortar_id_east) == 0);
    CHECK(mortar_next_temporal_ids.at(interface_mortar_id_south) == 0);
    const auto& mortar_meshes = get_tag(Tags::Mortars<Tags::Mesh<1>, 2>{});
    CHECK(mortar_meshes.at(boundary_mortar_id_west) ==
          Mesh<1>(2, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto));
    CHECK(mortar_meshes.at(boundary_mortar_id_north) ==
          Mesh<1>(3, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto));
    CHECK(mortar_meshes.at(interface_mortar_id_east) ==
          Mesh<1>(2, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto));
    CHECK(mortar_meshes.at(interface_mortar_id_south) ==
          Mesh<1>(3, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto));
    const auto& mortar_sizes = get_tag(Tags::Mortars<Tags::MortarSize<1>, 2>{});
    const std::array<Spectral::MortarSize, 1> expected_mortar_sizes{
        {Spectral::MortarSize::Full}};
    CHECK(mortar_sizes.at(boundary_mortar_id_west) == expected_mortar_sizes);
    CHECK(mortar_sizes.at(boundary_mortar_id_north) == expected_mortar_sizes);
    CHECK(mortar_sizes.at(interface_mortar_id_east) == expected_mortar_sizes);
    CHECK(mortar_sizes.at(interface_mortar_id_south) == expected_mortar_sizes);
    const auto& mortar_data = get_tag(Tags::VariablesBoundaryData{});
    // Just make sure this exists, it is not expected to hold any data
    mortar_data.at(boundary_mortar_id_west);
    mortar_data.at(boundary_mortar_id_north);
    mortar_data.at(interface_mortar_id_east);
    mortar_data.at(interface_mortar_id_south);
  }
  {
    INFO("3D");
    const ElementId<3> element_id{
        0, {{SegmentId{1, 0}, SegmentId{1, 1}, SegmentId{1, 0}}}};
    const domain::creators::Brick<Frame::Inertial> domain_creator{
        {{-0.5, 0., -1.}},
        {{1.5, 1., 3.}},
        {{false, false, false}},
        {{1, 1, 1}},
        {{2, 3, 4}}};
    // Register the coordinate map for serialization
    PUPable_reg(SINGLE_ARG(
        domain::CoordinateMap<
            Frame::Logical, Frame::Inertial,
            domain::CoordinateMaps::ProductOf3Maps<
                domain::CoordinateMaps::Affine, domain::CoordinateMaps::Affine,
                domain::CoordinateMaps::Affine>>));

    using metavariables = Metavariables<3>;
    using element_array = ElementArray<3, metavariables>;
    ActionTesting::MockRuntimeSystem<metavariables> runner{{}};
    ActionTesting::emplace_component_and_initialize<element_array>(
        &runner, element_id,
        {domain_creator.create_domain(), domain_creator.initial_extents(), 0});
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    runner.set_phase(metavariables::Phase::Testing);
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    const auto get_tag = [&runner, &element_id](auto tag_v) -> decltype(auto) {
      using tag = std::decay_t<decltype(tag_v)>;
      return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                                element_id);
    };

    const auto boundary_mortar_id_left = std::make_pair(
        Direction<3>::lower_xi(), ElementId<3>::external_boundary_id());
    const auto boundary_mortar_id_back = std::make_pair(
        Direction<3>::upper_eta(), ElementId<3>::external_boundary_id());
    const auto boundary_mortar_id_bottom = std::make_pair(
        Direction<3>::lower_zeta(), ElementId<3>::external_boundary_id());
    const auto interface_mortar_id_right = std::make_pair(
        Direction<3>::upper_xi(),
        ElementId<3>(0, {{SegmentId{1, 1}, SegmentId{1, 1}, SegmentId{1, 0}}}));
    const auto interface_mortar_id_front = std::make_pair(
        Direction<3>::lower_eta(),
        ElementId<3>(0, {{SegmentId{1, 0}, SegmentId{1, 0}, SegmentId{1, 0}}}));
    const auto interface_mortar_id_top = std::make_pair(
        Direction<3>::upper_zeta(),
        ElementId<3>(0, {{SegmentId{1, 0}, SegmentId{1, 1}, SegmentId{1, 1}}}));
    const auto& mortar_next_temporal_ids =
        get_tag(Tags::Mortars<Tags::Next<TemporalId>, 3>{});
    CHECK(mortar_next_temporal_ids.at(boundary_mortar_id_left) == 0);
    CHECK(mortar_next_temporal_ids.at(boundary_mortar_id_back) == 0);
    CHECK(mortar_next_temporal_ids.at(boundary_mortar_id_bottom) == 0);
    CHECK(mortar_next_temporal_ids.at(interface_mortar_id_right) == 0);
    CHECK(mortar_next_temporal_ids.at(interface_mortar_id_front) == 0);
    CHECK(mortar_next_temporal_ids.at(interface_mortar_id_top) == 0);
    const auto& mortar_meshes = get_tag(Tags::Mortars<Tags::Mesh<2>, 3>{});
    CHECK(mortar_meshes.at(boundary_mortar_id_left) ==
          Mesh<2>({{3, 4}}, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto));
    CHECK(mortar_meshes.at(boundary_mortar_id_back) ==
          Mesh<2>({{2, 4}}, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto));
    CHECK(mortar_meshes.at(boundary_mortar_id_bottom) ==
          Mesh<2>({{2, 3}}, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto));
    CHECK(mortar_meshes.at(interface_mortar_id_right) ==
          Mesh<2>({{3, 4}}, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto));
    CHECK(mortar_meshes.at(interface_mortar_id_front) ==
          Mesh<2>({{2, 4}}, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto));
    CHECK(mortar_meshes.at(interface_mortar_id_top) ==
          Mesh<2>({{2, 3}}, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto));
    const auto& mortar_sizes = get_tag(Tags::Mortars<Tags::MortarSize<2>, 3>{});
    const std::array<Spectral::MortarSize, 2> expected_mortar_sizes{
        {Spectral::MortarSize::Full, Spectral::MortarSize::Full}};
    CHECK(mortar_sizes.at(boundary_mortar_id_left) == expected_mortar_sizes);
    CHECK(mortar_sizes.at(boundary_mortar_id_back) == expected_mortar_sizes);
    CHECK(mortar_sizes.at(boundary_mortar_id_bottom) == expected_mortar_sizes);
    CHECK(mortar_sizes.at(interface_mortar_id_right) == expected_mortar_sizes);
    CHECK(mortar_sizes.at(interface_mortar_id_front) == expected_mortar_sizes);
    CHECK(mortar_sizes.at(interface_mortar_id_top) == expected_mortar_sizes);
    const auto& mortar_data = get_tag(Tags::VariablesBoundaryData{});
    // Just make sure this exists, it is not expected to hold any data
    mortar_data.at(boundary_mortar_id_left);
    mortar_data.at(boundary_mortar_id_back);
    mortar_data.at(boundary_mortar_id_bottom);
    mortar_data.at(interface_mortar_id_right);
    mortar_data.at(interface_mortar_id_front);
    mortar_data.at(interface_mortar_id_top);
  }
}
