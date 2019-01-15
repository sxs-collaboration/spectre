// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"                  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/SegmentId.hpp"
#include "Domain/Tags.hpp"  // IWYU pragma: keep
#include "Elliptic/Initialization/DiscontinuousGalerkin.hpp"
#include "Elliptic/Initialization/Domain.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Spectral/Projection.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/TMPL.hpp"
// IWYU pragma: no_forward_declare Tags::MortarSize
// IWYU pragma: no_forward_declare Tags::Mortars

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
};

template <size_t Dim>
struct Metavariables {
  using system = System<Dim>;
  using temporal_id = TemporalId;
  struct normal_dot_numerical_flux {
    struct type {
      using package_tags = tmpl::list<ScalarFieldTag>;
    };
  };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Initialization.DG",
                  "[Unit][Elliptic][Actions]") {
  {
    INFO("1D");
    // Reference element:
    // [X| | | ] -xi->
    const ElementId<1> element_id{0, {{SegmentId{2, 0}}}};
    const domain::creators::Interval<Frame::Inertial> domain_creator{
        {{-0.5}}, {{1.5}}, {{false}}, {{2}}, {{4}}};
    auto domain_box = Elliptic::Initialization::Domain<1>::initialize(
        db::DataBox<tmpl::list<>>{}, ElementIndex<1>{element_id},
        domain_creator.initial_extents(), domain_creator.create_domain());

    auto arguments_box = db::create_from<
        db::RemoveTags<>, db::AddSimpleTags<TemporalId>,
        db::AddComputeTags<
            Tags::InternalDirections<1>, Tags::BoundaryDirectionsInterior<1>,
            Tags::InterfaceComputeItem<Tags::InternalDirections<1>,
                                       Tags::Direction<1>>,
            Tags::InterfaceComputeItem<Tags::BoundaryDirectionsInterior<1>,
                                       Tags::Direction<1>>,
            Tags::InterfaceComputeItem<Tags::InternalDirections<1>,
                                       Tags::InterfaceMesh<1>>,
            Tags::InterfaceComputeItem<Tags::BoundaryDirectionsInterior<1>,
                                       Tags::InterfaceMesh<1>>>>(
        std::move(domain_box), 0);

    const auto box = Elliptic::Initialization::DiscontinuousGalerkin<
        Metavariables<1>>::initialize(std::move(arguments_box),
                                      domain_creator.initial_extents());

    // We are working with 2 mortars here: a domain boundary at lower xi and an
    // interface at upper xi.
    const auto boundary_mortar_id = std::make_pair(
        Direction<1>::lower_xi(), ElementId<1>::external_boundary_id());
    const auto interface_mortar_id = std::make_pair(
        Direction<1>::upper_xi(), ElementId<1>(0, {{SegmentId{2, 1}}}));
    const auto& mortar_next_temporal_ids =
        get<Tags::Mortars<Tags::Next<TemporalId>, 1>>(box);
    CHECK(mortar_next_temporal_ids.at(boundary_mortar_id) == 0);
    CHECK(mortar_next_temporal_ids.at(interface_mortar_id) == 0);
    const auto& mortar_meshes = get<Tags::Mortars<Tags::Mesh<0>, 1>>(box);
    CHECK(mortar_meshes.at(boundary_mortar_id) == Mesh<0>());
    CHECK(mortar_meshes.at(interface_mortar_id) == Mesh<0>());
    const auto& mortar_sizes = get<Tags::Mortars<Tags::MortarSize<0>, 1>>(box);
    CHECK(mortar_sizes.at(boundary_mortar_id).empty());
    CHECK(mortar_sizes.at(interface_mortar_id).empty());
    const auto& mortar_data = get<Tags::VariablesBoundaryData>(box);
    // Just make sure this exists, it is not expected to hold any data
    mortar_data.at(boundary_mortar_id);
    mortar_data.at(interface_mortar_id);

    // Test that the normal fluxes on the faces have been initialized
    const auto& boundary_normal_dot_fluxes = get<
        Tags::Interface<Tags::BoundaryDirectionsInterior<1>,
                        db::add_tag_prefix<Tags::NormalDotFlux,
                                           typename System<1>::variables_tag>>>(
        box);
    CHECK(boundary_normal_dot_fluxes.at(Direction<1>::lower_xi())
              .number_of_grid_points() == 1);
    const auto& interface_normal_dot_fluxes = get<
        Tags::Interface<Tags::InternalDirections<1>,
                        db::add_tag_prefix<Tags::NormalDotFlux,
                                           typename System<1>::variables_tag>>>(
        box);
    CHECK(interface_normal_dot_fluxes.at(Direction<1>::upper_xi())
              .number_of_grid_points() == 1);
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
    const ElementId<2> element_id{0, {{SegmentId{1, 0}, SegmentId{1, 1}}}};
    const domain::creators::Rectangle<Frame::Inertial> domain_creator{
        {{-0.5, 0.}}, {{1.5, 1.}}, {{false, false}}, {{1, 1}}, {{3, 2}}};
    auto domain_box = Elliptic::Initialization::Domain<2>::initialize(
        db::DataBox<tmpl::list<>>{}, ElementIndex<2>{element_id},
        domain_creator.initial_extents(), domain_creator.create_domain());

    auto arguments_box = db::create_from<
        db::RemoveTags<>, db::AddSimpleTags<TemporalId>,
        db::AddComputeTags<
            Tags::InternalDirections<2>, Tags::BoundaryDirectionsInterior<2>,
            Tags::InterfaceComputeItem<Tags::InternalDirections<2>,
                                       Tags::Direction<2>>,
            Tags::InterfaceComputeItem<Tags::BoundaryDirectionsInterior<2>,
                                       Tags::Direction<2>>,
            Tags::InterfaceComputeItem<Tags::InternalDirections<2>,
                                       Tags::InterfaceMesh<2>>,
            Tags::InterfaceComputeItem<Tags::BoundaryDirectionsInterior<2>,
                                       Tags::InterfaceMesh<2>>>>(
        std::move(domain_box), 0);

    const auto box = Elliptic::Initialization::DiscontinuousGalerkin<
        Metavariables<2>>::initialize(std::move(arguments_box),
                                      domain_creator.initial_extents());

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
        get<Tags::Mortars<Tags::Next<TemporalId>, 2>>(box);
    CHECK(mortar_next_temporal_ids.at(boundary_mortar_id_west) == 0);
    CHECK(mortar_next_temporal_ids.at(boundary_mortar_id_north) == 0);
    CHECK(mortar_next_temporal_ids.at(interface_mortar_id_east) == 0);
    CHECK(mortar_next_temporal_ids.at(interface_mortar_id_south) == 0);
    const auto& mortar_meshes = get<Tags::Mortars<Tags::Mesh<1>, 2>>(box);
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
    const auto& mortar_sizes = get<Tags::Mortars<Tags::MortarSize<1>, 2>>(box);
    const std::array<Spectral::MortarSize, 1> expected_mortar_sizes{
        {Spectral::MortarSize::Full}};
    CHECK(mortar_sizes.at(boundary_mortar_id_west) == expected_mortar_sizes);
    CHECK(mortar_sizes.at(boundary_mortar_id_north) == expected_mortar_sizes);
    CHECK(mortar_sizes.at(interface_mortar_id_east) == expected_mortar_sizes);
    CHECK(mortar_sizes.at(interface_mortar_id_south) == expected_mortar_sizes);
    const auto& mortar_data = get<Tags::VariablesBoundaryData>(box);
    // Just make sure this exists, it is not expected to hold any data
    mortar_data.at(boundary_mortar_id_west);
    mortar_data.at(boundary_mortar_id_north);
    mortar_data.at(interface_mortar_id_east);
    mortar_data.at(interface_mortar_id_south);

    // Test that the normal fluxes on the faces have been initialized
    const auto& boundary_normal_dot_fluxes = get<
        Tags::Interface<Tags::BoundaryDirectionsInterior<2>,
                        db::add_tag_prefix<Tags::NormalDotFlux,
                                           typename System<2>::variables_tag>>>(
        box);
    CHECK(boundary_normal_dot_fluxes.at(Direction<2>::lower_xi())
              .number_of_grid_points() == 2);
    CHECK(boundary_normal_dot_fluxes.at(Direction<2>::upper_eta())
              .number_of_grid_points() == 3);
    const auto& interface_normal_dot_fluxes = get<
        Tags::Interface<Tags::InternalDirections<2>,
                        db::add_tag_prefix<Tags::NormalDotFlux,
                                           typename System<2>::variables_tag>>>(
        box);
    CHECK(interface_normal_dot_fluxes.at(Direction<2>::upper_xi())
              .number_of_grid_points() == 2);
    CHECK(interface_normal_dot_fluxes.at(Direction<2>::lower_eta())
              .number_of_grid_points() == 3);
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
    auto domain_box = Elliptic::Initialization::Domain<3>::initialize(
        db::DataBox<tmpl::list<>>{}, ElementIndex<3>{element_id},
        domain_creator.initial_extents(), domain_creator.create_domain());

    auto arguments_box = db::create_from<
        db::RemoveTags<>, db::AddSimpleTags<TemporalId>,
        db::AddComputeTags<
            Tags::InternalDirections<3>, Tags::BoundaryDirectionsInterior<3>,
            Tags::InterfaceComputeItem<Tags::InternalDirections<3>,
                                       Tags::Direction<3>>,
            Tags::InterfaceComputeItem<Tags::BoundaryDirectionsInterior<3>,
                                       Tags::Direction<3>>,
            Tags::InterfaceComputeItem<Tags::InternalDirections<3>,
                                       Tags::InterfaceMesh<3>>,
            Tags::InterfaceComputeItem<Tags::BoundaryDirectionsInterior<3>,
                                       Tags::InterfaceMesh<3>>>>(
        std::move(domain_box), 0);

    const auto box = Elliptic::Initialization::DiscontinuousGalerkin<
        Metavariables<3>>::initialize(std::move(arguments_box),
                                      domain_creator.initial_extents());

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
        get<Tags::Mortars<Tags::Next<TemporalId>, 3>>(box);
    CHECK(mortar_next_temporal_ids.at(boundary_mortar_id_left) == 0);
    CHECK(mortar_next_temporal_ids.at(boundary_mortar_id_back) == 0);
    CHECK(mortar_next_temporal_ids.at(boundary_mortar_id_bottom) == 0);
    CHECK(mortar_next_temporal_ids.at(interface_mortar_id_right) == 0);
    CHECK(mortar_next_temporal_ids.at(interface_mortar_id_front) == 0);
    CHECK(mortar_next_temporal_ids.at(interface_mortar_id_top) == 0);
    const auto& mortar_meshes = get<Tags::Mortars<Tags::Mesh<2>, 3>>(box);
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
    const auto& mortar_sizes = get<Tags::Mortars<Tags::MortarSize<2>, 3>>(box);
    const std::array<Spectral::MortarSize, 2> expected_mortar_sizes{
        {Spectral::MortarSize::Full, Spectral::MortarSize::Full}};
    CHECK(mortar_sizes.at(boundary_mortar_id_left) == expected_mortar_sizes);
    CHECK(mortar_sizes.at(boundary_mortar_id_back) == expected_mortar_sizes);
    CHECK(mortar_sizes.at(boundary_mortar_id_bottom) == expected_mortar_sizes);
    CHECK(mortar_sizes.at(interface_mortar_id_right) == expected_mortar_sizes);
    CHECK(mortar_sizes.at(interface_mortar_id_front) == expected_mortar_sizes);
    CHECK(mortar_sizes.at(interface_mortar_id_top) == expected_mortar_sizes);
    const auto& mortar_data = get<Tags::VariablesBoundaryData>(box);
    // Just make sure this exists, it is not expected to hold any data
    mortar_data.at(boundary_mortar_id_left);
    mortar_data.at(boundary_mortar_id_back);
    mortar_data.at(boundary_mortar_id_bottom);
    mortar_data.at(interface_mortar_id_right);
    mortar_data.at(interface_mortar_id_front);
    mortar_data.at(interface_mortar_id_top);

    // Test that the normal fluxes on the faces have been initialized
    const auto& boundary_normal_dot_fluxes = get<
        Tags::Interface<Tags::BoundaryDirectionsInterior<3>,
                        db::add_tag_prefix<Tags::NormalDotFlux,
                                           typename System<3>::variables_tag>>>(
        box);
    CHECK(boundary_normal_dot_fluxes.at(Direction<3>::lower_xi())
              .number_of_grid_points() == 12);
    CHECK(boundary_normal_dot_fluxes.at(Direction<3>::upper_eta())
              .number_of_grid_points() == 8);
    CHECK(boundary_normal_dot_fluxes.at(Direction<3>::lower_zeta())
              .number_of_grid_points() == 6);
    const auto& interface_normal_dot_fluxes = get<
        Tags::Interface<Tags::InternalDirections<3>,
                        db::add_tag_prefix<Tags::NormalDotFlux,
                                           typename System<3>::variables_tag>>>(
        box);
    CHECK(interface_normal_dot_fluxes.at(Direction<3>::upper_xi())
              .number_of_grid_points() == 12);
    CHECK(interface_normal_dot_fluxes.at(Direction<3>::lower_eta())
              .number_of_grid_points() == 8);
    CHECK(interface_normal_dot_fluxes.at(Direction<3>::upper_zeta())
              .number_of_grid_points() == 6);
  }
}
