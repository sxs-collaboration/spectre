// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/SegmentId.hpp"
#include "Domain/Tags.hpp"  // IWYU pragma: keep
#include "Elliptic/Initialization/Domain.hpp"
#include "Elliptic/Initialization/Interface.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/TMPL.hpp"
// IWYU pragma: no_forward_declare Tags::deriv

namespace {
struct ScalarFieldTag : db::SimpleTag {
  static std::string name() noexcept { return "ScalarFieldTag"; };
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct System {
  static constexpr size_t volume_dim = Dim;
  using variables_tag = Tags::Variables<tmpl::list<ScalarFieldTag>>;
  using gradient_tags = tmpl::list<ScalarFieldTag>;
  template <typename Tag>
  using magnitude_tag = Tags::EuclideanMagnitude<Tag>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Initialization.Interface",
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

    using variables_tag = typename System<1>::variables_tag;
    using derivs_tag = db::add_tag_prefix<
        Tags::deriv,
        db::variables_tag_with_tags_list<variables_tag,
                                         typename System<1>::gradient_tags>,
        tmpl::size_t<1>, Frame::Inertial>;
    db::item_type<variables_tag> vars(DataVector{1., 2., 3., 4.});
    db::item_type<derivs_tag> derivs(DataVector{-1., -2., -3., -4.});
    auto arguments_box =
        db::create_from<db::RemoveTags<>,
                        db::AddSimpleTags<variables_tag, derivs_tag>>(
            std::move(domain_box), std::move(vars), std::move(derivs));

    const auto box = Elliptic::Initialization::Interface<System<1>>::initialize(
        std::move(arguments_box));

    const auto& internal_directions = get<Tags::InternalDirections<1>>(box);
    const std::unordered_set<Direction<1>> expected_internal_directions{
        Direction<1>::upper_xi()};
    CHECK(internal_directions == expected_internal_directions);
    const auto& external_directions =
        get<Tags::BoundaryDirectionsInterior<1>>(box);
    const std::unordered_set<Direction<1>> expected_external_directions{
        Direction<1>::lower_xi()};
    CHECK(external_directions == expected_external_directions);
    const auto& exterior_directions =
        get<Tags::BoundaryDirectionsExterior<1>>(box);
    const std::unordered_set<Direction<1>> expected_exterior_directions{
        Direction<1>::lower_xi()};
    CHECK(exterior_directions == expected_exterior_directions);

    const auto& interface_vars =
        get<Tags::Interface<Tags::InternalDirections<1>, variables_tag>>(box);
    const DataVector expected_field_upper_xi{4.};
    CHECK(get(get<ScalarFieldTag>(interface_vars.at(
              Direction<1>::upper_xi()))) == expected_field_upper_xi);
    const auto& external_vars = get<
        Tags::Interface<Tags::BoundaryDirectionsInterior<1>, variables_tag>>(
        box);
    const DataVector expected_field_lower_xi{1.};
    CHECK(get(get<ScalarFieldTag>(external_vars.at(
              Direction<1>::lower_xi()))) == expected_field_lower_xi);
    const auto& exterior_vars = get<
        Tags::Interface<Tags::BoundaryDirectionsExterior<1>, variables_tag>>(
        box);
    // Exterior vars are not computed, but only initialized
    CHECK(exterior_vars.at(Direction<1>::lower_xi()).number_of_grid_points() ==
          1);

    const auto& interface_derivs =
        get<Tags::Interface<Tags::InternalDirections<1>, derivs_tag>>(box);
    const DataVector expected_deriv_upper_xi{-4.};
    CHECK(
        get<0>(
            get<Tags::deriv<ScalarFieldTag, tmpl::size_t<1>, Frame::Inertial>>(
                interface_derivs.at(Direction<1>::upper_xi()))) ==
        expected_deriv_upper_xi);
    const auto& external_derivs =
        get<Tags::Interface<Tags::BoundaryDirectionsInterior<1>, derivs_tag>>(
            box);
    const DataVector expected_deriv_lower_xi{-1.};
    CHECK(
        get<0>(
            get<Tags::deriv<ScalarFieldTag, tmpl::size_t<1>, Frame::Inertial>>(
                external_derivs.at(Direction<1>::lower_xi()))) ==
        expected_deriv_lower_xi);
    const auto& exterior_derivs =
        get<Tags::Interface<Tags::BoundaryDirectionsExterior<1>, derivs_tag>>(
            box);
    // Exterior derivs are the same as the interior
    CHECK(
        get<0>(
            get<Tags::deriv<ScalarFieldTag, tmpl::size_t<1>, Frame::Inertial>>(
                exterior_derivs.at(Direction<1>::lower_xi()))) ==
        expected_deriv_lower_xi);
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
        {{-0.5, 0.}}, {{1.5, 2.}}, {{false, false}}, {{1, 1}}, {{3, 2}}};
    auto domain_box = Elliptic::Initialization::Domain<2>::initialize(
        db::DataBox<tmpl::list<>>{}, ElementIndex<2>{element_id},
        domain_creator.initial_extents(), domain_creator.create_domain());

    using variables_tag = typename System<2>::variables_tag;
    using derivs_tag = db::add_tag_prefix<
        Tags::deriv,
        db::variables_tag_with_tags_list<variables_tag,
                                         typename System<2>::gradient_tags>,
        tmpl::size_t<2>, Frame::Inertial>;
    db::item_type<variables_tag> vars(DataVector{1., 2., 3., 4., 5., 6.});
    db::item_type<derivs_tag> derivs(DataVector{-1., -2., -3., -4., -5., -6.});
    auto arguments_box =
        db::create_from<db::RemoveTags<>,
                        db::AddSimpleTags<variables_tag, derivs_tag>>(
            std::move(domain_box), std::move(vars), std::move(derivs));

    const auto box = Elliptic::Initialization::Interface<System<2>>::initialize(
        std::move(arguments_box));

    const auto& internal_directions = get<Tags::InternalDirections<2>>(box);
    const std::unordered_set<Direction<2>> expected_internal_directions{
        Direction<2>::upper_xi(), Direction<2>::lower_eta()};
    CHECK(internal_directions == expected_internal_directions);
    const auto& external_directions =
        get<Tags::BoundaryDirectionsInterior<2>>(box);
    const std::unordered_set<Direction<2>> expected_external_directions{
        Direction<2>::lower_xi(), Direction<2>::upper_eta()};
    CHECK(external_directions == expected_external_directions);
    const auto& exterior_directions =
        get<Tags::BoundaryDirectionsExterior<2>>(box);
    const std::unordered_set<Direction<2>> expected_exterior_directions{
        Direction<2>::lower_xi(), Direction<2>::upper_eta()};
    CHECK(exterior_directions == expected_exterior_directions);

    const auto& interface_vars =
        get<Tags::Interface<Tags::InternalDirections<2>, variables_tag>>(box);
    const DataVector expected_field_upper_xi{3., 6.};
    CHECK(get(get<ScalarFieldTag>(interface_vars.at(
              Direction<2>::upper_xi()))) == expected_field_upper_xi);
    const DataVector expected_field_lower_eta{1., 2., 3.};
    CHECK(get(get<ScalarFieldTag>(interface_vars.at(
              Direction<2>::lower_eta()))) == expected_field_lower_eta);
    const auto& external_vars = get<
        Tags::Interface<Tags::BoundaryDirectionsInterior<2>, variables_tag>>(
        box);
    const DataVector expected_field_lower_xi{1., 4.};
    CHECK(get(get<ScalarFieldTag>(external_vars.at(
              Direction<2>::lower_xi()))) == expected_field_lower_xi);
    const DataVector expected_field_upper_eta{4., 5., 6.};
    CHECK(get(get<ScalarFieldTag>(external_vars.at(
              Direction<2>::upper_eta()))) == expected_field_upper_eta);
    const auto& exterior_vars = get<
        Tags::Interface<Tags::BoundaryDirectionsExterior<2>, variables_tag>>(
        box);
    // Exterior vars are not computed, but only initialized
    CHECK(exterior_vars.at(Direction<2>::lower_xi()).number_of_grid_points() ==
          2);
    CHECK(exterior_vars.at(Direction<2>::upper_eta()).number_of_grid_points() ==
          3);
  }
  {
    INFO("3D");
    const ElementId<3> element_id{
        0, {{SegmentId{1, 0}, SegmentId{1, 1}, SegmentId{2, 0}}}};
    const domain::creators::Brick<Frame::Inertial> domain_creator{
        {{-0.5, 0., -1.}},
        {{1.5, 2., 4.}},
        {{false, false, false}},
        {{1, 1, 2}},
        {{3, 2, 2}}};
    auto domain_box = Elliptic::Initialization::Domain<3>::initialize(
        db::DataBox<tmpl::list<>>{}, ElementIndex<3>{element_id},
        domain_creator.initial_extents(), domain_creator.create_domain());

    using variables_tag = typename System<3>::variables_tag;
    using derivs_tag = db::add_tag_prefix<
        Tags::deriv,
        db::variables_tag_with_tags_list<variables_tag,
                                         typename System<3>::gradient_tags>,
        tmpl::size_t<3>, Frame::Inertial>;
    db::item_type<variables_tag> vars(
        DataVector{1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.});
    db::item_type<derivs_tag> derivs(DataVector{
        -1., -2., -3., -4., -5., -6., -7., -8., -9., -10., -11., -12.});
    auto arguments_box =
        db::create_from<db::RemoveTags<>,
                        db::AddSimpleTags<variables_tag, derivs_tag>>(
            std::move(domain_box), std::move(vars), std::move(derivs));

    const auto box = Elliptic::Initialization::Interface<System<3>>::initialize(
        std::move(arguments_box));

    const auto& internal_directions = get<Tags::InternalDirections<3>>(box);
    const std::unordered_set<Direction<3>> expected_internal_directions{
        Direction<3>::upper_xi(), Direction<3>::lower_eta(),
        Direction<3>::upper_zeta()};
    CHECK(internal_directions == expected_internal_directions);
    const auto& external_directions =
        get<Tags::BoundaryDirectionsInterior<3>>(box);
    const std::unordered_set<Direction<3>> expected_external_directions{
        Direction<3>::lower_xi(), Direction<3>::upper_eta(),
        Direction<3>::lower_zeta()};
    CHECK(external_directions == expected_external_directions);
    const auto& exterior_directions =
        get<Tags::BoundaryDirectionsExterior<3>>(box);
    const std::unordered_set<Direction<3>> expected_exterior_directions{
        Direction<3>::lower_xi(), Direction<3>::upper_eta(),
        Direction<3>::lower_zeta()};
    CHECK(exterior_directions == expected_exterior_directions);

    const auto& interface_vars =
        get<Tags::Interface<Tags::InternalDirections<3>, variables_tag>>(box);
    const DataVector expected_field_upper_xi{3., 6., 9., 12.};
    CHECK(get(get<ScalarFieldTag>(interface_vars.at(
              Direction<3>::upper_xi()))) == expected_field_upper_xi);
    const DataVector expected_field_lower_eta{1., 2., 3., 7., 8., 9.};
    CHECK(get(get<ScalarFieldTag>(interface_vars.at(
              Direction<3>::lower_eta()))) == expected_field_lower_eta);
    const DataVector expected_field_upper_zeta{7., 8., 9., 10., 11., 12.};
    CHECK(get(get<ScalarFieldTag>(interface_vars.at(
              Direction<3>::upper_zeta()))) == expected_field_upper_zeta);
    const auto& external_vars = get<
        Tags::Interface<Tags::BoundaryDirectionsInterior<3>, variables_tag>>(
        box);
    const DataVector expected_field_lower_xi{1., 4., 7., 10.};
    CHECK(get(get<ScalarFieldTag>(external_vars.at(
              Direction<3>::lower_xi()))) == expected_field_lower_xi);
    const DataVector expected_field_upper_eta{4., 5., 6., 10., 11., 12.};
    CHECK(get(get<ScalarFieldTag>(external_vars.at(
              Direction<3>::upper_eta()))) == expected_field_upper_eta);
    const DataVector expected_field_lower_zeta{1., 2., 3., 4., 5., 6.};
    CHECK(get(get<ScalarFieldTag>(external_vars.at(
              Direction<3>::lower_zeta()))) == expected_field_lower_zeta);
    const auto& exterior_vars = get<
        Tags::Interface<Tags::BoundaryDirectionsExterior<3>, variables_tag>>(
        box);
    // Exterior vars are not computed, but only initialized
    CHECK(exterior_vars.at(Direction<3>::lower_xi()).number_of_grid_points() ==
          4);
    CHECK(exterior_vars.at(Direction<3>::upper_eta()).number_of_grid_points() ==
          6);
    CHECK(
        exterior_vars.at(Direction<3>::lower_zeta()).number_of_grid_points() ==
        6);
  }
}
