// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/SegmentId.hpp"
#include "Domain/Tags.hpp"  // IWYU pragma: keep
#include "Elliptic/Initialization/Derivatives.hpp"
#include "Elliptic/Initialization/Domain.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"  // IWYU pragma: keep
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
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Initialization.Derivatives",
                  "[Unit][Elliptic][Actions]") {
  {
    INFO("1D");
    const ElementId<1> element_id{0, {{SegmentId{2, 1}}}};
    const domain::creators::Interval<Frame::Inertial> domain_creator{
        {{-0.5}}, {{1.5}}, {{false}}, {{2}}, {{4}}};
    auto domain_box = Elliptic::Initialization::Domain<1>::initialize(
        db::DataBox<tmpl::list<>>{}, ElementIndex<1>{element_id},
        domain_creator.initial_extents(), domain_creator.create_domain());

    const auto& mesh = get<Tags::Mesh<1>>(domain_box);
    Variables<tmpl::list<ScalarFieldTag>> vars{mesh.number_of_grid_points()};
    const auto& inertial_coords =
        get<Tags::Coordinates<1, Frame::Inertial>>(domain_box);
    get(get<ScalarFieldTag>(vars)) = get<0>(inertial_coords);

    auto argument_box =
        db::create_from<db::RemoveTags<>,
                        db::AddSimpleTags<typename System<1>::variables_tag>>(
            std::move(domain_box), std::move(vars));

    const auto box =
        Elliptic::Initialization::Derivatives<System<1>>::initialize(
            std::move(argument_box));

    const tnsr::i<DataVector, 1, Frame::Inertial> expected_derivs{{{{4, 1.}}}};
    CHECK_ITERABLE_APPROX(
        get<0>(
            get<Tags::deriv<ScalarFieldTag, tmpl::size_t<1>, Frame::Inertial>>(
                box)),
        get<0>(expected_derivs));
  }
  {
    INFO("2D");
    const ElementId<2> element_id{0, {{SegmentId{2, 1}, SegmentId{0, 0}}}};
    const domain::creators::Rectangle<Frame::Inertial> domain_creator{
        {{-0.5, 0.}}, {{1.5, 1.}}, {{false, false}}, {{2, 0}}, {{4, 2}}};
    auto domain_box = Elliptic::Initialization::Domain<2>::initialize(
        db::DataBox<tmpl::list<>>{}, ElementIndex<2>{element_id},
        domain_creator.initial_extents(), domain_creator.create_domain());

    const auto& mesh = get<Tags::Mesh<2>>(domain_box);
    Variables<tmpl::list<ScalarFieldTag>> vars{mesh.number_of_grid_points()};
    const auto& inertial_coords =
        get<Tags::Coordinates<2, Frame::Inertial>>(domain_box);
    get(get<ScalarFieldTag>(vars)) =
        get<0>(inertial_coords) + 2. * get<1>(inertial_coords);

    auto argument_box =
        db::create_from<db::RemoveTags<>,
                        db::AddSimpleTags<typename System<2>::variables_tag>>(
            std::move(domain_box), std::move(vars));

    const auto box =
        Elliptic::Initialization::Derivatives<System<2>>::initialize(
            std::move(argument_box));

    const tnsr::i<DataVector, 2, Frame::Inertial> expected_derivs{
        {{{8, 1.}, {8, 2.}}}};
    CHECK_ITERABLE_APPROX(
        get<0>(
            get<Tags::deriv<ScalarFieldTag, tmpl::size_t<2>, Frame::Inertial>>(
                box)),
        get<0>(expected_derivs));
    CHECK_ITERABLE_APPROX(
        get<1>(
            get<Tags::deriv<ScalarFieldTag, tmpl::size_t<2>, Frame::Inertial>>(
                box)),
        get<1>(expected_derivs));
  }
  {
    INFO("3D");
    const ElementId<3> element_id{
        0, {{SegmentId{2, 1}, SegmentId{0, 0}, SegmentId{1, 1}}}};
    const domain::creators::Brick<Frame::Inertial> domain_creator{
        {{-0.5, 0., -1.}},
        {{1.5, 1., 4.}},
        {{false, false, true}},
        {{2, 0, 1}},
        {{4, 2, 3}}};
    auto domain_box = Elliptic::Initialization::Domain<3>::initialize(
        db::DataBox<tmpl::list<>>{}, ElementIndex<3>{element_id},
        domain_creator.initial_extents(), domain_creator.create_domain());

    const auto& mesh = get<Tags::Mesh<3>>(domain_box);
    Variables<tmpl::list<ScalarFieldTag>> vars{mesh.number_of_grid_points()};
    const auto& inertial_coords =
        get<Tags::Coordinates<3, Frame::Inertial>>(domain_box);
    get(get<ScalarFieldTag>(vars)) = get<0>(inertial_coords) +
                                     2. * get<1>(inertial_coords) +
                                     3. * get<2>(inertial_coords);

    auto argument_box =
        db::create_from<db::RemoveTags<>,
                        db::AddSimpleTags<typename System<3>::variables_tag>>(
            std::move(domain_box), std::move(vars));

    const auto box =
        Elliptic::Initialization::Derivatives<System<3>>::initialize(
            std::move(argument_box));

    const tnsr::i<DataVector, 3, Frame::Inertial> expected_derivs{
        {{{24, 1.}, {24, 2.}, {24, 3.}}}};
    CHECK_ITERABLE_APPROX(
        get<0>(
            get<Tags::deriv<ScalarFieldTag, tmpl::size_t<3>, Frame::Inertial>>(
                box)),
        get<0>(expected_derivs));
    CHECK_ITERABLE_APPROX(
        get<1>(
            get<Tags::deriv<ScalarFieldTag, tmpl::size_t<3>, Frame::Inertial>>(
                box)),
        get<1>(expected_derivs));
    CHECK_ITERABLE_APPROX(
        get<2>(
            get<Tags::deriv<ScalarFieldTag, tmpl::size_t<3>, Frame::Inertial>>(
                box)),
        get<2>(expected_derivs));
  }
}
