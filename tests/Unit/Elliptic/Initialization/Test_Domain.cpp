// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <functional>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Initialization/Domain.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/TMPL.hpp"

SPECTRE_TEST_CASE("Unit.Elliptic.Initialization.Domain",
                  "[Unit][Elliptic][Actions]") {
  {
    INFO("1D");
    // Reference element:
    // [ |X| | ] -xi->
    const ElementId<1> element_id{0, {{SegmentId{2, 1}}}};
    const domain::creators::Interval<Frame::Inertial> domain_creator{
        {{-0.5}}, {{1.5}}, {{false}}, {{2}}, {{4}}};

    const auto box = Elliptic::Initialization::Domain<1>::initialize(
        db::DataBox<tmpl::list<>>{}, ElementIndex<1>{element_id},
        domain_creator.initial_extents(), domain_creator.create_domain());

    CHECK(get<Tags::Mesh<1>>(box) ==
          Mesh<1>{4, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto});
    CHECK(get<Tags::Element<1>>(box) ==
          Element<1>{element_id,
                     {{Direction<1>::lower_xi(),
                       {{{ElementId<1>{0, {{SegmentId{2, 0}}}}}}, {}}},
                      {Direction<1>::upper_xi(),
                       {{{ElementId<1>{0, {{SegmentId{2, 2}}}}}}, {}}}}});
    const auto& element_map = get<Tags::ElementMap<1>>(box);
    const tnsr::I<DataVector, 1, Frame::Logical> logical_coords_for_element_map{
        {{{-1., -0.5, 0., 0.1, 1.}}}};
    const auto inertial_coords_from_element_map =
        element_map(logical_coords_for_element_map);
    const tnsr::I<DataVector, 1, Frame::Logical> expected_inertial_coords{
        {{{0., 0.125, 0.25, 0.275, 0.5}}}};
    CHECK_ITERABLE_APPROX(get<0>(inertial_coords_from_element_map),
                          get<0>(expected_inertial_coords));
    const auto& logical_coords = get<Tags::Coordinates<1, Frame::Logical>>(box);
    CHECK(get<0>(logical_coords) ==
          Spectral::collocation_points<Spectral::Basis::Legendre,
                                       Spectral::Quadrature::GaussLobatto>(4));
    const auto& inertial_coords =
        get<Tags::Coordinates<1, Frame::Inertial>>(box);
    CHECK(inertial_coords == element_map(logical_coords));
    CHECK(get<Tags::InverseJacobian<Tags::ElementMap<1>,
                                    Tags::Coordinates<1, Frame::Logical>>>(
              box) == element_map.inv_jacobian(logical_coords));
  }
  {
    INFO("2D");
    // Reference element:
    // ^ eta
    // +-+-+-+-+> xi
    // | |X| | |
    // +-+-+-+-+
    const ElementId<2> element_id{0, {{SegmentId{2, 1}, SegmentId{0, 0}}}};
    const domain::creators::Rectangle<Frame::Inertial> domain_creator{
        {{-0.5, 0.}}, {{1.5, 2.}}, {{false, false}}, {{2, 0}}, {{4, 3}}};

    const auto box = Elliptic::Initialization::Domain<2>::initialize(
        db::DataBox<tmpl::list<>>{}, ElementIndex<2>{element_id},
        domain_creator.initial_extents(), domain_creator.create_domain());

    CHECK(get<Tags::Mesh<2>>(box) ==
          Mesh<2>{{{4, 3}},
                  Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto});
    CHECK(
        get<Tags::Element<2>>(box) ==
        Element<2>{
            element_id,
            {{Direction<2>::lower_xi(),
              {{{ElementId<2>{0, {{SegmentId{2, 0}, SegmentId{0, 0}}}}}}, {}}},
             {Direction<2>::upper_xi(),
              {{{ElementId<2>{0, {{SegmentId{2, 2}, SegmentId{0, 0}}}}}},
               {}}}}});
    const auto& element_map = get<Tags::ElementMap<2>>(box);
    const tnsr::I<DataVector, 2, Frame::Logical> logical_coords_for_element_map{
        {{{-1., -0.5, 0., 0.1, 1.}, {-1., -0.5, 0., 0.1, 1.}}}};
    const auto inertial_coords_from_element_map =
        element_map(logical_coords_for_element_map);
    const tnsr::I<DataVector, 2, Frame::Logical> expected_inertial_coords{
        {{{0., 0.125, 0.25, 0.275, 0.5}, {0., 0.5, 1., 1.1, 2.}}}};
    CHECK_ITERABLE_APPROX(get<0>(inertial_coords_from_element_map),
                          get<0>(expected_inertial_coords));
    CHECK_ITERABLE_APPROX(get<1>(inertial_coords_from_element_map),
                          get<1>(expected_inertial_coords));
    const auto& logical_coords = get<Tags::Coordinates<2, Frame::Logical>>(box);
    const auto& inertial_coords =
        get<Tags::Coordinates<2, Frame::Inertial>>(box);
    CHECK(inertial_coords == element_map(logical_coords));
    CHECK(get<Tags::InverseJacobian<Tags::ElementMap<2>,
                                    Tags::Coordinates<2, Frame::Logical>>>(
              box) == element_map.inv_jacobian(logical_coords));
  }
  {
    INFO("3D");
    const ElementId<3> element_id{
        0, {{SegmentId{2, 1}, SegmentId{0, 0}, SegmentId{1, 1}}}};
    const domain::creators::Brick<Frame::Inertial> domain_creator{
        {{-0.5, 0., -1.}},
        {{1.5, 2., 3.}},
        {{false, false, false}},
        {{2, 0, 1}},
        {{4, 3, 2}}};

    const auto box = Elliptic::Initialization::Domain<3>::initialize(
        db::DataBox<tmpl::list<>>{}, ElementIndex<3>{element_id},
        domain_creator.initial_extents(), domain_creator.create_domain());

    CHECK(get<Tags::Mesh<3>>(box) ==
          Mesh<3>{{{4, 3, 2}},
                  Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto});
    CHECK(
        get<Tags::Element<3>>(box) ==
        Element<3>{
            element_id,
            {{Direction<3>::lower_xi(),
              {{{ElementId<3>{
                   0, {{SegmentId{2, 0}, SegmentId{0, 0}, SegmentId{1, 1}}}}}},
               {}}},
             {Direction<3>::upper_xi(),
              {{{ElementId<3>{
                   0, {{SegmentId{2, 2}, SegmentId{0, 0}, SegmentId{1, 1}}}}}},
               {}}},
             {Direction<3>::lower_zeta(),
              {{{ElementId<3>{
                   0, {{SegmentId{2, 1}, SegmentId{0, 0}, SegmentId{1, 0}}}}}},
               {}}}}});
    const auto& element_map = get<Tags::ElementMap<3>>(box);
    const tnsr::I<DataVector, 3, Frame::Logical> logical_coords_for_element_map{
        {{{-1., -0.5, 0., 0.1, 1.},
          {-1., -0.5, 0., 0.1, 1.},
          {-1., -0.5, 0., 0.1, 1.}}}};
    const auto inertial_coords_from_element_map =
        element_map(logical_coords_for_element_map);
    const tnsr::I<DataVector, 3, Frame::Logical> expected_inertial_coords{
        {{{0., 0.125, 0.25, 0.275, 0.5},
          {0., 0.5, 1., 1.1, 2.},
          {1., 1.5, 2., 2.1, 3.}}}};
    CHECK_ITERABLE_APPROX(get<0>(inertial_coords_from_element_map),
                          get<0>(expected_inertial_coords));
    CHECK_ITERABLE_APPROX(get<1>(inertial_coords_from_element_map),
                          get<1>(expected_inertial_coords));
    CHECK_ITERABLE_APPROX(get<2>(inertial_coords_from_element_map),
                          get<2>(expected_inertial_coords));
    const auto& logical_coords = get<Tags::Coordinates<3, Frame::Logical>>(box);
    const auto& inertial_coords =
        get<Tags::Coordinates<3, Frame::Inertial>>(box);
    CHECK(inertial_coords == element_map(logical_coords));
    CHECK(get<Tags::InverseJacobian<Tags::ElementMap<3>,
                                    Tags::Coordinates<3, Frame::Logical>>>(
              box) == element_map.inv_jacobian(logical_coords));
  }
}
