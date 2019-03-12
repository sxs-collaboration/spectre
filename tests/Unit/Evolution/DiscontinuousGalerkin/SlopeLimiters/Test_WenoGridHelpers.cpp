// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <unordered_set>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Neighbors.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/Side.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/WenoGridHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"

namespace {

void test_check_element_no_href() {
  // Make a 2D element with neighbors:
  // - lower_xi: 1 neighbor, same refinement level => true
  // - upper_xi: 1 neighbor, same refinement level, with orientation => true
  // - lower_eta: 2 neighbors => false
  // - lower_eta: 1 neighbor, different refinement level => false
  const ElementId<2> self_id(0, {{{3, 7}, {4, 2}}});
  const ElementId<2> lower_xi_id(1, {{{3, 5}, {4, 11}}});
  const ElementId<2> upper_xi_id(2, {{{4, 7}, {3, 0}}});
  const ElementId<2> lower_eta_id_1(3, {{{3, 7}, {4, 2}}});
  const ElementId<2> lower_eta_id_2(4, {{{3, 7}, {4, 2}}});
  const ElementId<2> upper_eta_id(5, {{{3, 7}, {3, 2}}});

  const Neighbors<2> lower_xi_neighbors({lower_xi_id}, OrientationMap<2>{});
  const Neighbors<2> upper_xi_neighbors(
      {upper_xi_id},
      OrientationMap<2>(std::array<Direction<2>, 2>{
          {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}));
  const Neighbors<2> lower_eta_neighbors({lower_eta_id_1, lower_eta_id_2},
                                         OrientationMap<2>{});
  const Neighbors<2> upper_eta_neighbors({upper_eta_id}, OrientationMap<2>{});

  const Element<2> element(
      self_id, Element<2>::Neighbors_t{
                   {Direction<2>::lower_xi(), lower_xi_neighbors},
                   {Direction<2>::upper_xi(), upper_xi_neighbors},
                   {Direction<2>::lower_eta(), lower_eta_neighbors},
                   {Direction<2>::upper_eta(), upper_eta_neighbors}});

  CHECK(SlopeLimiters::Weno_detail::
            check_element_has_one_similar_neighbor_in_direction(
                element, Direction<2>::lower_xi()));
  CHECK(SlopeLimiters::Weno_detail::
            check_element_has_one_similar_neighbor_in_direction(
                element, Direction<2>::upper_xi()));
  CHECK_FALSE(SlopeLimiters::Weno_detail::
                  check_element_has_one_similar_neighbor_in_direction(
                      element, Direction<2>::lower_eta()));
  CHECK_FALSE(SlopeLimiters::Weno_detail::
                  check_element_has_one_similar_neighbor_in_direction(
                      element, Direction<2>::upper_eta()));
}

template <size_t VolumeDim>
Neighbors<VolumeDim> make_neighbor_with_id(const size_t id) noexcept {
  return {std::unordered_set<ElementId<VolumeDim>>{ElementId<VolumeDim>(id)},
          OrientationMap<VolumeDim>{}};
}

// Construct an element with one neighbor in each direction.
template <size_t VolumeDim>
Element<VolumeDim> make_element() noexcept;

template <>
Element<1> make_element() noexcept {
  return Element<1>{
      ElementId<1>{0},
      Element<1>::Neighbors_t{
          {Direction<1>::lower_xi(), make_neighbor_with_id<1>(1)},
          {Direction<1>::upper_xi(), make_neighbor_with_id<1>(2)}}};
}

template <>
Element<2> make_element() noexcept {
  return Element<2>{
      ElementId<2>{0},
      Element<2>::Neighbors_t{
          {Direction<2>::lower_xi(), make_neighbor_with_id<2>(1)},
          {Direction<2>::upper_xi(), make_neighbor_with_id<2>(2)},
          {Direction<2>::lower_eta(), make_neighbor_with_id<2>(3)},
          {Direction<2>::upper_eta(), make_neighbor_with_id<2>(4)}}};
}

template <>
Element<3> make_element() noexcept {
  return Element<3>{
      ElementId<3>{0},
      Element<3>::Neighbors_t{
          {Direction<3>::lower_xi(), make_neighbor_with_id<3>(1)},
          {Direction<3>::upper_xi(), make_neighbor_with_id<3>(2)},
          {Direction<3>::lower_eta(), make_neighbor_with_id<3>(3)},
          {Direction<3>::upper_eta(), make_neighbor_with_id<3>(4)},
          {Direction<3>::lower_zeta(), make_neighbor_with_id<3>(5)},
          {Direction<3>::upper_zeta(), make_neighbor_with_id<3>(6)}}};
}

template <size_t VolumeDim>
void check_grid_point_transform_no_href(
    const Mesh<VolumeDim>& local_mesh, const Mesh<VolumeDim>& neighbor_mesh,
    const Element<VolumeDim>& element) noexcept {
  const auto check = [&local_mesh, &neighbor_mesh ](
      const std::array<DataVector, VolumeDim>& transformed_coords,
      const bool local_mesh_provides_grid_points, const size_t dim,
      const double offset) noexcept {
    const Mesh<VolumeDim>& source_mesh =
        (local_mesh_provides_grid_points ? local_mesh : neighbor_mesh);
    for (size_t i = 0; i < VolumeDim; ++i) {
      const DataVector source_coords =
          get<0>(logical_coordinates(source_mesh.slice_through(i)));
      if (i == dim) {
        // Coordinates normal to the interface
        const DataVector expected_coords = source_coords + offset;
        CHECK(gsl::at(transformed_coords, i) == expected_coords);
      } else {
        // Coordinates parallel to the interface
        if (neighbor_mesh.slice_through(i) == local_mesh.slice_through(i)) {
          CHECK(gsl::at(transformed_coords, i).size() == 0);
        } else {
          const DataVector& expected_coords = source_coords;
          CHECK(gsl::at(transformed_coords, i) == expected_coords);
        }
      }
    }
  };

  for (size_t dim = 0; dim < VolumeDim; ++dim) {
    const auto from_lower = SlopeLimiters::Weno_detail::
        neighbor_grid_points_in_local_logical_coords(
            local_mesh, neighbor_mesh, element,
            Direction<VolumeDim>(dim, Side::Lower));
    check(from_lower, false, dim, -2.);

    const auto from_upper = SlopeLimiters::Weno_detail::
        neighbor_grid_points_in_local_logical_coords(
            local_mesh, neighbor_mesh, element,
            Direction<VolumeDim>(dim, Side::Upper));
    check(from_upper, false, dim, 2.);

    const auto to_lower = SlopeLimiters::Weno_detail::
        local_grid_points_in_neighbor_logical_coords(
            local_mesh, neighbor_mesh, element,
            Direction<VolumeDim>(dim, Side::Lower));
    check(to_lower, true, dim, 2.);

    const auto to_upper = SlopeLimiters::Weno_detail::
        local_grid_points_in_neighbor_logical_coords(
            local_mesh, neighbor_mesh, element,
            Direction<VolumeDim>(dim, Side::Upper));
    check(to_upper, true, dim, -2.);
  }
}

void test_grid_helpers_1d() {
  INFO("Testing WENO grid helpers in 1D");
  const auto element = make_element<1>();
  const Mesh<1> mesh({{6}}, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  check_grid_point_transform_no_href(mesh, mesh, element);

  const Mesh<1> other_mesh({{6}}, Spectral::Basis::Legendre,
                           Spectral::Quadrature::Gauss);
  check_grid_point_transform_no_href(mesh, other_mesh, element);
}

void test_grid_helpers_2d() {
  INFO("Testing WENO grid helpers in 2D");
  const auto element = make_element<2>();
  const Mesh<2> mesh({{5, 6}}, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  check_grid_point_transform_no_href(mesh, mesh, element);

  const Mesh<2> other_mesh({{4, 6}}, Spectral::Basis::Legendre,
                           Spectral::Quadrature::GaussLobatto);
  check_grid_point_transform_no_href(mesh, other_mesh, element);
}

void test_grid_helpers_3d() {
  INFO("Testing WENO grid helpers in 3D");
  const auto element = make_element<3>();
  const Mesh<3> mesh({{4, 5, 6}}, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  check_grid_point_transform_no_href(mesh, mesh, element);

  const Mesh<3> other_mesh({{4, 5, 3}}, Spectral::Basis::Legendre,
                           Spectral::Quadrature::GaussLobatto);
  check_grid_point_transform_no_href(mesh, other_mesh, element);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.SlopeLimiters.WenoGridHelpers",
                  "[SlopeLimiters][Unit]") {
  test_check_element_no_href();
  test_grid_helpers_1d();
  test_grid_helpers_2d();
  test_grid_helpers_3d();
}
