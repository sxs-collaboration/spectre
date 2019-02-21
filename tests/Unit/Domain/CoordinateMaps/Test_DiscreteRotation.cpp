// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <memory>
#include <pup.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/Rotation.hpp"
#include "Domain/Direction.hpp"
#include "Domain/OrientationMap.hpp"
#include "tests/Unit/Domain/CoordinateMaps/TestMapHelpers.hpp"

namespace domain {
namespace {
void test_1d() {
  INFO("1d");
  const CoordinateMaps::DiscreteRotation<1> identity_map1d{};
  const CoordinateMaps::DiscreteRotation<1> rotation_nx{OrientationMap<1>{
      std::array<Direction<1>, 1>{{Direction<1>::lower_xi()}}}};

  check_if_maps_are_equal(
      make_coordinate_map<Frame::Logical, Frame::Grid>(
          CoordinateMaps::Identity<1>{}),
      make_coordinate_map<Frame::Logical, Frame::Grid>(identity_map1d));
  check_if_maps_are_equal(
      make_coordinate_map<Frame::Logical, Frame::Grid>(
          CoordinateMaps::Affine{-1.0, 1.0, 1.0, -1.0}),
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotation_nx));

  const std::array<double, 1> point_px{{0.5}};
  const std::array<double, 1> point_nx{{-0.5}};
  CHECK(rotation_nx(point_px) == point_nx);
}

void test_2d() {
  INFO("2d");
  const CoordinateMaps::DiscreteRotation<2> identity_map2d{};
  const CoordinateMaps::DiscreteRotation<2> rotation_ny_px{
      OrientationMap<2>{std::array<Direction<2>, 2>{
          {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}}};

  check_if_maps_are_equal(
      make_coordinate_map<Frame::Logical, Frame::Grid>(
          CoordinateMaps::Identity<2>{}),
      make_coordinate_map<Frame::Logical, Frame::Grid>(identity_map2d));
  check_if_maps_are_equal(
      make_coordinate_map<Frame::Logical, Frame::Grid>(
          CoordinateMaps::Rotation<2>{M_PI_2}),
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotation_ny_px));

  const std::array<double, 2> point_px_py{{0.2, 0.5}};
  const std::array<double, 2> point_ny_px{{-0.5, 0.2}};
  CHECK(rotation_ny_px(point_px_py) == point_ny_px);
}

void test_3d() {
  INFO("3d");
  const CoordinateMaps::DiscreteRotation<3> identity_map3d{};
  const CoordinateMaps::DiscreteRotation<3> rotation_ny_nz_px{
      OrientationMap<3>{std::array<Direction<3>, 3>{
          {Direction<3>::lower_eta(), Direction<3>::lower_zeta(),
           Direction<3>::upper_xi()}}}};

  using Identity1D = CoordinateMaps::Identity<1>;
  using Identity2D = CoordinateMaps::Identity<2>;
  using Identity3D = CoordinateMaps::ProductOf2Maps<Identity1D, Identity2D>;
  const Identity1D id1d{};
  const Identity2D id2d{};
  const Identity3D id3d{id1d, id2d};

  check_if_maps_are_equal(
      make_coordinate_map<Frame::Logical, Frame::Grid>(id3d),
      make_coordinate_map<Frame::Logical, Frame::Grid>(identity_map3d));
  check_if_maps_are_equal(
      make_coordinate_map<Frame::Logical, Frame::Grid>(
          CoordinateMaps::Rotation<3>{M_PI_2, -M_PI_2, 0.0}),
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotation_ny_nz_px));

  const std::array<double, 3> point_px_py_pz{{0.2, 0.5, 0.3}};
  const std::array<double, 3> point_ny_nz_px{{-0.5, -0.3, 0.2}};
  CHECK(rotation_ny_nz_px(point_px_py_pz) == point_ny_nz_px);
}

void test_with_orientation() {
  INFO("With orientation");
  for (OrientationMapIterator<2> map_i{}; map_i; ++map_i) {
    const CoordinateMaps::DiscreteRotation<2> coord_map{map_i()};
    test_suite_for_map_on_unit_cube(coord_map);
  }
  for (OrientationMapIterator<3> map_i{}; map_i; ++map_i) {
    const CoordinateMaps::DiscreteRotation<3> coord_map{map_i()};
    test_suite_for_map_on_unit_cube(coord_map);
  }
}

void test_is_identity() {
  INFO("Is identity");
  check_if_map_is_identity(
      CoordinateMaps::DiscreteRotation<1>{OrientationMap<1>{}});
  CHECK(not CoordinateMaps::DiscreteRotation<1>{
      OrientationMap<1>{
          std::array<Direction<1>, 1>{{Direction<1>::lower_xi()}}}}
                .is_identity());

  check_if_map_is_identity(
      CoordinateMaps::DiscreteRotation<2>{OrientationMap<2>{}});
  CHECK(not CoordinateMaps::DiscreteRotation<2>{
      OrientationMap<2>{std::array<Direction<2>, 2>{
          {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}}}
                .is_identity());

  check_if_map_is_identity(
      CoordinateMaps::DiscreteRotation<3>{OrientationMap<3>{}});
  CHECK(not CoordinateMaps::DiscreteRotation<3>{
      OrientationMap<3>{std::array<Direction<3>, 3>{
          {Direction<3>::lower_eta(), Direction<3>::lower_zeta(),
           Direction<3>::upper_xi()}}}}
                .is_identity());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.DiscreteRotation",
                  "[Domain][Unit]") {
  test_1d();
  test_2d();
  test_3d();
  test_with_orientation();
  test_is_identity();
}
}  // namespace domain
