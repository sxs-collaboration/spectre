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

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.DiscreteRotation.Specific1D",
                  "[Domain][Unit]") {
  const domain::CoordinateMaps::DiscreteRotation<1> identity_map1d{};
  const domain::CoordinateMaps::DiscreteRotation<1> rotation_nx{
      domain::OrientationMap<1>{std::array<domain::Direction<1>, 1>{
          {domain::Direction<1>::lower_xi()}}}};

  check_if_maps_are_equal(
      domain::make_coordinate_map<Frame::Logical, Frame::Grid>(
          domain::CoordinateMaps::Identity<1>{}),
      domain::make_coordinate_map<Frame::Logical, Frame::Grid>(identity_map1d));
  check_if_maps_are_equal(
      domain::make_coordinate_map<Frame::Logical, Frame::Grid>(
          domain::CoordinateMaps::Affine{-1.0, 1.0, 1.0, -1.0}),
      domain::make_coordinate_map<Frame::Logical, Frame::Grid>(rotation_nx));

  const std::array<double, 1> point_px{{0.5}};
  const std::array<double, 1> point_nx{{-0.5}};
  CHECK(rotation_nx(point_px) == point_nx);
}

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.DiscreteRotation.Specific2D",
                  "[Domain][Unit]") {
  const domain::CoordinateMaps::DiscreteRotation<2> identity_map2d{};
  const domain::CoordinateMaps::DiscreteRotation<2> rotation_ny_px{
      domain::OrientationMap<2>{std::array<domain::Direction<2>, 2>{
          {domain::Direction<2>::lower_eta(),
           domain::Direction<2>::upper_xi()}}}};

  check_if_maps_are_equal(
      domain::make_coordinate_map<Frame::Logical, Frame::Grid>(
          domain::CoordinateMaps::Identity<2>{}),
      domain::make_coordinate_map<Frame::Logical, Frame::Grid>(identity_map2d));
  check_if_maps_are_equal(
      domain::make_coordinate_map<Frame::Logical, Frame::Grid>(
          domain::CoordinateMaps::Rotation<2>{M_PI_2}),
      domain::make_coordinate_map<Frame::Logical, Frame::Grid>(rotation_ny_px));

  const std::array<double, 2> point_px_py{{0.2, 0.5}};
  const std::array<double, 2> point_ny_px{{-0.5, 0.2}};
  CHECK(rotation_ny_px(point_px_py) == point_ny_px);
}

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.DiscreteRotation.Specific3D",
                  "[Domain][Unit]") {
  const domain::CoordinateMaps::DiscreteRotation<3> identity_map3d{};
  const domain::CoordinateMaps::DiscreteRotation<3> rotation_ny_nz_px{
      domain::OrientationMap<3>{std::array<domain::Direction<3>, 3>{
          {domain::Direction<3>::lower_eta(),
           domain::Direction<3>::lower_zeta(),
           domain::Direction<3>::upper_xi()}}}};

  using Identity1D = domain::CoordinateMaps::Identity<1>;
  using Identity2D = domain::CoordinateMaps::Identity<2>;
  using Identity3D =
      domain::CoordinateMaps::ProductOf2Maps<Identity1D, Identity2D>;
  const Identity1D id1d{};
  const Identity2D id2d{};
  const Identity3D id3d{id1d, id2d};

  check_if_maps_are_equal(
      domain::make_coordinate_map<Frame::Logical, Frame::Grid>(id3d),
      domain::make_coordinate_map<Frame::Logical, Frame::Grid>(identity_map3d));
  check_if_maps_are_equal(
      domain::make_coordinate_map<Frame::Logical, Frame::Grid>(
          domain::CoordinateMaps::Rotation<3>{M_PI_2, -M_PI_2, 0.0}),
      domain::make_coordinate_map<Frame::Logical, Frame::Grid>(
          rotation_ny_nz_px));

  const std::array<double, 3> point_px_py_pz{{0.2, 0.5, 0.3}};
  const std::array<double, 3> point_ny_nz_px{{-0.5, -0.3, 0.2}};
  CHECK(rotation_ny_nz_px(point_px_py_pz) == point_ny_nz_px);
}

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.DiscreteRotation",
                  "[Domain][Unit]") {
  for (OrientationMapIterator<2> map_i{}; map_i; ++map_i) {
    const domain::CoordinateMaps::DiscreteRotation<2> coord_map{map_i()};
    test_suite_for_map_on_unit_cube(coord_map);
  }
  for (OrientationMapIterator<3> map_i{}; map_i; ++map_i) {
    const domain::CoordinateMaps::DiscreteRotation<3> coord_map{map_i()};
    test_suite_for_map_on_unit_cube(coord_map);
  }
}
