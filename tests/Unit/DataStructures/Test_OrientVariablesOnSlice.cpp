// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <cmath>
#include <vector>

#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/Rotation.hpp"
#include "Domain/Domain.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/SegmentId.hpp"
#include "Utilities/MakeVector.hpp"
#include "tests/Unit/TestHelpers.hpp"

using Affine = CoordinateMaps::Affine;
using Affine2D = CoordinateMaps::ProductOf2Maps<Affine, Affine>;
using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
using AffineCoordMap = CoordinateMap<Frame::Logical, Frame::Grid, Affine>;
using AffineCoordMap2D = CoordinateMap<Frame::Logical, Frame::Grid, Affine2D>;
using AffineCoordMap3D = CoordinateMap<Frame::Logical, Frame::Grid, Affine3D>;
template <size_t Dim>
using Rotation = CoordinateMaps::Rotation<Dim>;

template <size_t SpatialDim>
struct PhysicalCoords {
  using type = tnsr::I<DataVector, SpatialDim, Frame::Grid>;
};

namespace {
template <size_t SpatialDim>
void check_orient_variables_on_slice(
    const Block<SpatialDim, Frame::Grid>& my_block,
    const Block<SpatialDim, Frame::Grid>& neighbor_block,
    const Direction<SpatialDim>& direction_to_neighbor,
    const Index<SpatialDim>& my_extents,
    const Index<SpatialDim>& neighbor_extents) {
  const size_t my_sliced_dim = direction_to_neighbor.logical_dimension();
  const Index<SpatialDim - 1> extents_on_my_slice(
      my_extents.slice_away(my_sliced_dim));
  Variables<tmpl::list<PhysicalCoords<SpatialDim>>> coords_on_my_slice(
      extents_on_my_slice.product());

  get<PhysicalCoords<SpatialDim>>(coords_on_my_slice) =
      my_block.coordinate_map()(interface_logical_coordinates(
          extents_on_my_slice, direction_to_neighbor));

  const OrientationMap<SpatialDim>& orientation =
      my_block.neighbors().at(direction_to_neighbor).orientation();
  const auto oriented_coords_on_neighbor_slice = orient_variables_on_slice(
      coords_on_my_slice, extents_on_my_slice, my_sliced_dim, orientation);

  const Index<SpatialDim - 1> extents_on_neighbor_slice(
      neighbor_extents.slice_away(orientation(my_sliced_dim)));
  const auto coords_on_neighbor_slice =
      neighbor_block.coordinate_map()(interface_logical_coordinates(
          extents_on_neighbor_slice,
          orientation(direction_to_neighbor).opposite()));

  const auto& computed_coords =
      get<PhysicalCoords<SpatialDim>>(oriented_coords_on_neighbor_slice);
  const auto& expected_coords = coords_on_neighbor_slice;
  for (size_t d = 0; d < SpatialDim; ++d) {
    for (size_t s = 0; s < extents_on_my_slice.product(); ++s) {
      INFO("d: " << d << "\t" << "s: " << s);
      CHECK(computed_coords.get(d)[s] == approx(expected_coords.get(d)[s]));
    }
  }
}

void test_1d_aligned() {
  auto maps = make_vector<
      std::unique_ptr<CoordinateMapBase<Frame::Logical, Frame::Grid, 1>>>(
      std::make_unique<AffineCoordMap>(
          make_coordinate_map<Frame::Logical, Frame::Grid>(
              Affine{-1.0, 1.0, -2.0, 2.0})),
      std::make_unique<AffineCoordMap>(
          make_coordinate_map<Frame::Logical, Frame::Grid>(
              Affine{-1.0, 1.0, 2.0, 5.0})),
      std::make_unique<AffineCoordMap>(
          make_coordinate_map<Frame::Logical, Frame::Grid>(
              Affine{-1.0, 1.0, 5.0, 6.0})));

  std::vector<std::array<size_t, 2>> corners{{{0, 1}}, {{1, 2}}, {{2, 3}}};

  const Domain<1, Frame::Grid> domain(std::move(maps), corners);
  const auto& blocks = domain.blocks();

  const Index<1> my_extents{4};
  const Index<1> lower_xi_neighbor_extents{5};
  const Index<1> upper_xi_neighbor_extents{6};

  check_orient_variables_on_slice(blocks[1], blocks[0],
                                  Direction<1>::lower_xi(), my_extents,
                                  lower_xi_neighbor_extents);
  check_orient_variables_on_slice(blocks[1], blocks[2],
                                  Direction<1>::upper_xi(), my_extents,
                                  upper_xi_neighbor_extents);
}

void test_1d_flipped() {
  auto maps = make_vector<
      std::unique_ptr<CoordinateMapBase<Frame::Logical, Frame::Grid, 1>>>(
      std::make_unique<AffineCoordMap>(
          make_coordinate_map<Frame::Logical, Frame::Grid>(
              Affine{-1.0, 1.0, 2.0, -2.0})),
      std::make_unique<AffineCoordMap>(
          make_coordinate_map<Frame::Logical, Frame::Grid>(
              Affine{-1.0, 1.0, 2.0, 5.0})),
      std::make_unique<AffineCoordMap>(
          make_coordinate_map<Frame::Logical, Frame::Grid>(
              Affine{-1.0, 1.0, 6.0, 5.0})));

  std::vector<std::array<size_t, 2>> corners{{{1, 0}}, {{1, 2}}, {{3, 2}}};

  const Domain<1, Frame::Grid> domain(std::move(maps), corners);
  const auto& blocks = domain.blocks();

  const Index<1> my_extents{4};
  const Index<1> lower_xi_neighbor_extents{5};
  const Index<1> upper_xi_neighbor_extents{6};

  check_orient_variables_on_slice(blocks[1], blocks[0],
                                  Direction<1>::lower_xi(), my_extents,
                                  lower_xi_neighbor_extents);
  check_orient_variables_on_slice(blocks[1], blocks[2],
                                  Direction<1>::upper_xi(), my_extents,
                                  upper_xi_neighbor_extents);
}

template <typename Map>
void test_2d_rotated(const Map& my_map, const std::array<size_t, 4>& my_corners,
                     const Index<2>& my_extents) {
  const Affine lower_x_map(-1.0, 1.0, -2.0, 2.0);
  const Affine center_x_map(-1.0, 1.0, 2.0, 5.0);
  const Affine upper_x_map(-1.0, 1.0, 5.0, 6.0);
  const Affine lower_y_map(-1.0, 1.0, -3.0, 0.0);
  const Affine center_y_map(-1.0, 1.0, 0.0, 4.0);
  const Affine upper_y_map(-1.0, 1.0, 4.0, 9.0);

  const auto lower_eta_neighbor_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(
          Affine2D{center_x_map, lower_y_map});
  const auto lower_xi_neighbor_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(
          Affine2D{lower_x_map, center_y_map});
  const auto upper_xi_neighbor_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(
          Affine2D{upper_x_map, center_y_map});
  const auto upper_eta_neighbor_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(
          Affine2D{center_x_map, upper_y_map});

  auto maps = make_vector<
      std::unique_ptr<CoordinateMapBase<Frame::Logical, Frame::Grid, 2>>>(
      std::make_unique<AffineCoordMap2D>(lower_eta_neighbor_map),
      std::make_unique<AffineCoordMap2D>(lower_xi_neighbor_map),
      std::make_unique<Map>(my_map),
      std::make_unique<AffineCoordMap2D>(upper_xi_neighbor_map),
      std::make_unique<AffineCoordMap2D>(upper_eta_neighbor_map));

  std::vector<std::array<size_t, 4>> corners{{{0, 1, 3, 4}},
                                             {{2, 3, 6, 7}},
                                             my_corners,
                                             {{4, 5, 8, 9}},
                                             {{7, 8, 10, 11}}};

  const Domain<2, Frame::Grid> domain(std::move(maps), corners);
  const auto& blocks = domain.blocks();

  const auto extents = [&my_extents]() {
    std::vector<Index<2>> ext;

    const size_t lower_x_extents = 5;
    const size_t center_x_extents = 4;
    const size_t upper_x_extents = 6;
    const size_t lower_y_extents = 3;
    const size_t center_y_extents = 7;
    const size_t upper_y_extents = 8;

    ext.emplace_back(center_x_extents, lower_y_extents);
    ext.emplace_back(lower_x_extents, center_y_extents);
    ext.emplace_back(my_extents);
    ext.emplace_back(upper_x_extents, center_y_extents);
    ext.emplace_back(center_x_extents, upper_y_extents);
    return ext;
  }();

  check_orient_variables_on_slice(
      blocks[1], blocks[2], Direction<2>::upper_xi(), extents[1], extents[2]);
  check_orient_variables_on_slice(
      blocks[0], blocks[2], Direction<2>::upper_eta(), extents[0], extents[2]);
  check_orient_variables_on_slice(
      blocks[3], blocks[2], Direction<2>::lower_xi(), extents[3], extents[2]);
  check_orient_variables_on_slice(
      blocks[4], blocks[2], Direction<2>::lower_eta(), extents[4], extents[2]);
}

void test_2d_aligned() {
  const Affine my_x_map(-1.0, 1.0, 2.0, 5.0);
  const Affine my_y_map(-1.0, 1.0, 0.0, 4.0);
  const Rotation<2> rotation(0.0);
  const Affine2D product{my_x_map, my_y_map};
  const auto my_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotation, product);
  const std::array<size_t, 4> my_corners{{3, 4, 7, 8}};
  const Index<2> my_extents(4, 7);
  test_2d_rotated(my_map, my_corners, my_extents);
}

void test_2d_flipped() {
  const Affine my_x_map(-1.0, 1.0, 2.0, 5.0);
  const Affine my_y_map(-1.0, 1.0, 0.0, 4.0);
  const Rotation<2> rotation(M_PI);
  const Affine2D product{my_x_map, my_y_map};
  const auto my_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotation, product);
  const std::array<size_t, 4> my_corners{{8, 7, 4, 3}};
  const Index<2> my_extents(4, 7);
  test_2d_rotated(my_map, my_corners, my_extents);
}

void test_2d_rotated_by_90() {
  const Affine my_x_map(-1.0, 1.0, 2.0, 5.0);
  const Affine my_y_map(-1.0, 1.0, 0.0, 4.0);
  const Rotation<2> rotation(M_PI_2);
  const Affine2D product{my_x_map, my_y_map};
  const auto my_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotation, product);
  const std::array<size_t, 4> my_corners{{4, 8, 3, 7}};
  const Index<2> my_extents(7, 4);
  test_2d_rotated(my_map, my_corners, my_extents);
}

void test_2d_rotated_by_270() {
  const Affine my_x_map(-1.0, 1.0, 2.0, 5.0);
  const Affine my_y_map(-1.0, 1.0, 0.0, 4.0);
  const Rotation<2> rotation(-M_PI_2);
  const Affine2D product{my_x_map, my_y_map};
  const auto my_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotation, product);
  const std::array<size_t, 4> my_corners{{7, 3, 8, 4}};
  const Index<2> my_extents(7, 4);
  test_2d_rotated(my_map, my_corners, my_extents);
}

template <typename Map>
void test_3d_rotated(const Map& my_map, const std::array<size_t, 8>& my_corners,
                     const Index<3>& my_extents) {
  const Affine lower_x_map(-1.0, 1.0, -2.0, 2.0);
  const Affine center_x_map(-1.0, 1.0, 2.0, 5.0);
  const Affine upper_x_map(-1.0, 1.0, 5.0, 6.0);
  const Affine lower_y_map(-1.0, 1.0, -3.0, 0.0);
  const Affine center_y_map(-1.0, 1.0, 0.0, 4.0);
  const Affine upper_y_map(-1.0, 1.0, 4.0, 9.0);
  const Affine lower_z_map(-1.0, 1.0, -5.0, 1.0);
  const Affine center_z_map(-1.0, 1.0, 1.0, 7.0);
  const Affine upper_z_map(-1.0, 1.0, 7.0, 12.0);

  const auto lower_zeta_neighbor_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(
          Affine3D{center_x_map, center_y_map, lower_z_map});
  const auto lower_eta_neighbor_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(
          Affine3D{center_x_map, lower_y_map, center_z_map});
  const auto lower_xi_neighbor_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(
          Affine3D{lower_x_map, center_y_map, center_z_map});
  const auto upper_xi_neighbor_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(
          Affine3D{upper_x_map, center_y_map, center_z_map});
  const auto upper_eta_neighbor_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(
          Affine3D{center_x_map, upper_y_map, center_z_map});
  const auto upper_zeta_neighbor_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(
          Affine3D{center_x_map, center_y_map, upper_z_map});

  auto maps = make_vector<
      std::unique_ptr<CoordinateMapBase<Frame::Logical, Frame::Grid, 3>>>(
      std::make_unique<AffineCoordMap3D>(lower_zeta_neighbor_map),
      std::make_unique<AffineCoordMap3D>(lower_eta_neighbor_map),
      std::make_unique<AffineCoordMap3D>(lower_xi_neighbor_map),
      std::make_unique<Map>(my_map),
      std::make_unique<AffineCoordMap3D>(upper_xi_neighbor_map),
      std::make_unique<AffineCoordMap3D>(upper_eta_neighbor_map),
      std::make_unique<AffineCoordMap3D>(upper_zeta_neighbor_map));

  std::vector<std::array<size_t, 8>> corners{
      {{0, 1, 2, 3, 7, 8, 11, 12}},      {{4, 5, 7, 8, 16, 17, 19, 20}},
      {{6, 7, 10, 11, 18, 19, 22, 23}},  my_corners,
      {{8, 9, 12, 13, 20, 21, 24, 25}},  {{11, 12, 14, 15, 23, 24, 26, 27}},
      {{19, 20, 23, 24, 28, 29, 30, 31}}};

  const Domain<3, Frame::Grid> domain(std::move(maps), corners);
  const auto& blocks = domain.blocks();

  const auto extents = [&my_extents]() {
    const size_t lower_x_extents = 5;
    const size_t center_x_extents = 4;
    const size_t upper_x_extents = 6;
    const size_t lower_y_extents = 3;
    const size_t center_y_extents = 7;
    const size_t upper_y_extents = 8;
    const size_t lower_z_extents = 2;
    const size_t center_z_extents = 10;
    const size_t upper_z_extents = 9;

    std::vector<Index<3>> ext;
    ext.emplace_back(center_x_extents, center_y_extents, lower_z_extents);
    ext.emplace_back(center_x_extents, lower_y_extents, center_z_extents);
    ext.emplace_back(lower_x_extents, center_y_extents, center_z_extents);
    ext.emplace_back(my_extents);
    ext.emplace_back(upper_x_extents, center_y_extents, center_z_extents);
    ext.emplace_back(center_x_extents, upper_y_extents, center_z_extents);
    ext.emplace_back(center_x_extents, center_y_extents, upper_z_extents);
    return ext;
  }();

  check_orient_variables_on_slice(
      blocks[0], blocks[3], Direction<3>::upper_zeta(), extents[0], extents[3]);
  check_orient_variables_on_slice(
      blocks[1], blocks[3], Direction<3>::upper_eta(), extents[1], extents[3]);
  check_orient_variables_on_slice(
      blocks[2], blocks[3], Direction<3>::upper_xi(), extents[2], extents[3]);
  check_orient_variables_on_slice(
      blocks[4], blocks[3], Direction<3>::lower_xi(), extents[4], extents[3]);
  check_orient_variables_on_slice(
      blocks[5], blocks[3], Direction<3>::lower_eta(), extents[5], extents[3]);
  check_orient_variables_on_slice(
      blocks[6], blocks[3], Direction<3>::lower_zeta(), extents[6], extents[3]);
}

void test_3d_aligned() {
  const Affine my_x_map(-1.0, 1.0, 2.0, 5.0);
  const Affine my_y_map(-1.0, 1.0, 0.0, 4.0);
  const Affine my_z_map(-1.0, 1.0, 1.0, 7.0);
  const Rotation<3> rotation(0.0, 0.0, 0.0);

  const Affine3D product{my_x_map, my_y_map, my_z_map};
  const auto my_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotation, product);

  const std::array<size_t, 8> my_corners{{7, 8, 11, 12, 19, 20, 23, 24}};
  const Index<3> my_extents(4, 7, 10);
  test_3d_rotated(my_map, my_corners, my_extents);
}

void test_3d_rotated_by_90_0_0() {
  const Affine my_x_map(-1.0, 1.0, 2.0, 5.0);
  const Affine my_y_map(-1.0, 1.0, 0.0, 4.0);
  const Affine my_z_map(-1.0, 1.0, 1.0, 7.0);
  const Rotation<3> rotation(M_PI_2, 0.0, 0.0);

  const Affine3D product{my_x_map, my_y_map, my_z_map};
  const auto my_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotation, product);

  const std::array<size_t, 8> my_corners{{8, 12, 7, 11, 20, 24, 19, 23}};
  const Index<3> my_extents(7_st, 4_st, 10_st);
  test_3d_rotated(my_map, my_corners, my_extents);
}

void test_3d_rotated_by_180_0_0() {
  const Affine my_x_map(-1.0, 1.0, 2.0, 5.0);
  const Affine my_y_map(-1.0, 1.0, 0.0, 4.0);
  const Affine my_z_map(-1.0, 1.0, 1.0, 7.0);
  const Rotation<3> rotation(M_PI, 0.0, 0.0);

  const Affine3D product{my_x_map, my_y_map, my_z_map};
  const auto my_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotation, product);

  const std::array<size_t, 8> my_corners{{12, 11, 8, 7, 24, 23, 20, 19}};
  const Index<3> my_extents(4, 7, 10);
  test_3d_rotated(my_map, my_corners, my_extents);
}

void test_3d_rotated_by_270_0_0() {
  const Affine my_x_map(-1.0, 1.0, 2.0, 5.0);
  const Affine my_y_map(-1.0, 1.0, 0.0, 4.0);
  const Affine my_z_map(-1.0, 1.0, 1.0, 7.0);
  const Rotation<3> rotation(-M_PI_2, 0.0, 0.0);
  const Affine3D product{my_x_map, my_y_map, my_z_map};
  const auto my_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotation, product);
  const std::array<size_t, 8> my_corners{{11, 7, 12, 8, 23, 19, 24, 20}};
  const Index<3> my_extents(7, 4, 10);
  test_3d_rotated(my_map, my_corners, my_extents);
}

void test_3d_rotated_by_0_90_0() {
  const Affine my_x_map(-1.0, 1.0, 2.0, 5.0);
  const Affine my_y_map(-1.0, 1.0, 0.0, 4.0);
  const Affine my_z_map(-1.0, 1.0, 1.0, 7.0);
  const Rotation<3> rotation(0.0, M_PI_2, 0.0);
  const Affine3D product{my_x_map, my_y_map, my_z_map};
  const auto my_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotation, product);
  const std::array<size_t, 8> my_corners{{19, 7, 23, 11, 20, 8, 24, 12}};
  const Index<3> my_extents(10, 7, 4);
  test_3d_rotated(my_map, my_corners, my_extents);
}

void test_3d_rotated_by_0_90_90() {
  const Affine my_x_map(-1.0, 1.0, 2.0, 5.0);
  const Affine my_y_map(-1.0, 1.0, 0.0, 4.0);
  const Affine my_z_map(-1.0, 1.0, 1.0, 7.0);
  const Rotation<3> rotation(0.0, M_PI_2, M_PI_2);
  const Affine3D product{my_x_map, my_y_map, my_z_map};
  const auto my_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotation, product);
  const std::array<size_t, 8> my_corners{{7, 11, 19, 23, 8, 12, 20, 24}};
  const Index<3> my_extents(7, 10, 4);
  test_3d_rotated(my_map, my_corners, my_extents);
}

void test_3d_rotated_by_0_90_180() {
  const Affine my_x_map(-1.0, 1.0, 2.0, 5.0);
  const Affine my_y_map(-1.0, 1.0, 0.0, 4.0);
  const Affine my_z_map(-1.0, 1.0, 1.0, 7.0);
  const Rotation<3> rotation(0.0, M_PI_2, M_PI);
  const Affine3D product{my_x_map, my_y_map, my_z_map};
  const auto my_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotation, product);
  const std::array<size_t, 8> my_corners{{11, 23, 7, 19, 12, 24, 8, 20}};
  const Index<3> my_extents(10, 7, 4);
  test_3d_rotated(my_map, my_corners, my_extents);
}

void test_3d_rotated_by_0_90_270() {
  const Affine my_x_map(-1.0, 1.0, 2.0, 5.0);
  const Affine my_y_map(-1.0, 1.0, 0.0, 4.0);
  const Affine my_z_map(-1.0, 1.0, 1.0, 7.0);
  const Rotation<3> rotation(0.0, M_PI_2, -M_PI_2);
  const Affine3D product{my_x_map, my_y_map, my_z_map};
  const auto my_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotation, product);
  const std::array<size_t, 8> my_corners{{23, 19, 11, 7, 24, 20, 12, 8}};
  const Index<3> my_extents(7, 10, 4);
  test_3d_rotated(my_map, my_corners, my_extents);
}

void test_3d_rotated_by_0_180_0() {
  const Affine my_x_map(-1.0, 1.0, 2.0, 5.0);
  const Affine my_y_map(-1.0, 1.0, 0.0, 4.0);
  const Affine my_z_map(-1.0, 1.0, 1.0, 7.0);
  const Rotation<3> rotation(0.0, M_PI, 0.0);
  const Affine3D product{my_x_map, my_y_map, my_z_map};
  const auto my_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotation, product);
  const std::array<size_t, 8> my_corners{{20, 19, 24, 23, 8, 7, 12, 11}};
  const Index<3> my_extents(4, 7, 10);
  test_3d_rotated(my_map, my_corners, my_extents);
}

void test_3d_rotated_by_0_180_90() {
  const Affine my_x_map(-1.0, 1.0, 2.0, 5.0);
  const Affine my_y_map(-1.0, 1.0, 0.0, 4.0);
  const Affine my_z_map(-1.0, 1.0, 1.0, 7.0);
  const Rotation<3> rotation(0.0, M_PI, M_PI_2);
  const Affine3D product{my_x_map, my_y_map, my_z_map};
  const auto my_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotation, product);
  const std::array<size_t, 8> my_corners{{19, 23, 20, 24, 7, 11, 8, 12}};
  const Index<3> my_extents(7, 4, 10);
  test_3d_rotated(my_map, my_corners, my_extents);
}

void test_3d_rotated_by_0_180_180() {
  const Affine my_x_map(-1.0, 1.0, 2.0, 5.0);
  const Affine my_y_map(-1.0, 1.0, 0.0, 4.0);
  const Affine my_z_map(-1.0, 1.0, 1.0, 7.0);
  const Rotation<3> rotation(0.0, M_PI, M_PI);
  const Affine3D product{my_x_map, my_y_map, my_z_map};
  const auto my_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotation, product);
  const std::array<size_t, 8> my_corners{{23, 24, 19, 20, 11, 12, 7, 8}};
  const Index<3> my_extents(4, 7, 10);
  test_3d_rotated(my_map, my_corners, my_extents);
}

void test_3d_rotated_by_0_180_270() {
  const Affine my_x_map(-1.0, 1.0, 2.0, 5.0);
  const Affine my_y_map(-1.0, 1.0, 0.0, 4.0);
  const Affine my_z_map(-1.0, 1.0, 1.0, 7.0);
  const Rotation<3> rotation(0.0, M_PI, -M_PI_2);
  const Affine3D product{my_x_map, my_y_map, my_z_map};
  const auto my_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotation, product);
  const std::array<size_t, 8> my_corners{{24, 20, 23, 19, 12, 8, 11, 7}};
  const Index<3> my_extents(7, 4, 10);
  test_3d_rotated(my_map, my_corners, my_extents);
}

void test_3d_rotated_by_0_270_0() {
  const Affine my_x_map(-1.0, 1.0, 2.0, 5.0);
  const Affine my_y_map(-1.0, 1.0, 0.0, 4.0);
  const Affine my_z_map(-1.0, 1.0, 1.0, 7.0);
  const Rotation<3> rotation(0.0, -M_PI_2, 0.0);
  const Affine3D product{my_x_map, my_y_map, my_z_map};
  const auto my_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotation, product);
  const std::array<size_t, 8> my_corners{{8, 20, 12, 24, 7, 19, 11, 23}};
  const Index<3> my_extents(10, 7, 4);
  test_3d_rotated(my_map, my_corners, my_extents);
}

void test_3d_rotated_by_0_270_90() {
  const Affine my_x_map(-1.0, 1.0, 2.0, 5.0);
  const Affine my_y_map(-1.0, 1.0, 0.0, 4.0);
  const Affine my_z_map(-1.0, 1.0, 1.0, 7.0);
  const Rotation<3> rotation(0.0, -M_PI_2, M_PI_2);
  const Affine3D product{my_x_map, my_y_map, my_z_map};
  const auto my_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotation, product);
  const std::array<size_t, 8> my_corners{{20, 24, 8, 12, 19, 23, 7, 11}};
  const Index<3> my_extents(7, 10, 4);
  test_3d_rotated(my_map, my_corners, my_extents);
}

void test_3d_rotated_by_0_270_180() {
  const Affine my_x_map(-1.0, 1.0, 2.0, 5.0);
  const Affine my_y_map(-1.0, 1.0, 0.0, 4.0);
  const Affine my_z_map(-1.0, 1.0, 1.0, 7.0);
  const Rotation<3> rotation(0.0, -M_PI_2, M_PI);
  const Affine3D product{my_x_map, my_y_map, my_z_map};
  const auto my_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotation, product);
  const std::array<size_t, 8> my_corners{{24, 12, 20, 8, 23, 11, 19, 7}};
  const Index<3> my_extents(10, 7, 4);
  test_3d_rotated(my_map, my_corners, my_extents);
}

void test_3d_rotated_by_0_270_270() {
  const Affine my_x_map(-1.0, 1.0, 2.0, 5.0);
  const Affine my_y_map(-1.0, 1.0, 0.0, 4.0);
  const Affine my_z_map(-1.0, 1.0, 1.0, 7.0);
  const Rotation<3> rotation(0.0, -M_PI_2, -M_PI_2);
  const Affine3D product{my_x_map, my_y_map, my_z_map};
  const auto my_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotation, product);
  const std::array<size_t, 8> my_corners{{12, 8, 24, 20, 11, 7, 23, 19}};
  const Index<3> my_extents(7, 10, 4);
  test_3d_rotated(my_map, my_corners, my_extents);
}

void test_3d_rotated_by_90_90_0() {
  const Affine my_x_map(-1.0, 1.0, 2.0, 5.0);
  const Affine my_y_map(-1.0, 1.0, 0.0, 4.0);
  const Affine my_z_map(-1.0, 1.0, 1.0, 7.0);
  const Rotation<3> rotation(M_PI_2, M_PI_2, 0.0);
  const Affine3D product{my_x_map, my_y_map, my_z_map};
  const auto my_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotation, product);
  const std::array<size_t, 8> my_corners{{20, 8, 19, 7, 24, 12, 23, 11}};
  const Index<3> my_extents(10, 4, 7);
  test_3d_rotated(my_map, my_corners, my_extents);
}

void test_3d_rotated_by_90_90_90() {
  const Affine my_x_map(-1.0, 1.0, 2.0, 5.0);
  const Affine my_y_map(-1.0, 1.0, 0.0, 4.0);
  const Affine my_z_map(-1.0, 1.0, 1.0, 7.0);
  const Rotation<3> rotation(M_PI_2, M_PI_2, M_PI_2);
  const Affine3D product{my_x_map, my_y_map, my_z_map};
  const auto my_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotation, product);
  const std::array<size_t, 8> my_corners{{8, 7, 20, 19, 12, 11, 24, 23}};
  const Index<3> my_extents(4, 10, 7);
  test_3d_rotated(my_map, my_corners, my_extents);
}

void test_3d_rotated_by_90_90_180() {
  const Affine my_x_map(-1.0, 1.0, 2.0, 5.0);
  const Affine my_y_map(-1.0, 1.0, 0.0, 4.0);
  const Affine my_z_map(-1.0, 1.0, 1.0, 7.0);
  const Rotation<3> rotation(M_PI_2, M_PI_2, M_PI);
  const Affine3D product{my_x_map, my_y_map, my_z_map};
  const auto my_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotation, product);
  const std::array<size_t, 8> my_corners{{7, 19, 8, 20, 11, 23, 12, 24}};
  const Index<3> my_extents(10, 4, 7);
  test_3d_rotated(my_map, my_corners, my_extents);
}

void test_3d_rotated_by_90_90_270() {
  const Affine my_x_map(-1.0, 1.0, 2.0, 5.0);
  const Affine my_y_map(-1.0, 1.0, 0.0, 4.0);
  const Affine my_z_map(-1.0, 1.0, 1.0, 7.0);
  const Rotation<3> rotation(M_PI_2, M_PI_2, -M_PI_2);
  const Affine3D product{my_x_map, my_y_map, my_z_map};
  const auto my_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotation, product);
  const std::array<size_t, 8> my_corners{{19, 20, 7, 8, 23, 24, 11, 12}};
  const Index<3> my_extents(4, 10, 7);
  test_3d_rotated(my_map, my_corners, my_extents);
}

void test_3d_rotated_by_90_270_0() {
  const Affine my_x_map(-1.0, 1.0, 2.0, 5.0);
  const Affine my_y_map(-1.0, 1.0, 0.0, 4.0);
  const Affine my_z_map(-1.0, 1.0, 1.0, 7.0);
  const Rotation<3> rotation(M_PI_2, -M_PI_2, 0.0);
  const Affine3D product{my_x_map, my_y_map, my_z_map};
  const auto my_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotation, product);
  const std::array<size_t, 8> my_corners{{12, 24, 11, 23, 8, 20, 7, 19}};
  const Index<3> my_extents(10, 4, 7);
  test_3d_rotated(my_map, my_corners, my_extents);
}

void test_3d_rotated_by_90_270_90() {
  const Affine my_x_map(-1.0, 1.0, 2.0, 5.0);
  const Affine my_y_map(-1.0, 1.0, 0.0, 4.0);
  const Affine my_z_map(-1.0, 1.0, 1.0, 7.0);
  const Rotation<3> rotation(M_PI_2, -M_PI_2, M_PI_2);
  const Affine3D product{my_x_map, my_y_map, my_z_map};
  const auto my_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotation, product);
  const std::array<size_t, 8> my_corners{{24, 23, 12, 11, 20, 19, 8, 7}};
  const Index<3> my_extents(4, 10, 7);
  test_3d_rotated(my_map, my_corners, my_extents);
}

void test_3d_rotated_by_90_270_180() {
  const Affine my_x_map(-1.0, 1.0, 2.0, 5.0);
  const Affine my_y_map(-1.0, 1.0, 0.0, 4.0);
  const Affine my_z_map(-1.0, 1.0, 1.0, 7.0);
  const Rotation<3> rotation(M_PI_2, -M_PI_2, M_PI);
  const Affine3D product{my_x_map, my_y_map, my_z_map};
  const auto my_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotation, product);
  const std::array<size_t, 8> my_corners{{23, 11, 24, 12, 19, 7, 20, 8}};
  const Index<3> my_extents(10, 4, 7);
  test_3d_rotated(my_map, my_corners, my_extents);
}

void test_3d_rotated_by_90_270_270() {
  const Affine my_x_map(-1.0, 1.0, 2.0, 5.0);
  const Affine my_y_map(-1.0, 1.0, 0.0, 4.0);
  const Affine my_z_map(-1.0, 1.0, 1.0, 7.0);
  const Rotation<3> rotation(M_PI_2, -M_PI_2, -M_PI_2);
  const Affine3D product{my_x_map, my_y_map, my_z_map};
  const auto my_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotation, product);
  const std::array<size_t, 8> my_corners{{11, 12, 23, 24, 7, 8, 19, 20}};
  const Index<3> my_extents(4, 10, 7);
  test_3d_rotated(my_map, my_corners, my_extents);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Variables.OrientVariablesOnSlice",
                  "[DataStructures][Unit]") {
  test_1d_aligned();
  test_1d_flipped();
  test_2d_aligned();
  test_2d_flipped();
  test_2d_rotated_by_90();
  test_2d_rotated_by_270();
  test_3d_aligned();
  test_3d_rotated_by_90_0_0();
  test_3d_rotated_by_180_0_0();
  test_3d_rotated_by_270_0_0();
  test_3d_rotated_by_0_90_0();
  test_3d_rotated_by_0_90_90();
  test_3d_rotated_by_0_90_180();
  test_3d_rotated_by_0_90_270();
  test_3d_rotated_by_0_180_0();
  test_3d_rotated_by_0_180_90();
  test_3d_rotated_by_0_180_180();
  test_3d_rotated_by_0_180_270();
  test_3d_rotated_by_0_270_0();
  test_3d_rotated_by_0_270_90();
  test_3d_rotated_by_0_270_180();
  test_3d_rotated_by_0_270_270();
  test_3d_rotated_by_90_90_0();
  test_3d_rotated_by_90_90_90();
  test_3d_rotated_by_90_90_180();
  test_3d_rotated_by_90_90_270();
  test_3d_rotated_by_90_270_0();
  test_3d_rotated_by_90_270_90();
  test_3d_rotated_by_90_270_180();
  test_3d_rotated_by_90_270_270();
}
