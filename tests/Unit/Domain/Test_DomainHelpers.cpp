// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <unordered_map>
#include <vector>

#include "Domain/BlockNeighbor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Wedge3D.hpp"
#include "Domain/Direction.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/OrientationMap.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"

namespace Frame {
struct Inertial;
struct Logical;
}  // namespace Frame

SPECTRE_TEST_CASE("Unit.Domain.DomainHelpers.Periodic.SameBlock",
                  "[Domain][Unit]") {
  const std::vector<std::array<size_t, 8>> corners_of_all_blocks{
      {{0, 1, 2, 3, 4, 5, 6, 7}}, {{8, 9, 10, 11, 0, 1, 2, 3}}};
  std::vector<std::unordered_map<Direction<3>, BlockNeighbor<3>>>
      neighbors_of_all_blocks;
  set_internal_boundaries<3>(corners_of_all_blocks, &neighbors_of_all_blocks);

  const OrientationMap<3> aligned{};
  CHECK(neighbors_of_all_blocks[0][Direction<3>::lower_zeta()].orientation() ==
        aligned);

  const PairOfFaces x_faces{{1, 3, 5, 7}, {0, 2, 4, 6}};

  const std::vector<PairOfFaces> identifications{x_faces};
  set_periodic_boundaries<3>(identifications, corners_of_all_blocks,
                             &neighbors_of_all_blocks);
  CHECK(neighbors_of_all_blocks[0][Direction<3>::upper_xi()].orientation() ==
        aligned);

  const std::vector<std::unordered_map<Direction<3>, BlockNeighbor<3>>>
      expected_block_neighbors{{{Direction<3>::upper_xi(), {0, aligned}},
                                {Direction<3>::lower_xi(), {0, aligned}},
                                {Direction<3>::lower_zeta(), {1, aligned}}},
                               {{Direction<3>::upper_zeta(), {0, aligned}}}};

  CHECK(neighbors_of_all_blocks == expected_block_neighbors);
}

SPECTRE_TEST_CASE("Unit.Domain.DomainHelpers.Periodic.DifferentBlocks",
                  "[Domain][Unit]") {
  const std::vector<std::array<size_t, 8>> corners_of_all_blocks{
      {{0, 1, 2, 3, 4, 5, 6, 7}}, {{8, 9, 10, 11, 0, 1, 2, 3}}};
  std::vector<std::unordered_map<Direction<3>, BlockNeighbor<3>>>
      neighbors_of_all_blocks;
  set_internal_boundaries<3>(corners_of_all_blocks, &neighbors_of_all_blocks);

  const OrientationMap<3> aligned{};
  CHECK(neighbors_of_all_blocks[0][Direction<3>::lower_zeta()].orientation() ==
        aligned);

  const PairOfFaces x_faces_on_different_blocks{{1, 3, 5, 7}, {8, 10, 0, 2}};

  const std::vector<PairOfFaces> identifications{x_faces_on_different_blocks};
  set_periodic_boundaries<3>(identifications, corners_of_all_blocks,
                             &neighbors_of_all_blocks);
  CHECK(neighbors_of_all_blocks[0][Direction<3>::upper_xi()].orientation() ==
        aligned);
  const std::vector<std::unordered_map<Direction<3>, BlockNeighbor<3>>>
      expected_block_neighbors{{{Direction<3>::upper_xi(), {1, aligned}},
                                {Direction<3>::lower_zeta(), {1, aligned}}},
                               {{Direction<3>::lower_xi(), {0, aligned}},
                                {Direction<3>::upper_zeta(), {0, aligned}}}};

  CHECK(neighbors_of_all_blocks == expected_block_neighbors);
}

SPECTRE_TEST_CASE("Unit.Domain.DomainHelpers.AllWedgeDirections",
                  "[Domain][Unit]") {
  using Wedge3DMap = CoordinateMaps::Wedge3D;
  const double inner_radius = 1.0;
  const double outer_radius = 2.0;
  const double inner_sphericity = 1.0;
  const double outer_sphericity = 1.0;
  const bool use_equiangular_map = true;

  const auto expected_coord_maps =
      make_vector_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Wedge3DMap{inner_radius, outer_radius, OrientationMap<3>{},
                     inner_sphericity, outer_sphericity, use_equiangular_map},
          Wedge3DMap{inner_radius, outer_radius,
                     OrientationMap<3>{std::array<Direction<3>, 3>{
                         {Direction<3>::upper_xi(), Direction<3>::lower_eta(),
                          Direction<3>::lower_zeta()}}},
                     inner_sphericity, outer_sphericity, use_equiangular_map},
          Wedge3DMap{inner_radius, outer_radius,
                     OrientationMap<3>{std::array<Direction<3>, 3>{
                         {Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
                          Direction<3>::lower_eta()}}},
                     inner_sphericity, outer_sphericity, use_equiangular_map},
          Wedge3DMap{inner_radius, outer_radius,
                     OrientationMap<3>{std::array<Direction<3>, 3>{
                         {Direction<3>::upper_xi(), Direction<3>::lower_zeta(),
                          Direction<3>::upper_eta()}}},
                     inner_sphericity, outer_sphericity, use_equiangular_map},
          Wedge3DMap{inner_radius, outer_radius,
                     OrientationMap<3>{std::array<Direction<3>, 3>{
                         {Direction<3>::upper_zeta(), Direction<3>::upper_xi(),
                          Direction<3>::upper_eta()}}},
                     inner_sphericity, outer_sphericity, use_equiangular_map},
          Wedge3DMap{inner_radius, outer_radius,
                     OrientationMap<3>{std::array<Direction<3>, 3>{
                         {Direction<3>::lower_zeta(), Direction<3>::lower_xi(),
                          Direction<3>::upper_eta()}}},
                     inner_sphericity, outer_sphericity, use_equiangular_map});
  const auto maps = wedge_coordinate_maps<Frame::Inertial>(
      inner_radius, outer_radius, inner_sphericity, outer_sphericity,
      use_equiangular_map);
  CHECK(*expected_coord_maps[0] == *maps[0]);
  CHECK(*expected_coord_maps[1] == *maps[1]);
  CHECK(*expected_coord_maps[2] == *maps[2]);
  CHECK(*expected_coord_maps[3] == *maps[3]);
  CHECK(*expected_coord_maps[4] == *maps[4]);
  CHECK(*expected_coord_maps[5] == *maps[5]);
}
