// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <unordered_map>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockNeighbor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Frustum.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/Wedge3D.hpp"
#include "Domain/Direction.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/OrientationMap.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/StdHelpers.hpp"

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

namespace {
std::vector<
    std::unique_ptr<CoordinateMapBase<Frame::Logical, Frame::Inertial, 3>>>
test_wedge_map_generation(double inner_radius, double outer_radius,
                          double inner_sphericity, double outer_sphericity,
                          bool use_equiangular_map,
                          double x_coord_of_shell_center = 0.0,
                          bool use_half_wedges = false) {
  using Wedge3DMap = CoordinateMaps::Wedge3D;
  if (use_half_wedges) {
    using Halves = Wedge3DMap::WedgeHalves;
    return make_vector_coordinate_map_base<Frame::Logical, Frame::Inertial>(
        Wedge3DMap{inner_radius, outer_radius, OrientationMap<3>{},
                   inner_sphericity, outer_sphericity, use_equiangular_map,
                   Halves::LowerOnly},
        Wedge3DMap{inner_radius, outer_radius, OrientationMap<3>{},
                   inner_sphericity, outer_sphericity, use_equiangular_map,
                   Halves::UpperOnly},
        Wedge3DMap{inner_radius, outer_radius,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_xi(), Direction<3>::lower_eta(),
                        Direction<3>::lower_zeta()}}},
                   inner_sphericity, outer_sphericity, use_equiangular_map,
                   Halves::LowerOnly},
        Wedge3DMap{inner_radius, outer_radius,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_xi(), Direction<3>::lower_eta(),
                        Direction<3>::lower_zeta()}}},
                   inner_sphericity, outer_sphericity, use_equiangular_map,
                   Halves::UpperOnly},
        Wedge3DMap{inner_radius, outer_radius,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
                        Direction<3>::lower_eta()}}},
                   inner_sphericity, outer_sphericity, use_equiangular_map,
                   Halves::LowerOnly},
        Wedge3DMap{inner_radius, outer_radius,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
                        Direction<3>::lower_eta()}}},
                   inner_sphericity, outer_sphericity, use_equiangular_map,
                   Halves::UpperOnly},
        Wedge3DMap{inner_radius, outer_radius,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_xi(), Direction<3>::lower_zeta(),
                        Direction<3>::upper_eta()}}},
                   inner_sphericity, outer_sphericity, use_equiangular_map,
                   Halves::LowerOnly},
        Wedge3DMap{inner_radius, outer_radius,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_xi(), Direction<3>::lower_zeta(),
                        Direction<3>::upper_eta()}}},
                   inner_sphericity, outer_sphericity, use_equiangular_map,
                   Halves::UpperOnly},
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
  }
  if (x_coord_of_shell_center == 0.0) {
    return make_vector_coordinate_map_base<Frame::Logical, Frame::Inertial>(
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
  }
  using Identity2D = CoordinateMaps::Identity<2>;
  using Affine = CoordinateMaps::Affine;
  const auto translation = CoordinateMaps::ProductOf2Maps<Affine, Identity2D>(
      Affine{-1.0, 1.0, -1.0 + x_coord_of_shell_center,
             1.0 + x_coord_of_shell_center},
      Identity2D{});

  return make_vector(
      make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Wedge3DMap{inner_radius, outer_radius, OrientationMap<3>{},
                     inner_sphericity, outer_sphericity, use_equiangular_map},
          translation),
      make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Wedge3DMap{inner_radius, outer_radius,
                     OrientationMap<3>{std::array<Direction<3>, 3>{
                         {Direction<3>::upper_xi(), Direction<3>::lower_eta(),
                          Direction<3>::lower_zeta()}}},
                     inner_sphericity, outer_sphericity, use_equiangular_map},
          translation),
      make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Wedge3DMap{inner_radius, outer_radius,
                     OrientationMap<3>{std::array<Direction<3>, 3>{
                         {Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
                          Direction<3>::lower_eta()}}},
                     inner_sphericity, outer_sphericity, use_equiangular_map},
          translation),
      make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Wedge3DMap{inner_radius, outer_radius,
                     OrientationMap<3>{std::array<Direction<3>, 3>{
                         {Direction<3>::upper_xi(), Direction<3>::lower_zeta(),
                          Direction<3>::upper_eta()}}},
                     inner_sphericity, outer_sphericity, use_equiangular_map},
          translation),
      make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Wedge3DMap{inner_radius, outer_radius,
                     OrientationMap<3>{std::array<Direction<3>, 3>{
                         {Direction<3>::upper_zeta(), Direction<3>::upper_xi(),
                          Direction<3>::upper_eta()}}},
                     inner_sphericity, outer_sphericity, use_equiangular_map},
          translation),
      make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Wedge3DMap{inner_radius, outer_radius,
                     OrientationMap<3>{std::array<Direction<3>, 3>{
                         {Direction<3>::lower_zeta(), Direction<3>::lower_xi(),
                          Direction<3>::upper_eta()}}},
                     inner_sphericity, outer_sphericity, use_equiangular_map},
          translation));
}
void test_wedge_map_generation_against_domain_helpers(
    double inner_radius, double outer_radius, double inner_sphericity,
    double outer_sphericity, bool use_equiangular_map,
    double x_coord_of_shell_center = 0.0, bool use_half_wedges = false) {
  const auto expected_coord_maps = test_wedge_map_generation(
      inner_radius, outer_radius, inner_sphericity, outer_sphericity,
      use_equiangular_map, x_coord_of_shell_center, use_half_wedges);
  const auto maps = wedge_coordinate_maps<Frame::Inertial>(
      inner_radius, outer_radius, inner_sphericity, outer_sphericity,
      use_equiangular_map, x_coord_of_shell_center, use_half_wedges);
  for (size_t i = 0; i < maps.size(); i++) {
    CHECK(*expected_coord_maps[i] == *maps[i]);
  }
}

}  // namespace
SPECTRE_TEST_CASE(
    "Unit.Domain.DomainHelpers.DefaultSixWedgeDirections.Equiangular",
    "[Domain][Unit]") {
  const double inner_radius = 1.2;
  const double outer_radius = 2.7;
  const double inner_sphericity = 0.8;
  const double outer_sphericity = 0.6;
  const bool use_equiangular_map = true;
  test_wedge_map_generation_against_domain_helpers(
      inner_radius, outer_radius, inner_sphericity, outer_sphericity,
      use_equiangular_map);
}

SPECTRE_TEST_CASE(
    "Unit.Domain.DomainHelpers.DefaultSixWedgeDirections.Equidistant",
    "[Domain][Unit]") {
  const double inner_radius = 0.8;
  const double outer_radius = 7.1;
  const double inner_sphericity = 0.2;
  const double outer_sphericity = 0.4;
  const bool use_equiangular_map = false;
  test_wedge_map_generation_against_domain_helpers(
      inner_radius, outer_radius, inner_sphericity, outer_sphericity,
      use_equiangular_map);
}

SPECTRE_TEST_CASE(
    "Unit.Domain.DomainHelpers.TranslatedSixWedgeDirections.Equiangular",
    "[Domain][Unit]") {
  const double inner_radius = 1.2;
  const double outer_radius = 3.1;
  const double inner_sphericity = 0.3;
  const double outer_sphericity = 0.6;
  const bool use_equiangular_map = true;
  const double x_coord_of_shell_center = 0.6;
  test_wedge_map_generation_against_domain_helpers(
      inner_radius, outer_radius, inner_sphericity, outer_sphericity,
      use_equiangular_map, x_coord_of_shell_center);
}

SPECTRE_TEST_CASE(
    "Unit.Domain.DomainHelpers.TranslatedSixWedgeDirections.Equidistant",
    "[Domain][Unit]") {
  const double inner_radius = 12.2;
  const double outer_radius = 31.1;
  const double inner_sphericity = 0.9;
  const double outer_sphericity = 0.1;
  const bool use_equiangular_map = false;
  const double x_coord_of_shell_center = -2.7;
  test_wedge_map_generation_against_domain_helpers(
      inner_radius, outer_radius, inner_sphericity, outer_sphericity,
      use_equiangular_map, x_coord_of_shell_center);
}

SPECTRE_TEST_CASE("Unit.Domain.DomainHelpers.TenWedgeDirections.Equiangular",
                  "[Domain][Unit]") {
  const double inner_radius = 0.2;
  const double outer_radius = 2.2;
  const double inner_sphericity = 0.0;
  const double outer_sphericity = 1.0;
  const bool use_equiangular_map = true;
  const bool use_half_wedges = true;
  test_wedge_map_generation_against_domain_helpers(
      inner_radius, outer_radius, inner_sphericity, outer_sphericity,
      use_equiangular_map, 0.0, use_half_wedges);
}

SPECTRE_TEST_CASE("Unit.Domain.DomainHelpers.TenWedgeDirections.Equidistant",
                  "[Domain][Unit]") {
  const double inner_radius = 0.2;
  const double outer_radius = 29.2;
  const double inner_sphericity = 0.01;
  const double outer_sphericity = 0.99;
  const bool use_equiangular_map = false;
  const bool use_half_wedges = true;
  test_wedge_map_generation_against_domain_helpers(
      inner_radius, outer_radius, inner_sphericity, outer_sphericity,
      use_equiangular_map, 0.0, use_half_wedges);
}

SPECTRE_TEST_CASE("Unit.Domain.DomainHelpers.AllFrustumDirections",
                  "[Domain][Unit]") {
  using FrustumMap = CoordinateMaps::Frustum;
  // half of the length of the inner cube in the binary compact object domain:
  const double lower = 1.7;
  // half of the length of the outer cube in the binary compact object domain:
  const double top = 5.2;
  for (const bool use_equiangular_map : {true, false}) {
    const auto expected_coord_maps = make_vector_coordinate_map_base<
        Frame::Logical, Frame::Inertial>(
        FrustumMap{{{{{-2.0 * lower, -lower}},
                     {{0.0, lower}},
                     {{-top, -top}},
                     {{0.0, top}}}},
                   lower,
                   top,
                   OrientationMap<3>{},
                   use_equiangular_map},
        FrustumMap{{{{{0.0, -lower}},
                     {{2.0 * lower, lower}},
                     {{0.0, -top}},
                     {{top, top}}}},
                   lower,
                   top,
                   OrientationMap<3>{},
                   use_equiangular_map},
        FrustumMap{{{{{-2.0 * lower, -lower}},
                     {{0.0, lower}},
                     {{-top, -top}},
                     {{0.0, top}}}},
                   lower,
                   top,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_xi(), Direction<3>::lower_eta(),
                        Direction<3>::lower_zeta()}}},
                   use_equiangular_map},
        FrustumMap{{{{{0.0, -lower}},
                     {{2.0 * lower, lower}},
                     {{0.0, -top}},
                     {{top, top}}}},
                   lower,
                   top,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_xi(), Direction<3>::lower_eta(),
                        Direction<3>::lower_zeta()}}},
                   use_equiangular_map},
        FrustumMap{{{{{-2.0 * lower, -lower}},
                     {{0.0, lower}},
                     {{-top, -top}},
                     {{0.0, top}}}},
                   lower,
                   top,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
                        Direction<3>::lower_eta()}}},
                   use_equiangular_map},
        FrustumMap{{{{{0.0, -lower}},
                     {{2.0 * lower, lower}},
                     {{0.0, -top}},
                     {{top, top}}}},
                   lower,
                   top,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
                        Direction<3>::lower_eta()}}},
                   use_equiangular_map},
        FrustumMap{{{{{-2.0 * lower, -lower}},
                     {{0.0, lower}},
                     {{-top, -top}},
                     {{0.0, top}}}},
                   lower,
                   top,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_xi(), Direction<3>::lower_zeta(),
                        Direction<3>::upper_eta()}}},
                   use_equiangular_map},
        FrustumMap{{{{{0.0, -lower}},
                     {{2.0 * lower, lower}},
                     {{0.0, -top}},
                     {{top, top}}}},
                   lower,
                   top,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_xi(), Direction<3>::lower_zeta(),
                        Direction<3>::upper_eta()}}},
                   use_equiangular_map},
        // Frustum on right half in the +x direction
        FrustumMap{{{{{-lower, -lower}},
                     {{lower, lower}},
                     {{-top, -top}},
                     {{top, top}}}},
                   2.0 * lower,
                   top,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_zeta(), Direction<3>::upper_xi(),
                        Direction<3>::upper_eta()}}},
                   use_equiangular_map},
        // Frustum on left half in the -x direction
        FrustumMap{{{{{-lower, -lower}},
                     {{lower, lower}},
                     {{-top, -top}},
                     {{top, top}}}},
                   2.0 * lower,
                   top,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::lower_zeta(), Direction<3>::lower_xi(),
                        Direction<3>::upper_eta()}}},
                   use_equiangular_map});

    const auto maps = frustum_coordinate_maps<Frame::Inertial>(
        2.0 * lower, 2.0 * top, use_equiangular_map);
    for (size_t i = 0; i < maps.size(); i++) {
      CHECK(*expected_coord_maps[i] == *maps[i]);
    }
  }
}
