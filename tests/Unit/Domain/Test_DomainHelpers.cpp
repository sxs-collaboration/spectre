// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
// IWYU pragma: no_include <initializer_list>
#include <memory>
#include <pup.h>
#include <vector>

#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockNeighbor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/Frustum.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/Wedge3D.hpp"
#include "Domain/Direction.hpp"
#include "Domain/DirectionMap.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/Side.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/StdHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Domain.DomainHelpers.Periodic.SameBlock",
                  "[Domain][Unit]") {
  const std::vector<std::array<size_t, 8>> corners_of_all_blocks{
      {{0, 1, 2, 3, 4, 5, 6, 7}}, {{8, 9, 10, 11, 0, 1, 2, 3}}};
  std::vector<DirectionMap<3, BlockNeighbor<3>>> neighbors_of_all_blocks;
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

  const std::vector<DirectionMap<3, BlockNeighbor<3>>> expected_block_neighbors{
      {{Direction<3>::upper_xi(), {0, aligned}},
       {Direction<3>::lower_xi(), {0, aligned}},
       {Direction<3>::lower_zeta(), {1, aligned}}},
      {{Direction<3>::upper_zeta(), {0, aligned}}}};

  CHECK(neighbors_of_all_blocks == expected_block_neighbors);
}

SPECTRE_TEST_CASE("Unit.Domain.DomainHelpers.Periodic.DifferentBlocks",
                  "[Domain][Unit]") {
  const std::vector<std::array<size_t, 8>> corners_of_all_blocks{
      {{0, 1, 2, 3, 4, 5, 6, 7}}, {{8, 9, 10, 11, 0, 1, 2, 3}}};
  std::vector<DirectionMap<3, BlockNeighbor<3>>> neighbors_of_all_blocks;
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
  const std::vector<DirectionMap<3, BlockNeighbor<3>>> expected_block_neighbors{
      {{Direction<3>::upper_xi(), {1, aligned}},
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

SPECTRE_TEST_CASE("Unit.Domain.DomainHelpers.ShellGraph", "[Domain][Unit]") {
  std::vector<std::array<size_t, 8>> expected_corners = {
      // Shell on left-hand side:
      {{5, 6, 7, 8, 13, 14, 15, 16}} /*+z*/,
      {{3, 4, 1, 2, 11, 12, 9, 10}} /*-z*/,
      {{7, 8, 3, 4, 15, 16, 11, 12}} /*+y*/,
      {{1, 2, 5, 6, 9, 10, 13, 14}} /*-y*/,
      {{2, 4, 6, 8, 10, 12, 14, 16}} /*+x*/,
      {{3, 1, 7, 5, 11, 9, 15, 13}} /*-x*/};
  const auto generated_corners = corners_for_radially_layered_domains(1, false);
  for (size_t i = 0; i < expected_corners.size(); i++) {
    INFO(i);
    CHECK(generated_corners[i] == expected_corners[i]);
  }
  CHECK(generated_corners == expected_corners);
}

SPECTRE_TEST_CASE("Unit.Domain.DomainHelpers.SphereGraph", "[Domain][Unit]") {
  std::vector<std::array<size_t, 8>> expected_corners = {
      // Shell on left-hand side:
      {{5, 6, 7, 8, 13, 14, 15, 16}} /*+z*/,
      {{3, 4, 1, 2, 11, 12, 9, 10}} /*-z*/,
      {{7, 8, 3, 4, 15, 16, 11, 12}} /*+y*/,
      {{1, 2, 5, 6, 9, 10, 13, 14}} /*-y*/,
      {{2, 4, 6, 8, 10, 12, 14, 16}} /*+x*/,
      {{3, 1, 7, 5, 11, 9, 15, 13}} /*-x*/,
      {{1, 2, 3, 4, 5, 6, 7, 8}} /*central block*/};
  const auto generated_corners = corners_for_radially_layered_domains(1, true);
  for (size_t i = 0; i < expected_corners.size(); i++) {
    INFO(i);
    CHECK(generated_corners[i] == expected_corners[i]);
  }
  CHECK(generated_corners == expected_corners);
}

namespace {
std::vector<std::array<size_t, 8>> expected_bbh_corners() {
  return {// Shell on left-hand side:
          {{5, 6, 7, 8, 13, 14, 15, 16}} /*+z*/,
          {{3, 4, 1, 2, 11, 12, 9, 10}} /*-z*/,
          {{7, 8, 3, 4, 15, 16, 11, 12}} /*+y*/,
          {{1, 2, 5, 6, 9, 10, 13, 14}} /*-y*/,
          {{2, 4, 6, 8, 10, 12, 14, 16}} /*+x*/,
          {{3, 1, 7, 5, 11, 9, 15, 13}} /*-x*/,
          // Cube on left-hand side:
          {{13, 14, 15, 16, 21, 22, 23, 24}} /*+z*/,
          {{11, 12, 9, 10, 19, 20, 17, 18}} /*-z*/,
          {{15, 16, 11, 12, 23, 24, 19, 20}} /*+y*/,
          {{9, 10, 13, 14, 17, 18, 21, 22}} /*-y*/,
          {{10, 12, 14, 16, 18, 20, 22, 24}} /*+x*/,
          {{11, 9, 15, 13, 19, 17, 23, 21}} /*-x*/,
          // Shell on right-hand side:
          {{45, 46, 47, 48, 53, 54, 55, 56}} /*+z*/,
          {{43, 44, 41, 42, 51, 52, 49, 50}} /*-z*/,
          {{47, 48, 43, 44, 55, 56, 51, 52}} /*+y*/,
          {{41, 42, 45, 46, 49, 50, 53, 54}} /*-y*/,
          {{42, 44, 46, 48, 50, 52, 54, 56}} /*+x*/,
          {{43, 41, 47, 45, 51, 49, 55, 53}} /*-x*/,
          // Cube on right-hand side:
          {{53, 54, 55, 56, 22, 62, 24, 64}} /*+z*/,
          {{51, 52, 49, 50, 20, 60, 18, 58}} /*-z*/,
          {{55, 56, 51, 52, 24, 64, 20, 60}} /*+y*/,
          {{49, 50, 53, 54, 18, 58, 22, 62}} /*-y*/,
          {{50, 52, 54, 56, 58, 60, 62, 64}} /*+x*/,
          {{51, 49, 55, 53, 20, 18, 24, 22}} /*-x*/,
          // Frustums on both sides:
          {{21, 22, 23, 24, 29, 30, 31, 32}} /*+zL*/,
          {{22, 62, 24, 64, 30, 70, 32, 72}} /*+zR*/,
          {{19, 20, 17, 18, 27, 28, 25, 26}} /*-zL*/,
          {{20, 60, 18, 58, 28, 68, 26, 66}} /*-zR*/,
          {{23, 24, 19, 20, 31, 32, 27, 28}} /*+yL*/,
          {{24, 64, 20, 60, 32, 72, 28, 68}} /*+yR*/,
          {{17, 18, 21, 22, 25, 26, 29, 30}} /*-yL*/,
          {{18, 58, 22, 62, 26, 66, 30, 70}} /*-yR*/,
          {{58, 60, 62, 64, 66, 68, 70, 72}} /*+xR*/,
          {{19, 17, 23, 21, 27, 25, 31, 29}} /*-xL*/,
          // Outermost Shell in the wave-zone:
          {{29, 30, 31, 32, 37, 38, 39, 40}} /*+zL*/,
          {{30, 70, 32, 72, 38, 78, 40, 80}} /*+zR*/,
          {{27, 28, 25, 26, 35, 36, 33, 34}} /*-zL*/,
          {{28, 68, 26, 66, 36, 76, 34, 74}} /*-zR*/,
          {{31, 32, 27, 28, 39, 40, 35, 36}} /*+yL*/,
          {{32, 72, 28, 68, 40, 80, 36, 76}} /*+yR*/,
          {{25, 26, 29, 30, 33, 34, 37, 38}} /*-yL*/,
          {{26, 66, 30, 70, 34, 74, 38, 78}} /*-yR*/,
          {{66, 68, 70, 72, 74, 76, 78, 80}} /*+xR*/,
          {{27, 25, 31, 29, 35, 33, 39, 37}} /*-xL*/};
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.DomainHelpers.BBHCorners", "[Domain][Unit]") {
  const auto generated_corners =
      corners_for_biradially_layered_domains(2, 2, false, false);
  for (size_t i = 0; i < expected_bbh_corners().size(); i++) {
    INFO(i);
    CHECK(generated_corners[i] == expected_bbh_corners()[i]);
  }
  CHECK(generated_corners == expected_bbh_corners());
}

SPECTRE_TEST_CASE("Unit.Domain.DomainHelpers.NSBHCorners", "[Domain][Unit]") {
  std::vector<std::array<size_t, 8>> expected_corners = expected_bbh_corners();
  expected_corners.push_back(std::array<size_t, 8>{{1, 2, 3, 4, 5, 6, 7, 8}});
  const auto generated_corners =
      corners_for_biradially_layered_domains(2, 2, true, false);
  for (size_t i = 0; i < expected_corners.size(); i++) {
    INFO(i);
    CHECK(generated_corners[i] == expected_corners[i]);
  }
  CHECK(generated_corners == expected_corners);
}

SPECTRE_TEST_CASE("Unit.Domain.DomainHelpers.BHNSCorners", "[Domain][Unit]") {
  std::vector<std::array<size_t, 8>> expected_corners = expected_bbh_corners();
  expected_corners.push_back(
      std::array<size_t, 8>{{41, 42, 43, 44, 45, 46, 47, 48}});
  const auto generated_corners =
      corners_for_biradially_layered_domains(2, 2, false, true);
  for (size_t i = 0; i < expected_corners.size(); i++) {
    INFO(i);
    CHECK(generated_corners[i] == expected_corners[i]);
  }
  CHECK(generated_corners == expected_corners);
}

SPECTRE_TEST_CASE("Unit.Domain.DomainHelpers.BNSCorners", "[Domain][Unit]") {
  std::vector<std::array<size_t, 8>> expected_corners = expected_bbh_corners();
  expected_corners.push_back(std::array<size_t, 8>{{1, 2, 3, 4, 5, 6, 7, 8}});
  expected_corners.push_back(
      std::array<size_t, 8>{{41, 42, 43, 44, 45, 46, 47, 48}});
  const auto generated_corners =
      corners_for_biradially_layered_domains(2, 2, true, true);
  for (size_t i = 0; i < expected_corners.size(); i++) {
    INFO(i);
    CHECK(generated_corners[i] == expected_corners[i]);
  }
  CHECK(generated_corners == expected_corners);
}

namespace {
void test_vci_1d() {
  VolumeCornerIterator<1> vci{};
  CHECK(vci);
  CHECK(vci() == std::array<Side, 1>{{Side::Lower}});
  CHECK(vci.coords_of_corner() == std::array<double, 1>{{-1.0}});
  CHECK(vci.directions_of_corner() ==
        std::array<Direction<1>, 1>{{Direction<1>::lower_xi()}});
  ++vci;
  CHECK(vci() == std::array<Side, 1>{{Side::Upper}});
  CHECK(vci.coords_of_corner() == std::array<double, 1>{{1.0}});
  CHECK(vci.directions_of_corner() ==
        std::array<Direction<1>, 1>{{Direction<1>::upper_xi()}});
  ++vci;
  CHECK(not vci);

  VolumeCornerIterator<1> vci2{Index<1>{2}, Index<1>{7}};
  CHECK(vci2);
  CHECK(vci2.global_corner_number() == 2);
  ++vci2;
  CHECK(vci2.global_corner_number() == 3);
  ++vci2;
  CHECK(not vci2);

  // Check setup_from_local_corner_number
  VolumeCornerIterator<1> vci3{1};
  CHECK(vci3() == std::array<Side, 1>{{Side::Upper}});
  CHECK(vci3.coords_of_corner() == std::array<double, 1>{{1.0}});
  CHECK(vci3.directions_of_corner() ==
        std::array<Direction<1>, 1>{{Direction<1>::upper_xi()}});
}

void test_vci_2d() {
  VolumeCornerIterator<2> vci{};
  CHECK(vci);
  CHECK(vci() == std::array<Side, 2>{{Side::Lower, Side::Lower}});
  CHECK(vci.coords_of_corner() == std::array<double, 2>{{-1.0, -1.0}});
  CHECK(vci.directions_of_corner() ==
        std::array<Direction<2>, 2>{
            {Direction<2>::lower_xi(), Direction<2>::lower_eta()}});
  ++vci;
  CHECK(vci() == std::array<Side, 2>{{Side::Upper, Side::Lower}});
  CHECK(vci.coords_of_corner() == std::array<double, 2>{{1.0, -1.0}});
  CHECK(vci.directions_of_corner() ==
        std::array<Direction<2>, 2>{
            {Direction<2>::upper_xi(), Direction<2>::lower_eta()}});
  ++vci;
  CHECK(vci() == std::array<Side, 2>{{Side::Lower, Side::Upper}});
  CHECK(vci.coords_of_corner() == std::array<double, 2>{{-1.0, 1.0}});
  CHECK(vci.directions_of_corner() ==
        std::array<Direction<2>, 2>{
            {Direction<2>::lower_xi(), Direction<2>::upper_eta()}});
  ++vci;
  CHECK(vci() == std::array<Side, 2>{{Side::Upper, Side::Upper}});
  CHECK(vci.coords_of_corner() == std::array<double, 2>{{1.0, 1.0}});
  CHECK(vci.directions_of_corner() ==
        std::array<Direction<2>, 2>{
            {Direction<2>::upper_xi(), Direction<2>::upper_eta()}});
  ++vci;
  CHECK(not vci);

  VolumeCornerIterator<2> vci2{Index<2>{1, 2}, Index<2>{3, 4}};
  CHECK(vci2);
  CHECK(vci2.global_corner_number() == 7);
  ++vci2;
  CHECK(vci2.global_corner_number() == 8);
  ++vci2;
  CHECK(vci2.global_corner_number() == 10);
  ++vci2;
  CHECK(vci2.global_corner_number() == 11);
  ++vci2;
  CHECK(not vci2);

  // Check setup_from_local_corner_number
  VolumeCornerIterator<2> vci3{2};
  CHECK(vci3() == std::array<Side, 2>{{Side::Lower, Side::Upper}});
  CHECK(vci3.coords_of_corner() == std::array<double, 2>{{-1.0, 1.0}});
  CHECK(vci3.directions_of_corner() ==
        std::array<Direction<2>, 2>{
            {Direction<2>::lower_xi(), Direction<2>::upper_eta()}});
}

void test_vci_3d() {
  VolumeCornerIterator<3> vci{};
  CHECK(vci);
  CHECK(vci() == std::array<Side, 3>{{Side::Lower, Side::Lower, Side::Lower}});
  CHECK(vci.coords_of_corner() == std::array<double, 3>{{-1.0, -1.0, -1.0}});
  CHECK(vci.directions_of_corner() ==
        std::array<Direction<3>, 3>{{Direction<3>::lower_xi(),
                                     Direction<3>::lower_eta(),
                                     Direction<3>::lower_zeta()}});
  ++vci;
  CHECK(vci() == std::array<Side, 3>{{Side::Upper, Side::Lower, Side::Lower}});
  CHECK(vci.coords_of_corner() == std::array<double, 3>{{1.0, -1.0, -1.0}});
  CHECK(vci.directions_of_corner() ==
        std::array<Direction<3>, 3>{{Direction<3>::upper_xi(),
                                     Direction<3>::lower_eta(),
                                     Direction<3>::lower_zeta()}});
  ++vci;
  CHECK(vci() == std::array<Side, 3>{{Side::Lower, Side::Upper, Side::Lower}});
  CHECK(vci.coords_of_corner() == std::array<double, 3>{{-1.0, 1.0, -1.0}});
  CHECK(vci.directions_of_corner() ==
        std::array<Direction<3>, 3>{{Direction<3>::lower_xi(),
                                     Direction<3>::upper_eta(),
                                     Direction<3>::lower_zeta()}});
  ++vci;
  CHECK(vci() == std::array<Side, 3>{{Side::Upper, Side::Upper, Side::Lower}});
  CHECK(vci.coords_of_corner() == std::array<double, 3>{{1.0, 1.0, -1.0}});
  CHECK(vci.directions_of_corner() ==
        std::array<Direction<3>, 3>{{Direction<3>::upper_xi(),
                                     Direction<3>::upper_eta(),
                                     Direction<3>::lower_zeta()}});
  ++vci;
  CHECK(vci() == std::array<Side, 3>{{Side::Lower, Side::Lower, Side::Upper}});
  CHECK(vci.coords_of_corner() == std::array<double, 3>{{-1.0, -1.0, 1.0}});
  CHECK(vci.directions_of_corner() ==
        std::array<Direction<3>, 3>{{Direction<3>::lower_xi(),
                                     Direction<3>::lower_eta(),
                                     Direction<3>::upper_zeta()}});
  ++vci;
  CHECK(vci() == std::array<Side, 3>{{Side::Upper, Side::Lower, Side::Upper}});
  CHECK(vci.coords_of_corner() == std::array<double, 3>{{1.0, -1.0, 1.0}});
  CHECK(vci.directions_of_corner() ==
        std::array<Direction<3>, 3>{{Direction<3>::upper_xi(),
                                     Direction<3>::lower_eta(),
                                     Direction<3>::upper_zeta()}});
  ++vci;
  CHECK(vci() == std::array<Side, 3>{{Side::Lower, Side::Upper, Side::Upper}});
  CHECK(vci.coords_of_corner() == std::array<double, 3>{{-1.0, 1.0, 1.0}});
  CHECK(vci.directions_of_corner() ==
        std::array<Direction<3>, 3>{{Direction<3>::lower_xi(),
                                     Direction<3>::upper_eta(),
                                     Direction<3>::upper_zeta()}});
  ++vci;
  CHECK(vci() == std::array<Side, 3>{{Side::Upper, Side::Upper, Side::Upper}});
  CHECK(vci.coords_of_corner() == std::array<double, 3>{{1.0, 1.0, 1.0}});
  CHECK(vci.directions_of_corner() ==
        std::array<Direction<3>, 3>{{Direction<3>::upper_xi(),
                                     Direction<3>::upper_eta(),
                                     Direction<3>::upper_zeta()}});
  ++vci;
  CHECK(not vci);

  VolumeCornerIterator<3> vci2{Index<3>{1, 1, 1}, Index<3>{4, 4, 4}};
  CHECK(vci2);
  CHECK(vci2.global_corner_number() == 21);
  ++vci2;
  CHECK(vci2.global_corner_number() == 22);
  ++vci2;
  CHECK(vci2.global_corner_number() == 25);
  ++vci2;
  CHECK(vci2.global_corner_number() == 26);
  ++vci2;
  CHECK(vci2.global_corner_number() == 37);
  ++vci2;
  CHECK(vci2.global_corner_number() == 38);
  ++vci2;
  CHECK(vci2.global_corner_number() == 41);
  ++vci2;
  CHECK(vci2.global_corner_number() == 42);
  ++vci2;
  CHECK(not vci2);

  // Check setup_from_local_corner_number
  VolumeCornerIterator<3> vci3{5};
  CHECK(vci3() == std::array<Side, 3>{{Side::Upper, Side::Lower, Side::Upper}});
  CHECK(vci3.coords_of_corner() == std::array<double, 3>{{1.0, -1.0, 1.0}});
  CHECK(vci3.directions_of_corner() ==
        std::array<Direction<3>, 3>{{Direction<3>::upper_xi(),
                                     Direction<3>::lower_eta(),
                                     Direction<3>::upper_zeta()}});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.DomainHelpers.VolumeCornerIterator",
                  "[Domain][Unit]") {
  test_vci_1d();
  test_vci_2d();
  test_vci_3d();
}

// [[OutputRegex, Index out of range.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Domain.DomainHelpers.VolumeCornerIterator.Assert", "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  auto extents = Index<3>{4, 4, 4};
  auto failed_index_with = Index<3>{2, 3, 4};
  static_cast<void>(VolumeCornerIterator<3>{failed_index_with, extents});
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

SPECTRE_TEST_CASE("Unit.Domain.DomainHelpers.CornersForRectilinearDomains",
                  "[Domain][Unit]") {
  std::vector<std::array<size_t, 2>> corners_for_a_1d_road{
      {{0, 1}}, {{1, 2}}, {{2, 3}}};
  std::vector<std::array<size_t, 4>> corners_for_a_2d_vertical_tower{
      {{0, 1, 2, 3}}, {{2, 3, 4, 5}}, {{4, 5, 6, 7}}};
  std::vector<std::array<size_t, 4>> corners_for_a_2d_horizontal_wall{
      {{0, 1, 4, 5}}, {{1, 2, 5, 6}}, {{2, 3, 6, 7}}};
  std::vector<std::array<size_t, 4>> corners_for_a_2d_field{
      {{0, 1, 3, 4}}, {{1, 2, 4, 5}}, {{3, 4, 6, 7}}, {{4, 5, 7, 8}}};
  // This 2D net of a cube does not have its boundaries identified.
  std::vector<std::array<size_t, 4>> corners_for_a_2d_net_of_a_cube{
      {{1, 2, 5, 6}},    {{5, 6, 9, 10}},    {{8, 9, 12, 13}},
      {{9, 10, 13, 14}}, {{10, 11, 14, 15}}, {{13, 14, 17, 18}}};
  std::vector<std::array<size_t, 8>> corners_for_a_3d_cube{
      {{0, 1, 3, 4, 9, 10, 12, 13}},      {{1, 2, 4, 5, 10, 11, 13, 14}},
      {{3, 4, 6, 7, 12, 13, 15, 16}},     {{4, 5, 7, 8, 13, 14, 16, 17}},
      {{9, 10, 12, 13, 18, 19, 21, 22}},  {{10, 11, 13, 14, 19, 20, 22, 23}},
      {{12, 13, 15, 16, 21, 22, 24, 25}}, {{13, 14, 16, 17, 22, 23, 25, 26}}};
  std::vector<std::array<size_t, 8>> corners_for_a_rubiks_cube_with_hole{
      {{0, 1, 4, 5, 16, 17, 20, 21}},
      {{1, 2, 5, 6, 17, 18, 21, 22}},
      {{2, 3, 6, 7, 18, 19, 22, 23}},
      {{4, 5, 8, 9, 20, 21, 24, 25}},
      {{5, 6, 9, 10, 21, 22, 25, 26}},
      {{6, 7, 10, 11, 22, 23, 26, 27}},
      {{8, 9, 12, 13, 24, 25, 28, 29}},
      {{9, 10, 13, 14, 25, 26, 29, 30}},
      {{10, 11, 14, 15, 26, 27, 30, 31}},

      {{16, 17, 20, 21, 32, 33, 36, 37}},
      {{17, 18, 21, 22, 33, 34, 37, 38}},
      {{18, 19, 22, 23, 34, 35, 38, 39}},
      {{20, 21, 24, 25, 36, 37, 40, 41}},
      /*central block is skipped!*/
      {{22, 23, 26, 27, 38, 39, 42, 43}},
      {{24, 25, 28, 29, 40, 41, 44, 45}},
      {{25, 26, 29, 30, 41, 42, 45, 46}},
      {{26, 27, 30, 31, 42, 43, 46, 47}},

      {{32, 33, 36, 37, 48, 49, 52, 53}},
      {{33, 34, 37, 38, 49, 50, 53, 54}},
      {{34, 35, 38, 39, 50, 51, 54, 55}},
      {{36, 37, 40, 41, 52, 53, 56, 57}},
      {{37, 38, 41, 42, 53, 54, 57, 58}},
      {{38, 39, 42, 43, 54, 55, 58, 59}},
      {{40, 41, 44, 45, 56, 57, 60, 61}},
      {{41, 42, 45, 46, 57, 58, 61, 62}},
      {{42, 43, 46, 47, 58, 59, 62, 63}}};

  CHECK(corners_for_rectilinear_domains(Index<1>{3}) == corners_for_a_1d_road);
  CHECK(corners_for_rectilinear_domains(Index<2>{1, 3}) ==
        corners_for_a_2d_vertical_tower);
  CHECK(corners_for_rectilinear_domains(Index<2>{3, 1}) ==
        corners_for_a_2d_horizontal_wall);
  CHECK(corners_for_rectilinear_domains(Index<2>{2, 2}) ==
        corners_for_a_2d_field);
  CHECK(corners_for_rectilinear_domains(
            Index<2>{3, 4},
            std::vector<Index<2>>{Index<2>{0, 0}, Index<2>{2, 0},
                                  Index<2>{0, 1}, Index<2>{2, 1},
                                  Index<2>{0, 3}, Index<2>{2, 3}}) ==
        corners_for_a_2d_net_of_a_cube);
  CHECK(corners_for_rectilinear_domains(Index<3>{2, 2, 2}) ==
        corners_for_a_3d_cube);
  CHECK(corners_for_rectilinear_domains(
            Index<3>{3, 3, 3}, std::vector<Index<3>>{Index<3>{1, 1, 1}}) ==
        corners_for_a_rubiks_cube_with_hole);
}

SPECTRE_TEST_CASE("Unit.Domain.DomainHelpers.DiscreteRotation.CornerNumbers",
                  "[Domain][Unit]") {
  CHECK(std::array<size_t, 2>{{0, 1}} ==
        discrete_rotation(OrientationMap<1>{std::array<Direction<1>, 1>{
                              {Direction<1>::upper_xi()}}},
                          std::array<size_t, 2>{{0, 1}}));
  CHECK(std::array<size_t, 2>{{1, 0}} ==
        discrete_rotation(OrientationMap<1>{std::array<Direction<1>, 1>{
                              {Direction<1>::lower_xi()}}},
                          std::array<size_t, 2>{{0, 1}}));

  CHECK(std::array<size_t, 4>{{1, 4, 0, 3}} ==
        discrete_rotation(
            OrientationMap<2>(std::array<Direction<2>, 2>{
                {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}),
            std::array<size_t, 4>{{0, 1, 3, 4}}));
  CHECK(std::array<size_t, 4>{{4, 0, 5, 1}} ==
        discrete_rotation(
            OrientationMap<2>(std::array<Direction<2>, 2>{
                {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}),
            std::array<size_t, 4>{{0, 1, 4, 5}}));
  CHECK(std::array<size_t, 4>{{3, 1, 2, 0}} ==
        discrete_rotation(
            OrientationMap<2>(std::array<Direction<2>, 2>{
                {Direction<2>::lower_eta(), Direction<2>::lower_xi()}}),
            std::array<size_t, 4>{{0, 1, 2, 3}}));

  CHECK(std::array<size_t, 8>{{9, 0, 12, 3, 10, 1, 13, 4}} ==
        discrete_rotation(
            OrientationMap<3>(std::array<Direction<3>, 3>{
                {Direction<3>::lower_zeta(), Direction<3>::upper_eta(),
                 Direction<3>::upper_xi()}}),
            std::array<size_t, 8>{{0, 1, 3, 4, 9, 10, 12, 13}}));
  CHECK(std::array<size_t, 8>{{10, 13, 9, 12, 1, 4, 0, 3}} ==
        discrete_rotation(
            OrientationMap<3>(std::array<Direction<3>, 3>{
                {Direction<3>::upper_eta(), Direction<3>::lower_xi(),
                 Direction<3>::lower_zeta()}}),
            std::array<size_t, 8>{{0, 1, 3, 4, 9, 10, 12, 13}}));
  CHECK(std::array<size_t, 8>{{12, 3, 13, 4, 9, 0, 10, 1}} ==
        discrete_rotation(
            OrientationMap<3>(std::array<Direction<3>, 3>{
                {Direction<3>::lower_zeta(), Direction<3>::upper_xi(),
                 Direction<3>::lower_eta()}}),
            std::array<size_t, 8>{{0, 1, 3, 4, 9, 10, 12, 13}}));
}

SPECTRE_TEST_CASE("Unit.Domain.DomainHelpers.MapsForRectilinearDomains",
                  "[Domain][Unit]") {
  using Affine = CoordinateMaps::Affine;
  using Affine2D = CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  using Equiangular = CoordinateMaps::Equiangular;
  using Equiangular2D =
      CoordinateMaps::ProductOf2Maps<Equiangular, Equiangular>;
  using Equiangular3D =
      CoordinateMaps::ProductOf3Maps<Equiangular, Equiangular, Equiangular>;

  const std::vector<
      std::unique_ptr<CoordinateMapBase<Frame::Logical, Frame::Inertial, 1>>>
      affine_maps_1d = maps_for_rectilinear_domains<Frame::Inertial>(
          Index<1>{3},
          std::array<std::vector<double>, 1>{{{0.0, 0.5, 1.7, 2.0}}},
          {Index<1>{0}}, false);
  const auto expected_affine_maps_1d =
      make_vector_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Affine{-1., 1., 0.5, 1.7}, Affine{-1., 1., 1.7, 2.0});
  for (size_t i = 0; i < affine_maps_1d.size(); i++) {
    CHECK(*affine_maps_1d[i] == *expected_affine_maps_1d[i]);
  }

  const std::vector<
      std::unique_ptr<CoordinateMapBase<Frame::Logical, Frame::Inertial, 1>>>
      equiangular_maps_1d = maps_for_rectilinear_domains<Frame::Inertial>(
          Index<1>{3},
          std::array<std::vector<double>, 1>{{{0.0, 0.5, 1.7, 2.0}}},
          {Index<1>{1}}, true);
  const auto expected_equiangular_maps_1d =
      make_vector_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Equiangular{-1., 1., 0.0, 0.5}, Equiangular{-1., 1., 1.7, 2.0});
  for (size_t i = 0; i < equiangular_maps_1d.size(); i++) {
    CHECK(*equiangular_maps_1d[i] == *expected_equiangular_maps_1d[i]);
  }

  const std::vector<
      std::unique_ptr<CoordinateMapBase<Frame::Logical, Frame::Inertial, 2>>>
      affine_maps_2d = maps_for_rectilinear_domains<Frame::Inertial>(
          Index<2>{3, 2},
          std::array<std::vector<double>, 2>{
              {{0.0, 0.5, 1.7, 2.0}, {0.0, 1.0, 2.0}}},
          {Index<2>{}}, false);
  const auto expected_affine_maps_2d =
      make_vector_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Affine2D{Affine{-1., 1., 0.0, 0.5}, Affine{-1., 1., 0.0, 1.0}},
          Affine2D{Affine{-1., 1., 0.5, 1.7}, Affine{-1., 1., 0.0, 1.0}},
          Affine2D{Affine{-1., 1., 1.7, 2.0}, Affine{-1., 1., 0.0, 1.0}},
          Affine2D{Affine{-1., 1., 0.0, 0.5}, Affine{-1., 1., 1.0, 2.0}},
          Affine2D{Affine{-1., 1., 0.5, 1.7}, Affine{-1., 1., 1.0, 2.0}},
          Affine2D{Affine{-1., 1., 1.7, 2.0}, Affine{-1., 1., 1.0, 2.0}});
  for (size_t i = 0; i < affine_maps_2d.size(); i++) {
    CHECK(*affine_maps_2d[i] == *expected_affine_maps_2d[i]);
  }

  const std::vector<
      std::unique_ptr<CoordinateMapBase<Frame::Logical, Frame::Inertial, 2>>>
      equiangular_maps_2d = maps_for_rectilinear_domains<Frame::Inertial>(
          Index<2>{3, 2},
          std::array<std::vector<double>, 2>{
              {{0.0, 0.5, 1.7, 2.0}, {0.0, 1.0, 2.0}}},
          {Index<2>{2, 1}}, true);
  const auto expected_equiangular_maps_2d =
      make_vector_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Equiangular2D{Equiangular{-1., 1., 0.0, 0.5},
                        Equiangular{-1., 1., 0.0, 1.0}},
          Equiangular2D{Equiangular{-1., 1., 0.5, 1.7},
                        Equiangular{-1., 1., 0.0, 1.0}},
          Equiangular2D{Equiangular{-1., 1., 1.7, 2.0},
                        Equiangular{-1., 1., 0.0, 1.0}},
          Equiangular2D{Equiangular{-1., 1., 0.0, 0.5},
                        Equiangular{-1., 1., 1.0, 2.0}},
          Equiangular2D{Equiangular{-1., 1., 0.5, 1.7},
                        Equiangular{-1., 1., 1.0, 2.0}});
  for (size_t i = 0; i < equiangular_maps_2d.size(); i++) {
    CHECK(*equiangular_maps_2d[i] == *expected_equiangular_maps_2d[i]);
  }

  const std::vector<
      std::unique_ptr<CoordinateMapBase<Frame::Logical, Frame::Inertial, 3>>>
      affine_maps_3d = maps_for_rectilinear_domains<Frame::Inertial>(
          Index<3>{2, 2, 1},
          std::array<std::vector<double>, 3>{
              {{0.0, 0.5, 2.0}, {0.0, 1.0, 2.0}, {-0.4, 0.3}}},
          {Index<3>{}}, false);
  const auto expected_affine_maps_3d =
      make_vector_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Affine3D{Affine{-1., 1., 0.0, 0.5}, Affine{-1., 1., 0.0, 1.0},
                   Affine{-1., 1., -0.4, 0.3}},
          Affine3D{Affine{-1., 1., 0.5, 2.0}, Affine{-1., 1., 0.0, 1.0},
                   Affine{-1., 1., -0.4, 0.3}},
          Affine3D{Affine{-1., 1., 0.0, 0.5}, Affine{-1., 1., 1.0, 2.0},
                   Affine{-1., 1., -0.4, 0.3}},
          Affine3D{Affine{-1., 1., 0.5, 2.0}, Affine{-1., 1., 1.0, 2.0},
                   Affine{-1., 1., -0.4, 0.3}});
  for (size_t i = 0; i < affine_maps_3d.size(); i++) {
    CHECK(*affine_maps_3d[i] == *expected_affine_maps_3d[i]);
  }

  const std::vector<
      std::unique_ptr<CoordinateMapBase<Frame::Logical, Frame::Inertial, 3>>>
      equiangular_maps_3d = maps_for_rectilinear_domains<Frame::Inertial>(
          Index<3>{2, 2, 1},
          std::array<std::vector<double>, 3>{
              {{0.0, 0.5, 2.0}, {0.0, 1.0, 2.0}, {-0.4, 0.3}}},
          {Index<3>{0, 0, 0}}, true);
  const auto expected_equiangular_maps_3d =
      make_vector_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Equiangular3D{Equiangular{-1., 1., 0.5, 2.0},
                        Equiangular{-1., 1., 0.0, 1.0},
                        Equiangular{-1., 1., -0.4, 0.3}},
          Equiangular3D{Equiangular{-1., 1., 0.0, 0.5},
                        Equiangular{-1., 1., 1.0, 2.0},
                        Equiangular{-1., 1., -0.4, 0.3}},
          Equiangular3D{Equiangular{-1., 1., 0.5, 2.0},
                        Equiangular{-1., 1., 1.0, 2.0},
                        Equiangular{-1., 1., -0.4, 0.3}});
  for (size_t i = 0; i < equiangular_maps_3d.size(); i++) {
    CHECK(*equiangular_maps_3d[i] == *expected_equiangular_maps_3d[i]);
  }
}
