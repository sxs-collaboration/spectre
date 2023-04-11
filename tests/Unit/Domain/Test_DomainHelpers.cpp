// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <unordered_set>
#include <vector>

#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/EquatorialCompression.hpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/Frustum.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/Side.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/StdHelpers.hpp"

namespace domain {
namespace {

void test_periodic_same_block() {
  const std::vector<std::array<size_t, 8>> corners_of_all_blocks{
      {{0, 1, 2, 3, 4, 5, 6, 7}}, {{8, 9, 10, 11, 0, 1, 2, 3}}};
  std::vector<DirectionMap<3, BlockNeighbor<3>>> neighbors_of_all_blocks;
  set_internal_boundaries<3>(&neighbors_of_all_blocks, corners_of_all_blocks);

  const OrientationMap<3> aligned{};
  CHECK(neighbors_of_all_blocks[0][Direction<3>::lower_zeta()].orientation() ==
        aligned);

  const PairOfFaces x_faces{{1, 3, 5, 7}, {0, 2, 4, 6}};

  const std::vector<PairOfFaces> identifications{x_faces};
  set_identified_boundaries<3>(identifications, corners_of_all_blocks,
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

void test_periodic_different_blocks() {
  const std::vector<std::array<size_t, 8>> corners_of_all_blocks{
      {{0, 1, 2, 3, 4, 5, 6, 7}}, {{8, 9, 10, 11, 0, 1, 2, 3}}};
  std::vector<DirectionMap<3, BlockNeighbor<3>>> neighbors_of_all_blocks;
  set_internal_boundaries<3>(&neighbors_of_all_blocks, corners_of_all_blocks);

  const OrientationMap<3> aligned{};
  CHECK(neighbors_of_all_blocks[0][Direction<3>::lower_zeta()].orientation() ==
        aligned);

  const PairOfFaces x_faces_on_different_blocks{{1, 3, 5, 7}, {8, 10, 0, 2}};

  const std::vector<PairOfFaces> identifications{x_faces_on_different_blocks};
  set_identified_boundaries<3>(identifications, corners_of_all_blocks,
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

std::vector<CoordinateMaps::Wedge<3>> test_wedge_map_generation(
    double inner_radius, double outer_radius, double inner_sphericity,
    double outer_sphericity, bool use_equiangular_map,
    bool use_half_wedges = false) {
  using Wedge3DMap = CoordinateMaps::Wedge<3>;

  if (use_half_wedges) {
    using Halves = Wedge3DMap::WedgeHalves;
    return make_vector(
        Wedge3DMap{inner_radius, outer_radius, inner_sphericity,
                   outer_sphericity, OrientationMap<3>{}, use_equiangular_map,
                   Halves::LowerOnly},
        Wedge3DMap{inner_radius, outer_radius, inner_sphericity,
                   outer_sphericity, OrientationMap<3>{}, use_equiangular_map,
                   Halves::UpperOnly},
        Wedge3DMap{inner_radius, outer_radius, inner_sphericity,
                   outer_sphericity,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_xi(), Direction<3>::lower_eta(),
                        Direction<3>::lower_zeta()}}},
                   use_equiangular_map, Halves::LowerOnly},
        Wedge3DMap{inner_radius, outer_radius, inner_sphericity,
                   outer_sphericity,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_xi(), Direction<3>::lower_eta(),
                        Direction<3>::lower_zeta()}}},
                   use_equiangular_map, Halves::UpperOnly},
        Wedge3DMap{inner_radius, outer_radius, inner_sphericity,
                   outer_sphericity,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
                        Direction<3>::lower_eta()}}},
                   use_equiangular_map, Halves::LowerOnly},
        Wedge3DMap{inner_radius, outer_radius, inner_sphericity,
                   outer_sphericity,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
                        Direction<3>::lower_eta()}}},
                   use_equiangular_map, Halves::UpperOnly},
        Wedge3DMap{inner_radius, outer_radius, inner_sphericity,
                   outer_sphericity,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_xi(), Direction<3>::lower_zeta(),
                        Direction<3>::upper_eta()}}},
                   use_equiangular_map, Halves::LowerOnly},
        Wedge3DMap{inner_radius, outer_radius, inner_sphericity,
                   outer_sphericity,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_xi(), Direction<3>::lower_zeta(),
                        Direction<3>::upper_eta()}}},
                   use_equiangular_map, Halves::UpperOnly},
        Wedge3DMap{inner_radius, outer_radius, inner_sphericity,
                   outer_sphericity,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_zeta(), Direction<3>::upper_xi(),
                        Direction<3>::upper_eta()}}},
                   use_equiangular_map},
        Wedge3DMap{inner_radius, outer_radius, inner_sphericity,
                   outer_sphericity,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::lower_zeta(), Direction<3>::lower_xi(),
                        Direction<3>::upper_eta()}}},
                   use_equiangular_map});
  }

  return make_vector(
      Wedge3DMap{inner_radius, outer_radius, inner_sphericity, outer_sphericity,
                 OrientationMap<3>{}, use_equiangular_map},
      Wedge3DMap{inner_radius, outer_radius, inner_sphericity, outer_sphericity,
                 OrientationMap<3>{std::array<Direction<3>, 3>{
                     {Direction<3>::upper_xi(), Direction<3>::lower_eta(),
                      Direction<3>::lower_zeta()}}},
                 use_equiangular_map},
      Wedge3DMap{inner_radius, outer_radius, inner_sphericity, outer_sphericity,
                 OrientationMap<3>{std::array<Direction<3>, 3>{
                     {Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
                      Direction<3>::lower_eta()}}},
                 use_equiangular_map},
      Wedge3DMap{inner_radius, outer_radius, inner_sphericity, outer_sphericity,
                 OrientationMap<3>{std::array<Direction<3>, 3>{
                     {Direction<3>::upper_xi(), Direction<3>::lower_zeta(),
                      Direction<3>::upper_eta()}}},
                 use_equiangular_map},
      Wedge3DMap{inner_radius, outer_radius, inner_sphericity, outer_sphericity,
                 OrientationMap<3>{std::array<Direction<3>, 3>{
                     {Direction<3>::upper_zeta(), Direction<3>::upper_xi(),
                      Direction<3>::upper_eta()}}},
                 use_equiangular_map},
      Wedge3DMap{inner_radius, outer_radius, inner_sphericity, outer_sphericity,
                 OrientationMap<3>{std::array<Direction<3>, 3>{
                     {Direction<3>::lower_zeta(), Direction<3>::lower_xi(),
                      Direction<3>::upper_eta()}}},
                 use_equiangular_map});
}

void test_wedge_map_generation_against_domain_helpers(
    double inner_radius, double outer_radius, double inner_sphericity,
    double outer_sphericity, bool use_equiangular_map,
    bool use_half_wedges = false) {
  const auto expected_coord_maps = test_wedge_map_generation(
      inner_radius, outer_radius, inner_sphericity, outer_sphericity,
      use_equiangular_map, use_half_wedges);
  const auto maps = sph_wedge_coordinate_maps(
      inner_radius, outer_radius, inner_sphericity, outer_sphericity,
      use_equiangular_map, use_half_wedges);
  CHECK(maps == expected_coord_maps);
}

void test_wedge_errors() {
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      ([]() {
        const double inner_radius = 0.5;
        const double outer_radius = 2.0;
        const double inner_sphericity = 1.0;
        const double outer_sphericity = 1.0;
        const bool use_equiangular_map = true;
        const bool use_half_wedges = true;
        const std::vector<domain::CoordinateMaps::Distribution>
            radial_distribution{
                domain::CoordinateMaps::Distribution::Logarithmic};
        const ShellWedges which_wedges = ShellWedges::FourOnEquator;
        static_cast<void>(sph_wedge_coordinate_maps(
            inner_radius, outer_radius, inner_sphericity, outer_sphericity,
            use_equiangular_map, use_half_wedges, {}, radial_distribution,
            which_wedges));
      }()),
      Catch::Contains("If we are using half wedges we must also be using "
                      "ShellWedges::All."));
#endif
}

void test_six_wedge_directions_equiangular() {
  INFO("Default six wedge directions equiangular");
  const double inner_radius = 1.2;
  const double outer_radius = 2.7;
  const double inner_sphericity = 0.8;
  const double outer_sphericity = 0.6;
  const bool use_equiangular_map = true;
  test_wedge_map_generation_against_domain_helpers(
      inner_radius, outer_radius, inner_sphericity, outer_sphericity,
      use_equiangular_map);
}

void test_six_wedge_directions_equidistant() {
  INFO("Default six wedge directions equidistant");
  const double inner_radius = 0.8;
  const double outer_radius = 7.1;
  const double inner_sphericity = 0.2;
  const double outer_sphericity = 0.4;
  const bool use_equiangular_map = false;
  test_wedge_map_generation_against_domain_helpers(
      inner_radius, outer_radius, inner_sphericity, outer_sphericity,
      use_equiangular_map);
}

void test_ten_wedge_directions_equiangular() {
  INFO("Ten wedge directions equiangular");
  const double inner_radius = 0.2;
  const double outer_radius = 2.2;
  const double inner_sphericity = 0.0;
  const double outer_sphericity = 1.0;
  const bool use_equiangular_map = true;
  const bool use_half_wedges = true;
  test_wedge_map_generation_against_domain_helpers(
      inner_radius, outer_radius, inner_sphericity, outer_sphericity,
      use_equiangular_map, use_half_wedges);
}

void test_ten_wedge_directions_equidistant() {
  INFO("Ten wedge directions equidistant");
  const double inner_radius = 0.2;
  const double outer_radius = 29.2;
  const double inner_sphericity = 0.01;
  const double outer_sphericity = 0.99;
  const bool use_equiangular_map = false;
  const bool use_half_wedges = true;
  test_wedge_map_generation_against_domain_helpers(
      inner_radius, outer_radius, inner_sphericity, outer_sphericity,
      use_equiangular_map, use_half_wedges);
}

void test_wedge_map_generation() {
  test_six_wedge_directions_equiangular();
  test_six_wedge_directions_equidistant();
  test_ten_wedge_directions_equiangular();
  test_ten_wedge_directions_equidistant();
}

void test_all_frustum_directions() {
  using FrustumMap = CoordinateMaps::Frustum;
  // half of the length of the inner cube in the binary compact object domain:
  const double lower = 1.7;
  // half of the length of the outer cube in the binary compact object domain:
  const double top = 5.2;
  const std::array<double, 3> origin_preimage{{0.2, 0.3, -0.1}};
  const auto displacement1 =
      discrete_rotation(OrientationMap<3>{}.inverse_map(), origin_preimage);
  const auto displacement2 =
      discrete_rotation(OrientationMap<3>{}.inverse_map(), origin_preimage);
  const auto displacement3 = discrete_rotation(
      OrientationMap<3>{
          std::array<Direction<3>, 3>{{Direction<3>::upper_xi(),
                                       Direction<3>::lower_eta(),
                                       Direction<3>::lower_zeta()}}}
          .inverse_map(),
      origin_preimage);
  const auto displacement4 = discrete_rotation(
      OrientationMap<3>{
          std::array<Direction<3>, 3>{{Direction<3>::upper_xi(),
                                       Direction<3>::lower_eta(),
                                       Direction<3>::lower_zeta()}}}
          .inverse_map(),
      origin_preimage);
  const auto displacement5 = discrete_rotation(
      OrientationMap<3>{
          std::array<Direction<3>, 3>{{Direction<3>::upper_xi(),
                                       Direction<3>::upper_zeta(),
                                       Direction<3>::lower_eta()}}}
          .inverse_map(),
      origin_preimage);
  const auto displacement6 = discrete_rotation(
      OrientationMap<3>{
          std::array<Direction<3>, 3>{{Direction<3>::upper_xi(),
                                       Direction<3>::upper_zeta(),
                                       Direction<3>::lower_eta()}}}
          .inverse_map(),
      origin_preimage);
  const auto displacement7 = discrete_rotation(
      OrientationMap<3>{
          std::array<Direction<3>, 3>{{Direction<3>::upper_xi(),
                                       Direction<3>::lower_zeta(),
                                       Direction<3>::upper_eta()}}}
          .inverse_map(),
      origin_preimage);
  const auto displacement8 = discrete_rotation(
      OrientationMap<3>{
          std::array<Direction<3>, 3>{{Direction<3>::upper_xi(),
                                       Direction<3>::lower_zeta(),
                                       Direction<3>::upper_eta()}}}
          .inverse_map(),
      origin_preimage);
  const auto displacement9 = discrete_rotation(
      OrientationMap<3>{
          std::array<Direction<3>, 3>{{Direction<3>::upper_zeta(),
                                       Direction<3>::upper_xi(),
                                       Direction<3>::upper_eta()}}}
          .inverse_map(),
      origin_preimage);
  const auto displacement10 = discrete_rotation(
      OrientationMap<3>{
          std::array<Direction<3>, 3>{{Direction<3>::lower_zeta(),
                                       Direction<3>::lower_xi(),
                                       Direction<3>::upper_eta()}}}
          .inverse_map(),
      origin_preimage);

  const double projective_scale_factor = 0.3;
  const double sphericity = 0.;
  for (const bool use_equiangular_map : {true, false}) {
    const double stretch = tan(0.5 * M_PI_2);
    const auto expected_coord_maps = make_vector(
        FrustumMap{
            {{{{-2.0 * lower - displacement1[0], -lower - displacement1[1]}},
              {{-displacement1[0], lower - displacement1[1]}},
              {{-stretch * top, -top}},
              {{0.0, top}}}},
            lower - displacement1[2],
            top,
            OrientationMap<3>{},
            use_equiangular_map,
            projective_scale_factor,
            false,
            sphericity,
            -1.},
        FrustumMap{
            {{{{-displacement2[0], -lower - displacement2[1]}},
              {{2.0 * lower - displacement2[0], lower - displacement2[1]}},
              {{0.0, -top}},
              {{stretch * top, top}}}},
            lower - displacement2[2],
            top,
            OrientationMap<3>{},
            use_equiangular_map,
            projective_scale_factor,
            false,
            sphericity,
            1.},
        FrustumMap{
            {{{{-2.0 * lower - displacement3[0], -lower - displacement3[1]}},
              {{-displacement3[0], lower - displacement3[1]}},
              {{-stretch * top, -top}},
              {{0.0, top}}}},
            lower - displacement3[2],
            top,
            OrientationMap<3>{std::array<Direction<3>, 3>{
                {Direction<3>::upper_xi(), Direction<3>::lower_eta(),
                 Direction<3>::lower_zeta()}}},
            use_equiangular_map,
            projective_scale_factor,
            false,
            sphericity,
            -1.},
        FrustumMap{
            {{{{-displacement4[0], -lower - displacement4[1]}},
              {{2.0 * lower - displacement4[0], lower - displacement4[1]}},
              {{0.0, -top}},
              {{stretch * top, top}}}},
            lower - displacement4[2],
            top,
            OrientationMap<3>{std::array<Direction<3>, 3>{
                {Direction<3>::upper_xi(), Direction<3>::lower_eta(),
                 Direction<3>::lower_zeta()}}},
            use_equiangular_map,
            projective_scale_factor,
            false,
            sphericity,
            1.},
        FrustumMap{
            {{{{-2.0 * lower - displacement5[0], -lower - displacement5[1]}},
              {{-displacement5[0], lower - displacement5[1]}},
              {{-stretch * top, -top}},
              {{0.0, top}}}},
            lower - displacement5[2],
            top,
            OrientationMap<3>{std::array<Direction<3>, 3>{
                {Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
                 Direction<3>::lower_eta()}}},
            use_equiangular_map,
            projective_scale_factor,
            false,
            sphericity,
            -1.},
        FrustumMap{
            {{{{-displacement6[0], -lower - displacement6[1]}},
              {{2.0 * lower - displacement6[0], lower - displacement6[1]}},
              {{0.0, -top}},
              {{stretch * top, top}}}},
            lower - displacement6[2],
            top,
            OrientationMap<3>{std::array<Direction<3>, 3>{
                {Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
                 Direction<3>::lower_eta()}}},
            use_equiangular_map,
            projective_scale_factor,
            false,
            sphericity,
            1.},
        FrustumMap{
            {{{{-2.0 * lower - displacement7[0], -lower - displacement7[1]}},
              {{-displacement7[0], lower - displacement7[1]}},
              {{-stretch * top, -top}},
              {{0.0, top}}}},
            lower - displacement7[2],
            top,
            OrientationMap<3>{std::array<Direction<3>, 3>{
                {Direction<3>::upper_xi(), Direction<3>::lower_zeta(),
                 Direction<3>::upper_eta()}}},
            use_equiangular_map,
            projective_scale_factor,
            false,
            sphericity,
            -1.},
        FrustumMap{
            {{{{-displacement8[0], -lower - displacement8[1]}},
              {{2.0 * lower - displacement8[0], lower - displacement8[1]}},
              {{0.0, -top}},
              {{stretch * top, top}}}},
            lower - displacement8[2],
            top,
            OrientationMap<3>{std::array<Direction<3>, 3>{
                {Direction<3>::upper_xi(), Direction<3>::lower_zeta(),
                 Direction<3>::upper_eta()}}},
            use_equiangular_map,
            projective_scale_factor,
            false,
            sphericity,
            1.},
        // Frustum on right half in the +x direction
        FrustumMap{{{{{-lower - displacement9[0], -lower - displacement9[1]}},
                     {{lower - displacement9[0], lower - displacement9[1]}},
                     {{-top, -top}},
                     {{top, top}}}},
                   2.0 * lower - displacement9[2],
                   stretch * top,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::upper_zeta(), Direction<3>::upper_xi(),
                        Direction<3>::upper_eta()}}},
                   use_equiangular_map,
                   projective_scale_factor},
        // Frustum on left half in the -x direction
        FrustumMap{{{{{-lower - displacement10[0], -lower - displacement10[1]}},
                     {{lower - displacement10[0], lower - displacement10[1]}},
                     {{-top, -top}},
                     {{top, top}}}},
                   2.0 * lower - displacement10[2],
                   stretch * top,
                   OrientationMap<3>{std::array<Direction<3>, 3>{
                       {Direction<3>::lower_zeta(), Direction<3>::lower_xi(),
                        Direction<3>::upper_eta()}}},
                   use_equiangular_map,
                   projective_scale_factor});

    const auto maps = frustum_coordinate_maps(
        2.0 * lower, 2.0 * top, use_equiangular_map, origin_preimage, 0.3);
    CHECK(maps == expected_coord_maps);
  }
}

void test_frustrum_errors() {
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      ([]() {
        const double length_inner_cube = 0.9;
        const double length_outer_cube = 1.5;
        const bool use_equiangular_map = true;
        const std::array<double, 3> origin_preimage = {{0.0, 0.0, 0.0}};
        static_cast<void>(
            frustum_coordinate_maps(length_inner_cube, length_outer_cube,
                                    use_equiangular_map, origin_preimage));
      }()),
      Catch::Contains("The outer cube is too small! The inner cubes will "
                      "pierce the surface of the outer cube."));

  CHECK_THROWS_WITH(
      ([]() {
        const double length_inner_cube = 1.0;
        const double length_outer_cube = 3.0;
        const bool use_equiangular_map = true;
        const std::array<double, 3> origin_preimage = {{0.6, 0.0, 0.0}};
        static_cast<void>(
            frustum_coordinate_maps(length_inner_cube, length_outer_cube,
                                    use_equiangular_map, origin_preimage));
      }()),
      Catch::Contains("The current choice for `origin_preimage` results in the "
                      "inner cubes piercing the surface of the outer cube."));
#endif
}

void test_shell_graph() {
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

void test_sphere_graph() {
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

void test_bbh_corners() {
  const auto generated_corners =
      corners_for_biradially_layered_domains(2, 2, false, false);
  for (size_t i = 0; i < expected_bbh_corners().size(); i++) {
    INFO(i);
    CHECK(generated_corners[i] == expected_bbh_corners()[i]);
  }
  CHECK(generated_corners == expected_bbh_corners());
}

void test_nsbh_corners() {
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

void test_bhns_corners() {
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

void test_bns_corners() {
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

void test_volume_corner_iterator() {
  test_vci_1d();
  test_vci_2d();
  test_vci_3d();
}

void test_fci_1d() {
  FaceCornerIterator<1> fci{Direction<1>::upper_xi()};
  CHECK(fci);
  CHECK(fci.volume_index() == 1);
  CHECK(fci.face_index() == 0);
  ++fci;
  CHECK(not fci);

  FaceCornerIterator<1> fci2{Direction<1>::lower_xi()};
  CHECK(fci2);
  CHECK(fci2.volume_index() == 0);
  CHECK(fci2.face_index() == 0);
  ++fci2;
  CHECK(not fci2);
}

void test_fci_2d() {
  FaceCornerIterator<2> fci{Direction<2>::upper_xi()};
  CHECK(fci);
  CHECK(fci.volume_index() == 1);
  CHECK(fci.face_index() == 0);
  ++fci;
  CHECK(fci.volume_index() == 3);
  CHECK(fci.face_index() == 1);
  ++fci;
  CHECK(not fci);

  FaceCornerIterator<2> fci2{Direction<2>::lower_xi()};
  CHECK(fci2);
  CHECK(fci2.volume_index() == 0);
  CHECK(fci2.face_index() == 0);
  ++fci2;
  CHECK(fci2.volume_index() == 2);
  CHECK(fci2.face_index() == 1);
  ++fci2;
  CHECK(not fci2);

  FaceCornerIterator<2> fci3{Direction<2>::upper_eta()};
  CHECK(fci3);
  CHECK(fci3.volume_index() == 2);
  CHECK(fci3.face_index() == 0);
  ++fci3;
  CHECK(fci3.volume_index() == 3);
  CHECK(fci3.face_index() == 1);
  ++fci3;
  CHECK(not fci3);

  FaceCornerIterator<2> fci4{Direction<2>::lower_eta()};
  CHECK(fci4);
  CHECK(fci4.volume_index() == 0);
  CHECK(fci4.face_index() == 0);
  ++fci4;
  CHECK(fci4.volume_index() == 1);
  CHECK(fci4.face_index() == 1);
  ++fci4;
  CHECK(not fci4);
}
void test_fci_3d() {
  FaceCornerIterator<3> fci{Direction<3>::upper_xi()};
  CHECK(fci);
  CHECK(fci.volume_index() == 1);
  CHECK(fci.face_index() == 0);
  ++fci;
  CHECK(fci.volume_index() == 3);
  CHECK(fci.face_index() == 1);
  ++fci;
  CHECK(fci.volume_index() == 5);
  CHECK(fci.face_index() == 2);
  ++fci;
  CHECK(fci.volume_index() == 7);
  CHECK(fci.face_index() == 3);
  ++fci;
  CHECK(not fci);

  FaceCornerIterator<3> fci2{Direction<3>::lower_xi()};
  CHECK(fci2);
  CHECK(fci2.volume_index() == 0);
  CHECK(fci2.face_index() == 0);
  ++fci2;
  CHECK(fci2.volume_index() == 2);
  CHECK(fci2.face_index() == 1);
  ++fci2;
  CHECK(fci2.volume_index() == 4);
  CHECK(fci2.face_index() == 2);
  ++fci2;
  CHECK(fci2.volume_index() == 6);
  CHECK(fci2.face_index() == 3);
  ++fci2;
  CHECK(not fci2);

  FaceCornerIterator<3> fci3{Direction<3>::upper_eta()};
  CHECK(fci3);
  CHECK(fci3.volume_index() == 2);
  CHECK(fci3.face_index() == 0);
  ++fci3;
  CHECK(fci3.volume_index() == 3);
  CHECK(fci3.face_index() == 1);
  ++fci3;
  CHECK(fci3.volume_index() == 6);
  CHECK(fci3.face_index() == 2);
  ++fci3;
  CHECK(fci3.volume_index() == 7);
  CHECK(fci3.face_index() == 3);
  ++fci3;
  CHECK(not fci3);

  FaceCornerIterator<3> fci4{Direction<3>::lower_eta()};
  CHECK(fci4);
  CHECK(fci4.volume_index() == 0);
  CHECK(fci4.face_index() == 0);
  ++fci4;
  CHECK(fci4.volume_index() == 1);
  CHECK(fci4.face_index() == 1);
  ++fci4;
  CHECK(fci4.volume_index() == 4);
  CHECK(fci4.face_index() == 2);
  ++fci4;
  CHECK(fci4.volume_index() == 5);
  CHECK(fci4.face_index() == 3);
  ++fci4;
  CHECK(not fci4);

  FaceCornerIterator<3> fci5{Direction<3>::upper_zeta()};
  CHECK(fci5);
  CHECK(fci5.volume_index() == 4);
  CHECK(fci5.face_index() == 0);
  ++fci5;
  CHECK(fci5.volume_index() == 5);
  CHECK(fci5.face_index() == 1);
  ++fci5;
  CHECK(fci5.volume_index() == 6);
  CHECK(fci5.face_index() == 2);
  ++fci5;
  CHECK(fci5.volume_index() == 7);
  CHECK(fci5.face_index() == 3);
  ++fci5;
  CHECK(not fci5);

  FaceCornerIterator<3> fci6{Direction<3>::lower_zeta()};
  CHECK(fci6);
  CHECK(fci6.volume_index() == 0);
  CHECK(fci6.face_index() == 0);
  ++fci6;
  CHECK(fci6.volume_index() == 1);
  CHECK(fci6.face_index() == 1);
  ++fci6;
  CHECK(fci6.volume_index() == 2);
  CHECK(fci6.face_index() == 2);
  ++fci6;
  CHECK(fci6.volume_index() == 3);
  CHECK(fci6.face_index() == 3);
  ++fci6;
  CHECK(not fci6);
}

void test_face_corner_iterator() {
  test_fci_1d();
  test_fci_2d();
  test_fci_3d();
}

void test_indices_for_rectilinear_domains() {
  std::vector<Index<1>> indices_for_a_1d_road{Index<1>{0}, Index<1>{1},
                                              Index<1>{2}};
  std::vector<Index<2>> indices_for_a_2d_vertical_tower{
      Index<2>{0, 0}, Index<2>{0, 1}, Index<2>{0, 2}};
  std::vector<Index<2>> indices_for_a_2d_horizontal_wall{
      Index<2>{0, 0}, Index<2>{1, 0}, Index<2>{2, 0}};
  std::vector<Index<2>> indices_for_a_2d_field{Index<2>{0, 0}, Index<2>{1, 0},
                                               Index<2>{0, 1}, Index<2>{1, 1}};
  std::vector<Index<2>> indices_for_a_2d_net_of_a_cube{
      Index<2>{1, 0}, Index<2>{1, 1}, Index<2>{0, 2},
      Index<2>{1, 2}, Index<2>{2, 2}, Index<2>{1, 3}};
  std::vector<Index<3>> indices_for_a_3d_cube{
      Index<3>{0, 0, 0}, Index<3>{1, 0, 0}, Index<3>{0, 1, 0},
      Index<3>{1, 1, 0}, Index<3>{0, 0, 1}, Index<3>{1, 0, 1},
      Index<3>{0, 1, 1}, Index<3>{1, 1, 1}};
  std::vector<Index<3>> indices_for_a_rubiks_cube_with_hole{
      Index<3>{0, 0, 0}, Index<3>{1, 0, 0}, Index<3>{2, 0, 0},
      Index<3>{0, 1, 0}, Index<3>{1, 1, 0}, Index<3>{2, 1, 0},
      Index<3>{0, 2, 0}, Index<3>{1, 2, 0}, Index<3>{2, 2, 0},
      Index<3>{0, 0, 1}, Index<3>{1, 0, 1}, Index<3>{2, 0, 1},
      Index<3>{0, 1, 1},
      /*central block is skipped!*/
      Index<3>{2, 1, 1}, Index<3>{0, 2, 1}, Index<3>{1, 2, 1},
      Index<3>{2, 2, 1}, Index<3>{0, 0, 2}, Index<3>{1, 0, 2},
      Index<3>{2, 0, 2}, Index<3>{0, 1, 2}, Index<3>{1, 1, 2},
      Index<3>{2, 1, 2}, Index<3>{0, 2, 2}, Index<3>{1, 2, 2},
      Index<3>{2, 2, 2}};

  CHECK(indices_for_rectilinear_domains(Index<1>{3}) == indices_for_a_1d_road);
  CHECK(indices_for_rectilinear_domains(Index<2>{1, 3}) ==
        indices_for_a_2d_vertical_tower);
  CHECK(indices_for_rectilinear_domains(Index<2>{3, 1}) ==
        indices_for_a_2d_horizontal_wall);
  CHECK(indices_for_rectilinear_domains(Index<2>{2, 2}) ==
        indices_for_a_2d_field);
  CHECK(indices_for_rectilinear_domains(
            Index<2>{3, 4},
            std::vector<Index<2>>{Index<2>{0, 0}, Index<2>{2, 0},
                                  Index<2>{0, 1}, Index<2>{2, 1},
                                  Index<2>{0, 3}, Index<2>{2, 3}}) ==
        indices_for_a_2d_net_of_a_cube);
  CHECK(indices_for_rectilinear_domains(Index<3>{2, 2, 2}) ==
        indices_for_a_3d_cube);
  CHECK(indices_for_rectilinear_domains(
            Index<3>{3, 3, 3}, std::vector<Index<3>>{Index<3>{1, 1, 1}}) ==
        indices_for_a_rubiks_cube_with_hole);
}

void test_corners_for_rectilinear_domains() {
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

void test_discrete_rotation_corner_numbers() {
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
                {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}),
            std::array<size_t, 4>{{0, 1, 3, 4}}));
  CHECK(std::array<size_t, 4>{{4, 0, 5, 1}} ==
        discrete_rotation(
            OrientationMap<2>(std::array<Direction<2>, 2>{
                {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}),
            std::array<size_t, 4>{{0, 1, 4, 5}}));
  CHECK(std::array<size_t, 4>{{3, 1, 2, 0}} ==
        discrete_rotation(
            OrientationMap<2>(std::array<Direction<2>, 2>{
                {Direction<2>::lower_eta(), Direction<2>::lower_xi()}}),
            std::array<size_t, 4>{{0, 1, 2, 3}}));

  CHECK(std::array<size_t, 8>{{9, 0, 12, 3, 10, 1, 13, 4}} ==
        discrete_rotation(
            OrientationMap<3>(std::array<Direction<3>, 3>{
                {Direction<3>::upper_zeta(), Direction<3>::upper_eta(),
                 Direction<3>::lower_xi()}}),
            std::array<size_t, 8>{{0, 1, 3, 4, 9, 10, 12, 13}}));
  CHECK(std::array<size_t, 8>{{10, 13, 9, 12, 1, 4, 0, 3}} ==
        discrete_rotation(
            OrientationMap<3>(std::array<Direction<3>, 3>{
                {Direction<3>::lower_eta(), Direction<3>::upper_xi(),
                 Direction<3>::lower_zeta()}}),
            std::array<size_t, 8>{{0, 1, 3, 4, 9, 10, 12, 13}}));
  CHECK(std::array<size_t, 8>{{12, 3, 13, 4, 9, 0, 10, 1}} ==
        discrete_rotation(
            OrientationMap<3>(std::array<Direction<3>, 3>{
                {Direction<3>::upper_eta(), Direction<3>::lower_zeta(),
                 Direction<3>::lower_xi()}}),
            std::array<size_t, 8>{{0, 1, 3, 4, 9, 10, 12, 13}}));
}

void test_maps_for_rectilinear_domains() {
  using Affine = CoordinateMaps::Affine;
  using Affine2D = CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  using Equiangular = CoordinateMaps::Equiangular;
  using Equiangular2D =
      CoordinateMaps::ProductOf2Maps<Equiangular, Equiangular>;
  using Equiangular3D =
      CoordinateMaps::ProductOf3Maps<Equiangular, Equiangular, Equiangular>;

  const std::vector<std::unique_ptr<
      CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 1>>>
      affine_maps_1d = maps_for_rectilinear_domains<Frame::Inertial>(
          Index<1>{3},
          std::array<std::vector<double>, 1>{{{0.0, 0.5, 1.7, 2.0}}},
          {Index<1>{0}}, {}, false);
  const auto expected_affine_maps_1d =
      make_vector_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          Affine{-1., 1., 0.5, 1.7}, Affine{-1., 1., 1.7, 2.0});
  for (size_t i = 0; i < affine_maps_1d.size(); i++) {
    CHECK(*affine_maps_1d[i] == *expected_affine_maps_1d[i]);
  }

  const std::vector<std::unique_ptr<
      CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 1>>>
      equiangular_maps_1d = maps_for_rectilinear_domains<Frame::Inertial>(
          Index<1>{3},
          std::array<std::vector<double>, 1>{{{0.0, 0.5, 1.7, 2.0}}},
          {Index<1>{1}}, {}, true);
  const auto expected_equiangular_maps_1d =
      make_vector_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          Equiangular{-1., 1., 0.0, 0.5}, Equiangular{-1., 1., 1.7, 2.0});
  for (size_t i = 0; i < equiangular_maps_1d.size(); i++) {
    CHECK(*equiangular_maps_1d[i] == *expected_equiangular_maps_1d[i]);
  }

  const std::vector<std::unique_ptr<
      CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 2>>>
      affine_maps_2d = maps_for_rectilinear_domains<Frame::Inertial>(
          Index<2>{3, 2},
          std::array<std::vector<double>, 2>{
              {{0.0, 0.5, 1.7, 2.0}, {0.0, 1.0, 2.0}}},
          {Index<2>{}}, {}, false);
  const auto expected_affine_maps_2d =
      make_vector_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          Affine2D{Affine{-1., 1., 0.0, 0.5}, Affine{-1., 1., 0.0, 1.0}},
          Affine2D{Affine{-1., 1., 0.5, 1.7}, Affine{-1., 1., 0.0, 1.0}},
          Affine2D{Affine{-1., 1., 1.7, 2.0}, Affine{-1., 1., 0.0, 1.0}},
          Affine2D{Affine{-1., 1., 0.0, 0.5}, Affine{-1., 1., 1.0, 2.0}},
          Affine2D{Affine{-1., 1., 0.5, 1.7}, Affine{-1., 1., 1.0, 2.0}},
          Affine2D{Affine{-1., 1., 1.7, 2.0}, Affine{-1., 1., 1.0, 2.0}});
  for (size_t i = 0; i < affine_maps_2d.size(); i++) {
    CHECK(*affine_maps_2d[i] == *expected_affine_maps_2d[i]);
  }

  const std::vector<std::unique_ptr<
      CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 2>>>
      equiangular_maps_2d = maps_for_rectilinear_domains<Frame::Inertial>(
          Index<2>{3, 2},
          std::array<std::vector<double>, 2>{
              {{0.0, 0.5, 1.7, 2.0}, {0.0, 1.0, 2.0}}},
          {Index<2>{2, 1}}, {}, true);
  const auto expected_equiangular_maps_2d =
      make_vector_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
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

  // [show_maps_for_rectilinear_domains]
  const std::vector<std::unique_ptr<
      CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 3>>>
      affine_maps_3d = maps_for_rectilinear_domains<Frame::Inertial>(
          Index<3>{2, 2, 1},
          std::array<std::vector<double>, 3>{
              {{0.0, 0.5, 2.0}, {0.0, 1.0, 2.0}, {-0.4, 0.3}}},
          {Index<3>{}}, {}, false);
  // [show_maps_for_rectilinear_domains]
  const auto expected_affine_maps_3d =
      make_vector_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
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

  const std::vector<std::unique_ptr<
      CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 3>>>
      equiangular_maps_3d = maps_for_rectilinear_domains<Frame::Inertial>(
          Index<3>{2, 2, 1},
          std::array<std::vector<double>, 3>{
              {{0.0, 0.5, 2.0}, {0.0, 1.0, 2.0}, {-0.4, 0.3}}},
          {Index<3>{0, 0, 0}}, {}, true);
  const auto expected_equiangular_maps_3d =
      make_vector_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
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

void test_set_cartesian_periodic_boundaries_1() {
  // 9x9 cube with the central block removed. Periodic is xi.
  const std::vector<std::unordered_set<Direction<3>>>
      expected_external_boundaries{
          {{Direction<3>::lower_zeta(), Direction<3>::lower_eta()}},
          {{Direction<3>::lower_zeta(), Direction<3>::lower_eta()}},
          {{Direction<3>::lower_zeta(), Direction<3>::lower_eta()}},
          {{Direction<3>::lower_zeta()}},
          {{Direction<3>::lower_zeta(), Direction<3>::upper_zeta()}},
          {{Direction<3>::lower_zeta()}},
          {{Direction<3>::lower_zeta(), Direction<3>::upper_eta()}},
          {{Direction<3>::lower_zeta(), Direction<3>::upper_eta()}},
          {{Direction<3>::lower_zeta(), Direction<3>::upper_eta()}},
          {{Direction<3>::lower_eta()}},
          {{Direction<3>::lower_eta(), Direction<3>::upper_eta()}},
          {{Direction<3>::lower_eta()}},
          {},
          {},
          {{Direction<3>::upper_eta()}},
          {{Direction<3>::upper_eta(), Direction<3>::lower_eta()}},
          {{Direction<3>::upper_eta()}},
          {{Direction<3>::upper_zeta(), Direction<3>::lower_eta()}},
          {{Direction<3>::upper_zeta(), Direction<3>::lower_eta()}},
          {{Direction<3>::upper_zeta(), Direction<3>::lower_eta()}},
          {{Direction<3>::upper_zeta()}},
          {{Direction<3>::upper_zeta(), Direction<3>::lower_zeta()}},
          {{Direction<3>::upper_zeta()}},
          {{Direction<3>::upper_zeta(), Direction<3>::upper_eta()}},
          {{Direction<3>::upper_zeta(), Direction<3>::upper_eta()}},
          {{Direction<3>::upper_zeta(), Direction<3>::upper_eta()}}};
  const auto domain = rectilinear_domain<3>(
      Index<3>{3, 3, 3},
      std::array<std::vector<double>, 3>{
          {{0.0, 1.0, 2.0, 3.0}, {0.0, 1.0, 2.0, 3.0}, {0.0, 1.0, 2.0, 3.0}}},
      {Index<3>{1, 1, 1}}, {}, std::array<bool, 3>{{true, false, false}}, {});

  // Issue 1018 describes a memory bug that is caused by incorrect indexing
  // in the function `maps_for_rectilinear_domains` called through the function
  // `rectilinear_domain`. When called through `rectilinear_domain`, the empty
  // `orientations_of_all_blocks` vector passed to `rectilinear_domain`
  // triggers the construction of a vector filled with default-constructed
  // `OrientationMaps` which is then passed to `maps_for_rectilinear_domains`.
  // In `maps_for_rectilinear_domains` the non-empty vector of `OrientationMaps`
  // is then indexed into using `block_orientation_index`.
  // The maps created in this fashion are then compared to the maps created by
  // calling the `maps_for_rectilinear_domains` directly, with an empty vector
  // of `OrientationMaps`. In this case no indexing is done, so if an incorrect
  // map is created via incorrect indexing, the maps created by the two
  // function calls will differ.
  const std::vector<std::unique_ptr<
      CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 3>>>
      expected_coordinate_maps = maps_for_rectilinear_domains<Frame::Inertial>(
          Index<3>{3, 3, 3},
          std::array<std::vector<double>, 3>{{{0.0, 1.0, 2.0, 3.0},
                                              {0.0, 1.0, 2.0, 3.0},
                                              {0.0, 1.0, 2.0, 3.0}}},
          {Index<3>{1, 1, 1}}, {});

  for (size_t i = 0; i < domain.blocks().size(); i++) {
    CAPTURE(i);
    CHECK(domain.blocks()[i].external_boundaries() ==
          expected_external_boundaries[i]);
    CHECK(domain.blocks()[i].stationary_map() == *expected_coordinate_maps[i]);
  }
}

void test_set_cartesian_periodic_boundaries_2() {
  const auto rotation = OrientationMap<2>{std::array<Direction<2>, 2>{
      {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}};
  auto orientations_of_all_blocks =
      std::vector<OrientationMap<2>>{4, OrientationMap<2>{}};
  orientations_of_all_blocks[0] = rotation;
  const std::vector<std::unordered_set<Direction<2>>>
      expected_external_boundaries{{{Direction<2>::upper_xi()}},
                                   {{Direction<2>::lower_eta()}},
                                   {{Direction<2>::upper_eta()}},
                                   {{Direction<2>::upper_eta()}}};

  const auto domain = rectilinear_domain<2>(
      Index<2>{2, 2},
      std::array<std::vector<double>, 2>{{{0.0, 1.0, 2.0}, {0.0, 1.0, 2.0}}},
      {}, orientations_of_all_blocks, std::array<bool, 2>{{true, false}}, {},
      false);

  const std::vector<std::unique_ptr<
      CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 2>>>
      expected_coordinate_maps = maps_for_rectilinear_domains<Frame::Inertial>(
          Index<2>{2, 2},
          std::array<std::vector<double>, 2>{
              {{0.0, 1.0, 2.0}, {0.0, 1.0, 2.0}}},
          {}, orientations_of_all_blocks, false);

  for (size_t i = 0; i < domain.blocks().size(); i++) {
    CAPTURE(i);
    CHECK(domain.blocks()[i].external_boundaries() ==
          expected_external_boundaries[i]);
    CHECK(domain.blocks()[i].stationary_map() == *expected_coordinate_maps[i]);
  }
}

void test_set_cartesian_periodic_boundaries_3() {
  const OrientationMap<2> flipped{std::array<Direction<2>, 2>{
      {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}};
  const OrientationMap<2> quarter_turn_cw{std::array<Direction<2>, 2>{
      {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}};
  const OrientationMap<2> quarter_turn_ccw{std::array<Direction<2>, 2>{
      {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}};
  auto orientations_of_all_blocks =
      std::vector<OrientationMap<2>>{4, OrientationMap<2>{}};
  orientations_of_all_blocks[1] = flipped;
  orientations_of_all_blocks[2] = quarter_turn_cw;
  orientations_of_all_blocks[3] = quarter_turn_ccw;
  const std::vector<std::unordered_set<Direction<2>>>
      expected_external_boundaries{{}, {}, {}, {}};

  const auto domain = rectilinear_domain<2>(
      Index<2>{2, 2},
      std::array<std::vector<double>, 2>{{{0.0, 1.0, 2.0}, {0.0, 1.0, 2.0}}},
      {}, orientations_of_all_blocks, std::array<bool, 2>{{true, true}}, {},
      false);

  const std::vector<DirectionMap<2, BlockNeighbor<2>>> expected_block_neighbors{
      {{Direction<2>::upper_xi(), {1, flipped}},
       {Direction<2>::upper_eta(), {2, quarter_turn_cw}},
       {Direction<2>::lower_xi(), {1, flipped}},
       {Direction<2>::lower_eta(), {2, quarter_turn_cw}}},
      {{Direction<2>::upper_xi(), {0, flipped}},
       {Direction<2>::lower_eta(), {3, quarter_turn_cw}},
       {Direction<2>::lower_xi(), {0, flipped}},
       {Direction<2>::upper_eta(), {3, quarter_turn_cw}}},
      {{Direction<2>::upper_xi(), {0, quarter_turn_ccw}},
       {Direction<2>::upper_eta(), {3, flipped}},
       {Direction<2>::lower_xi(), {0, quarter_turn_ccw}},
       {Direction<2>::lower_eta(), {3, flipped}}},
      {{Direction<2>::lower_xi(), {1, quarter_turn_ccw}},
       {Direction<2>::upper_eta(), {2, flipped}},
       {Direction<2>::upper_xi(), {1, quarter_turn_ccw}},
       {Direction<2>::lower_eta(), {2, flipped}}}};
  const std::vector<std::unique_ptr<
      CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 2>>>
      expected_coordinate_maps = maps_for_rectilinear_domains<Frame::Inertial>(
          Index<2>{2, 2},
          std::array<std::vector<double>, 2>{
              {{0.0, 1.0, 2.0}, {0.0, 1.0, 2.0}}},
          {}, orientations_of_all_blocks, false);

  for (size_t i = 0; i < domain.blocks().size(); i++) {
    CAPTURE(i);
    CHECK(domain.blocks()[i].external_boundaries() ==
          expected_external_boundaries[i]);
    CHECK(domain.blocks()[i].stationary_map() == *expected_coordinate_maps[i]);
    CHECK(domain.blocks()[i].neighbors() == expected_block_neighbors[i]);
  }
}

void test_which_wedges() {
  CHECK(get_output(ShellWedges::All) == "All");
  CHECK(get_output(ShellWedges::FourOnEquator) == "FourOnEquator");
  CHECK(get_output(ShellWedges::OneAlongMinusX) == "OneAlongMinusX");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.DomainHelpers", "[Domain][Unit]") {
  test_periodic_same_block();
  test_periodic_different_blocks();
  test_wedge_map_generation();
  test_wedge_errors();
  test_all_frustum_directions();
  test_frustrum_errors();
  test_shell_graph();
  test_sphere_graph();
  test_bbh_corners();
  test_nsbh_corners();
  test_bhns_corners();
  test_bns_corners();
  test_volume_corner_iterator();
  test_face_corner_iterator();
  test_indices_for_rectilinear_domains();
  test_corners_for_rectilinear_domains();
  test_discrete_rotation_corner_numbers();
  test_maps_for_rectilinear_domains();
  test_set_cartesian_periodic_boundaries_1();
  test_set_cartesian_periodic_boundaries_2();
  test_set_cartesian_periodic_boundaries_3();
  test_which_wedges();
}
}  // namespace domain
