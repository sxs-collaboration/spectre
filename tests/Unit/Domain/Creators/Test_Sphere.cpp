// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <unordered_set>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"          // IWYU pragma: keep
#include "Domain/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/Wedge3D.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Direction.hpp"
#include "Domain/DirectionMap.hpp"
#include "Domain/Domain.hpp"
#include "Domain/OrientationMap.hpp"
#include "Utilities/MakeArray.hpp"
#include "tests/Unit/Domain/DomainTestHelpers.hpp"
#include "tests/Unit/TestCreation.hpp"

namespace domain {
namespace {
void test_sphere_construction(
    const creators::Sphere<Frame::Inertial>& sphere, const double inner_radius,
    const double outer_radius, const bool use_equiangular_map,
    const std::array<size_t, 2>& expected_sphere_extents,
    const std::vector<std::array<size_t, 3>>& expected_refinement_level) {
  const auto domain = sphere.create_domain();
  const OrientationMap<3> aligned_orientation{};
  const OrientationMap<3> quarter_turn_ccw_about_zeta(
      std::array<Direction<3>, 3>{{Direction<3>::lower_eta(),
                                   Direction<3>::upper_xi(),
                                   Direction<3>::upper_zeta()}});
  const OrientationMap<3> half_turn_about_zeta(std::array<Direction<3>, 3>{
      {Direction<3>::lower_xi(), Direction<3>::lower_eta(),
       Direction<3>::upper_zeta()}});
  const OrientationMap<3> quarter_turn_cw_about_zeta(
      std::array<Direction<3>, 3>{{Direction<3>::upper_eta(),
                                   Direction<3>::lower_xi(),
                                   Direction<3>::upper_zeta()}});
  const OrientationMap<3> center_relative_to_minus_z(
      std::array<Direction<3>, 3>{{Direction<3>::upper_xi(),
                                   Direction<3>::lower_eta(),
                                   Direction<3>::lower_zeta()}});
  const OrientationMap<3> center_relative_to_plus_y(std::array<Direction<3>, 3>{
      {Direction<3>::upper_xi(), Direction<3>::lower_zeta(),
       Direction<3>::upper_eta()}});
  const OrientationMap<3> center_relative_to_minus_y(
      std::array<Direction<3>, 3>{{Direction<3>::upper_xi(),
                                   Direction<3>::upper_zeta(),
                                   Direction<3>::lower_eta()}});
  const OrientationMap<3> center_relative_to_plus_x(std::array<Direction<3>, 3>{
      {Direction<3>::upper_eta(), Direction<3>::upper_zeta(),
       Direction<3>::upper_xi()}});
  const OrientationMap<3> center_relative_to_minus_x(
      std::array<Direction<3>, 3>{{Direction<3>::lower_eta(),
                                   Direction<3>::upper_zeta(),
                                   Direction<3>::lower_xi()}});

  const std::vector<DirectionMap<3, BlockNeighbor<3>>> expected_block_neighbors{
      {{Direction<3>::upper_xi(), {4, quarter_turn_ccw_about_zeta}},
       {Direction<3>::upper_eta(), {2, aligned_orientation}},
       {Direction<3>::lower_xi(), {5, quarter_turn_cw_about_zeta}},
       {Direction<3>::lower_eta(), {3, aligned_orientation}},
       {Direction<3>::lower_zeta(), {6, aligned_orientation}}},
      {{Direction<3>::upper_xi(), {4, quarter_turn_cw_about_zeta}},
       {Direction<3>::upper_eta(), {3, aligned_orientation}},
       {Direction<3>::lower_xi(), {5, quarter_turn_ccw_about_zeta}},
       {Direction<3>::lower_eta(), {2, aligned_orientation}},
       {Direction<3>::lower_zeta(), {6, center_relative_to_minus_z}}},
      {{Direction<3>::upper_xi(), {4, half_turn_about_zeta}},
       {Direction<3>::upper_eta(), {1, aligned_orientation}},
       {Direction<3>::lower_xi(), {5, half_turn_about_zeta}},
       {Direction<3>::lower_eta(), {0, aligned_orientation}},
       {Direction<3>::lower_zeta(), {6, center_relative_to_plus_y}}},
      {{Direction<3>::upper_xi(), {4, aligned_orientation}},
       {Direction<3>::upper_eta(), {0, aligned_orientation}},
       {Direction<3>::lower_xi(), {5, aligned_orientation}},
       {Direction<3>::lower_eta(), {1, aligned_orientation}},
       {Direction<3>::lower_zeta(), {6, center_relative_to_minus_y}}},
      {{Direction<3>::upper_xi(), {2, half_turn_about_zeta}},
       {Direction<3>::upper_eta(), {0, quarter_turn_cw_about_zeta}},
       {Direction<3>::lower_xi(), {3, aligned_orientation}},
       {Direction<3>::lower_eta(), {1, quarter_turn_ccw_about_zeta}},
       {Direction<3>::lower_zeta(), {6, center_relative_to_plus_x}}},
      {{Direction<3>::upper_xi(), {3, aligned_orientation}},
       {Direction<3>::upper_eta(), {0, quarter_turn_ccw_about_zeta}},
       {Direction<3>::lower_xi(), {2, half_turn_about_zeta}},
       {Direction<3>::lower_eta(), {1, quarter_turn_cw_about_zeta}},
       {Direction<3>::lower_zeta(), {6, center_relative_to_minus_x}}},
      {{Direction<3>::upper_zeta(), {0, aligned_orientation}},
       {Direction<3>::lower_zeta(),
        {1, center_relative_to_minus_z.inverse_map()}},
       {Direction<3>::upper_eta(),
        {2, center_relative_to_plus_y.inverse_map()}},
       {Direction<3>::lower_eta(),
        {3, center_relative_to_minus_y.inverse_map()}},
       {Direction<3>::upper_xi(), {4, center_relative_to_plus_x.inverse_map()}},
       {Direction<3>::lower_xi(),
        {5, center_relative_to_minus_x.inverse_map()}}}};

  const std::vector<std::unordered_set<Direction<3>>>
      expected_external_boundaries{{{Direction<3>::upper_zeta()}},
                                   {{Direction<3>::upper_zeta()}},
                                   {{Direction<3>::upper_zeta()}},
                                   {{Direction<3>::upper_zeta()}},
                                   {{Direction<3>::upper_zeta()}},
                                   {{Direction<3>::upper_zeta()}},
                                   {}};

  std::vector<std::array<size_t, 3>> expected_extents{
      6,
      {{expected_sphere_extents[1], expected_sphere_extents[1],
        expected_sphere_extents[0]}}};
  expected_extents.push_back(
      {{expected_sphere_extents[1], expected_sphere_extents[1],
        expected_sphere_extents[1]}});

  CHECK(sphere.initial_extents() == expected_extents);
  CHECK(sphere.initial_refinement_levels() == expected_refinement_level);
  using Wedge3DMap = CoordinateMaps::Wedge3D;
  using Affine = CoordinateMaps::Affine;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  using Equiangular = CoordinateMaps::Equiangular;
  using Equiangular3D =
      CoordinateMaps::ProductOf3Maps<Equiangular, Equiangular, Equiangular>;

  auto coord_maps =
      make_vector_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Wedge3DMap{inner_radius, outer_radius, OrientationMap<3>{}, 0.0, 1.0,
                     use_equiangular_map},
          Wedge3DMap{inner_radius, outer_radius,
                     OrientationMap<3>{std::array<Direction<3>, 3>{
                         {Direction<3>::upper_xi(), Direction<3>::lower_eta(),
                          Direction<3>::lower_zeta()}}},
                     0.0, 1.0, use_equiangular_map},
          Wedge3DMap{inner_radius, outer_radius,
                     OrientationMap<3>{std::array<Direction<3>, 3>{
                         {Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
                          Direction<3>::lower_eta()}}},
                     0.0, 1.0, use_equiangular_map},
          Wedge3DMap{inner_radius, outer_radius,
                     OrientationMap<3>{std::array<Direction<3>, 3>{
                         {Direction<3>::upper_xi(), Direction<3>::lower_zeta(),
                          Direction<3>::upper_eta()}}},
                     0.0, 1.0, use_equiangular_map},
          Wedge3DMap{inner_radius, outer_radius,
                     OrientationMap<3>{std::array<Direction<3>, 3>{
                         {Direction<3>::upper_zeta(), Direction<3>::upper_xi(),
                          Direction<3>::upper_eta()}}},
                     0.0, 1.0, use_equiangular_map},
          Wedge3DMap{inner_radius, outer_radius,
                     OrientationMap<3>{std::array<Direction<3>, 3>{
                         {Direction<3>::lower_zeta(), Direction<3>::lower_xi(),
                          Direction<3>::upper_eta()}}},
                     0.0, 1.0, use_equiangular_map});
  if (use_equiangular_map) {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::Logical, Frame::Inertial>(Equiangular3D{
            Equiangular(-1.0, 1.0, -1.0 * inner_radius / sqrt(3.0),
                        inner_radius / sqrt(3.0)),
            Equiangular(-1.0, 1.0, -1.0 * inner_radius / sqrt(3.0),
                        inner_radius / sqrt(3.0)),
            Equiangular(-1.0, 1.0, -1.0 * inner_radius / sqrt(3.0),
                        inner_radius / sqrt(3.0))}));
  } else {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
            Affine3D{Affine(-1.0, 1.0, -1.0 * inner_radius / sqrt(3.0),
                            inner_radius / sqrt(3.0)),
                     Affine(-1.0, 1.0, -1.0 * inner_radius / sqrt(3.0),
                            inner_radius / sqrt(3.0)),
                     Affine(-1.0, 1.0, -1.0 * inner_radius / sqrt(3.0),
                            inner_radius / sqrt(3.0))}));
  }
  test_domain_construction(domain, expected_block_neighbors,
                           expected_external_boundaries, coord_maps);
  test_initial_domain(domain, sphere.initial_refinement_levels());
}

void test_sphere_boundaries_equiangular() {
  INFO("Sphere boundaries equiangular");
  const double inner_radius = 1.0, outer_radius = 2.0;
  const size_t refinement = 2;
  const std::array<size_t, 2> grid_points_r_angular{{4, 4}};

  const creators::Sphere<Frame::Inertial> sphere{
      inner_radius, outer_radius, refinement, grid_points_r_angular, true};
  test_physical_separation(sphere.create_domain().blocks());

  test_sphere_construction(sphere, inner_radius, outer_radius, true,
                           grid_points_r_angular,
                           {7, make_array<3>(refinement)});
}

void test_sphere_factory_equiangular() {
  INFO("Sphere factory equiangular");
  const auto sphere = test_factory_creation<DomainCreator<3, Frame::Inertial>>(
      "  Sphere:\n"
      "    InnerRadius: 1\n"
      "    OuterRadius: 3\n"
      "    InitialRefinement: 2\n"
      "    InitialGridPoints: [2,3]\n"
      "    UseEquiangularMap: true\n");
  const double inner_radius = 1.0, outer_radius = 3.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 2> grid_points_r_angular{{2, 3}};
  test_sphere_construction(
      dynamic_cast<const creators::Sphere<Frame::Inertial>&>(*sphere),
      inner_radius, outer_radius, true, grid_points_r_angular,
      {7, make_array<3>(refinement_level)});
}

void test_sphere_boundaries_equidistant() {
  INFO("Sphere boundaries equidistant");
  const double inner_radius = 1.0, outer_radius = 2.0;
  const size_t refinement = 2;
  const std::array<size_t, 2> grid_points_r_angular{{4, 4}};

  const creators::Sphere<Frame::Inertial> sphere{
      inner_radius, outer_radius, refinement, grid_points_r_angular, false};
  test_physical_separation(sphere.create_domain().blocks());

  test_sphere_construction(sphere, inner_radius, outer_radius, false,
                           grid_points_r_angular,
                           {7, make_array<3>(refinement)});
}

void test_sphere_factory_equidistant() {
  INFO("Sphere factory equidistant");
  const auto sphere = test_factory_creation<DomainCreator<3, Frame::Inertial>>(
      "  Sphere:\n"
      "    InnerRadius: 1\n"
      "    OuterRadius: 3\n"
      "    InitialRefinement: 2\n"
      "    InitialGridPoints: [2,3]\n"
      "    UseEquiangularMap: false\n");
  const double inner_radius = 1.0, outer_radius = 3.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 2> grid_points_r_angular{{2, 3}};
  test_sphere_construction(
      dynamic_cast<const creators::Sphere<Frame::Inertial>&>(*sphere),
      inner_radius, outer_radius, false, grid_points_r_angular,
      {7, make_array<3>(refinement_level)});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.Sphere", "[Domain][Unit]") {
  test_sphere_boundaries_equiangular();
  test_sphere_factory_equiangular();
  test_sphere_boundaries_equidistant();
  test_sphere_factory_equidistant();
}
}  // namespace domain
