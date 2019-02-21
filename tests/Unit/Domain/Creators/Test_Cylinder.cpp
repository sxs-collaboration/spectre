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
#include "Domain/CoordinateMaps/Wedge2D.hpp"
#include "Domain/Creators/Cylinder.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Direction.hpp"
#include "Domain/DirectionMap.hpp"
#include "Domain/Domain.hpp"
#include "Domain/OrientationMap.hpp"
#include "Utilities/MakeArray.hpp"
#include "tests/Unit/Domain/DomainTestHelpers.hpp"
#include "tests/Unit/TestCreation.hpp"

namespace domain {
namespace {
void test_cylinder_construction(
    const creators::Cylinder<Frame::Inertial>& cylinder,
    const double inner_radius, const double outer_radius,
    const double lower_bound, const double upper_bound,
    const bool is_periodic_in_z,
    const std::array<size_t, 3>& expected_wedge_extents,
    const std::vector<std::array<size_t, 3>>& expected_refinement_level,
    const bool use_equiangular_map) {
  const auto domain = cylinder.create_domain();
  const OrientationMap<3> aligned_orientation{};
  const OrientationMap<3> quarter_turn_ccw(std::array<Direction<3>, 3>{
      {Direction<3>::lower_eta(), Direction<3>::upper_xi(),
       Direction<3>::upper_zeta()}});
  const OrientationMap<3> half_turn(std::array<Direction<3>, 3>{
      {Direction<3>::lower_xi(), Direction<3>::lower_eta(),
       Direction<3>::upper_zeta()}});
  const OrientationMap<3> quarter_turn_cw(std::array<Direction<3>, 3>{
      {Direction<3>::upper_eta(), Direction<3>::lower_xi(),
       Direction<3>::upper_zeta()}});
  std::vector<DirectionMap<3, BlockNeighbor<3>>> expected_block_neighbors{};
  std::vector<std::unordered_set<Direction<3>>> expected_external_boundaries{};
  if (not is_periodic_in_z) {
    expected_block_neighbors = std::vector<DirectionMap<3, BlockNeighbor<3>>>{
        {{Direction<3>::lower_eta(), {3, aligned_orientation}},
         {Direction<3>::upper_eta(), {1, aligned_orientation}},
         {Direction<3>::lower_xi(), {4, aligned_orientation}}},
        {{Direction<3>::lower_eta(), {0, aligned_orientation}},
         {Direction<3>::upper_eta(), {2, aligned_orientation}},
         {Direction<3>::lower_xi(), {4, quarter_turn_cw}}},
        {{Direction<3>::lower_eta(), {1, aligned_orientation}},
         {Direction<3>::upper_eta(), {3, aligned_orientation}},
         {Direction<3>::lower_xi(), {4, half_turn}}},
        {{Direction<3>::lower_eta(), {2, aligned_orientation}},
         {Direction<3>::upper_eta(), {0, aligned_orientation}},
         {Direction<3>::lower_xi(), {4, quarter_turn_ccw}}},
        {{Direction<3>::upper_xi(), {0, aligned_orientation}},
         {Direction<3>::upper_eta(), {1, quarter_turn_ccw}},
         {Direction<3>::lower_xi(), {2, half_turn}},
         {Direction<3>::lower_eta(), {3, quarter_turn_cw}}}};
    expected_external_boundaries =
        std::vector<std::unordered_set<Direction<3>>>{
            {{Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
              Direction<3>::lower_zeta()}},
            {{Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
              Direction<3>::lower_zeta()}},
            {{Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
              Direction<3>::lower_zeta()}},
            {{Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
              Direction<3>::lower_zeta()}},
            {Direction<3>::upper_zeta(), Direction<3>::lower_zeta()}};
  } else {
    expected_block_neighbors = std::vector<DirectionMap<3, BlockNeighbor<3>>>{
        {{Direction<3>::lower_eta(), {3, aligned_orientation}},
         {Direction<3>::upper_eta(), {1, aligned_orientation}},
         {Direction<3>::lower_xi(), {4, aligned_orientation}},
         {Direction<3>::upper_zeta(), {0, aligned_orientation}},
         {Direction<3>::lower_zeta(), {0, aligned_orientation}}},
        {{Direction<3>::lower_eta(), {0, aligned_orientation}},
         {Direction<3>::upper_eta(), {2, aligned_orientation}},
         {Direction<3>::lower_xi(), {4, quarter_turn_cw}},
         {Direction<3>::upper_zeta(), {1, aligned_orientation}},
         {Direction<3>::lower_zeta(), {1, aligned_orientation}}},
        {{Direction<3>::lower_eta(), {1, aligned_orientation}},
         {Direction<3>::upper_eta(), {3, aligned_orientation}},
         {Direction<3>::lower_xi(), {4, half_turn}},
         {Direction<3>::upper_zeta(), {2, aligned_orientation}},
         {Direction<3>::lower_zeta(), {2, aligned_orientation}}},
        {{Direction<3>::lower_eta(), {2, aligned_orientation}},
         {Direction<3>::upper_eta(), {0, aligned_orientation}},
         {Direction<3>::lower_xi(), {4, quarter_turn_ccw}},
         {Direction<3>::upper_zeta(), {3, aligned_orientation}},
         {Direction<3>::lower_zeta(), {3, aligned_orientation}}},
        {{Direction<3>::upper_xi(), {0, aligned_orientation}},
         {Direction<3>::upper_eta(), {1, quarter_turn_ccw}},
         {Direction<3>::lower_xi(), {2, half_turn}},
         {Direction<3>::lower_eta(), {3, quarter_turn_cw}},
         {Direction<3>::upper_zeta(), {4, aligned_orientation}},
         {Direction<3>::lower_zeta(), {4, aligned_orientation}}}};

    expected_external_boundaries =
        std::vector<std::unordered_set<Direction<3>>>{
            {{Direction<3>::upper_xi()}},
            {{Direction<3>::upper_xi()}},
            {{Direction<3>::upper_xi()}},
            {{Direction<3>::upper_xi()}},
            {}};
  }
  const std::vector<std::array<size_t, 3>>& expected_extents{
      expected_wedge_extents,
      expected_wedge_extents,
      expected_wedge_extents,
      expected_wedge_extents,
      {{expected_wedge_extents[1], expected_wedge_extents[1],
        expected_wedge_extents[2]}}};

  CHECK(cylinder.initial_extents() == expected_extents);
  CHECK(cylinder.initial_refinement_levels() == expected_refinement_level);
  using TargetFrame = Frame::Inertial;
  using Affine = CoordinateMaps::Affine;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  using Equiangular = CoordinateMaps::Equiangular;
  using Equiangular3DPrism =
      CoordinateMaps::ProductOf3Maps<Equiangular, Equiangular, Affine>;
  using Wedge2D = CoordinateMaps::Wedge2D;
  using Wedge3DPrism = CoordinateMaps::ProductOf2Maps<Wedge2D, Affine>;

  auto coord_maps =
      make_vector_coordinate_map_base<Frame::Logical, TargetFrame>(
          Wedge3DPrism{Wedge2D{inner_radius, outer_radius, 0.0, 1.0,
                               OrientationMap<2>{std::array<Direction<2>, 2>{
                                   {Direction<2>::upper_xi(),
                                    Direction<2>::upper_eta()}}},
                               use_equiangular_map},
                       Affine{-1.0, 1.0, lower_bound, upper_bound}},
          Wedge3DPrism{Wedge2D{inner_radius, outer_radius, 0.0, 1.0,
                               OrientationMap<2>{std::array<Direction<2>, 2>{
                                   {Direction<2>::lower_eta(),
                                    Direction<2>::upper_xi()}}},
                               use_equiangular_map},
                       Affine{-1.0, 1.0, lower_bound, upper_bound}},
          Wedge3DPrism{Wedge2D{inner_radius, outer_radius, 0.0, 1.0,
                               OrientationMap<2>{std::array<Direction<2>, 2>{
                                   {Direction<2>::lower_xi(),
                                    Direction<2>::lower_eta()}}},
                               use_equiangular_map},
                       Affine{-1.0, 1.0, lower_bound, upper_bound}},
          Wedge3DPrism{Wedge2D{inner_radius, outer_radius, 0.0, 1.0,
                               OrientationMap<2>{std::array<Direction<2>, 2>{
                                   {Direction<2>::upper_eta(),
                                    Direction<2>::lower_xi()}}},
                               use_equiangular_map},
                       Affine{-1.0, 1.0, lower_bound, upper_bound}});

  if (use_equiangular_map) {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::Logical, TargetFrame>(
            Equiangular3DPrism{
                Equiangular(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                            inner_radius / sqrt(2.0)),
                Equiangular(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                            inner_radius / sqrt(2.0)),
                Affine{-1.0, 1.0, lower_bound, upper_bound}}));
  } else {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::Logical, TargetFrame>(
            Affine3D{Affine(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                            inner_radius / sqrt(2.0)),
                     Affine(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                            inner_radius / sqrt(2.0)),
                     Affine{-1.0, 1.0, lower_bound, upper_bound}}));
  }

  test_domain_construction(domain, expected_block_neighbors,
                           expected_external_boundaries, coord_maps);

  test_initial_domain(domain, cylinder.initial_refinement_levels());
}

void test_cylinder_boundaries_equiangular() {
  INFO("Cylinder boundaries equiangular");
  const double inner_radius = 1.0, outer_radius = 2.0;
  const double lower_bound = -2.5, upper_bound = 5.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 3> grid_points{{4, 4, 3}};

  const creators::Cylinder<Frame::Inertial> cylinder{
      inner_radius, outer_radius,     lower_bound, upper_bound,
      true,         refinement_level, grid_points, true};
  test_physical_separation(cylinder.create_domain().blocks());
  test_cylinder_construction(cylinder, inner_radius, outer_radius, lower_bound,
                             upper_bound, true, grid_points,
                             {5, make_array<3>(refinement_level)}, true);
}

void test_cylinder_factory_equiangular() {
  INFO("Cylinder factory equiangular");
  const auto cylinder =
      test_factory_creation<DomainCreator<3, Frame::Inertial>>(
          "  Cylinder:\n"
          "    InnerRadius: 1.0\n"
          "    OuterRadius: 3.0\n"
          "    LowerBound: -1.2\n"
          "    UpperBound: 3.7\n"
          "    InitialRefinement: 2\n"
          "    InitialGridPoints: [2,3,4]\n"
          "    UseEquiangularMap: true\n");

  const double inner_radius = 1.0, outer_radius = 3.0;
  const double lower_bound = -1.2, upper_bound = 3.7;
  const size_t refinement_level = 2;
  const std::array<size_t, 3> grid_points{{2, 3, 4}};
  test_cylinder_construction(
      dynamic_cast<const creators::Cylinder<Frame::Inertial>&>(*cylinder),
      inner_radius, outer_radius, lower_bound, upper_bound, true, grid_points,
      {5, make_array<3>(refinement_level)}, true);
}

void test_cylinder_boundaries_equidistant() {
  INFO("Cylinder boundaries equidistant");
  const double inner_radius = 1.0, outer_radius = 2.0;
  const double lower_bound = -2.5, upper_bound = 5.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 3> grid_points{{4, 4, 3}};

  const creators::Cylinder<Frame::Inertial> cylinder{
      inner_radius, outer_radius,     lower_bound, upper_bound,
      true,         refinement_level, grid_points, false};
  test_physical_separation(cylinder.create_domain().blocks());
  test_cylinder_construction(cylinder, inner_radius, outer_radius, lower_bound,
                             upper_bound, true, grid_points,
                             {5, make_array<3>(refinement_level)}, false);
}

void test_cylinder_factory_equidistant() {
  INFO("Cylinder factory equidistant");
  const auto cylinder =
      test_factory_creation<DomainCreator<3, Frame::Inertial>>(
          "  Cylinder:\n"
          "    InnerRadius: 1.0\n"
          "    OuterRadius: 3.0\n"
          "    LowerBound: -1.2\n"
          "    UpperBound: 3.7\n"
          "    InitialRefinement: 2\n"
          "    InitialGridPoints: [2,3,4]\n"
          "    UseEquiangularMap: false\n");

  const double inner_radius = 1.0, outer_radius = 3.0;
  const double lower_bound = -1.2, upper_bound = 3.7;
  const size_t refinement_level = 2;
  const std::array<size_t, 3> grid_points{{2, 3, 4}};
  test_cylinder_construction(
      dynamic_cast<const creators::Cylinder<Frame::Inertial>&>(*cylinder),
      inner_radius, outer_radius, lower_bound, upper_bound, true, grid_points,
      {5, make_array<3>(refinement_level)}, false);
}

void test_cylinder_boundaries_equiangular_not_periodic_in_z() {
  INFO("Cylinder boundaries equiangular not periodic in z");
  const double inner_radius = 1.0, outer_radius = 2.0;
  const double lower_bound = -2.5, upper_bound = 5.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 3> grid_points{{4, 4, 3}};

  const creators::Cylinder<Frame::Inertial> cylinder{
      inner_radius, outer_radius,     lower_bound, upper_bound,
      false,        refinement_level, grid_points, true};
  test_physical_separation(cylinder.create_domain().blocks());
  test_cylinder_construction(cylinder, inner_radius, outer_radius, lower_bound,
                             upper_bound, false, grid_points,
                             {5, make_array<3>(refinement_level)}, true);
}

void test_cylinder_factory_equiangular_not_periodic_in_z() {
  INFO("Cylinder factory equiangular not periodic in z");
  const auto cylinder =
      test_factory_creation<DomainCreator<3, Frame::Inertial>>(
          "  Cylinder:\n"
          "    InnerRadius: 1.0\n"
          "    OuterRadius: 3.0\n"
          "    LowerBound: -1.2\n"
          "    UpperBound: 3.7\n"
          "    IsPeriodicInZ: false\n"
          "    InitialRefinement: 2\n"
          "    InitialGridPoints: [2,3,4]\n"
          "    UseEquiangularMap: true\n");

  const double inner_radius = 1.0, outer_radius = 3.0;
  const double lower_bound = -1.2, upper_bound = 3.7;
  const size_t refinement_level = 2;
  const std::array<size_t, 3> grid_points{{2, 3, 4}};
  test_cylinder_construction(
      dynamic_cast<const creators::Cylinder<Frame::Inertial>&>(*cylinder),
      inner_radius, outer_radius, lower_bound, upper_bound, false, grid_points,
      {5, make_array<3>(refinement_level)}, true);
}

void test_cylinder_boundaries_equidistant_not_periodic_in_z() {
  INFO("Cylinder boundaries equidistant not periodic in z");
  const double inner_radius = 1.0, outer_radius = 2.0;
  const double lower_bound = -2.5, upper_bound = 5.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 3> grid_points{{4, 4, 3}};

  const creators::Cylinder<Frame::Inertial> cylinder{
      inner_radius, outer_radius,     lower_bound, upper_bound,
      false,        refinement_level, grid_points, false};
  test_physical_separation(cylinder.create_domain().blocks());
  test_cylinder_construction(cylinder, inner_radius, outer_radius, lower_bound,
                             upper_bound, false, grid_points,
                             {5, make_array<3>(refinement_level)}, false);
}

void test_cylinder_factory_equidistant_not_periodic_in_z() {
  INFO("Cylinder factory equidistant not periodic in z");
  const auto cylinder =
      test_factory_creation<DomainCreator<3, Frame::Inertial>>(
          "  Cylinder:\n"
          "    InnerRadius: 1.0\n"
          "    OuterRadius: 3.0\n"
          "    LowerBound: -1.2\n"
          "    UpperBound: 3.7\n"
          "    IsPeriodicInZ: false\n"
          "    InitialRefinement: 2\n"
          "    InitialGridPoints: [2,3,4]\n"
          "    UseEquiangularMap: false\n");

  const double inner_radius = 1.0, outer_radius = 3.0;
  const double lower_bound = -1.2, upper_bound = 3.7;
  const size_t refinement_level = 2;
  const std::array<size_t, 3> grid_points{{2, 3, 4}};
  test_cylinder_construction(
      dynamic_cast<const creators::Cylinder<Frame::Inertial>&>(*cylinder),
      inner_radius, outer_radius, lower_bound, upper_bound, false, grid_points,
      {5, make_array<3>(refinement_level)}, false);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.Cylinder", "[Domain][Unit]") {
  test_cylinder_boundaries_equiangular();
  test_cylinder_factory_equiangular();
  test_cylinder_boundaries_equidistant();
  test_cylinder_factory_equidistant();
  test_cylinder_boundaries_equiangular_not_periodic_in_z();
  test_cylinder_factory_equiangular_not_periodic_in_z();
  test_cylinder_boundaries_equidistant_not_periodic_in_z();
  test_cylinder_factory_equidistant_not_periodic_in_z();
}
}  // namespace domain
