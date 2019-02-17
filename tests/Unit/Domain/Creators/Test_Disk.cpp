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
#include "Domain/Creators/Disk.hpp"
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
void test_disk_construction(
    const creators::Disk<Frame::Inertial>& disk, const double inner_radius,
    const double outer_radius,
    const std::array<size_t, 2>& expected_wedge_extents,
    const std::vector<std::array<size_t, 2>>& expected_refinement_level,
    const bool use_equiangular_map) {
  const auto domain = disk.create_domain();
  const OrientationMap<2> aligned_orientation{};
  const OrientationMap<2> quarter_turn_ccw(std::array<Direction<2>, 2>{
      {Direction<2>::lower_eta(), Direction<2>::upper_xi()}});
  const OrientationMap<2> half_turn(std::array<Direction<2>, 2>{
      {Direction<2>::lower_xi(), Direction<2>::lower_eta()}});
  const OrientationMap<2> quarter_turn_cw(std::array<Direction<2>, 2>{
      {Direction<2>::upper_eta(), Direction<2>::lower_xi()}});
  const std::vector<DirectionMap<2, BlockNeighbor<2>>> expected_block_neighbors{
      {{Direction<2>::lower_eta(), {3, aligned_orientation}},
       {Direction<2>::upper_eta(), {1, aligned_orientation}},
       {Direction<2>::lower_xi(), {4, aligned_orientation}}},
      {{Direction<2>::lower_eta(), {0, aligned_orientation}},
       {Direction<2>::upper_eta(), {2, aligned_orientation}},
       {Direction<2>::lower_xi(), {4, quarter_turn_cw}}},
      {{Direction<2>::lower_eta(), {1, aligned_orientation}},
       {Direction<2>::upper_eta(), {3, aligned_orientation}},
       {Direction<2>::lower_xi(), {4, half_turn}}},
      {{Direction<2>::lower_eta(), {2, aligned_orientation}},
       {Direction<2>::upper_eta(), {0, aligned_orientation}},
       {Direction<2>::lower_xi(), {4, quarter_turn_ccw}}},
      {{Direction<2>::upper_xi(), {0, aligned_orientation}},
       {Direction<2>::upper_eta(), {1, quarter_turn_ccw}},
       {Direction<2>::lower_xi(), {2, half_turn}},
       {Direction<2>::lower_eta(), {3, quarter_turn_cw}}}};
  const std::vector<std::unordered_set<Direction<2>>>
      expected_external_boundaries{{{Direction<2>::upper_xi()}},
                                   {{Direction<2>::upper_xi()}},
                                   {{Direction<2>::upper_xi()}},
                                   {{Direction<2>::upper_xi()}},
                                   {}};

  const std::vector<std::array<size_t, 2>>& expected_extents{
      expected_wedge_extents,
      expected_wedge_extents,
      expected_wedge_extents,
      expected_wedge_extents,
      {{expected_wedge_extents[1], expected_wedge_extents[1]}}};

  CHECK(disk.initial_extents() == expected_extents);
  CHECK(disk.initial_refinement_levels() == expected_refinement_level);
  using TargetFrame = Frame::Inertial;
  using Wedge2DMap = CoordinateMaps::Wedge2D;
  using Affine = CoordinateMaps::Affine;
  using Affine2D = CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  using Equiangular = CoordinateMaps::Equiangular;
  using Equiangular2D =
      CoordinateMaps::ProductOf2Maps<Equiangular, Equiangular>;

  auto coord_maps = make_vector_coordinate_map_base<Frame::Logical,
                                                    TargetFrame>(
      Wedge2DMap{inner_radius, outer_radius, 0.0, 1.0,
                 OrientationMap<2>{std::array<Direction<2>, 2>{
                     {Direction<2>::upper_xi(), Direction<2>::upper_eta()}}},
                 use_equiangular_map},
      Wedge2DMap{inner_radius, outer_radius, 0.0, 1.0,
                 OrientationMap<2>{std::array<Direction<2>, 2>{
                     {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}},
                 use_equiangular_map},
      Wedge2DMap{inner_radius, outer_radius, 0.0, 1.0,
                 OrientationMap<2>{std::array<Direction<2>, 2>{
                     {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}},
                 use_equiangular_map},
      Wedge2DMap{inner_radius, outer_radius, 0.0, 1.0,
                 OrientationMap<2>{std::array<Direction<2>, 2>{
                     {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}},
                 use_equiangular_map});

  if (use_equiangular_map) {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::Logical, TargetFrame>(Equiangular2D{
            Equiangular(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                        inner_radius / sqrt(2.0)),
            Equiangular(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                        inner_radius / sqrt(2.0))}));
  } else {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::Logical, TargetFrame>(
            Affine2D{Affine(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                            inner_radius / sqrt(2.0)),
                     Affine(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                            inner_radius / sqrt(2.0))}));
  }
  test_domain_construction(domain, expected_block_neighbors,
                           expected_external_boundaries, coord_maps);

  test_initial_domain(domain, disk.initial_refinement_levels());
}

void test_disk_boundaries_equiangular() {
  INFO("Disk boundaries equiangular");
  const double inner_radius = 1.0, outer_radius = 2.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 2> grid_points{{4, 4}};

  const creators::Disk<Frame::Inertial> disk{
      inner_radius, outer_radius, refinement_level, grid_points, true};
  test_physical_separation(disk.create_domain().blocks());
  test_disk_construction(disk, inner_radius, outer_radius, grid_points,
                         {5, make_array<2>(refinement_level)}, true);
}

void test_disk_factory_equiangular() {
  INFO("Disk factory equiangular");
  const auto disk = test_factory_creation<DomainCreator<2, Frame::Inertial>>(
      "  Disk:\n"
      "    InnerRadius: 1\n"
      "    OuterRadius: 3\n"
      "    InitialRefinement: 2\n"
      "    InitialGridPoints: [2,3]\n"
      "    UseEquiangularMap: true\n");

  const double inner_radius = 1.0, outer_radius = 3.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 2> grid_points{{2, 3}};
  test_disk_construction(
      dynamic_cast<const creators::Disk<Frame::Inertial>&>(*disk), inner_radius,
      outer_radius, grid_points, {5, make_array<2>(refinement_level)}, true);
}

void test_disk_boundaries_equidistant() {
  INFO("Disk boundaries equidistant");
  const double inner_radius = 1.0, outer_radius = 2.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 2> grid_points{{4, 4}};

  const creators::Disk<Frame::Inertial> disk{
      inner_radius, outer_radius, refinement_level, grid_points, false};
  test_physical_separation(disk.create_domain().blocks());
  test_disk_construction(disk, inner_radius, outer_radius, grid_points,
                         {5, make_array<2>(refinement_level)}, false);
}

void test_disk_factory_equidistant() {
  INFO("Disk factory equidistant");
  const auto disk = test_factory_creation<DomainCreator<2, Frame::Inertial>>(
      "  Disk:\n"
      "    InnerRadius: 1\n"
      "    OuterRadius: 3\n"
      "    InitialRefinement: 2\n"
      "    InitialGridPoints: [2,3]\n"
      "    UseEquiangularMap: false\n");

  const double inner_radius = 1.0, outer_radius = 3.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 2> grid_points{{2, 3}};
  test_disk_construction(
      dynamic_cast<const creators::Disk<Frame::Inertial>&>(*disk), inner_radius,
      outer_radius, grid_points, {5, make_array<2>(refinement_level)}, false);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.Disk.Factory.Equidistant",
                  "[Domain][Unit]") {
  test_disk_boundaries_equiangular();
  test_disk_factory_equiangular();
  test_disk_boundaries_equidistant();
  test_disk_factory_equidistant();
}
}  // namespace domain
