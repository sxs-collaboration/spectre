// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <unordered_map>
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
#include "Domain/Direction.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainCreators/Disk.hpp"
#include "Domain/DomainCreators/DomainCreator.hpp"
#include "Domain/OrientationMap.hpp"
#include "Utilities/MakeArray.hpp"
#include "tests/Unit/Domain/DomainTestHelpers.hpp"
#include "tests/Unit/TestCreation.hpp"

namespace {
void test_disk_construction(
    const domain::creators::Disk<Frame::Inertial>& disk,
    const double inner_radius, const double outer_radius,
    const std::array<size_t, 2>& expected_wedge_extents,
    const std::vector<std::array<size_t, 2>>& expected_refinement_level,
    const bool use_equiangular_map) {
  const auto domain = disk.create_domain();
  const domain::OrientationMap<2> aligned_orientation{};
  const domain::OrientationMap<2> quarter_turn_ccw(
      std::array<domain::Direction<2>, 2>{{domain::Direction<2>::lower_eta(),
                                           domain::Direction<2>::upper_xi()}});
  const domain::OrientationMap<2> half_turn(std::array<domain::Direction<2>, 2>{
      {domain::Direction<2>::lower_xi(), domain::Direction<2>::lower_eta()}});
  const domain::OrientationMap<2> quarter_turn_cw(
      std::array<domain::Direction<2>, 2>{{domain::Direction<2>::upper_eta(),
                                           domain::Direction<2>::lower_xi()}});
  const std::vector<
      std::unordered_map<domain::Direction<2>, domain::BlockNeighbor<2>>>
      expected_block_neighbors{
          {{domain::Direction<2>::lower_eta(), {3, aligned_orientation}},
           {domain::Direction<2>::upper_eta(), {1, aligned_orientation}},
           {domain::Direction<2>::lower_xi(), {4, aligned_orientation}}},
          {{domain::Direction<2>::lower_eta(), {0, aligned_orientation}},
           {domain::Direction<2>::upper_eta(), {2, aligned_orientation}},
           {domain::Direction<2>::lower_xi(), {4, quarter_turn_cw}}},
          {{domain::Direction<2>::lower_eta(), {1, aligned_orientation}},
           {domain::Direction<2>::upper_eta(), {3, aligned_orientation}},
           {domain::Direction<2>::lower_xi(), {4, half_turn}}},
          {{domain::Direction<2>::lower_eta(), {2, aligned_orientation}},
           {domain::Direction<2>::upper_eta(), {0, aligned_orientation}},
           {domain::Direction<2>::lower_xi(), {4, quarter_turn_ccw}}},
          {{domain::Direction<2>::upper_xi(), {0, aligned_orientation}},
           {domain::Direction<2>::upper_eta(), {1, quarter_turn_ccw}},
           {domain::Direction<2>::lower_xi(), {2, half_turn}},
           {domain::Direction<2>::lower_eta(), {3, quarter_turn_cw}}}};
  const std::vector<std::unordered_set<domain::Direction<2>>>
      expected_external_boundaries{{{domain::Direction<2>::upper_xi()}},
                                   {{domain::Direction<2>::upper_xi()}},
                                   {{domain::Direction<2>::upper_xi()}},
                                   {{domain::Direction<2>::upper_xi()}},
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
  using Wedge2DMap = domain::CoordinateMaps::Wedge2D;
  using Affine = domain::CoordinateMaps::Affine;
  using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  using Equiangular = domain::CoordinateMaps::Equiangular;
  using Equiangular2D =
      domain::CoordinateMaps::ProductOf2Maps<Equiangular, Equiangular>;

  auto coord_maps = domain::make_vector_coordinate_map_base<Frame::Logical,
                                                            TargetFrame>(
      Wedge2DMap{inner_radius, outer_radius, 0.0, 1.0,
                 domain::OrientationMap<2>{std::array<domain::Direction<2>, 2>{
                     {domain::Direction<2>::upper_xi(),
                      domain::Direction<2>::upper_eta()}}},
                 use_equiangular_map},
      Wedge2DMap{inner_radius, outer_radius, 0.0, 1.0,
                 domain::OrientationMap<2>{std::array<domain::Direction<2>, 2>{
                     {domain::Direction<2>::lower_eta(),
                      domain::Direction<2>::upper_xi()}}},
                 use_equiangular_map},
      Wedge2DMap{inner_radius, outer_radius, 0.0, 1.0,
                 domain::OrientationMap<2>{std::array<domain::Direction<2>, 2>{
                     {domain::Direction<2>::lower_xi(),
                      domain::Direction<2>::lower_eta()}}},
                 use_equiangular_map},
      Wedge2DMap{inner_radius, outer_radius, 0.0, 1.0,
                 domain::OrientationMap<2>{std::array<domain::Direction<2>, 2>{
                     {domain::Direction<2>::upper_eta(),
                      domain::Direction<2>::lower_xi()}}},
                 use_equiangular_map});

  if (use_equiangular_map) {
    coord_maps.emplace_back(
        domain::make_coordinate_map_base<Frame::Logical, TargetFrame>(
            Equiangular2D{
                Equiangular(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                            inner_radius / sqrt(2.0)),
                Equiangular(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                            inner_radius / sqrt(2.0))}));
  } else {
    coord_maps.emplace_back(
        domain::make_coordinate_map_base<Frame::Logical, TargetFrame>(
            Affine2D{Affine(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                            inner_radius / sqrt(2.0)),
                     Affine(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                            inner_radius / sqrt(2.0))}));
  }
  test_domain_construction(domain, expected_block_neighbors,
                           expected_external_boundaries, coord_maps);

  test_initial_domain(domain, disk.initial_refinement_levels());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.DomainCreators.Disk.Boundaries.Equiangular",
                  "[Domain][Unit]") {
  const double inner_radius = 1.0, outer_radius = 2.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 2> grid_points{{4, 4}};

  const domain::creators::Disk<Frame::Inertial> disk{
      inner_radius, outer_radius, refinement_level, grid_points, true};
  test_physical_separation(disk.create_domain().blocks());
  test_disk_construction(disk, inner_radius, outer_radius, grid_points,
                         {5, make_array<2>(refinement_level)}, true);
}

SPECTRE_TEST_CASE("Unit.Domain.DomainCreators.Disk.Factory.Equiangular",
                  "[Domain][Unit]") {
  const auto disk =
      test_factory_creation<domain::DomainCreator<2, Frame::Inertial>>(
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
      dynamic_cast<const domain::creators::Disk<Frame::Inertial>&>(*disk),
      inner_radius, outer_radius, grid_points,
      {5, make_array<2>(refinement_level)}, true);
}

SPECTRE_TEST_CASE("Unit.Domain.DomainCreators.Disk.Boundaries.Equidistant",
                  "[Domain][Unit]") {
  const double inner_radius = 1.0, outer_radius = 2.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 2> grid_points{{4, 4}};

  const domain::creators::Disk<Frame::Inertial> disk{
      inner_radius, outer_radius, refinement_level, grid_points, false};
  test_physical_separation(disk.create_domain().blocks());
  test_disk_construction(disk, inner_radius, outer_radius, grid_points,
                         {5, make_array<2>(refinement_level)}, false);
}

SPECTRE_TEST_CASE("Unit.Domain.DomainCreators.Disk.Factory.Equidistant",
                  "[Domain][Unit]") {
  const auto disk =
      test_factory_creation<domain::DomainCreator<2, Frame::Inertial>>(
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
      dynamic_cast<const domain::creators::Disk<Frame::Inertial>&>(*disk),
      inner_radius, outer_radius, grid_points,
      {5, make_array<2>(refinement_level)}, false);
}
