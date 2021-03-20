// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <unordered_set>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"  // IWYU pragma: keep
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Creators/Disk.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/OptionTags.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/MakeArray.hpp"

namespace domain {
namespace {
void test_disk_construction(
    const creators::Disk& disk, const double inner_radius,
    const double outer_radius,
    const std::array<size_t, 2>& expected_wedge_extents,
    const std::vector<std::array<size_t, 2>>& expected_refinement_level,
    const bool use_equiangular_map,
    const std::vector<DirectionMap<
        2, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>&
        expected_boundary_conditions = {}) {
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
  using Wedge2DMap = CoordinateMaps::Wedge<2>;
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
                           expected_external_boundaries, coord_maps,
                           std::numeric_limits<double>::signaling_NaN(), {}, {},
                           expected_boundary_conditions);

  test_initial_domain(domain, disk.initial_refinement_levels());

  Parallel::register_classes_with_charm(typename creators::Disk::maps_list{});
  test_serialization(domain);
}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
create_boundary_condition() {
  return std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<2>>(
      Direction<2>::lower_xi(), 2);
}

auto create_boundary_conditions() {
  std::vector<DirectionMap<
      2, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions_all_blocks{};
  const auto boundary_condition = create_boundary_condition();
  for (size_t block_id = 0; block_id < 4; ++block_id) {
    DirectionMap<2,
                 std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>
        boundary_conditions{};
    boundary_conditions[Direction<2>::upper_xi()] =
        boundary_condition->get_clone();
    boundary_conditions_all_blocks.push_back(std::move(boundary_conditions));
  }
  boundary_conditions_all_blocks.emplace_back();

  return boundary_conditions_all_blocks;
}

void test_disk_boundaries_equiangular() {
  INFO("Disk boundaries equiangular");
  const double inner_radius = 1.0;
  const double outer_radius = 2.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 2> grid_points{{4, 4}};

  const creators::Disk disk{inner_radius, outer_radius, refinement_level,
                            grid_points, true};
  test_physical_separation(disk.create_domain().blocks());
  test_disk_construction(disk, inner_radius, outer_radius, grid_points,
                         {5, make_array<2>(refinement_level)}, true);

  const creators::Disk disk_boundary_conditions{
      inner_radius, outer_radius, refinement_level,
      grid_points,  true,         create_boundary_condition()};
  test_physical_separation(disk_boundary_conditions.create_domain().blocks());
  test_disk_construction(disk_boundary_conditions, inner_radius, outer_radius,
                         grid_points, {5, make_array<2>(refinement_level)},
                         true, create_boundary_conditions());
}

void test_disk_factory_equiangular() {
  INFO("Disk factory equiangular");
  const auto disk = TestHelpers::test_creation<
      std::unique_ptr<DomainCreator<2>>, domain::OptionTags::DomainCreator<2>,
      TestHelpers::domain::BoundaryConditions::
          MetavariablesWithoutBoundaryConditions<2>>(
      "Disk:\n"
      "  InnerRadius: 1\n"
      "  OuterRadius: 3\n"
      "  InitialRefinement: 2\n"
      "  InitialGridPoints: [2,3]\n"
      "  UseEquiangularMap: true\n");

  const double inner_radius = 1.0;
  const double outer_radius = 3.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 2> grid_points{{2, 3}};
  test_disk_construction(dynamic_cast<const creators::Disk&>(*disk),
                         inner_radius, outer_radius, grid_points,
                         {5, make_array<2>(refinement_level)}, true);

  const auto disk_boundary_conditions = TestHelpers::test_creation<
      std::unique_ptr<DomainCreator<2>>, domain::OptionTags::DomainCreator<2>,
      TestHelpers::domain::BoundaryConditions::
          MetavariablesWithBoundaryConditions<2>>(
      "Disk:\n"
      "  InnerRadius: 1\n"
      "  OuterRadius: 3\n"
      "  InitialRefinement: 2\n"
      "  InitialGridPoints: [2,3]\n"
      "  UseEquiangularMap: true\n"
      "  BoundaryCondition:\n"
      "    TestBoundaryCondition:\n"
      "      Direction: lower-xi\n"
      "      BlockId: 2\n");
  test_disk_construction(
      dynamic_cast<const creators::Disk&>(*disk_boundary_conditions),
      inner_radius, outer_radius, grid_points,
      {5, make_array<2>(refinement_level)}, true, create_boundary_conditions());
}

void test_disk_boundaries_equidistant() {
  INFO("Disk boundaries equidistant");
  const double inner_radius = 1.0;
  const double outer_radius = 2.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 2> grid_points{{4, 4}};

  const creators::Disk disk{inner_radius, outer_radius, refinement_level,
                            grid_points, false};
  test_physical_separation(disk.create_domain().blocks());
  test_disk_construction(disk, inner_radius, outer_radius, grid_points,
                         {5, make_array<2>(refinement_level)}, false);

  const creators::Disk disk_boundary_conditions{
      inner_radius, outer_radius, refinement_level,
      grid_points,  false,        create_boundary_condition()};
  test_physical_separation(disk_boundary_conditions.create_domain().blocks());
  test_disk_construction(disk_boundary_conditions, inner_radius, outer_radius,
                         grid_points, {5, make_array<2>(refinement_level)},
                         false, create_boundary_conditions());

  CHECK_THROWS_WITH(
      creators::Disk(inner_radius, outer_radius, refinement_level, grid_points,
                     false,
                     std::make_unique<TestHelpers::domain::BoundaryConditions::
                                          TestPeriodicBoundaryCondition<2>>(),
                     Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Cannot have periodic boundary conditions on a disk"));
  CHECK_THROWS_WITH(
      creators::Disk(inner_radius, outer_radius, refinement_level, grid_points,
                     false,
                     std::make_unique<TestHelpers::domain::BoundaryConditions::
                                          TestNoneBoundaryCondition<2>>(),
                     Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "None boundary condition is not supported. If you would like an "
          "outflow boundary condition, you must use that."));
}

void test_disk_factory_equidistant() {
  INFO("Disk factory equidistant");
  const auto disk = TestHelpers::test_creation<
      std::unique_ptr<DomainCreator<2>>, domain::OptionTags::DomainCreator<2>,
      TestHelpers::domain::BoundaryConditions::
          MetavariablesWithoutBoundaryConditions<2>>(
      "Disk:\n"
      "  InnerRadius: 1\n"
      "  OuterRadius: 3\n"
      "  InitialRefinement: 2\n"
      "  InitialGridPoints: [2,3]\n"
      "  UseEquiangularMap: false\n");

  const double inner_radius = 1.0;
  const double outer_radius = 3.0;
  const size_t refinement_level = 2;
  const std::array<size_t, 2> grid_points{{2, 3}};
  test_disk_construction(dynamic_cast<const creators::Disk&>(*disk),
                         inner_radius, outer_radius, grid_points,
                         {5, make_array<2>(refinement_level)}, false);

  const auto disk_boundary_conditions = TestHelpers::test_creation<
      std::unique_ptr<DomainCreator<2>>, domain::OptionTags::DomainCreator<2>,
      TestHelpers::domain::BoundaryConditions::
          MetavariablesWithBoundaryConditions<2>>(
      "Disk:\n"
      "  InnerRadius: 1\n"
      "  OuterRadius: 3\n"
      "  InitialRefinement: 2\n"
      "  InitialGridPoints: [2,3]\n"
      "  UseEquiangularMap: false\n"
      "  BoundaryCondition:\n"
      "    TestBoundaryCondition:\n"
      "      Direction: lower-xi\n"
      "      BlockId: 2\n");
  test_disk_construction(
      dynamic_cast<const creators::Disk&>(*disk_boundary_conditions),
      inner_radius, outer_radius, grid_points,
      {5, make_array<2>(refinement_level)}, false,
      create_boundary_conditions());
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
