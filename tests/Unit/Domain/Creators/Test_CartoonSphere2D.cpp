// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numpy/npy_common.h>
#include <pup.h>
#include <unordered_set>
#include <vector>

#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Creators/CartoonSphere2D.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/Structure/BlockNeighbors.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace domain {
namespace {

template <typename... FuncsOfTime>
void test_sphere_construction(
    const creators::CartoonSphere2D& sphere, const double inner_radius,
    const double outer_radius,
    const std::vector<std::array<size_t, 2>>& expected_refinement_levels_param,
    const std::vector<std::array<size_t, 2>>& expected_extents_param,
    const std::vector<double>& expected_radial_partitioning,
    const bool use_equiangular_map,
    const std::variant<domain::creators::detail::Excision,
                       domain::creators::detail::InnerSquare>
        interior,
    const bool expect_boundary_conditions = false,
    const std::tuple<std::pair<std::string, FuncsOfTime>...>&
        expected_functions_of_time = {},
    const std::vector<std::unique_ptr<domain::CoordinateMapBase<
        Frame::Grid, Frame::Inertial, 3>>>& expected_grid_to_inertial_maps = {},
    const std::unordered_map<std::string, double>& initial_expiration_times =
        {}) {
  const auto domain = TestHelpers::domain::creators::test_domain_creator(
      sphere, expect_boundary_conditions, false, std::vector<double>{5.0});
  const bool fill_interior =
      std::holds_alternative<domain::creators::detail::InnerSquare>(interior);
  const size_t num_shells = expected_radial_partitioning.size() + 1;
  const size_t num_blocks = num_shells * 3 + (fill_interior ? 1 : 0);
  CHECK(sphere.grid_anchors().empty());

  const auto block_groups = sphere.block_groups();
  std::vector<std::string> expected_block_names{};
  expected_block_names.reserve(num_blocks);
  std::unordered_map<std::string, std::unordered_set<std::string>>
      expected_block_groups{num_shells};
  for (size_t i = 0; i < num_shells; ++i) {
    const std::string shell = "Shell" + std::to_string(i);
    expected_block_groups[shell];
    expected_block_names.push_back(shell + "_LowerY");
    expected_block_groups[shell].insert(expected_block_names.back());
    expected_block_names.push_back(shell + "_UpperX");
    expected_block_groups[shell].insert(expected_block_names.back());
    expected_block_names.push_back(shell + "_UpperY");
    expected_block_groups[shell].insert(expected_block_names.back());
  }
  if (fill_interior) {
    const std::string shell = "Shell" + std::to_string(num_shells - 1);
    expected_block_names.emplace_back(shell + "_HalfSquare");
    expected_block_groups[shell].insert(expected_block_names.back());
  }
  CHECK(sphere.block_names() == expected_block_names);
  for (auto& [key, value] : expected_block_groups) {
    CHECK(block_groups.at(key) == value);
  }

  const OrientationMap<3> aligned = OrientationMap<3>::create_aligned();
  const OrientationMap<3> turn_ccw(std::array<Direction<3>, 3>{
      {Direction<3>::lower_eta(), Direction<3>::upper_xi(),
       Direction<3>::upper_zeta()}});
  const OrientationMap<3> half_turn(std::array<Direction<3>, 3>{
      {Direction<3>::lower_xi(), Direction<3>::lower_eta(),
       Direction<3>::upper_zeta()}});
  const OrientationMap<3> turn_cw(std::array<Direction<3>, 3>{
      {Direction<3>::upper_eta(), Direction<3>::lower_xi(),
       Direction<3>::upper_zeta()}});
  std::vector<DirectionMap<3, BlockNeighbors<3>>> expected_neighbors{
      num_blocks};
  for (size_t i = 0; i < num_shells; ++i) {
    // between this shell
    expected_neighbors[3 * i + 0].emplace(std::pair(
        Direction<3>::upper_xi(), BlockNeighbors<3>(3 * i + 1, half_turn)));

    expected_neighbors[3 * i + 1].emplace(std::pair(
        Direction<3>::upper_xi(), BlockNeighbors<3>(3 * i + 0, half_turn)));
    expected_neighbors[3 * i + 1].emplace(std::pair(
        Direction<3>::lower_xi(), BlockNeighbors<3>(3 * i + 2, aligned)));

    expected_neighbors[3 * i + 2].emplace(std::pair(
        Direction<3>::upper_xi(), BlockNeighbors<3>(3 * i + 1, aligned)));

    // +r direction
    if (i != 0) {
      expected_neighbors[3 * i + 0].emplace(
          std::pair(Direction<3>::lower_eta(),
                    BlockNeighbors<3>((i - 1) * 3 + 0, aligned)));
      expected_neighbors[3 * i + 1].emplace(
          std::pair(Direction<3>::upper_eta(),
                    BlockNeighbors<3>((i - 1) * 3 + 1, aligned)));
      expected_neighbors[3 * i + 2].emplace(
          std::pair(Direction<3>::upper_eta(),
                    BlockNeighbors<3>((i - 1) * 3 + 2, aligned)));
    }
    // -r direction
    if (i != num_shells - 1) {
      expected_neighbors[3 * i + 0].emplace(
          std::pair(Direction<3>::upper_eta(),
                    BlockNeighbors<3>((i + 1) * 3 + 0, aligned)));
      expected_neighbors[3 * i + 1].emplace(
          std::pair(Direction<3>::lower_eta(),
                    BlockNeighbors<3>((i + 1) * 3 + 1, aligned)));
      expected_neighbors[3 * i + 2].emplace(
          std::pair(Direction<3>::lower_eta(),
                    BlockNeighbors<3>((i + 1) * 3 + 2, aligned)));
    } else if (fill_interior) {
      expected_neighbors[3 * i + 0].emplace(std::pair(
          Direction<3>::upper_eta(), BlockNeighbors<3>(3 * i + 3, aligned)));
      expected_neighbors[3 * i + 1].emplace(std::pair(
          Direction<3>::lower_eta(), BlockNeighbors<3>(3 * i + 3, turn_ccw)));
      expected_neighbors[3 * i + 2].emplace(std::pair(
          Direction<3>::lower_eta(), BlockNeighbors<3>(3 * i + 3, aligned)));

      expected_neighbors[3 * i + 3].emplace(std::pair(
          Direction<3>::lower_eta(), BlockNeighbors<3>(3 * i + 0, aligned)));
      expected_neighbors[3 * i + 3].emplace(std::pair(
          Direction<3>::upper_xi(), BlockNeighbors<3>(3 * i + 1, turn_cw)));
      expected_neighbors[3 * i + 3].emplace(std::pair(
          Direction<3>::upper_eta(), BlockNeighbors<3>(3 * i + 2, aligned)));
    }
  }

  std::vector<std::unordered_set<Direction<3>>> expected_external_boundaries{
      num_blocks};
  expected_external_boundaries[0].emplace(Direction<3>::lower_eta());
  expected_external_boundaries[1].emplace(Direction<3>::upper_eta());
  expected_external_boundaries[2].emplace(Direction<3>::upper_eta());

  for (size_t i = 0; i < num_shells; ++i) {
    expected_external_boundaries[3 * i + 0].emplace(Direction<3>::lower_xi());
    expected_external_boundaries[3 * i + 2].emplace(Direction<3>::lower_xi());
  }
  if (fill_interior) {
    expected_external_boundaries[num_blocks - 1].emplace(
        Direction<3>::lower_xi());
  } else {
    expected_external_boundaries[(num_shells - 1) * 3 + 0].emplace(
        Direction<3>::upper_eta());
    expected_external_boundaries[(num_shells - 1) * 3 + 1].emplace(
        Direction<3>::lower_eta());
    expected_external_boundaries[(num_shells - 1) * 3 + 2].emplace(
        Direction<3>::lower_eta());
  }

  std::vector<std::array<size_t, 2>> expected_extents_temp{};
  std::vector<std::array<size_t, 3>> expected_extents{};
  expected_extents_temp.reserve(num_blocks);
  expected_extents.reserve(num_blocks);
  if (fill_interior) {
    expected_extents_temp.push_back(expected_extents_param[0]);
  }
  for (size_t i = 0; i < num_shells; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      expected_extents_temp.push_back(expected_extents_param[i]);
    }
  }
  std::reverse(expected_extents_temp.begin(), expected_extents_temp.end());
  std::transform(expected_extents_temp.begin(), expected_extents_temp.end(),
                 std::back_inserter(expected_extents),
                 [](const std::array<size_t, 2>& arr) -> std::array<size_t, 3> {
                   return {arr[1], arr[0], 1};
                 });
  if (fill_interior) {
    expected_extents.back()[1] = expected_extents.back()[0];
  }
  CHECK(sphere.initial_extents() == expected_extents);

  std::vector<std::array<size_t, 2>> expected_refinement_levels_temp{};
  std::vector<std::array<size_t, 3>> expected_refinement_levels{};
  expected_refinement_levels_temp.reserve(num_blocks);
  expected_refinement_levels.reserve(num_blocks);
  if (fill_interior) {
    expected_refinement_levels_temp.push_back(
        expected_refinement_levels_param[0]);
  }
  for (size_t i = 0; i < num_shells; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      expected_refinement_levels_temp.push_back(
          expected_refinement_levels_param[i]);
    }
  }
  std::reverse(expected_refinement_levels_temp.begin(),
               expected_refinement_levels_temp.end());
  size_t n = 0;
  std::transform(
      expected_refinement_levels_temp.begin(),
      expected_refinement_levels_temp.end(),
      std::back_inserter(expected_refinement_levels),
      [&n](const std::array<size_t, 2>& arr) -> std::array<size_t, 3> {
        const size_t shift = (n % 3 == 0 or n % 3 == 2) and arr[1] != 0 ? 1 : 0;
        ++n;
        return {arr[1] - shift, arr[0], 0};
      });
  if (fill_interior) {
    expected_refinement_levels.back()[1] =
        expected_refinement_levels_temp.back()[1];
    expected_refinement_levels.back()[0] =
        expected_refinement_levels.back()[1] - 1;
  }
  CHECK(sphere.initial_refinement_levels() == expected_refinement_levels);

  using Identity1D = CoordinateMaps::Identity<1>;
  using Wedge2DMap = CoordinateMaps::Wedge<2>;
  using Wedge3DPrism =
      domain::CoordinateMaps::ProductOf2Maps<Wedge2DMap, Identity1D>;
  using Affine = CoordinateMaps::Affine;
  using Affine2D = CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  using Equiangular = CoordinateMaps::Equiangular;
  using Equiangular2D =
      CoordinateMaps::ProductOf2Maps<Equiangular, Equiangular>;
  using Rotation3D = CoordinateMaps::DiscreteRotation<3>;

  using TargetFrame = tmpl::conditional_t<sizeof...(FuncsOfTime) == 0,
                                          Frame::Inertial, Frame::Grid>;
  std::vector<
      std::unique_ptr<CoordinateMapBase<Frame::BlockLogical, TargetFrame, 3>>>
      coord_maps{};
  coord_maps.reserve(num_blocks);

  for (size_t i = 0; i < num_shells; ++i) {
    const bool on_inner = i == num_shells - 1;
    const double inner_radius_i =
        on_inner
            ? inner_radius
            : expected_radial_partitioning[expected_radial_partitioning.size() -
                                           1 - i];
    const double outer_radius_i =
        i == 0
            ? outer_radius
            : expected_radial_partitioning[expected_radial_partitioning.size() -
                                           i];
    const double inner_sphericity = fill_interior and on_inner ? 0.0 : 1.0;
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::BlockLogical, TargetFrame>(
            Rotation3D{OrientationMap<3>{std::array<Direction<3>, 3>{
                {Direction<3>::lower_eta(), Direction<3>::upper_xi(),
                 Direction<3>::upper_zeta()}}}},
            Wedge3DPrism{
                Wedge2DMap{
                    inner_radius_i, outer_radius_i, inner_sphericity, 1.0,
                    OrientationMap<2>{std::array<Direction<2>, 2>{
                        {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}},
                    use_equiangular_map,
                    domain::CoordinateMaps::Wedge<2>::WedgeHalves::UpperOnly},
                Identity1D{}}));
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::BlockLogical, TargetFrame>(
            Rotation3D{OrientationMap<3>{std::array<Direction<3>, 3>{
                {Direction<3>::upper_eta(), Direction<3>::lower_xi(),
                 Direction<3>::upper_zeta()}}}},
            Wedge3DPrism{
                Wedge2DMap{
                    inner_radius_i, outer_radius_i, inner_sphericity, 1.0,
                    OrientationMap<2>{std::array<Direction<2>, 2>{
                        {Direction<2>::upper_xi(), Direction<2>::upper_eta()}}},
                    use_equiangular_map},
                Identity1D{}}));
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::BlockLogical, TargetFrame>(
            Rotation3D{OrientationMap<3>{std::array<Direction<3>, 3>{
                {Direction<3>::upper_eta(), Direction<3>::lower_xi(),
                 Direction<3>::upper_zeta()}}}},
            Wedge3DPrism{
                Wedge2DMap{
                    inner_radius_i, outer_radius_i, inner_sphericity, 1.0,
                    OrientationMap<2>{std::array<Direction<2>, 2>{
                        {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}},
                    use_equiangular_map,
                    domain::CoordinateMaps::Wedge<2>::WedgeHalves::LowerOnly},
                Identity1D{}}));
  }
  if (fill_interior) {
    if (use_equiangular_map) {
      coord_maps.emplace_back(
          make_coordinate_map_base<Frame::BlockLogical, TargetFrame>(
              CoordinateMaps::ProductOf2Maps<Equiangular2D, Identity1D>{
                  Equiangular2D{
                      Equiangular(-3.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                                  inner_radius / sqrt(2.0)),
                      Equiangular(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                                  inner_radius / sqrt(2.0))},
                  Identity1D{}}));
    } else {
      coord_maps.emplace_back(
          make_coordinate_map_base<Frame::BlockLogical, TargetFrame>(
              CoordinateMaps::ProductOf2Maps<Affine2D, Identity1D>{
                  Affine2D{Affine(-1.0, 1.0, 0.0, inner_radius / sqrt(2.0)),
                           Affine(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                                  inner_radius / sqrt(2.0))},
                  Identity1D{}}));
    }
  }
  test_domain_construction(
      domain, expected_neighbors, expected_external_boundaries, coord_maps, 9.0,
      sphere.functions_of_time(), expected_grid_to_inertial_maps);
  TestHelpers::domain::creators::test_functions_of_time(
      sphere, expected_functions_of_time, initial_expiration_times);
}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
create_boundary_condition1() {
  return std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>>(
      Direction<3>::upper_eta(), 1);
}
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
create_boundary_condition2() {
  return std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>>(
      Direction<3>::lower_xi(), 2);
}
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
create_boundary_condition3() {
  return std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>>(
      Direction<3>::lower_eta(), 3);
}

void test_sphere_boundaries() {
  INFO("CartoonSphere2D boundaries");
  const double inner_radius = 1.0;
  const double outer_radius = 2.0;
  const std::array<size_t, 2> refinement_level_arr{{2, 3}};
  const std::vector<std::array<size_t, 2>> refinement_level_vec{
      {4, 4}, {3, 5}, {2, 6}};
  const std::vector<double> radial_partitioning{{1.3, 1.8}};
  const std::vector<double> radial_partitioning_empty{};
  const std::array<size_t, 2> grid_points_arr{{4, 4}};
  const std::vector<std::array<size_t, 2>> grid_points_vec{
      {4, 4}, {5, 3}, {6, 7}};
  const domain::creators::detail::InnerSquare fill_center{0.0};

  {
    INFO("No BC, no refinement, equiangular");
    const creators::CartoonSphere2D sphere{inner_radius,
                                           outer_radius,
                                           refinement_level_arr,
                                           grid_points_arr,
                                           radial_partitioning_empty,
                                           true,
                                           fill_center};
    test_sphere_construction(
        sphere, inner_radius, outer_radius,
        std::vector<std::array<size_t, 2>>{1, refinement_level_arr},
        std::vector<std::array<size_t, 2>>{1, grid_points_arr},
        radial_partitioning_empty, true, fill_center);
  }
  {
    INFO("No BC, no refinement, affine");
    const creators::CartoonSphere2D sphere{inner_radius,
                                           outer_radius,
                                           refinement_level_arr,
                                           grid_points_arr,
                                           radial_partitioning_empty,
                                           false,
                                           fill_center};
    test_sphere_construction(
        sphere, inner_radius, outer_radius,
        std::vector<std::array<size_t, 2>>{1, refinement_level_arr},
        std::vector<std::array<size_t, 2>>{1, grid_points_arr},
        radial_partitioning_empty, false, fill_center);
  }
  {
    INFO("No BC, with refinement and extents array");
    const creators::CartoonSphere2D sphere{inner_radius,
                                           outer_radius,
                                           refinement_level_vec,
                                           grid_points_arr,
                                           radial_partitioning,
                                           true,
                                           fill_center};
    test_sphere_construction(
        sphere, inner_radius, outer_radius, refinement_level_vec,
        std::vector<std::array<size_t, 2>>{3, grid_points_arr},
        radial_partitioning, true, fill_center);
  }
  {
    INFO("No BC, with refinement and refinement array");
    const creators::CartoonSphere2D sphere{inner_radius,
                                           outer_radius,
                                           refinement_level_arr,
                                           grid_points_vec,
                                           radial_partitioning,
                                           true,
                                           fill_center};
    test_sphere_construction(
        sphere, inner_radius, outer_radius,
        std::vector<std::array<size_t, 2>>{3, refinement_level_arr},
        grid_points_vec, radial_partitioning, true, fill_center);
  }
  {
    INFO("With BC, no excision");
    const creators::CartoonSphere2D sphere{inner_radius,
                                           outer_radius,
                                           refinement_level_vec,
                                           grid_points_vec,
                                           radial_partitioning,
                                           true,
                                           fill_center,
                                           nullptr,
                                           create_boundary_condition1(),
                                           create_boundary_condition2()};
    test_sphere_construction(sphere, inner_radius, outer_radius,
                             refinement_level_vec, grid_points_vec,
                             radial_partitioning, true, fill_center, true);
  }
  {
    INFO("With BC, with excision");
    domain::creators::detail::Excision excise1{create_boundary_condition3()};
    domain::creators::detail::Excision excise2{create_boundary_condition3()};
    const creators::CartoonSphere2D sphere{inner_radius,
                                           outer_radius,
                                           refinement_level_vec,
                                           grid_points_vec,
                                           radial_partitioning,
                                           true,
                                           std::move(excise1),
                                           nullptr,
                                           create_boundary_condition1(),
                                           create_boundary_condition2()};
    test_sphere_construction(
        sphere, inner_radius, outer_radius, refinement_level_vec,
        grid_points_vec, radial_partitioning, true, std::move(excise2), true);
  }
  {
    INFO("With BC, with excision and no partitioning");
    domain::creators::detail::Excision excise1{create_boundary_condition3()};
    domain::creators::detail::Excision excise2{create_boundary_condition3()};
    const creators::CartoonSphere2D sphere{inner_radius,
                                           outer_radius,
                                           refinement_level_arr,
                                           grid_points_arr,
                                           radial_partitioning_empty,
                                           true,
                                           std::move(excise1),
                                           nullptr,
                                           create_boundary_condition1(),
                                           create_boundary_condition2()};
    test_sphere_construction(
        sphere, inner_radius, outer_radius,
        std::vector<std::array<size_t, 2>>{1, refinement_level_arr},
        std::vector<std::array<size_t, 2>>{1, grid_points_arr},
        radial_partitioning_empty, true, std::move(excise2), true);
  }
}

void test_sphere_factory() {
  INFO("CartoonSphere2D factory");
  using Translation3D = CoordinateMaps::TimeDependent::Translation<3>;
  const auto sphere = TestHelpers::test_option_tag<
      domain::OptionTags::DomainCreator<3>,
      TestHelpers::domain::BoundaryConditions::
          MetavariablesWithoutBoundaryConditions<
              3, domain::creators::CartoonSphere2D>>(
      "CartoonSphere2D:\n"
      "  InnerRadius: 1.0\n"
      "  OuterRadius: 5.0\n"
      "  InitialRefinement:\n"
      "    - [3, 4]\n"
      "    - [3, 3]\n"
      "    - [2, 4]\n"
      "  InitialGridPoints: [2,3]\n"
      "  RadialPartitioning: [3.5, 4.5]\n"
      "  UseEquiangularMap: true\n"
      "  Interior:\n"
      "    FillWithSphericity: 0.0\n"
      "  TimeDependence: None\n");
  const double inner_radius = 1.0;
  const double outer_radius = 5.0;
  const std::vector<std::array<size_t, 2>> refinement_levels{
      {3, 4}, {3, 3}, {2, 4}};
  const std::vector<std::array<size_t, 2>> grid_points{3, {2, 3}};
  const std::vector<double> radial_partition{{3.5, 4.5}};
  const domain::creators::detail::InnerSquare fill_center{0.0};
  test_sphere_construction(
      dynamic_cast<const creators::CartoonSphere2D&>(*sphere), inner_radius,
      outer_radius, refinement_levels, grid_points, radial_partition, true,
      fill_center);

  const auto sphere_boundary_conditions = TestHelpers::test_option_tag<
      domain::OptionTags::DomainCreator<3>,
      TestHelpers::domain::BoundaryConditions::
          MetavariablesWithBoundaryConditions<
              3, domain::creators::CartoonSphere2D>>(
      "CartoonSphere2D:\n"
      "  InnerRadius: 1.0\n"
      "  OuterRadius: 5.0\n"
      "  InitialRefinement:\n"
      "    - [3, 4]\n"
      "    - [3, 3]\n"
      "    - [2, 4]\n"
      "  InitialGridPoints: [2,3]\n"
      "  RadialPartitioning: [3.5, 4.5]\n"
      "  UseEquiangularMap: false\n"
      "  TimeDependence: None\n"
      "  Interior:\n"
      "    ExciseWithBoundaryCondition:\n"
      "      TestBoundaryCondition:\n"
      "        Direction: lower-eta\n"
      "        BlockId: 3\n"
      "  YAxisBoundaryCondition:\n"
      "    TestBoundaryCondition:\n"
      "      Direction: upper-eta\n"
      "      BlockId: 1\n"
      "  OuterBoundaryCondition:\n"
      "    TestBoundaryCondition:\n"
      "      Direction: lower-xi\n"
      "      BlockId: 2\n");
  domain::creators::detail::Excision excise1{create_boundary_condition3()};
  test_sphere_construction(dynamic_cast<const creators::CartoonSphere2D&>(
                               *sphere_boundary_conditions),
                           inner_radius, outer_radius, refinement_levels,
                           grid_points, radial_partition, false,
                           std::move(excise1), true);

  INFO("With TimeDependent Map");
  const auto sphere_time_dependent = TestHelpers::test_option_tag<
      domain::OptionTags::DomainCreator<3>,
      TestHelpers::domain::BoundaryConditions::
          MetavariablesWithBoundaryConditions<
              3, domain::creators::CartoonSphere2D>>(
      "CartoonSphere2D:\n"
      "  InnerRadius: 1.0\n"
      "  OuterRadius: 5.0\n"
      "  InitialRefinement:\n"
      "    - [3, 4]\n"
      "    - [3, 3]\n"
      "    - [2, 4]\n"
      "  InitialGridPoints: [2,3]\n"
      "  RadialPartitioning: [3.5, 4.5]\n"
      "  UseEquiangularMap: false\n"
      "  TimeDependence:\n"
      "    UniformTranslation:\n"
      "      InitialTime: 2.3\n"
      "      Velocity: [1.1, -0.1, 0.3]\n"
      "  Interior:\n"
      "    ExciseWithBoundaryCondition:\n"
      "      TestBoundaryCondition:\n"
      "        Direction: lower-eta\n"
      "        BlockId: 3\n"
      "  YAxisBoundaryCondition:\n"
      "    TestBoundaryCondition:\n"
      "      Direction: upper-eta\n"
      "      BlockId: 1\n"
      "  OuterBoundaryCondition:\n"
      "    TestBoundaryCondition:\n"
      "      Direction: lower-xi\n"
      "      BlockId: 2\n");
  domain::creators::detail::Excision excise2{create_boundary_condition3()};
  domain::creators::detail::Excision excise3{create_boundary_condition3()};
  const double initial_time = 2.3;
  const DataVector velocity{{1.1, -0.1, 0.3}};
  const std::string f_of_t_name = "Translation";
  std::unordered_map<std::string, double> initial_expiration_times{};
  initial_expiration_times[f_of_t_name] = 9.0;
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>>
      time_map_vec{};
  time_map_vec.reserve(9);
  for (size_t i = 0; i < 9; ++i) {
    time_map_vec.push_back(
        make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            Translation3D{f_of_t_name}));
  }
  // without expiration times
  test_sphere_construction(
      dynamic_cast<const creators::CartoonSphere2D&>(*sphere_time_dependent),
      inner_radius, outer_radius, refinement_levels, grid_points,
      radial_partition, false, std::move(excise2), true,
      std::make_tuple(
          std::pair<std::string,
                    domain::FunctionsOfTime::PiecewisePolynomial<2>>{
              f_of_t_name,
              {initial_time,
               std::array<DataVector, 3>{{{3, 0.0}, velocity, {3, 0.0}}},
               std::numeric_limits<double>::infinity()}}),
      time_map_vec);
  // with expiration times
  test_sphere_construction(
      dynamic_cast<const creators::CartoonSphere2D&>(*sphere_time_dependent),
      inner_radius, outer_radius, refinement_levels, grid_points,
      radial_partition, false, std::move(excise3), true,
      std::make_tuple(
          std::pair<std::string,
                    domain::FunctionsOfTime::PiecewisePolynomial<2>>{
              f_of_t_name,
              {initial_time,
               std::array<DataVector, 3>{{{3, 0.0}, velocity, {3, 0.0}}},
               initial_expiration_times[f_of_t_name]}}),
      time_map_vec, initial_expiration_times);
}

void test_sphere_errors() {
  INFO("CartoonSphere2D testing errors");
  const double inner_radius = 1.0;
  const double high_inner_radius = 3.0;
  const double outer_radius = 2.0;
  const std::vector<std::array<size_t, 2>> refinement_level_vec{
      {3, 4}, {3, 5}, {4, 6}};
  const std::vector<std::array<size_t, 2>> refinement_level_short{{3, 4},
                                                                  {3, 2}};
  const std::vector<std::array<size_t, 2>> grid_points_vec{
      {4, 4}, {5, 3}, {6, 7}};
  const std::vector<double> radial_partitioning{{1.3, 1.8}};
  const std::vector<double> radial_partitioning_unordered{{1.6, 1.3}};
  const std::vector<double> radial_partitioning_low{{0.8, 1.3}};
  const std::vector<double> radial_partitioning_high{{1.6, 3.3}};
  const domain::creators::detail::InnerSquare fill_center{0.0};

  CHECK_THROWS_WITH(
      creators::CartoonSphere2D(
          high_inner_radius, outer_radius, refinement_level_vec,
          grid_points_vec, radial_partitioning, false, fill_center, nullptr,
          nullptr, nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("Inner radius must be smaller than "
                                         "outer radius, but inner radius is "));
  CHECK_THROWS_WITH(
      creators::CartoonSphere2D(
          inner_radius, outer_radius, refinement_level_vec, grid_points_vec,
          radial_partitioning_unordered, false, fill_center, nullptr, nullptr,
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Specify radial partitioning in ascending order."));
  CHECK_THROWS_WITH(
      creators::CartoonSphere2D(
          inner_radius, outer_radius, refinement_level_vec, grid_points_vec,
          radial_partitioning_low, false, fill_center, nullptr, nullptr,
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "First radial partition must be larger than the inner"));
  CHECK_THROWS_WITH(
      creators::CartoonSphere2D(
          inner_radius, outer_radius, refinement_level_vec, grid_points_vec,
          radial_partitioning_high, false, fill_center, nullptr, nullptr,
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Last radial partition must be smaller than the outer"));
  CHECK_THROWS_WITH(
      creators::CartoonSphere2D(
          inner_radius, outer_radius, refinement_level_short, grid_points_vec,
          radial_partitioning, false, fill_center, nullptr, nullptr, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("InitialRefinement must be one larger "
                                         "than RadialPartitioning (size"));
  CHECK_THROWS_WITH(
      creators::CartoonSphere2D(
          inner_radius, outer_radius, refinement_level_vec, grid_points_vec,
          radial_partitioning, false, fill_center, nullptr,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Must specify either both inner and outer boundary conditions "));
  CHECK_THROWS_WITH(
      creators::CartoonSphere2D(
          inner_radius, outer_radius, refinement_level_vec, grid_points_vec,
          radial_partitioning, false, fill_center, nullptr,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestNoneBoundaryCondition<3>>(),
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestNoneBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "None boundary condition is not supported. If you would like an "));
  CHECK_THROWS_WITH(
      creators::CartoonSphere2D(
          inner_radius, outer_radius, refinement_level_vec, grid_points_vec,
          radial_partitioning, false, fill_center, nullptr,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Cannot have periodic boundary conditions on a 2D sphere."));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.CartoonSphere2D", "[Domain][Unit]") {
  test_sphere_boundaries();
  test_sphere_factory();
  test_sphere_errors();
}
}  // namespace domain
