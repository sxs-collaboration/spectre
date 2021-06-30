// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <unordered_set>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/RotatedRectangles.hpp"
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

namespace domain {
namespace {
void test_rotated_rectangles_construction(
    const creators::RotatedRectangles& rotated_rectangles,
    const std::array<double, 2>& lower_bound,
    const std::array<double, 2>& midpoint,
    const std::array<double, 2>& upper_bound,
    const std::vector<std::array<size_t, 2>>& expected_extents,
    const std::vector<std::array<size_t, 2>>& expected_refinement_level,
    const std::vector<DirectionMap<2, BlockNeighbor<2>>>&
        expected_block_neighbors,
    const std::vector<std::unordered_set<Direction<2>>>&
        expected_external_boundaries,
    const std::vector<DirectionMap<
        2, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>&
        expected_boundary_conditions = {}) noexcept {
  const auto domain = rotated_rectangles.create_domain();

  CHECK(domain.blocks().size() == expected_extents.size());
  CHECK(domain.blocks().size() == expected_refinement_level.size());
  CHECK(rotated_rectangles.initial_extents() == expected_extents);
  CHECK(rotated_rectangles.initial_refinement_levels() ==
        expected_refinement_level);

  using Affine = CoordinateMaps::Affine;
  using Affine2D = CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  using DiscreteRotation2D = CoordinateMaps::DiscreteRotation<2>;

  const Affine lower_x_map(-1.0, 1.0, lower_bound[0], midpoint[0]);
  const Affine upper_x_map(-1.0, 1.0, midpoint[0], upper_bound[0]);
  const Affine lower_y_map(-1.0, 1.0, lower_bound[1], midpoint[1]);
  const Affine upper_y_map(-1.0, 1.0, midpoint[1], upper_bound[1]);
  std::vector<
      std::unique_ptr<CoordinateMapBase<Frame::Logical, Frame::Inertial, 2>>>
      coord_maps;
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Affine2D(lower_x_map, lower_y_map)));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          DiscreteRotation2D{OrientationMap<2>{std::array<Direction<2>, 2>{
              {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}}},
          Affine2D(upper_x_map, lower_y_map)));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          DiscreteRotation2D{OrientationMap<2>{std::array<Direction<2>, 2>{
              {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}}},
          Affine2D(lower_x_map, upper_y_map)));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          DiscreteRotation2D{OrientationMap<2>{std::array<Direction<2>, 2>{
              {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}}},
          Affine2D(upper_x_map, upper_y_map)));
  test_domain_construction(domain, expected_block_neighbors,
                           expected_external_boundaries, coord_maps,
                           std::numeric_limits<double>::signaling_NaN(), {}, {},
                           expected_boundary_conditions);
  test_initial_domain(domain, rotated_rectangles.initial_refinement_levels());

  Parallel::register_classes_with_charm(
      typename domain::creators::RotatedRectangles::maps_list{});
  test_serialization(domain);
}

auto create_boundary_conditions() {
  std::vector<DirectionMap<
      2, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      expected_boundary_conditions{4};
  const auto boundary_condition = std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<2>>(
      Direction<2>::lower_xi(), 2);
  expected_boundary_conditions[0][Direction<2>::lower_xi()] =
      boundary_condition->get_clone();
  expected_boundary_conditions[0][Direction<2>::lower_eta()] =
      boundary_condition->get_clone();

  expected_boundary_conditions[1][Direction<2>::lower_xi()] =
      boundary_condition->get_clone();
  expected_boundary_conditions[1][Direction<2>::upper_eta()] =
      boundary_condition->get_clone();

  expected_boundary_conditions[2][Direction<2>::upper_xi()] =
      boundary_condition->get_clone();
  expected_boundary_conditions[2][Direction<2>::upper_eta()] =
      boundary_condition->get_clone();

  expected_boundary_conditions[3][Direction<2>::lower_xi()] =
      boundary_condition->get_clone();
  expected_boundary_conditions[3][Direction<2>::upper_eta()] =
      boundary_condition->get_clone();
  return expected_boundary_conditions;
}

void test_rotated_rectangles() {
  INFO("Rotated rectangles");
  const std::vector<std::array<size_t, 2>> grid_points{
      {{4, 2}}, {{1, 2}}, {{3, 4}}, {{3, 1}}};
  const std::vector<std::array<size_t, 2>> refinement_level{
      {{0, 1}}, {{0, 1}}, {{1, 0}}, {{1, 0}}};
  const std::array<double, 2> lower_bound{{-1.2, -2.0}};
  const std::array<double, 2> midpoint{{-0.6, 0.2}};
  const std::array<double, 2> upper_bound{{0.8, 3.0}};
  const OrientationMap<2> half_turn{std::array<Direction<2>, 2>{
      {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}};
  const OrientationMap<2> quarter_turn_cw{std::array<Direction<2>, 2>{
      {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}};
  const OrientationMap<2> quarter_turn_ccw{std::array<Direction<2>, 2>{
      {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}};

  const creators::RotatedRectangles rotated_rectangles{
      lower_bound,
      midpoint,
      upper_bound,
      {{refinement_level[0][0], refinement_level[0][1]}},
      {{{{grid_points[0][0], grid_points[1][0]}},
        {{grid_points[0][1], grid_points[2][0]}}}},
      {{false, false}}};
  test_rotated_rectangles_construction(
      rotated_rectangles, lower_bound, midpoint, upper_bound, grid_points,
      refinement_level,
      std::vector<DirectionMap<2, BlockNeighbor<2>>>{
          {{Direction<2>::upper_xi(), {1, half_turn}},
           {Direction<2>::upper_eta(), {2, quarter_turn_ccw}}},
          {{Direction<2>::upper_xi(), {0, half_turn}},
           {Direction<2>::lower_eta(), {3, quarter_turn_ccw}}},
          {{Direction<2>::lower_xi(), {0, quarter_turn_cw}},
           {Direction<2>::lower_eta(), {3, half_turn}}},
          {{Direction<2>::upper_xi(), {1, quarter_turn_cw}},
           {Direction<2>::lower_eta(), {2, half_turn}}}},
      std::vector<std::unordered_set<Direction<2>>>{
          {Direction<2>::lower_xi(), Direction<2>::lower_eta()},
          {Direction<2>::lower_xi(), Direction<2>::upper_eta()},
          {Direction<2>::upper_xi(), Direction<2>::upper_eta()},
          {Direction<2>::lower_xi(), Direction<2>::upper_eta()}});
  test_physical_separation(rotated_rectangles.create_domain().blocks());

  const creators::RotatedRectangles rotated_rectangles_boundary_conditions{
      lower_bound,
      midpoint,
      upper_bound,
      {{refinement_level[0][0], refinement_level[0][1]}},
      {{{{grid_points[0][0], grid_points[1][0]}},
        {{grid_points[0][1], grid_points[2][0]}}}},
      std::make_unique<
          TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<2>>(
          Direction<2>::lower_xi(), 2)};
  const std::vector<DirectionMap<
      2, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      expected_boundary_conditions = create_boundary_conditions();

  test_rotated_rectangles_construction(
      rotated_rectangles_boundary_conditions, lower_bound, midpoint,
      upper_bound, grid_points, refinement_level,
      std::vector<DirectionMap<2, BlockNeighbor<2>>>{
          {{Direction<2>::upper_xi(), {1, half_turn}},
           {Direction<2>::upper_eta(), {2, quarter_turn_ccw}}},
          {{Direction<2>::upper_xi(), {0, half_turn}},
           {Direction<2>::lower_eta(), {3, quarter_turn_ccw}}},
          {{Direction<2>::lower_xi(), {0, quarter_turn_cw}},
           {Direction<2>::lower_eta(), {3, half_turn}}},
          {{Direction<2>::upper_xi(), {1, quarter_turn_cw}},
           {Direction<2>::lower_eta(), {2, half_turn}}}},
      std::vector<std::unordered_set<Direction<2>>>{
          {Direction<2>::lower_xi(), Direction<2>::lower_eta()},
          {Direction<2>::lower_xi(), Direction<2>::upper_eta()},
          {Direction<2>::upper_xi(), Direction<2>::upper_eta()},
          {Direction<2>::lower_xi(), Direction<2>::upper_eta()}},
      expected_boundary_conditions);

  const creators::RotatedRectangles rotated_periodic_rectangles{
      lower_bound,
      midpoint,
      upper_bound,
      {{refinement_level[0][0], refinement_level[0][1]}},
      {{{{grid_points[0][0], grid_points[1][0]}},
        {{grid_points[0][1], grid_points[2][0]}}}},
      {{true, true}}};
  test_rotated_rectangles_construction(
      rotated_periodic_rectangles, lower_bound, midpoint, upper_bound,
      grid_points, refinement_level,
      std::vector<DirectionMap<2, BlockNeighbor<2>>>{
          {{Direction<2>::upper_xi(), {1, half_turn}},
           {Direction<2>::upper_eta(), {2, quarter_turn_ccw}},
           {Direction<2>::lower_xi(), {1, half_turn}},
           {Direction<2>::lower_eta(), {2, quarter_turn_ccw}}},
          {{Direction<2>::upper_xi(), {0, half_turn}},
           {Direction<2>::lower_eta(), {3, quarter_turn_ccw}},
           {Direction<2>::lower_xi(), {0, half_turn}},
           {Direction<2>::upper_eta(), {3, quarter_turn_ccw}}},
          {{Direction<2>::lower_xi(), {0, quarter_turn_cw}},
           {Direction<2>::lower_eta(), {3, half_turn}},
           {Direction<2>::upper_xi(), {0, quarter_turn_cw}},
           {Direction<2>::upper_eta(), {3, half_turn}}},
          {{Direction<2>::upper_xi(), {1, quarter_turn_cw}},
           {Direction<2>::lower_eta(), {2, half_turn}},
           {Direction<2>::lower_xi(), {1, quarter_turn_cw}},
           {Direction<2>::upper_eta(), {2, half_turn}}}},
      std::vector<std::unordered_set<Direction<2>>>{{}, {}, {}, {}});

  const creators::RotatedRectangles
      periodic_rotated_rectangles_boundary_conditions{
          lower_bound,
          midpoint,
          upper_bound,
          {{refinement_level[0][0], refinement_level[0][1]}},
          {{{{grid_points[0][0], grid_points[1][0]}},
            {{grid_points[0][1], grid_points[2][0]}}}},
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<2>>()};
  test_rotated_rectangles_construction(
      periodic_rotated_rectangles_boundary_conditions, lower_bound, midpoint,
      upper_bound, grid_points, refinement_level,
      std::vector<DirectionMap<2, BlockNeighbor<2>>>{
          {{Direction<2>::upper_xi(), {1, half_turn}},
           {Direction<2>::upper_eta(), {2, quarter_turn_ccw}},
           {Direction<2>::lower_xi(), {1, half_turn}},
           {Direction<2>::lower_eta(), {2, quarter_turn_ccw}}},
          {{Direction<2>::upper_xi(), {0, half_turn}},
           {Direction<2>::lower_eta(), {3, quarter_turn_ccw}},
           {Direction<2>::lower_xi(), {0, half_turn}},
           {Direction<2>::upper_eta(), {3, quarter_turn_ccw}}},
          {{Direction<2>::lower_xi(), {0, quarter_turn_cw}},
           {Direction<2>::lower_eta(), {3, half_turn}},
           {Direction<2>::upper_xi(), {0, quarter_turn_cw}},
           {Direction<2>::upper_eta(), {3, half_turn}}},
          {{Direction<2>::upper_xi(), {1, quarter_turn_cw}},
           {Direction<2>::lower_eta(), {2, half_turn}},
           {Direction<2>::lower_xi(), {1, quarter_turn_cw}},
           {Direction<2>::upper_eta(), {2, half_turn}}}},
      std::vector<std::unordered_set<Direction<2>>>{{}, {}, {}, {}});

  CHECK_THROWS_WITH(
      creators::RotatedRectangles(
          lower_bound, midpoint, upper_bound,
          {{refinement_level[0][0], refinement_level[0][1]}},
          {{{{grid_points[0][0], grid_points[1][0]}},
            {{grid_points[0][1], grid_points[2][0]}}}},
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestNoneBoundaryCondition<2>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "None boundary condition is not supported. If you would like "
          "an outflow boundary condition, you must use that."));
}

void test_rotated_rectangles_factory() {
  INFO("Rotated rectangles factory");
  const OrientationMap<2> half_turn{std::array<Direction<2>, 2>{
      {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}};
  const OrientationMap<2> quarter_turn_cw{std::array<Direction<2>, 2>{
      {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}};
  const OrientationMap<2> quarter_turn_ccw{std::array<Direction<2>, 2>{
      {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}};

  {
    INFO("No boundary condition");
    const auto domain_creator = TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<2>,
        TestHelpers::domain::BoundaryConditions::
            MetavariablesWithoutBoundaryConditions<
                2, domain::creators::RotatedRectangles>>(
        "RotatedRectangles:\n"
        "  LowerBound: [0.1, -0.4]\n"
        "  Midpoint:   [2.6, 3.2]\n"
        "  UpperBound: [5.1, 6.2]\n"
        "  IsPeriodicIn: [false, false]\n"
        "  InitialGridPoints: [[3,2],[1,4]]\n"
        "  InitialRefinement: [2,1]\n");
    const auto* rotated_rectangles_creator =
        dynamic_cast<const creators::RotatedRectangles*>(domain_creator.get());
    test_rotated_rectangles_construction(
        *rotated_rectangles_creator, {{0.1, -0.4}}, {{2.6, 3.2}}, {{5.1, 6.2}},
        {{{3, 1}}, {{2, 1}}, {{4, 3}}, {{4, 2}}},
        {{{2, 1}}, {{2, 1}}, {{1, 2}}, {{1, 2}}},
        std::vector<DirectionMap<2, BlockNeighbor<2>>>{
            {{Direction<2>::upper_xi(), {1, half_turn}},
             {Direction<2>::upper_eta(), {2, quarter_turn_ccw}}},
            {{Direction<2>::upper_xi(), {0, half_turn}},
             {Direction<2>::lower_eta(), {3, quarter_turn_ccw}}},
            {{Direction<2>::lower_xi(), {0, quarter_turn_cw}},
             {Direction<2>::lower_eta(), {3, half_turn}}},
            {{Direction<2>::upper_xi(), {1, quarter_turn_cw}},
             {Direction<2>::lower_eta(), {2, half_turn}}}},
        std::vector<std::unordered_set<Direction<2>>>{
            {Direction<2>::lower_xi(), Direction<2>::lower_eta()},
            {Direction<2>::lower_xi(), Direction<2>::upper_eta()},
            {Direction<2>::upper_xi(), Direction<2>::upper_eta()},
            {Direction<2>::lower_xi(), Direction<2>::upper_eta()}});
  }
  {
    INFO("With boundary condition");
    const std::vector<DirectionMap<
        2, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
        expected_boundary_conditions = create_boundary_conditions();
    const auto domain_creator = TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<2>,
        TestHelpers::domain::BoundaryConditions::
            MetavariablesWithBoundaryConditions<
                2, domain::creators::RotatedRectangles>>(
        "RotatedRectangles:\n"
        "  LowerBound: [0.1, -0.4]\n"
        "  Midpoint:   [2.6, 3.2]\n"
        "  UpperBound: [5.1, 6.2]\n"
        "  InitialGridPoints: [[3,2],[1,4]]\n"
        "  InitialRefinement: [2,1]\n"
        "  BoundaryCondition:\n"
        "    TestBoundaryCondition:\n"
        "      Direction: lower-xi\n"
        "      BlockId: 2\n");
    const auto* rotated_rectangles_creator =
        dynamic_cast<const creators::RotatedRectangles*>(domain_creator.get());
    test_rotated_rectangles_construction(
        *rotated_rectangles_creator, {{0.1, -0.4}}, {{2.6, 3.2}}, {{5.1, 6.2}},
        {{{3, 1}}, {{2, 1}}, {{4, 3}}, {{4, 2}}},
        {{{2, 1}}, {{2, 1}}, {{1, 2}}, {{1, 2}}},
        std::vector<DirectionMap<2, BlockNeighbor<2>>>{
            {{Direction<2>::upper_xi(), {1, half_turn}},
             {Direction<2>::upper_eta(), {2, quarter_turn_ccw}}},
            {{Direction<2>::upper_xi(), {0, half_turn}},
             {Direction<2>::lower_eta(), {3, quarter_turn_ccw}}},
            {{Direction<2>::lower_xi(), {0, quarter_turn_cw}},
             {Direction<2>::lower_eta(), {3, half_turn}}},
            {{Direction<2>::upper_xi(), {1, quarter_turn_cw}},
             {Direction<2>::lower_eta(), {2, half_turn}}}},
        std::vector<std::unordered_set<Direction<2>>>{
            {Direction<2>::lower_xi(), Direction<2>::lower_eta()},
            {Direction<2>::lower_xi(), Direction<2>::upper_eta()},
            {Direction<2>::upper_xi(), Direction<2>::upper_eta()},
            {Direction<2>::lower_xi(), Direction<2>::upper_eta()}},
        expected_boundary_conditions);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.RotatedRectangles.Factory",
                  "[Domain][Unit]") {
  test_rotated_rectangles();
  test_rotated_rectangles_factory();
}
}  // namespace domain
