// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainCreators/DomainCreator.hpp"
#include "Domain/DomainCreators/RotatedRectangles.hpp"
#include "Domain/OrientationMap.hpp"
#include "tests/Unit/Domain/DomainTestHelpers.hpp"
#include "tests/Unit/TestCreation.hpp"

namespace {
void test_rotated_rectangles_construction(
    const domain::creators::RotatedRectangles<Frame::Inertial>&
        rotated_rectangles,
    const std::array<double, 2>& lower_bound,
    const std::array<double, 2>& midpoint,
    const std::array<double, 2>& upper_bound,
    const std::vector<std::array<size_t, 2>>& expected_extents,
    const std::vector<std::array<size_t, 2>>& expected_refinement_level,
    const std::vector<
        std::unordered_map<domain::Direction<2>, domain::BlockNeighbor<2>>>&
        expected_block_neighbors,
    const std::vector<std::unordered_set<domain::Direction<2>>>&
        expected_external_boundaries) noexcept {
  const auto domain = rotated_rectangles.create_domain();

  CHECK(domain.blocks().size() == expected_extents.size());
  CHECK(domain.blocks().size() == expected_refinement_level.size());
  CHECK(rotated_rectangles.initial_extents() == expected_extents);
  CHECK(rotated_rectangles.initial_refinement_levels() ==
        expected_refinement_level);

  using Affine = domain::CoordinateMaps::Affine;
  using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  using DiscreteRotation2D = domain::CoordinateMaps::DiscreteRotation<2>;

  const Affine lower_x_map(-1.0, 1.0, lower_bound[0], midpoint[0]);
  const Affine upper_x_map(-1.0, 1.0, midpoint[0], upper_bound[0]);
  const Affine lower_y_map(-1.0, 1.0, lower_bound[1], midpoint[1]);
  const Affine upper_y_map(-1.0, 1.0, midpoint[1], upper_bound[1]);
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::Logical, Frame::Inertial, 2>>>
      coord_maps;
  coord_maps.emplace_back(
      domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Affine2D(lower_x_map, lower_y_map)));
  coord_maps.emplace_back(
      domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          DiscreteRotation2D{
              domain::OrientationMap<2>{std::array<domain::Direction<2>, 2>{
                  {domain::Direction<2>::lower_xi(),
                   domain::Direction<2>::lower_eta()}}}},
          Affine2D(upper_x_map, lower_y_map)));
  coord_maps.emplace_back(
      domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          DiscreteRotation2D{
              domain::OrientationMap<2>{std::array<domain::Direction<2>, 2>{
                  {domain::Direction<2>::lower_eta(),
                   domain::Direction<2>::upper_xi()}}}},
          Affine2D(lower_x_map, upper_y_map)));
  coord_maps.emplace_back(
      domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          DiscreteRotation2D{
              domain::OrientationMap<2>{std::array<domain::Direction<2>, 2>{
                  {domain::Direction<2>::upper_eta(),
                   domain::Direction<2>::lower_xi()}}}},
          Affine2D(upper_x_map, upper_y_map)));
  test_domain_construction(domain, expected_block_neighbors,
                           expected_external_boundaries, coord_maps);
  test_initial_domain(domain, rotated_rectangles.initial_refinement_levels());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.DomainCreators.RotatedRectangles",
                  "[Domain][Unit]") {
  const std::vector<std::array<size_t, 2>> grid_points{
      {{4, 2}}, {{1, 2}}, {{3, 4}}, {{3, 1}}},
      refinement_level{{{0, 1}}, {{0, 1}}, {{1, 0}}, {{1, 0}}};
  const std::array<double, 2> lower_bound{{-1.2, -2.0}}, midpoint{{-0.6, 0.2}},
      upper_bound{{0.8, 3.0}};
  const domain::OrientationMap<2> half_turn{std::array<domain::Direction<2>, 2>{
      {domain::Direction<2>::lower_xi(), domain::Direction<2>::lower_eta()}}};
  const domain::OrientationMap<2> quarter_turn_cw{
      std::array<domain::Direction<2>, 2>{{domain::Direction<2>::upper_eta(),
                                           domain::Direction<2>::lower_xi()}}};
  const domain::OrientationMap<2> quarter_turn_ccw{
      std::array<domain::Direction<2>, 2>{{domain::Direction<2>::lower_eta(),
                                           domain::Direction<2>::upper_xi()}}};

  const domain::creators::RotatedRectangles<Frame::Inertial> rotated_rectangles{
      lower_bound,
      midpoint,
      upper_bound,
      {{false, false}},
      {{refinement_level[0][0], refinement_level[0][1]}},
      {{{{grid_points[0][0], grid_points[1][0]}},
        {{grid_points[0][1], grid_points[2][0]}}}}};
  test_rotated_rectangles_construction(
      rotated_rectangles, lower_bound, midpoint, upper_bound, grid_points,
      refinement_level,
      std::vector<
          std::unordered_map<domain::Direction<2>, domain::BlockNeighbor<2>>>{
          {{domain::Direction<2>::upper_xi(), {1, half_turn}},
           {domain::Direction<2>::upper_eta(), {2, quarter_turn_ccw}}},
          {{domain::Direction<2>::upper_xi(), {0, half_turn}},
           {domain::Direction<2>::lower_eta(), {3, quarter_turn_ccw}}},
          {{domain::Direction<2>::lower_xi(), {0, quarter_turn_cw}},
           {domain::Direction<2>::lower_eta(), {3, half_turn}}},
          {{domain::Direction<2>::upper_xi(), {1, quarter_turn_cw}},
           {domain::Direction<2>::lower_eta(), {2, half_turn}}}},
      std::vector<std::unordered_set<domain::Direction<2>>>{
          {domain::Direction<2>::lower_xi(), domain::Direction<2>::lower_eta()},
          {domain::Direction<2>::lower_xi(), domain::Direction<2>::upper_eta()},
          {domain::Direction<2>::upper_xi(), domain::Direction<2>::upper_eta()},
          {domain::Direction<2>::lower_xi(),
           domain::Direction<2>::upper_eta()}});
  test_physical_separation(rotated_rectangles.create_domain().blocks());

  const domain::creators::RotatedRectangles<Frame::Inertial>
      rotated_periodic_rectangles{
          lower_bound,
          midpoint,
          upper_bound,
          {{true, true}},
          {{refinement_level[0][0], refinement_level[0][1]}},
          {{{{grid_points[0][0], grid_points[1][0]}},
            {{grid_points[0][1], grid_points[2][0]}}}}};
  test_rotated_rectangles_construction(
      rotated_periodic_rectangles, lower_bound, midpoint, upper_bound,
      grid_points, refinement_level,
      std::vector<
          std::unordered_map<domain::Direction<2>, domain::BlockNeighbor<2>>>{
          {{domain::Direction<2>::upper_xi(), {1, half_turn}},
           {domain::Direction<2>::upper_eta(), {2, quarter_turn_ccw}},
           {domain::Direction<2>::lower_xi(), {1, half_turn}},
           {domain::Direction<2>::lower_eta(), {2, quarter_turn_ccw}}},
          {{domain::Direction<2>::upper_xi(), {0, half_turn}},
           {domain::Direction<2>::lower_eta(), {3, quarter_turn_ccw}},
           {domain::Direction<2>::lower_xi(), {0, half_turn}},
           {domain::Direction<2>::upper_eta(), {3, quarter_turn_ccw}}},
          {{domain::Direction<2>::lower_xi(), {0, quarter_turn_cw}},
           {domain::Direction<2>::lower_eta(), {3, half_turn}},
           {domain::Direction<2>::upper_xi(), {0, quarter_turn_cw}},
           {domain::Direction<2>::upper_eta(), {3, half_turn}}},
          {{domain::Direction<2>::upper_xi(), {1, quarter_turn_cw}},
           {domain::Direction<2>::lower_eta(), {2, half_turn}},
           {domain::Direction<2>::lower_xi(), {1, quarter_turn_cw}},
           {domain::Direction<2>::upper_eta(), {2, half_turn}}}},
      std::vector<std::unordered_set<domain::Direction<2>>>{{}, {}, {}, {}});
}

SPECTRE_TEST_CASE("Unit.Domain.DomainCreators.RotatedRectangles.Factory",
                  "[Domain][Unit]") {
  const domain::OrientationMap<2> half_turn{std::array<domain::Direction<2>, 2>{
      {domain::Direction<2>::lower_xi(), domain::Direction<2>::lower_eta()}}};
  const domain::OrientationMap<2> quarter_turn_cw{
      std::array<domain::Direction<2>, 2>{{domain::Direction<2>::upper_eta(),
                                           domain::Direction<2>::lower_xi()}}};
  const domain::OrientationMap<2> quarter_turn_ccw{
      std::array<domain::Direction<2>, 2>{{domain::Direction<2>::lower_eta(),
                                           domain::Direction<2>::upper_xi()}}};

  const auto domain_creator =
      test_factory_creation<domain::DomainCreator<2, Frame::Inertial>>(
          "  RotatedRectangles:\n"
          "    LowerBound: [0.1, -0.4]\n"
          "    Midpoint:   [2.6, 3.2]\n"
          "    UpperBound: [5.1, 6.2]\n"
          "    IsPeriodicIn: [false, false]\n"
          "    InitialGridPoints: [[3,2],[1,4]]\n"
          "    InitialRefinement: [2,1]\n");
  const auto* rotated_rectangles_creator =
      dynamic_cast<const domain::creators::RotatedRectangles<Frame::Inertial>*>(
          domain_creator.get());
  test_rotated_rectangles_construction(
      *rotated_rectangles_creator, {{0.1, -0.4}}, {{2.6, 3.2}}, {{5.1, 6.2}},
      {{{3, 1}}, {{2, 1}}, {{4, 3}}, {{4, 2}}},
      {{{2, 1}}, {{2, 1}}, {{1, 2}}, {{1, 2}}},
      std::vector<
          std::unordered_map<domain::Direction<2>, domain::BlockNeighbor<2>>>{
          {{domain::Direction<2>::upper_xi(), {1, half_turn}},
           {domain::Direction<2>::upper_eta(), {2, quarter_turn_ccw}}},
          {{domain::Direction<2>::upper_xi(), {0, half_turn}},
           {domain::Direction<2>::lower_eta(), {3, quarter_turn_ccw}}},
          {{domain::Direction<2>::lower_xi(), {0, quarter_turn_cw}},
           {domain::Direction<2>::lower_eta(), {3, half_turn}}},
          {{domain::Direction<2>::upper_xi(), {1, quarter_turn_cw}},
           {domain::Direction<2>::lower_eta(), {2, half_turn}}}},
      std::vector<std::unordered_set<domain::Direction<2>>>{
          {domain::Direction<2>::lower_xi(), domain::Direction<2>::lower_eta()},
          {domain::Direction<2>::lower_xi(), domain::Direction<2>::upper_eta()},
          {domain::Direction<2>::upper_xi(), domain::Direction<2>::upper_eta()},
          {domain::Direction<2>::lower_xi(),
           domain::Direction<2>::upper_eta()}});
}
