// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "Domain/CoordinateMaps/AffineMap.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainCreators/Interval.hpp"
#include "Domain/DomainCreators/RegisterDerivedWithCharm.hpp"
#include "Utilities/MakeVector.hpp"
#include "tests/Unit/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "tests/Unit/Domain/DomainTestHelpers.hpp"
#include "tests/Unit/TestFactoryCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
void test_interval(
    const DomainCreators::Interval& interval,
    const std::array<double, 1>& lower_bound,
    const std::array<double, 1>& upper_bound,
    const std::vector<std::array<size_t, 1>>& expected_extents,
    const std::vector<std::array<size_t, 1>>& expected_refinement_level) {
  const auto domain = interval.create_domain();
  const auto& block = domain.blocks()[0];
  const auto& neighbors = block.neighbors();
  const auto& external_boundaries = block.external_boundaries();

  CHECK(block.id() == 0);
  CHECK(interval.initial_extents() == expected_extents);
  CHECK(interval.initial_refinement_levels() == expected_refinement_level);

  PUPable_reg(SINGLE_ARG(CoordinateMap<Frame::Logical, Frame::Inertial,
                         CoordinateMaps::AffineMap>));
  test_domain_construction(
      domain,
      std::vector<std::unordered_map<Direction<1>, BlockNeighbor<1>>>{{}},
      std::vector<std::unordered_set<Direction<1>>>{
          {{Direction<1>::lower_xi()}, {Direction<1>::upper_xi()}}},
      make_vector(make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          CoordinateMaps::AffineMap{-1., 1., lower_bound[0], upper_bound[0]})));
}

void test_periodic_interval(
    const DomainCreators::Interval& interval,
    const std::array<double, 1>& lower_bound,
    const std::array<double, 1>& upper_bound,
    const std::vector<std::array<size_t, 1>>& expected_extents,
    const std::vector<std::array<size_t, 1>>& expected_refinement_level) {
  const auto domain = interval.create_domain();
  const auto& block = domain.blocks()[0];
  const auto& neighbors = block.neighbors();
  const auto& external_boundaries = block.external_boundaries();

  CHECK(block.id() == 0);
  CHECK(interval.initial_extents() == expected_extents);
  CHECK(interval.initial_refinement_levels() == expected_refinement_level);

  const Orientation<1> aligned_orientation{{{Direction<1>::lower_xi()}},
                                           {{Direction<1>::lower_xi()}}};

  test_domain_construction(
      domain,
      std::vector<std::unordered_map<Direction<1>, BlockNeighbor<1>>>{
          {{Direction<1>::lower_xi(), {0, aligned_orientation}},
           {Direction<1>::upper_xi(), {0, aligned_orientation}}}},
      std::vector<std::unordered_set<Direction<1>>>{{}},
      make_vector(make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          CoordinateMaps::AffineMap{-1., 1., lower_bound[0], upper_bound[0]})));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.DomainCreators.Interval", "[Domain][Unit]") {
  const std::vector<std::array<size_t, 1>> grid_points{{{4}}},
      refinement_level{{{3}}};
  const std::array<double, 1> lower_bound{{-1.2}}, upper_bound{{0.8}};

  const DomainCreators::Interval interval{lower_bound, upper_bound,
                                          std::array<bool, 1>{{false}},
                                          refinement_level[0], grid_points[0]};

  test_interval(interval, lower_bound, upper_bound, grid_points,
                refinement_level);

  const DomainCreators::Interval periodic_interval{
      lower_bound, upper_bound, std::array<bool, 1>{{true}},
      refinement_level[0], grid_points[0]};
  test_periodic_interval(periodic_interval, lower_bound, upper_bound,
                         grid_points, refinement_level);

  // Test serialization of the map
  DomainCreators::register_derived_with_charm();
  const auto base_map =
      make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          CoordinateMaps::AffineMap{-1., 1., lower_bound[0], upper_bound[0]});
  const auto base_map_deserialized = serialize_and_deserialize(base_map);
  using MapType = const CoordinateMap<Frame::Logical, Frame::Inertial,
                                      CoordinateMaps::AffineMap>*;
  REQUIRE(dynamic_cast<MapType>(base_map.get()) != nullptr);
  const auto coord_map = make_coordinate_map<Frame::Logical, Frame::Inertial>(
      CoordinateMaps::AffineMap{-1., 1., lower_bound[0], upper_bound[0]});
  CHECK(*dynamic_cast<MapType>(base_map.get()) == coord_map);
}

SPECTRE_TEST_CASE("Unit.Domain.DomainCreators.Interval.Factory",
                  "[Domain][Unit]") {
  test_factory_creation<DomainCreator<1, Frame::Inertial>>(
      "  Interval:\n"
      "    LowerBound: [0]\n"
      "    UpperBound: [1]\n"
      "    IsPeriodicIn: [True]\n"
      "    InitialGridPoints: [3]\n"
      "    InitialRefinement: [2]\n");
}
