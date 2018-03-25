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
#include "Domain/Block.hpp"          // IWYU pragma: keep
#include "Domain/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainCreators/DomainCreator.hpp"
#include "Domain/DomainCreators/Interval.hpp"
#include "Domain/DomainCreators/RegisterDerivedWithCharm.hpp"
#include "Domain/OrientationMap.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Utilities/MakeVector.hpp"
#include "tests/Unit/Domain/DomainTestHelpers.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
void test_interval_construction(
    const DomainCreators::Interval<Frame::Inertial>& interval,
    const std::array<double, 1>& lower_bound,
    const std::array<double, 1>& upper_bound,
    const std::vector<std::array<size_t, 1>>& expected_extents,
    const std::vector<std::array<size_t, 1>>& expected_refinement_level,
    const std::vector<std::unordered_map<Direction<1>, BlockNeighbor<1>>>&
        expected_block_neighbors,
    const std::vector<std::unordered_set<Direction<1>>>&
        expected_external_boundaries) noexcept {
  const auto domain = interval.create_domain();

  CHECK(interval.initial_extents() == expected_extents);
  CHECK(interval.initial_refinement_levels() == expected_refinement_level);

  test_domain_construction(
      domain, expected_block_neighbors, expected_external_boundaries,
      make_vector(make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          CoordinateMaps::Affine{-1., 1., lower_bound[0], upper_bound[0]})));
  test_initial_domain(domain, interval.initial_refinement_levels());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.DomainCreators.Interval", "[Domain][Unit]") {
  const std::vector<std::array<size_t, 1>> grid_points{{{4}}},
      refinement_level{{{3}}};
  const std::array<double, 1> lower_bound{{-1.2}}, upper_bound{{0.8}};
  // default Orientation is aligned
  const OrientationMap<1> aligned_orientation{};

  const DomainCreators::Interval<Frame::Inertial> interval{
      lower_bound, upper_bound, std::array<bool, 1>{{false}},
      refinement_level[0], grid_points[0]};
  test_interval_construction(
      interval, lower_bound, upper_bound, grid_points, refinement_level,
      std::vector<std::unordered_map<Direction<1>, BlockNeighbor<1>>>{{}},
      std::vector<std::unordered_set<Direction<1>>>{
          {{Direction<1>::lower_xi()}, {Direction<1>::upper_xi()}}});

  const DomainCreators::Interval<Frame::Inertial> periodic_interval{
      lower_bound, upper_bound, std::array<bool, 1>{{true}},
      refinement_level[0], grid_points[0]};
  test_interval_construction(
      periodic_interval, lower_bound, upper_bound, grid_points,
      refinement_level,
      std::vector<std::unordered_map<Direction<1>, BlockNeighbor<1>>>{
          {{Direction<1>::lower_xi(), {0, aligned_orientation}},
           {Direction<1>::upper_xi(), {0, aligned_orientation}}}},
      std::vector<std::unordered_set<Direction<1>>>{{}});

  // Test serialization of the map
  DomainCreators::register_derived_with_charm();

  const auto base_map =
      make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          CoordinateMaps::Affine{-1., 1., lower_bound[0], upper_bound[0]});
  const auto base_map_deserialized = serialize_and_deserialize(base_map);
  using MapType = const CoordinateMap<Frame::Logical, Frame::Inertial,
                                      CoordinateMaps::Affine>*;
  REQUIRE(dynamic_cast<MapType>(base_map.get()) != nullptr);
  const auto coord_map = make_coordinate_map<Frame::Logical, Frame::Inertial>(
      CoordinateMaps::Affine{-1., 1., lower_bound[0], upper_bound[0]});
  CHECK(*dynamic_cast<MapType>(base_map.get()) == coord_map);
}

SPECTRE_TEST_CASE("Unit.Domain.DomainCreators.Interval.Factory",
                  "[Domain][Unit]") {
  const auto domain_creator =
      test_factory_creation<DomainCreator<1, Frame::Inertial>>(
          "  Interval:\n"
          "    LowerBound: [0]\n"
          "    UpperBound: [1]\n"
          "    IsPeriodicIn: [True]\n"
          "    InitialGridPoints: [3]\n"
          "    InitialRefinement: [2]\n");
  const auto* interval_creator =
      dynamic_cast<const DomainCreators::Interval<Frame::Inertial>*>(
          domain_creator.get());
  test_interval_construction(
      *interval_creator, {{0.}}, {{1.}}, {{{3}}}, {{{2}}},
      std::vector<std::unordered_map<Direction<1>, BlockNeighbor<1>>>{
          {{Direction<1>::lower_xi(), {0, {}}},
           {Direction<1>::upper_xi(), {0, {}}}}},
      std::vector<std::unordered_set<Direction<1>>>{{}});
}
