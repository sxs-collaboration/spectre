// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "Domain/CoordinateMaps/AffineMap.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainCreators/Interval.hpp"
#include "Utilities/MakeVector.hpp"
#include "tests/Unit/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "tests/Unit/Domain/TestDomainHelpers.hpp"
#include "tests/Unit/TestFactoryCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
void test_interval(const DomainCreators::Interval& interval,
                   const std::array<double, 1>& lower_bound,
                   const std::array<double, 1>& upper_bound,
                   const std::array<size_t, 1>& expected_extents,
                   const std::array<size_t, 1>& expected_refinement_level) {
  const auto domain = interval.create_domain();
  const auto& block = domain.blocks()[0];
  const auto& neighbors = block.neighbors();
  const auto& external_boundaries = block.external_boundaries();

  CHECK(block.id() == 0);
  CHECK(interval.initial_extents(0) == expected_extents);
  CHECK(interval.initial_refinement_levels(0) == expected_refinement_level);

  test_domain_construction(
      domain,
      std::vector<std::unordered_map<Direction<1>, BlockNeighbor<1>>>{{}},
      std::vector<std::unordered_set<Direction<1>>>{
          {{Direction<1>::lower_xi()}, {Direction<1>::upper_xi()}}},
      make_vector(make_coordinate_map<Frame::Logical, Frame::Inertial>(
          CoordinateMaps::AffineMap{-1., 1., lower_bound[0], upper_bound[0]})));
}

void test_periodic_interval(
    const DomainCreators::Interval& interval,
    const std::array<double, 1>& lower_bound,
    const std::array<double, 1>& upper_bound,
    const std::array<size_t, 1>& expected_extents,
    const std::array<size_t, 1>& expected_refinement_level) {
  const auto domain = interval.create_domain();
  const auto& block = domain.blocks()[0];
  const auto& neighbors = block.neighbors();
  const auto& external_boundaries = block.external_boundaries();

  CHECK(block.id() == 0);
  CHECK(interval.initial_extents(0) == expected_extents);
  CHECK(interval.initial_refinement_levels(0) == expected_refinement_level);

  const Orientation<1> aligned_orientation{{{Direction<1>::lower_xi()}},
                                           {{Direction<1>::lower_xi()}}};

  test_domain_construction(
      domain,
      std::vector<std::unordered_map<Direction<1>, BlockNeighbor<1>>>{
          {{Direction<1>::lower_xi(), {0, aligned_orientation}},
           {Direction<1>::upper_xi(), {0, aligned_orientation}}}},
      std::vector<std::unordered_set<Direction<1>>>{{}},
      make_vector(make_coordinate_map<Frame::Logical, Frame::Inertial>(
          CoordinateMaps::AffineMap{-1., 1., lower_bound[0], upper_bound[0]})));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.DomainCreators.Interval", "[Domain][Unit]") {
  const std::array<size_t, 1> grid_points{{4}}, num_elements{{3}};
  const std::array<double, 1> lower_bound{{-1.2}}, upper_bound{{0.8}};

  const DomainCreators::Interval interval{lower_bound, upper_bound,
                                          std::array<bool, 1>{{false}},
                                          num_elements, grid_points};

  test_interval(interval, lower_bound, upper_bound, grid_points, num_elements);

  const DomainCreators::Interval periodic_interval{lower_bound, upper_bound,
                                                   std::array<bool, 1>{{true}},
                                                   num_elements, grid_points};
  test_periodic_interval(periodic_interval, lower_bound, upper_bound,
                         grid_points, num_elements);
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

// [[OutputRegex, index = 1]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Domain.DomainCreators.Interval.Extents",
                               "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  DomainCreators::Interval default_interval{};
  default_interval.initial_extents(1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
// [[OutputRegex, index = 2]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Domain.DomainCreators.Interval.Refinement",
                               "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  DomainCreators::Interval default_interval{};
  default_interval.initial_refinement_levels(2);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
