// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <memory>
#include <pup.h>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Structure/BlockNeighbors.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StdHelpers.hpp"

namespace {
void test_create_initial_element(
    const ElementId<2>& element_id, const std::vector<Block<2>>& blocks,
    const std::vector<std::array<size_t, 2>>& refinement_levels,
    const DirectionMap<2, Neighbors<2>>& expected_neighbors,
    const std::array<domain::Topology, 2>& topologies =
        domain::topologies::hypercube<2>) {
  const auto created_element =
      domain::create_initial_element(element_id, blocks, refinement_levels);
  const Element<2> expected_element{element_id, expected_neighbors, topologies};
  CHECK(created_element == expected_element);
}

void test_h_refinement() {
  const auto make_check = [](const ElementId<3>& self_id,
                             const Direction<3>& neighbor_direction) {
    return [self_id, neighbor_direction](
               const OrientationMap<3>& neighbor_orientation,
               const std::array<size_t, 3>& neighbor_refinement,
               const std::unordered_set<ElementId<3>>& expected_neighbors) {
      CAPTURE(neighbor_orientation);
      CAPTURE(neighbor_refinement);
      std::vector<Block<3>> blocks;
      blocks.emplace_back(
          Block<3>(domain::make_coordinate_map_base<Frame::BlockLogical,
                                                    Frame::Inertial>(
                       domain::CoordinateMaps::Identity<3>{}),
                   0, {{neighbor_direction, {1, neighbor_orientation}}}));
      blocks.emplace_back(
          Block<3>(domain::make_coordinate_map_base<Frame::BlockLogical,
                                                    Frame::Inertial>(
                       domain::CoordinateMaps::Identity<3>{}),
                   1,
                   {{neighbor_orientation(neighbor_direction).opposite(),
                     {1, neighbor_orientation.inverse_map()}}}));
      const std::vector<std::array<size_t, 3>> refinement_levels{
          {{1, 1, 1}}, neighbor_refinement};

      const auto refined_neighbors =
          domain::create_initial_element(self_id, blocks, refinement_levels)
              .neighbors()
              .at(neighbor_direction)
              .ids();
      CHECK(refined_neighbors == expected_neighbors);
    };
  };

  const auto check_upper =
      make_check({0, {{{1, 1}, {1, 0}, {1, 1}}}}, Direction<3>::upper_xi());
  const auto check_lower =
      make_check({0, {{{1, 0}, {1, 0}, {1, 1}}}}, Direction<3>::lower_xi());

  const OrientationMap<3> aligned = OrientationMap<3>::create_aligned();
  const OrientationMap<3> rotated{
      {{Direction<3>::upper_zeta(), Direction<3>::upper_xi(),
        Direction<3>::upper_eta()}}};
  const OrientationMap<3> reflected{
      {{Direction<3>::lower_xi(), Direction<3>::lower_eta(),
        Direction<3>::upper_zeta()}}};

  // Same tangential refinement
  check_upper(aligned, {{1, 1, 1}}, {{1, {{{1, 0}, {1, 0}, {1, 1}}}}});
  check_lower(aligned, {{1, 1, 1}}, {{1, {{{1, 1}, {1, 0}, {1, 1}}}}});
  check_upper(rotated, {{1, 1, 1}}, {{1, {{{1, 0}, {1, 1}, {1, 0}}}}});
  check_lower(rotated, {{1, 1, 1}}, {{1, {{{1, 0}, {1, 1}, {1, 1}}}}});
  check_upper(reflected, {{1, 1, 1}}, {{1, {{{1, 1}, {1, 1}, {1, 1}}}}});
  check_lower(reflected, {{1, 1, 1}}, {{1, {{{1, 0}, {1, 1}, {1, 1}}}}});

  check_upper(aligned, {{0, 1, 1}}, {{1, {{{0, 0}, {1, 0}, {1, 1}}}}});
  check_lower(aligned, {{0, 1, 1}}, {{1, {{{0, 0}, {1, 0}, {1, 1}}}}});
  check_upper(rotated, {{1, 1, 0}}, {{1, {{{1, 0}, {1, 1}, {0, 0}}}}});
  check_lower(rotated, {{1, 1, 0}}, {{1, {{{1, 0}, {1, 1}, {0, 0}}}}});
  check_upper(reflected, {{0, 1, 1}}, {{1, {{{0, 0}, {1, 1}, {1, 1}}}}});
  check_lower(reflected, {{0, 1, 1}}, {{1, {{{0, 0}, {1, 1}, {1, 1}}}}});

  check_upper(aligned, {{2, 1, 1}}, {{1, {{{2, 0}, {1, 0}, {1, 1}}}}});
  check_lower(aligned, {{2, 1, 1}}, {{1, {{{2, 3}, {1, 0}, {1, 1}}}}});
  check_upper(rotated, {{1, 1, 2}}, {{1, {{{1, 0}, {1, 1}, {2, 0}}}}});
  check_lower(rotated, {{1, 1, 2}}, {{1, {{{1, 0}, {1, 1}, {2, 3}}}}});
  check_upper(reflected, {{2, 1, 1}}, {{1, {{{2, 3}, {1, 1}, {1, 1}}}}});
  check_lower(reflected, {{2, 1, 1}}, {{1, {{{2, 0}, {1, 1}, {1, 1}}}}});

  // Larger in both dimensions
  check_upper(aligned, {{1, 0, 0}}, {{1, {{{1, 0}, {0, 0}, {0, 0}}}}});
  check_lower(aligned, {{1, 0, 0}}, {{1, {{{1, 1}, {0, 0}, {0, 0}}}}});
  check_upper(rotated, {{0, 0, 1}}, {{1, {{{0, 0}, {0, 0}, {1, 0}}}}});
  check_lower(rotated, {{0, 0, 1}}, {{1, {{{0, 0}, {0, 0}, {1, 1}}}}});
  check_upper(reflected, {{1, 0, 0}}, {{1, {{{1, 1}, {0, 0}, {0, 0}}}}});
  check_lower(reflected, {{1, 0, 0}}, {{1, {{{1, 0}, {0, 0}, {0, 0}}}}});

  check_upper(aligned, {{0, 0, 0}}, {{1, {{{0, 0}, {0, 0}, {0, 0}}}}});
  check_lower(aligned, {{0, 0, 0}}, {{1, {{{0, 0}, {0, 0}, {0, 0}}}}});
  check_upper(rotated, {{0, 0, 0}}, {{1, {{{0, 0}, {0, 0}, {0, 0}}}}});
  check_lower(rotated, {{0, 0, 0}}, {{1, {{{0, 0}, {0, 0}, {0, 0}}}}});
  check_upper(reflected, {{0, 0, 0}}, {{1, {{{0, 0}, {0, 0}, {0, 0}}}}});
  check_lower(reflected, {{0, 0, 0}}, {{1, {{{0, 0}, {0, 0}, {0, 0}}}}});

  check_upper(aligned, {{2, 0, 0}}, {{1, {{{2, 0}, {0, 0}, {0, 0}}}}});
  check_lower(aligned, {{2, 0, 0}}, {{1, {{{2, 3}, {0, 0}, {0, 0}}}}});
  check_upper(rotated, {{0, 0, 2}}, {{1, {{{0, 0}, {0, 0}, {2, 0}}}}});
  check_lower(rotated, {{0, 0, 2}}, {{1, {{{0, 0}, {0, 0}, {2, 3}}}}});
  check_upper(reflected, {{2, 0, 0}}, {{1, {{{2, 3}, {0, 0}, {0, 0}}}}});
  check_lower(reflected, {{2, 0, 0}}, {{1, {{{2, 0}, {0, 0}, {0, 0}}}}});

  // Smaller in both dimensions
  check_upper(aligned, {{1, 2, 2}},
              {{1, {{{1, 0}, {2, 0}, {2, 2}}}},
               {1, {{{1, 0}, {2, 0}, {2, 3}}}},
               {1, {{{1, 0}, {2, 1}, {2, 2}}}},
               {1, {{{1, 0}, {2, 1}, {2, 3}}}}});
  check_lower(aligned, {{1, 2, 2}},
              {{1, {{{1, 1}, {2, 0}, {2, 2}}}},
               {1, {{{1, 1}, {2, 0}, {2, 3}}}},
               {1, {{{1, 1}, {2, 1}, {2, 2}}}},
               {1, {{{1, 1}, {2, 1}, {2, 3}}}}});
  check_upper(rotated, {{2, 2, 1}},
              {{1, {{{2, 0}, {2, 2}, {1, 0}}}},
               {1, {{{2, 0}, {2, 3}, {1, 0}}}},
               {1, {{{2, 1}, {2, 2}, {1, 0}}}},
               {1, {{{2, 1}, {2, 3}, {1, 0}}}}});
  check_lower(rotated, {{2, 2, 1}},
              {{1, {{{2, 0}, {2, 2}, {1, 1}}}},
               {1, {{{2, 0}, {2, 3}, {1, 1}}}},
               {1, {{{2, 1}, {2, 2}, {1, 1}}}},
               {1, {{{2, 1}, {2, 3}, {1, 1}}}}});
  check_upper(reflected, {{1, 2, 2}},
              {{1, {{{1, 1}, {2, 3}, {2, 2}}}},
               {1, {{{1, 1}, {2, 3}, {2, 3}}}},
               {1, {{{1, 1}, {2, 2}, {2, 2}}}},
               {1, {{{1, 1}, {2, 2}, {2, 3}}}}});
  check_lower(reflected, {{1, 2, 2}},
              {{1, {{{1, 0}, {2, 3}, {2, 2}}}},
               {1, {{{1, 0}, {2, 3}, {2, 3}}}},
               {1, {{{1, 0}, {2, 2}, {2, 2}}}},
               {1, {{{1, 0}, {2, 2}, {2, 3}}}}});

  check_upper(aligned, {{0, 2, 2}},
              {{1, {{{0, 0}, {2, 0}, {2, 2}}}},
               {1, {{{0, 0}, {2, 0}, {2, 3}}}},
               {1, {{{0, 0}, {2, 1}, {2, 2}}}},
               {1, {{{0, 0}, {2, 1}, {2, 3}}}}});
  check_lower(aligned, {{0, 2, 2}},
              {{1, {{{0, 0}, {2, 0}, {2, 2}}}},
               {1, {{{0, 0}, {2, 0}, {2, 3}}}},
               {1, {{{0, 0}, {2, 1}, {2, 2}}}},
               {1, {{{0, 0}, {2, 1}, {2, 3}}}}});
  check_upper(rotated, {{2, 2, 0}},
              {{1, {{{2, 0}, {2, 2}, {0, 0}}}},
               {1, {{{2, 0}, {2, 3}, {0, 0}}}},
               {1, {{{2, 1}, {2, 2}, {0, 0}}}},
               {1, {{{2, 1}, {2, 3}, {0, 0}}}}});
  check_lower(rotated, {{2, 2, 0}},
              {{1, {{{2, 0}, {2, 2}, {0, 0}}}},
               {1, {{{2, 0}, {2, 3}, {0, 0}}}},
               {1, {{{2, 1}, {2, 2}, {0, 0}}}},
               {1, {{{2, 1}, {2, 3}, {0, 0}}}}});
  check_upper(reflected, {{0, 2, 2}},
              {{1, {{{0, 0}, {2, 3}, {2, 2}}}},
               {1, {{{0, 0}, {2, 3}, {2, 3}}}},
               {1, {{{0, 0}, {2, 2}, {2, 2}}}},
               {1, {{{0, 0}, {2, 2}, {2, 3}}}}});
  check_lower(reflected, {{0, 2, 2}},
              {{1, {{{0, 0}, {2, 3}, {2, 2}}}},
               {1, {{{0, 0}, {2, 3}, {2, 3}}}},
               {1, {{{0, 0}, {2, 2}, {2, 2}}}},
               {1, {{{0, 0}, {2, 2}, {2, 3}}}}});

  check_upper(aligned, {{2, 2, 2}},
              {{1, {{{2, 0}, {2, 0}, {2, 2}}}},
               {1, {{{2, 0}, {2, 0}, {2, 3}}}},
               {1, {{{2, 0}, {2, 1}, {2, 2}}}},
               {1, {{{2, 0}, {2, 1}, {2, 3}}}}});
  check_lower(aligned, {{2, 2, 2}},
              {{1, {{{2, 3}, {2, 0}, {2, 2}}}},
               {1, {{{2, 3}, {2, 0}, {2, 3}}}},
               {1, {{{2, 3}, {2, 1}, {2, 2}}}},
               {1, {{{2, 3}, {2, 1}, {2, 3}}}}});
  check_upper(rotated, {{2, 2, 2}},
              {{1, {{{2, 0}, {2, 2}, {2, 0}}}},
               {1, {{{2, 0}, {2, 3}, {2, 0}}}},
               {1, {{{2, 1}, {2, 2}, {2, 0}}}},
               {1, {{{2, 1}, {2, 3}, {2, 0}}}}});
  check_lower(rotated, {{2, 2, 2}},
              {{1, {{{2, 0}, {2, 2}, {2, 3}}}},
               {1, {{{2, 0}, {2, 3}, {2, 3}}}},
               {1, {{{2, 1}, {2, 2}, {2, 3}}}},
               {1, {{{2, 1}, {2, 3}, {2, 3}}}}});
  check_upper(reflected, {{2, 2, 2}},
              {{1, {{{2, 3}, {2, 3}, {2, 2}}}},
               {1, {{{2, 3}, {2, 3}, {2, 3}}}},
               {1, {{{2, 3}, {2, 2}, {2, 2}}}},
               {1, {{{2, 3}, {2, 2}, {2, 3}}}}});
  check_lower(reflected, {{2, 2, 2}},
              {{1, {{{2, 0}, {2, 3}, {2, 2}}}},
               {1, {{{2, 0}, {2, 3}, {2, 3}}}},
               {1, {{{2, 0}, {2, 2}, {2, 2}}}},
               {1, {{{2, 0}, {2, 2}, {2, 3}}}}});

  // Larger in one dimension
  check_upper(aligned, {{1, 0, 1}}, {{1, {{{1, 0}, {0, 0}, {1, 1}}}}});
  check_lower(aligned, {{1, 0, 1}}, {{1, {{{1, 1}, {0, 0}, {1, 1}}}}});
  check_upper(rotated, {{0, 1, 1}}, {{1, {{{0, 0}, {1, 1}, {1, 0}}}}});
  check_lower(rotated, {{0, 1, 1}}, {{1, {{{0, 0}, {1, 1}, {1, 1}}}}});
  check_upper(reflected, {{1, 0, 1}}, {{1, {{{1, 1}, {0, 0}, {1, 1}}}}});
  check_lower(reflected, {{1, 0, 1}}, {{1, {{{1, 0}, {0, 0}, {1, 1}}}}});

  check_upper(aligned, {{0, 0, 1}}, {{1, {{{0, 0}, {0, 0}, {1, 1}}}}});
  check_lower(aligned, {{0, 0, 1}}, {{1, {{{0, 0}, {0, 0}, {1, 1}}}}});
  check_upper(rotated, {{0, 1, 0}}, {{1, {{{0, 0}, {1, 1}, {0, 0}}}}});
  check_lower(rotated, {{0, 1, 0}}, {{1, {{{0, 0}, {1, 1}, {0, 0}}}}});
  check_upper(reflected, {{0, 0, 1}}, {{1, {{{0, 0}, {0, 0}, {1, 1}}}}});
  check_lower(reflected, {{0, 0, 1}}, {{1, {{{0, 0}, {0, 0}, {1, 1}}}}});

  check_upper(aligned, {{2, 0, 1}}, {{1, {{{2, 0}, {0, 0}, {1, 1}}}}});
  check_lower(aligned, {{2, 0, 1}}, {{1, {{{2, 3}, {0, 0}, {1, 1}}}}});
  check_upper(rotated, {{0, 1, 2}}, {{1, {{{0, 0}, {1, 1}, {2, 0}}}}});
  check_lower(rotated, {{0, 1, 2}}, {{1, {{{0, 0}, {1, 1}, {2, 3}}}}});
  check_upper(reflected, {{2, 0, 1}}, {{1, {{{2, 3}, {0, 0}, {1, 1}}}}});
  check_lower(reflected, {{2, 0, 1}}, {{1, {{{2, 0}, {0, 0}, {1, 1}}}}});

  // Smaller in one dimension
  check_upper(
      aligned, {{1, 2, 1}},
      {{1, {{{1, 0}, {2, 0}, {1, 1}}}}, {1, {{{1, 0}, {2, 1}, {1, 1}}}}});
  check_lower(
      aligned, {{1, 2, 1}},
      {{1, {{{1, 1}, {2, 0}, {1, 1}}}}, {1, {{{1, 1}, {2, 1}, {1, 1}}}}});
  check_upper(
      rotated, {{2, 1, 1}},
      {{1, {{{2, 0}, {1, 1}, {1, 0}}}}, {1, {{{2, 1}, {1, 1}, {1, 0}}}}});
  check_lower(
      rotated, {{2, 1, 1}},
      {{1, {{{2, 0}, {1, 1}, {1, 1}}}}, {1, {{{2, 1}, {1, 1}, {1, 1}}}}});
  check_upper(
      reflected, {{1, 2, 1}},
      {{1, {{{1, 1}, {2, 3}, {1, 1}}}}, {1, {{{1, 1}, {2, 2}, {1, 1}}}}});
  check_lower(
      reflected, {{1, 2, 1}},
      {{1, {{{1, 0}, {2, 3}, {1, 1}}}}, {1, {{{1, 0}, {2, 2}, {1, 1}}}}});

  check_upper(
      aligned, {{0, 2, 1}},
      {{1, {{{0, 0}, {2, 0}, {1, 1}}}}, {1, {{{0, 0}, {2, 1}, {1, 1}}}}});
  check_lower(
      aligned, {{0, 2, 1}},
      {{1, {{{0, 0}, {2, 0}, {1, 1}}}}, {1, {{{0, 0}, {2, 1}, {1, 1}}}}});
  check_upper(
      rotated, {{2, 1, 0}},
      {{1, {{{2, 0}, {1, 1}, {0, 0}}}}, {1, {{{2, 1}, {1, 1}, {0, 0}}}}});
  check_lower(
      rotated, {{2, 1, 0}},
      {{1, {{{2, 0}, {1, 1}, {0, 0}}}}, {1, {{{2, 1}, {1, 1}, {0, 0}}}}});
  check_upper(
      reflected, {{0, 2, 1}},
      {{1, {{{0, 0}, {2, 3}, {1, 1}}}}, {1, {{{0, 0}, {2, 2}, {1, 1}}}}});
  check_lower(
      reflected, {{0, 2, 1}},
      {{1, {{{0, 0}, {2, 3}, {1, 1}}}}, {1, {{{0, 0}, {2, 2}, {1, 1}}}}});

  check_upper(
      aligned, {{2, 2, 1}},
      {{1, {{{2, 0}, {2, 0}, {1, 1}}}}, {1, {{{2, 0}, {2, 1}, {1, 1}}}}});
  check_lower(
      aligned, {{2, 2, 1}},
      {{1, {{{2, 3}, {2, 0}, {1, 1}}}}, {1, {{{2, 3}, {2, 1}, {1, 1}}}}});
  check_upper(
      rotated, {{2, 1, 2}},
      {{1, {{{2, 0}, {1, 1}, {2, 0}}}}, {1, {{{2, 1}, {1, 1}, {2, 0}}}}});
  check_lower(
      rotated, {{2, 1, 2}},
      {{1, {{{2, 0}, {1, 1}, {2, 3}}}}, {1, {{{2, 1}, {1, 1}, {2, 3}}}}});
  check_upper(
      reflected, {{2, 2, 1}},
      {{1, {{{2, 3}, {2, 3}, {1, 1}}}}, {1, {{{2, 3}, {2, 2}, {1, 1}}}}});
  check_lower(
      reflected, {{2, 2, 1}},
      {{1, {{{2, 0}, {2, 3}, {1, 1}}}}, {1, {{{2, 0}, {2, 2}, {1, 1}}}}});

  // Larger in one dimension and smaller in another dimension
  check_upper(
      aligned, {{1, 2, 0}},
      {{1, {{{1, 0}, {2, 0}, {0, 0}}}}, {1, {{{1, 0}, {2, 1}, {0, 0}}}}});
  check_lower(
      aligned, {{1, 2, 0}},
      {{1, {{{1, 1}, {2, 0}, {0, 0}}}}, {1, {{{1, 1}, {2, 1}, {0, 0}}}}});
  check_upper(
      rotated, {{2, 0, 1}},
      {{1, {{{2, 0}, {0, 0}, {1, 0}}}}, {1, {{{2, 1}, {0, 0}, {1, 0}}}}});
  check_lower(
      rotated, {{2, 0, 1}},
      {{1, {{{2, 0}, {0, 0}, {1, 1}}}}, {1, {{{2, 1}, {0, 0}, {1, 1}}}}});
  check_upper(
      reflected, {{1, 2, 0}},
      {{1, {{{1, 1}, {2, 3}, {0, 0}}}}, {1, {{{1, 1}, {2, 2}, {0, 0}}}}});
  check_lower(
      reflected, {{1, 2, 0}},
      {{1, {{{1, 0}, {2, 3}, {0, 0}}}}, {1, {{{1, 0}, {2, 2}, {0, 0}}}}});

  check_upper(
      aligned, {{0, 2, 0}},
      {{1, {{{0, 0}, {2, 0}, {0, 0}}}}, {1, {{{0, 0}, {2, 1}, {0, 0}}}}});
  check_lower(
      aligned, {{0, 2, 0}},
      {{1, {{{0, 0}, {2, 0}, {0, 0}}}}, {1, {{{0, 0}, {2, 1}, {0, 0}}}}});
  check_upper(
      rotated, {{2, 0, 0}},
      {{1, {{{2, 0}, {0, 0}, {0, 0}}}}, {1, {{{2, 1}, {0, 0}, {0, 0}}}}});
  check_lower(
      rotated, {{2, 0, 0}},
      {{1, {{{2, 0}, {0, 0}, {0, 0}}}}, {1, {{{2, 1}, {0, 0}, {0, 0}}}}});
  check_upper(
      reflected, {{0, 2, 0}},
      {{1, {{{0, 0}, {2, 3}, {0, 0}}}}, {1, {{{0, 0}, {2, 2}, {0, 0}}}}});
  check_lower(
      reflected, {{0, 2, 0}},
      {{1, {{{0, 0}, {2, 3}, {0, 0}}}}, {1, {{{0, 0}, {2, 2}, {0, 0}}}}});

  check_upper(
      aligned, {{2, 2, 0}},
      {{1, {{{2, 0}, {2, 0}, {0, 0}}}}, {1, {{{2, 0}, {2, 1}, {0, 0}}}}});
  check_lower(
      aligned, {{2, 2, 0}},
      {{1, {{{2, 3}, {2, 0}, {0, 0}}}}, {1, {{{2, 3}, {2, 1}, {0, 0}}}}});
  check_upper(
      rotated, {{2, 0, 2}},
      {{1, {{{2, 0}, {0, 0}, {2, 0}}}}, {1, {{{2, 1}, {0, 0}, {2, 0}}}}});
  check_lower(
      rotated, {{2, 0, 2}},
      {{1, {{{2, 0}, {0, 0}, {2, 3}}}}, {1, {{{2, 1}, {0, 0}, {2, 3}}}}});
  check_upper(
      reflected, {{2, 2, 0}},
      {{1, {{{2, 3}, {2, 3}, {0, 0}}}}, {1, {{{2, 3}, {2, 2}, {0, 0}}}}});
  check_lower(
      reflected, {{2, 2, 0}},
      {{1, {{{2, 0}, {2, 3}, {0, 0}}}}, {1, {{{2, 0}, {2, 2}, {0, 0}}}}});

  // Larger perpendicular refinement
  const auto check_perpendicular_refinement_upper =
      make_check({0, {{{3, 7}, {0, 0}, {0, 0}}}}, Direction<3>::upper_xi());
  const auto check_perpendicular_refinement_lower =
      make_check({0, {{{3, 0}, {0, 0}, {0, 0}}}}, Direction<3>::lower_xi());

  check_perpendicular_refinement_upper(aligned, {{1, 0, 0}},
                                       {{1, {{{1, 0}, {0, 0}, {0, 0}}}}});
  check_perpendicular_refinement_lower(aligned, {{1, 0, 0}},
                                       {{1, {{{1, 1}, {0, 0}, {0, 0}}}}});
  check_perpendicular_refinement_upper(rotated, {{0, 0, 1}},
                                       {{1, {{{0, 0}, {0, 0}, {1, 0}}}}});
  check_perpendicular_refinement_lower(rotated, {{0, 0, 1}},
                                       {{1, {{{0, 0}, {0, 0}, {1, 1}}}}});
  check_perpendicular_refinement_upper(reflected, {{1, 0, 0}},
                                       {{1, {{{1, 1}, {0, 0}, {0, 0}}}}});
  check_perpendicular_refinement_lower(reflected, {{1, 0, 0}},
                                       {{1, {{{1, 0}, {0, 0}, {0, 0}}}}});

  check_perpendicular_refinement_upper(aligned, {{5, 0, 0}},
                                       {{1, {{{5, 0}, {0, 0}, {0, 0}}}}});
  check_perpendicular_refinement_lower(aligned, {{5, 0, 0}},
                                       {{1, {{{5, 31}, {0, 0}, {0, 0}}}}});
  check_perpendicular_refinement_upper(rotated, {{0, 0, 5}},
                                       {{1, {{{0, 0}, {0, 0}, {5, 0}}}}});
  check_perpendicular_refinement_lower(rotated, {{0, 0, 5}},
                                       {{1, {{{0, 0}, {0, 0}, {5, 31}}}}});
  check_perpendicular_refinement_upper(reflected, {{5, 0, 0}},
                                       {{1, {{{5, 31}, {0, 0}, {0, 0}}}}});
  check_perpendicular_refinement_lower(reflected, {{5, 0, 0}},
                                       {{1, {{{5, 0}, {0, 0}, {0, 0}}}}});
}

void test_nonconforming_blocks() {
  const OrientationMap<2> aligned = OrientationMap<2>::create_aligned();
  std::vector<Block<2>> blocks;
  blocks.emplace_back(
      nullptr, 0,
      DirectionMap<2, BlockNeighbors<2>>{
          {Direction<2>::upper_xi(),
           BlockNeighbors<2>{
               {1, 2, 3, 4},
               {{1, aligned}, {2, aligned}, {3, aligned}, {4, aligned}},
               false}}},
      "Annulus", std::array{domain::Topology::I1, domain::Topology::S1});
  blocks.emplace_back(
      nullptr, 1,
      DirectionMap<2, BlockNeighbors<2>>{
          {Direction<2>::lower_xi(), BlockNeighbors<2>{0, aligned}},
          {Direction<2>::lower_eta(), BlockNeighbors<2>{2, aligned}},
          {Direction<2>::upper_eta(), BlockNeighbors<2>{4, aligned}}},
      "North", std::array{domain::Topology::I1, domain::Topology::I1});
  blocks.emplace_back(
      nullptr, 2,
      DirectionMap<2, BlockNeighbors<2>>{
          {Direction<2>::lower_xi(), BlockNeighbors<2>{0, aligned}},
          {Direction<2>::lower_eta(), BlockNeighbors<2>{3, aligned}},
          {Direction<2>::upper_eta(), BlockNeighbors<2>{1, aligned}}},
      "East", std::array{domain::Topology::I1, domain::Topology::I1});
  blocks.emplace_back(
      nullptr, 3,
      DirectionMap<2, BlockNeighbors<2>>{
          {Direction<2>::lower_xi(), BlockNeighbors<2>{0, aligned}},
          {Direction<2>::lower_eta(), BlockNeighbors<2>{4, aligned}},
          {Direction<2>::upper_eta(), BlockNeighbors<2>{2, aligned}}},
      "South", std::array{domain::Topology::I1, domain::Topology::I1});
  blocks.emplace_back(
      nullptr, 4,
      DirectionMap<2, BlockNeighbors<2>>{
          {Direction<2>::lower_xi(), BlockNeighbors<2>{0, aligned}},
          {Direction<2>::lower_eta(), BlockNeighbors<2>{1, aligned}},
          {Direction<2>::upper_eta(), BlockNeighbors<2>{3, aligned}}},
      "West", std::array{domain::Topology::I1, domain::Topology::I1});
  const std::vector<std::array<size_t, 2>> initial_refinement_levels{
      std::array{2_st, 0_st}, std::array{0_st, 1_st}, std::array{0_st, 1_st},
      std::array{0_st, 1_st}, std::array{0_st, 1_st}};
  const ElementId<2> annulus_u{0, std::array{SegmentId{2, 3}, SegmentId{0, 0}}};
  const ElementId<2> annulus_m{0, std::array{SegmentId{2, 2}, SegmentId{0, 0}}};
  const ElementId<2> north_l{1, std::array{SegmentId{0, 0}, SegmentId{1, 0}}};
  const ElementId<2> north_u{1, std::array{SegmentId{0, 0}, SegmentId{1, 1}}};
  const ElementId<2> east_l{2, std::array{SegmentId{0, 0}, SegmentId{1, 0}}};
  const ElementId<2> east_u{2, std::array{SegmentId{0, 0}, SegmentId{1, 1}}};
  const ElementId<2> south_l{3, std::array{SegmentId{0, 0}, SegmentId{1, 0}}};
  const ElementId<2> south_u{3, std::array{SegmentId{0, 0}, SegmentId{1, 1}}};
  const ElementId<2> west_l{4, std::array{SegmentId{0, 0}, SegmentId{1, 0}}};
  const ElementId<2> west_u{4, std::array{SegmentId{0, 0}, SegmentId{1, 1}}};
  test_create_initial_element(
      annulus_u, blocks, initial_refinement_levels,
      {{Direction<2>::lower_xi(), Neighbors<2>{annulus_m, aligned}},
       {Direction<2>::upper_xi(),
        Neighbors<2>{{north_l, north_u, east_l, east_u, south_l, south_u,
                      west_l, west_u},
                     {{1, aligned}, {2, aligned}, {3, aligned}, {4, aligned}},
                     false}}},
      domain::topologies::annulus);
  test_create_initial_element(
      north_l, blocks, initial_refinement_levels,
      {{Direction<2>::lower_xi(), Neighbors<2>{annulus_u, aligned}},
       {Direction<2>::lower_eta(), Neighbors<2>{east_u, aligned}},
       {Direction<2>::upper_eta(), Neighbors<2>{north_u, aligned}}},
      domain::topologies::hypercube<2>);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CreateInitialElement", "[Domain][Unit]") {
  OrientationMap<2> aligned(
      make_array(Direction<2>::upper_xi(), Direction<2>::upper_eta()));
  OrientationMap<2> unaligned(
      make_array(Direction<2>::lower_eta(), Direction<2>::upper_xi()));
  std::vector<Block<2>> blocks;
  blocks.emplace_back(Block<2>(
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          domain::CoordinateMaps::Identity<2>{}),
      0,
      {{Direction<2>::upper_xi(), BlockNeighbors<2>{1, aligned}},
       {Direction<2>::upper_eta(), BlockNeighbors<2>{2, unaligned}}}));
  blocks.emplace_back(Block<2>(
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          domain::CoordinateMaps::Identity<2>{}),
      1, {{Direction<2>::lower_xi(), BlockNeighbors<2>{0, aligned}}}));
  blocks.emplace_back(Block<2>(
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          domain::CoordinateMaps::Identity<2>{}),
      2,
      {{Direction<2>::lower_xi(),
        BlockNeighbors<2>{0, unaligned.inverse_map()}}}));
  std::vector<std::array<size_t, 2>> refinement{{{2, 3}}, {{2, 3}}, {{3, 2}}};

  // interior element
  test_create_initial_element(
      ElementId<2>{0, {{SegmentId{2, 2}, SegmentId{3, 4}}}}, blocks, refinement,
      {{Direction<2>::upper_xi(),
        Neighbors<2>{{ElementId<2>{0, {{SegmentId{2, 3}, SegmentId{3, 4}}}}},
                     aligned}},
       {Direction<2>::lower_xi(),
        Neighbors<2>{{ElementId<2>{0, {{SegmentId{2, 1}, SegmentId{3, 4}}}}},
                     aligned}},
       {Direction<2>::upper_eta(),
        Neighbors<2>{{ElementId<2>{0, {{SegmentId{2, 2}, SegmentId{3, 5}}}}},
                     aligned}},
       {Direction<2>::lower_eta(),
        Neighbors<2>{{ElementId<2>{0, {{SegmentId{2, 2}, SegmentId{3, 3}}}}},
                     aligned}}});

  // element on external boundary
  test_create_initial_element(
      ElementId<2>{0, {{SegmentId{2, 0}, SegmentId{3, 0}}}}, blocks, refinement,
      {{Direction<2>::upper_xi(),
        Neighbors<2>{{ElementId<2>{0, {{SegmentId{2, 1}, SegmentId{3, 0}}}}},
                     aligned}},
       {Direction<2>::upper_eta(),
        Neighbors<2>{{ElementId<2>{0, {{SegmentId{2, 0}, SegmentId{3, 1}}}}},
                     aligned}}});

  // element bounding aligned neighbor block
  test_create_initial_element(
      ElementId<2>{0, {{SegmentId{2, 3}, SegmentId{3, 4}}}}, blocks, refinement,
      {{Direction<2>::upper_xi(),
        Neighbors<2>{{ElementId<2>{1, {{SegmentId{2, 0}, SegmentId{3, 4}}}}},
                     aligned}},
       {Direction<2>::lower_xi(),
        Neighbors<2>{{ElementId<2>{0, {{SegmentId{2, 2}, SegmentId{3, 4}}}}},
                     aligned}},
       {Direction<2>::upper_eta(),
        Neighbors<2>{{ElementId<2>{0, {{SegmentId{2, 3}, SegmentId{3, 5}}}}},
                     aligned}},
       {Direction<2>::lower_eta(),
        Neighbors<2>{{ElementId<2>{0, {{SegmentId{2, 3}, SegmentId{3, 3}}}}},
                     aligned}}});

  // element bounding unaligned neighbor block
  test_create_initial_element(
      ElementId<2>{0, {{SegmentId{2, 2}, SegmentId{3, 7}}}}, blocks, refinement,
      {{Direction<2>::upper_xi(),
        Neighbors<2>{{ElementId<2>{0, {{SegmentId{2, 3}, SegmentId{3, 7}}}}},
                     aligned}},
       {Direction<2>::lower_xi(),
        Neighbors<2>{{ElementId<2>{0, {{SegmentId{2, 1}, SegmentId{3, 7}}}}},
                     aligned}},
       {Direction<2>::upper_eta(),
        Neighbors<2>{{ElementId<2>{2, {{SegmentId{3, 0}, SegmentId{2, 1}}}}},
                     unaligned}},
       {Direction<2>::lower_eta(),
        Neighbors<2>{{ElementId<2>{0, {{SegmentId{2, 2}, SegmentId{3, 6}}}}},
                     aligned}}});

  // element bounding both neighbor blocks
  test_create_initial_element(
      ElementId<2>{0, {{SegmentId{2, 3}, SegmentId{3, 7}}}}, blocks, refinement,
      {{Direction<2>::upper_xi(),
        Neighbors<2>{{ElementId<2>{1, {{SegmentId{2, 0}, SegmentId{3, 7}}}}},
                     aligned}},
       {Direction<2>::lower_xi(),
        Neighbors<2>{{ElementId<2>{0, {{SegmentId{2, 2}, SegmentId{3, 7}}}}},
                     aligned}},
       {Direction<2>::upper_eta(),
        Neighbors<2>{{ElementId<2>{2, {{SegmentId{3, 0}, SegmentId{2, 0}}}}},
                     unaligned}},
       {Direction<2>::lower_eta(),
        Neighbors<2>{{ElementId<2>{0, {{SegmentId{2, 3}, SegmentId{3, 6}}}}},
                     aligned}}});

  {
    // element with a non-zero grid index
    const size_t grid_index = 3;
    test_create_initial_element(
        ElementId<2>{0, {{SegmentId{2, 2}, SegmentId{3, 4}}}, grid_index},
        blocks, refinement,
        {{Direction<2>::upper_xi(),
          Neighbors<2>{
              {ElementId<2>{
                  0, {{SegmentId{2, 3}, SegmentId{3, 4}}}, grid_index}},
              aligned}},
         {Direction<2>::lower_xi(),
          Neighbors<2>{
              {ElementId<2>{
                  0, {{SegmentId{2, 1}, SegmentId{3, 4}}}, grid_index}},
              aligned}},
         {Direction<2>::upper_eta(),
          Neighbors<2>{
              {ElementId<2>{
                  0, {{SegmentId{2, 2}, SegmentId{3, 5}}}, grid_index}},
              aligned}},
         {Direction<2>::lower_eta(),
          Neighbors<2>{
              {ElementId<2>{
                  0, {{SegmentId{2, 2}, SegmentId{3, 3}}}, grid_index}},
              aligned}}});
  }

  {
    // Test refine_Bn_topology: elements at xi_index=0 retain the inner
    // topology, while elements at xi_index!=0 get the refined topology.
    const OrientationMap<2> aligned_2d = OrientationMap<2>::create_aligned();
    const OrientationMap<3> aligned_3d = OrientationMap<3>::create_aligned();

    // Helper: one 2D block, no neighbors. Refinement {2, 0} gives two element
    // ids to check (xi_index=0 and xi_index=1).
    const auto test_topology_refinement_2d = [&aligned_2d](
                                                 const std::array<
                                                     domain::Topology, 2>&
                                                     block_topologies,
                                                 const std::array<
                                                     domain::Topology, 2>&
                                                     expected_outer) {
      const std::array<domain::Topology, 2>& expected_inner = block_topologies;
      const ElementId<2> element_id_0{0, {{SegmentId{2, 0}, SegmentId{0, 0}}}};
      const ElementId<2> element_id_1{0, {{SegmentId{2, 1}, SegmentId{0, 0}}}};
      std::vector<Block<2>> local_blocks;
      local_blocks.emplace_back(nullptr, 0,
                                DirectionMap<2, BlockNeighbors<2>>{}, "",
                                block_topologies);
      const std::vector<std::array<size_t, 2>> refinement_levels{{2, 0}};

      CHECK(domain::create_initial_element(element_id_0, local_blocks,
                                           refinement_levels) ==
            Element<2>{
                element_id_0,
                {{Direction<2>::upper_xi(),
                  Neighbors<2>{ElementId<2>{0, std::array{SegmentId{2, 1},
                                                          SegmentId{0, 0}}},
                               aligned_2d}}},
                expected_inner});

      CHECK(domain::create_initial_element(element_id_1, local_blocks,
                                           refinement_levels) ==
            Element<2>{
                element_id_1,
                {{Direction<2>::lower_xi(),
                  Neighbors<2>{ElementId<2>{0, std::array{SegmentId{2, 0},
                                                          SegmentId{0, 0}}},
                               aligned_2d}},
                 {Direction<2>::upper_xi(),
                  Neighbors<2>{ElementId<2>{0, std::array{SegmentId{2, 2},
                                                          SegmentId{0, 0}}},
                               aligned_2d}}},
                expected_outer});
    };

    // Helper: one 3D block, no neighbors. Refinement {2, 1, 0} gives two
    // element ids to check (xi_index=0 and xi_index=1).
    const auto test_topology_refinement_3d =
        [&aligned_3d](const std::array<domain::Topology, 3>& block_topologies,
                      const std::array<domain::Topology, 3>& expected_outer) {
          const std::array<domain::Topology, 3>& expected_inner =
              block_topologies;
          const ElementId<3> element_id_0{
              0, {{SegmentId{2, 0}, SegmentId{0, 0}, SegmentId{0, 0}}}};
          const ElementId<3> element_id_1{
              0, {{SegmentId{2, 1}, SegmentId{0, 0}, SegmentId{0, 0}}}};
          std::vector<Block<3>> local_blocks;
          local_blocks.emplace_back(nullptr, 0,
                                    DirectionMap<3, BlockNeighbors<3>>{}, "",
                                    block_topologies);
          const std::vector<std::array<size_t, 3>> refinement_levels{{2, 1, 0}};

          CHECK(domain::create_initial_element(element_id_0, local_blocks,
                                               refinement_levels) ==
                Element<3>{
                    element_id_0,
                    {{Direction<3>::upper_xi(),
                      Neighbors<3>{ElementId<3>{0, std::array{SegmentId{2, 1},
                                                              SegmentId{0, 0},
                                                              SegmentId{0, 0}}},
                                   aligned_3d}}},
                    expected_inner});

          CHECK(domain::create_initial_element(element_id_1, local_blocks,
                                               refinement_levels) ==
                Element<3>{
                    element_id_1,
                    {{Direction<3>::lower_xi(),
                      Neighbors<3>{ElementId<3>{0, std::array{SegmentId{2, 0},
                                                              SegmentId{0, 0},
                                                              SegmentId{0, 0}}},
                                   aligned_3d}},
                     {Direction<3>::upper_xi(),
                      Neighbors<3>{ElementId<3>{0, std::array{SegmentId{2, 2},
                                                              SegmentId{0, 0},
                                                              SegmentId{0, 0}}},
                                   aligned_3d}}},
                    expected_outer});
        };

    // 2D: disk center stays disk; off-axis becomes annulus.
    test_topology_refinement_2d(domain::topologies::disk,
                                domain::topologies::annulus);

    // 3D: cartoon cylinder inner stays when on-axis; becomes cartoon cylinder.
    test_topology_refinement_3d(domain::topologies::cartoon_cylinder_inner,
                                domain::topologies::cartoon_cylinder);

    // 3D: cartoon sphere inner stays when on-axis; becomes cartoon sphere.
    test_topology_refinement_3d(domain::topologies::cartoon_sphere_inner,
                                domain::topologies::cartoon_sphere);

    // 3D: full cylinder stays when on-axis; becomes cylindrical shell.
    test_topology_refinement_3d(domain::topologies::full_cylinder,
                                domain::topologies::cylindrical_shell);

    // 3D: full sphere stays when on-axis; becomes spherical shell.
    test_topology_refinement_3d(domain::topologies::full_sphere,
                                domain::topologies::spherical_shell);
  }

  test_h_refinement();
  test_nonconforming_blocks();
}
