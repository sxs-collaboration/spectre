// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/BlockNeighbors.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/EqualRateRegionGenerator.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/NonconformingEqualRateRegions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace {
static_assert(evolution::dg::equal_rate_region_generator<
              evolution::dg::NonconformingEqualRateRegions<1>, 1>);
static_assert(evolution::dg::equal_rate_region_generator<
              evolution::dg::NonconformingEqualRateRegions<2>, 2>);
static_assert(evolution::dg::equal_rate_region_generator<
              evolution::dg::NonconformingEqualRateRegions<3>, 3>);

class NonconformingCreator : public DomainCreator<2> {
 public:
  Domain<2> create_domain() const override {
    // This map is nonsense, but we only care about the topology.
    const auto map =
        domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            domain::CoordinateMaps::Identity<2>{});

    const auto aligned = OrientationMap<2>::create_aligned();
    const OrientationMap<2> rotated{
        {{Direction<2>::upper_eta(), Direction<2>::lower_xi()}}};

    std::vector<Block<2>> blocks{};

    // A layer of two squares, the second rotated
    const auto add_squares = [&](const size_t id1,
                                 const std::optional<size_t>& inner_neighbor,
                                 const std::optional<size_t>& outer_neighbor) {
      // First block
      {
        DirectionMap<2, BlockNeighbors<2>> neighbors{};
        neighbors.emplace(Direction<2>::upper_eta(),
                          BlockNeighbors<2>(id1 + 1, rotated));
        neighbors.emplace(Direction<2>::lower_eta(),
                          BlockNeighbors<2>(id1 + 1, rotated));
        if (inner_neighbor.has_value()) {
          neighbors.emplace(
              Direction<2>::lower_xi(),
              BlockNeighbors<2>({*inner_neighbor}, {{*inner_neighbor, aligned}},
                                false));
        }
        if (outer_neighbor.has_value()) {
          neighbors.emplace(
              Direction<2>::upper_xi(),
              BlockNeighbors<2>({*outer_neighbor}, {{*outer_neighbor, aligned}},
                                false));
        }
        blocks.emplace_back(map->get_clone(), id1, std::move(neighbors),
                            std::to_string(id1),
                            domain::topologies::hypercube<2>);
      }
      // Second block
      {
        DirectionMap<2, BlockNeighbors<2>> neighbors{};
        neighbors.emplace(rotated(Direction<2>::upper_eta()),
                          BlockNeighbors<2>(id1, rotated.inverse_map()));
        neighbors.emplace(rotated(Direction<2>::lower_eta()),
                          BlockNeighbors<2>(id1 + 1, rotated.inverse_map()));
        if (inner_neighbor.has_value()) {
          neighbors.emplace(
              rotated(Direction<2>::lower_xi()),
              BlockNeighbors<2>({*inner_neighbor},
                                {{*inner_neighbor, rotated.inverse_map()}},
                                false));
        }
        if (outer_neighbor.has_value()) {
          neighbors.emplace(
              rotated(Direction<2>::upper_xi()),
              BlockNeighbors<2>({*outer_neighbor},
                                {{*outer_neighbor, rotated.inverse_map()}},
                                false));
        }
        blocks.emplace_back(map->get_clone(), id1 + 1, std::move(neighbors),
                            std::to_string(id1 + 1),
                            domain::topologies::hypercube<2>);
      }
    };

    const auto add_annulus = [&](const size_t id,
                                 const std::vector<size_t>& inner_neighbors,
                                 const std::vector<size_t>& outer_neighbors) {
      DirectionMap<2, BlockNeighbors<2>> neighbors{};
      if (not inner_neighbors.empty()) {
        std::unordered_set<size_t> neighbor_ids{};
        std::unordered_map<size_t, OrientationMap<2>> orientations{};
        neighbor_ids.emplace(inner_neighbors[0]);
        orientations.emplace(inner_neighbors[0], aligned);
        if (inner_neighbors.size() > 1) {
          neighbor_ids.emplace(inner_neighbors[1]);
          orientations.emplace(inner_neighbors[1], rotated);
        }
        neighbors.emplace(
            Direction<2>::lower_xi(),
            BlockNeighbors<2>(std::move(neighbor_ids), std::move(orientations),
                              inner_neighbors.size() == 1));
      }
      if (not outer_neighbors.empty()) {
        std::unordered_set<size_t> neighbor_ids{};
        std::unordered_map<size_t, OrientationMap<2>> orientations{};
        neighbor_ids.emplace(outer_neighbors[0]);
        orientations.emplace(outer_neighbors[0], aligned);
        if (outer_neighbors.size() > 1) {
          neighbor_ids.emplace(outer_neighbors[1]);
          orientations.emplace(outer_neighbors[1], rotated);
        }
        neighbors.emplace(
            Direction<2>::upper_xi(),
            BlockNeighbors<2>(std::move(neighbor_ids), std::move(orientations),
                              outer_neighbors.size() == 1));
      }
      blocks.emplace_back(map->get_clone(), id, std::move(neighbors),
                          std::to_string(id), domain::topologies::annulus);
    };

    add_squares(0, std::nullopt, 2);
    add_annulus(2, {0, 1}, {3, 4});
    add_squares(3, 2, 5);
    add_annulus(5, {3, 4}, {6, 7});
    add_squares(6, 5, 8);
    add_annulus(8, {6, 7}, {9});
    add_annulus(9, {8}, {});

    return Domain{std::move(blocks)};
  }

  std::unordered_map<std::string, tnsr::I<double, 2, Frame::Grid>>
  grid_anchors() const override {
    ERROR("");
  }

  std::vector<DirectionMap<
      2, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
  external_boundary_conditions() const override {
    ERROR("");
  }

  std::vector<std::string> block_names() const override { ERROR(""); }

  std::vector<std::array<size_t, 2>> initial_extents() const override {
    ERROR("");
  }

  std::vector<std::array<size_t, 2>> initial_refinement_levels()
      const override {
    return {
        {{2, 0}}, {{0, 2}},  // squares 0, 1
        {{0, 0}},            // annulus 2
        {{2, 0}}, {{0, 2}},  // squares 3, 4
        {{2, 0}},            // annulus 5
        {{2, 0}}, {{0, 2}},  // squares 6, 7
        {{0, 0}},            // annulus 8
        {{0, 0}},            // annulus 9
    };
  }
};

void check_regions(const evolution::dg::NonconformingEqualRateRegions<2>&
                       nonconforming_regions) {
  const auto regions = nonconforming_regions.regions();
  CHECK(regions.size() == 4);
  REQUIRE(regions.contains("Nonconforming2"));
  REQUIRE(regions.contains("Nonconforming5-"));
  REQUIRE(regions.contains("Nonconforming5+"));
  REQUIRE(regions.contains("Nonconforming8"));

  const auto region2 = regions.at("Nonconforming2");
  const auto region5m = regions.at("Nonconforming5-");
  const auto region5p = regions.at("Nonconforming5+");
  const auto region8 = regions.at("Nonconforming8");

  const std::unordered_set<ElementId<2>> expected_region2_ids{
      {0, {{{2, 3}, {0, 0}}}},
      {1, {{{0, 0}, {2, 3}}}},
      {2, {{{0, 0}, {0, 0}}}},
      {3, {{{2, 0}, {0, 0}}}},
      {4, {{{0, 0}, {2, 0}}}}};
  const std::unordered_set<ElementId<2>> expected_region5m_ids{
      {3, {{{2, 3}, {0, 0}}}},
      {4, {{{0, 0}, {2, 3}}}},
      {5, {{{2, 0}, {0, 0}}}}};
  const std::unordered_set<ElementId<2>> expected_region5p_ids{
      {5, {{{2, 3}, {0, 0}}}},
      {6, {{{2, 0}, {0, 0}}}},
      {7, {{{0, 0}, {2, 0}}}}};
  const std::unordered_set<ElementId<2>> expected_region8_ids{
      {6, {{{2, 3}, {0, 0}}}},
      {7, {{{0, 0}, {2, 3}}}},
      {8, {{{0, 0}, {0, 0}}}}};

  const auto all_elements =
      initial_element_ids(NonconformingCreator{}.initial_refinement_levels());

  for (const auto& id : all_elements) {
    CAPTURE(id);
    CHECK(nonconforming_regions.is_in_region(region2, id) ==
          expected_region2_ids.contains(id));
    CHECK(nonconforming_regions.is_in_region(region5m, id) ==
          expected_region5m_ids.contains(id));
    CHECK(nonconforming_regions.is_in_region(region5p, id) ==
          expected_region5p_ids.contains(id));
    CHECK(nonconforming_regions.is_in_region(region8, id) ==
          expected_region8_ids.contains(id));
  }
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.DG.EqualRateLts.NonconformingEqualRateRegions",
    "[Unit][Evolution]") {
  const std::unique_ptr<DomainCreator<2>> domain_creator =
      std::make_unique<NonconformingCreator>();

  const evolution::dg::NonconformingEqualRateRegions<2> regions(domain_creator);
  check_regions(regions);
  check_regions(serialize_and_deserialize(regions));
}
}  // namespace
