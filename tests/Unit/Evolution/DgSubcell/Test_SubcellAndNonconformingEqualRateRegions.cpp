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
#include "Evolution/DgSubcell/ReconstructionMethod.hpp"
#include "Evolution/DgSubcell/SubcellAndNonconformingEqualRateRegions.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/EqualRateRegionGenerator.hpp"
#include "NumericalAlgorithms/FiniteDifference/DerivativeOrder.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace {
static_assert(
    evolution::dg::equal_rate_region_generator<
        evolution::dg::subcell::SubcellAndNonconformingEqualRateRegions<1>, 1>);
static_assert(
    evolution::dg::equal_rate_region_generator<
        evolution::dg::subcell::SubcellAndNonconformingEqualRateRegions<2>, 2>);
static_assert(
    evolution::dg::equal_rate_region_generator<
        evolution::dg::subcell::SubcellAndNonconformingEqualRateRegions<3>, 3>);

// A layer of two squares, the second rotated
void add_squares(gsl::not_null<std::vector<Block<2>>*> blocks, const size_t id1,
                 const std::optional<size_t>& inner_neighbor,
                 const std::optional<size_t>& outer_neighbor) {
  const auto map =
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          domain::CoordinateMaps::Identity<2>{});
  const auto aligned = OrientationMap<2>::create_aligned();
  const OrientationMap<2> rotated{
      {{Direction<2>::upper_eta(), Direction<2>::lower_xi()}}};
  // First block
  {
    DirectionMap<2, BlockNeighbors<2>> neighbors{};
    neighbors.emplace(Direction<2>::upper_eta(),
                      BlockNeighbors<2>(id1 + 1, rotated));
    neighbors.emplace(Direction<2>::lower_eta(),
                      BlockNeighbors<2>(id1 + 1, rotated));
    if (inner_neighbor.has_value()) {
      neighbors.emplace(Direction<2>::lower_xi(),
                        BlockNeighbors<2>({*inner_neighbor},
                                          {{*inner_neighbor, aligned}}, false));
    }
    if (outer_neighbor.has_value()) {
      neighbors.emplace(Direction<2>::upper_xi(),
                        BlockNeighbors<2>({*outer_neighbor},
                                          {{*outer_neighbor, aligned}}, false));
    }
    blocks->emplace_back(map->get_clone(), id1, std::move(neighbors),
                         std::to_string(id1), domain::topologies::hypercube<2>);
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
                            {{*inner_neighbor, rotated.inverse_map()}}, false));
    }
    if (outer_neighbor.has_value()) {
      neighbors.emplace(
          rotated(Direction<2>::upper_xi()),
          BlockNeighbors<2>({*outer_neighbor},
                            {{*outer_neighbor, rotated.inverse_map()}}, false));
    }
    blocks->emplace_back(map->get_clone(), id1 + 1, std::move(neighbors),
                         std::to_string(id1 + 1),
                         domain::topologies::hypercube<2>);
  }
}

void add_annulus(gsl::not_null<std::vector<Block<2>>*> blocks, const size_t id,
                 const std::vector<size_t>& inner_neighbors,
                 const std::vector<size_t>& outer_neighbors) {
  const auto map =
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          domain::CoordinateMaps::Identity<2>{});
  const auto aligned = OrientationMap<2>::create_aligned();
  const OrientationMap<2> rotated{
      {{Direction<2>::upper_eta(), Direction<2>::lower_xi()}}};
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
  blocks->emplace_back(map->get_clone(), id, std::move(neighbors),
                       std::to_string(id), domain::topologies::annulus);
}

// Domain matching the one in Test_NonconformingEqualRateRegions:
//   Blocks 0,1  - square pair (hypercube, subcell-capable)
//   Block  2    - annulus (DG-only)
//   Blocks 3,4  - square pair (hypercube, subcell-capable)
//   Block  5    - annulus (DG-only, two nonconforming sides)
//   Blocks 6,7  - square pair (hypercube, subcell-capable)
//   Block  8    - annulus (DG-only)
//   Block  9    - annulus (DG-only, only inner neighbor)
//
// Nonconforming interfaces (annulus side is the "one" side):
//   Block 2  <-> blocks 0,1 (lower_xi) and blocks 3,4 (upper_xi)
//   Block 5  <-> blocks 3,4 (lower_xi) and blocks 6,7 (upper_xi)
//   Block 8  <-> blocks 6,7 (lower_xi) and block  9  (upper_xi)
//
// All interfaces involve at least one subcell-capable block, so no
// purely-DG nonconforming regions are created.
class NonconformingCreator : public DomainCreator<2> {
 public:
  Domain<2> create_domain() const override {
    std::vector<Block<2>> blocks{};
    add_squares(make_not_null(&blocks), 0, std::nullopt, 2);
    add_annulus(make_not_null(&blocks), 2, {0, 1}, {3, 4});
    add_squares(make_not_null(&blocks), 3, 2, 5);
    add_annulus(make_not_null(&blocks), 5, {3, 4}, {6, 7});
    add_squares(make_not_null(&blocks), 6, 5, 8);
    add_annulus(make_not_null(&blocks), 8, {6, 7}, {9});
    add_annulus(make_not_null(&blocks), 9, {8}, {});
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

  std::vector<std::string> block_names() const override {
    return {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
  }

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

// Build SubcellOptions.  Annulus blocks are excluded from subcell
// automatically by topology; square blocks can be forced DG-only by passing
// their names in `only_dg`.
evolution::dg::subcell::SubcellOptions make_subcell_options(
    std::optional<std::vector<std::string>> only_dg = std::nullopt) {
  return {4.0,
          1,
          2.0e-3,
          2.0e-4,
          false,
          false,
          evolution::dg::subcell::fd::ReconstructionMethod::DimByDim,
          false,
          std::move(only_dg),
          fd::DerivativeOrder::Two,
          1,
          1,
          1,
          1};
}

void check_regions(
    const evolution::dg::subcell::SubcellAndNonconformingEqualRateRegions<2>&
        combined) {
  // All nonconforming interfaces in this domain are subcell-adjacent, so only
  // the "Subcell" region (index 0) should exist: no NonconformingN regions.
  const auto regions = combined.regions();
  CHECK(regions.size() == 1);
  REQUIRE(regions.contains("Subcell"));
  CHECK(regions.at("Subcell") == 0);

  const std::vector<size_t> dg_only_blocks{2, 5, 8, 9};

  // The DG-only elements folded into the Subcell region are those sitting
  // at the outermost position in the direction perpendicular to a
  // subcell-adjacent nonconforming face.
  const std::unordered_set<ElementId<2>> expected_subcell_dg_only_elements{
      {2, {{{0, 0}, {0, 0}}}},
      {5, {{{2, 0}, {0, 0}}}},
      {5, {{{2, 3}, {0, 0}}}},
      {8, {{{0, 0}, {0, 0}}}}};

  const auto all_elements =
      initial_element_ids(NonconformingCreator{}.initial_refinement_levels());

  for (const auto& id : all_elements) {
    CAPTURE(id);
    const bool in_dg_block = alg::found(dg_only_blocks, id.block_id());
    if (not in_dg_block) {
      // All subcell-capable elements are always in the Subcell region.
      CHECK(combined.is_in_region(0, id));
    } else {
      // DG-only elements: in Subcell only at subcell-adjacent nonconforming
      // faces.
      CHECK(combined.is_in_region(0, id) ==
            expected_subcell_dg_only_elements.contains(id));
    }
  }
}

// Domain for testing transitive propagation of subcell-adjacency:
//   Same structure as NonconformingCreator except block 8 has TWO outer
//   nonconforming neighbors - a square pair (blocks 9, 10) that are forced
//   DG-only via subcell options.
//
// Block 8 is subcell-adjacent on its inner face (toward subcell-capable
// blocks 6,7) and sees a purely-DG nonconforming face on its outer side
// (toward DG-only blocks 9,10).  The main loop creates "Nonconforming8".
// The post-processing pass detects that "Nonconforming8" overlaps
// subcell_adjacent_dg_faces_ (via block 8) and merges blocks 9 and 10
// into the Subcell region transitively, leaving no NonconformingN regions.
class TransitiveNonconformingCreator : public DomainCreator<2> {
 public:
  Domain<2> create_domain() const override {
    std::vector<Block<2>> blocks{};
    add_squares(make_not_null(&blocks), 0, std::nullopt, 2);
    add_annulus(make_not_null(&blocks), 2, {0, 1}, {3, 4});
    add_squares(make_not_null(&blocks), 3, 2, 5);
    add_annulus(make_not_null(&blocks), 5, {3, 4}, {6, 7});
    add_squares(make_not_null(&blocks), 6, 5, 8);
    // Block 8: inner face subcell-adjacent (neighbors 6,7), outer face
    // nonconforming with two DG-only square neighbors (9,10).
    add_annulus(make_not_null(&blocks), 8, {6, 7}, {9, 10});
    add_squares(make_not_null(&blocks), 9, 8, std::nullopt);  // forced DG-only
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

  std::vector<std::string> block_names() const override {
    return {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"};
  }

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
        {{0, 0}},            // annulus 10 (new)
    };
  }
};

// Same domain as TransitiveNonconformingCreator but block 8 has refinement 1
// in xi, giving it two distinct elements (inner index 0, outer index 1).
class SeparableNonconformingCreator : public TransitiveNonconformingCreator {
 public:
  std::vector<std::array<size_t, 2>> initial_refinement_levels()
      const override {
    return {
        {{2, 0}}, {{0, 2}},  // squares 0, 1
        {{0, 0}},            // annulus 2
        {{2, 0}}, {{0, 2}},  // squares 3, 4
        {{2, 0}},            // annulus 5
        {{2, 0}}, {{0, 2}},  // squares 6, 7
        {{1, 0}},            // annulus 8 - 2 elements in xi, separable faces
        {{0, 0}},            // annulus 9
        {{0, 0}},            // annulus 10
    };
  }
};

// Block 8 is subcell-adjacent (inner face toward blocks 6,7) so it lands in
// subcell_adjacent_dg_faces_.  Its outer face toward blocks 9,10 is initially
// classified as a purely-DG nonconforming interface ("Nonconforming8").  The
// post-processing pass detects the overlap (block 8 in both sets) and merges
// "Nonconforming8" into Subcell, pulling blocks 9 and 10 along.
void check_transitive_regions() {
  const std::unique_ptr<DomainCreator<2>> domain_creator =
      std::make_unique<TransitiveNonconformingCreator>();
  // Blocks 9 and 10 are squares (hypercubes) so must be forced DG-only.
  const auto subcell_opts = make_subcell_options({{"9", "10"}});

  const evolution::dg::subcell::SubcellAndNonconformingEqualRateRegions<2>
      combined(subcell_opts, domain_creator);

  // Post-processing must have absorbed "Nonconforming8", leaving only Subcell.
  const auto regions = combined.regions();
  CHECK(regions.size() == 1);
  REQUIRE(regions.contains("Subcell"));
  CHECK(regions.at("Subcell") == 0);

  const std::vector<size_t> dg_only_blocks{2, 5, 8, 9, 10};

  // Same outermost DG-only elements as NonconformingCreator, plus blocks 9 and
  // 10 (each a single element at index 0, refinement_level 0) which reach
  // Subcell only via the transitive merge of block 8's outer face.
  const std::unordered_set<ElementId<2>>
      expected_subcell_dg_only_elements{
          {2, {{{0, 0}, {0, 0}}}}, {5, {{{2, 0}, {0, 0}}}},
          {5, {{{2, 3}, {0, 0}}}}, {8, {{{0, 0}, {0, 0}}}},
          {9, {{{0, 0}, {0, 0}}}},    // reached via transitivity
          {10, {{{0, 0}, {0, 0}}}}};  // reached via transitivity

  const auto all_elements = initial_element_ids(
      TransitiveNonconformingCreator{}.initial_refinement_levels());

  for (const auto& id : all_elements) {
    CAPTURE(id);
    const bool in_dg_block = alg::found(dg_only_blocks, id.block_id());
    if (not in_dg_block) {
      CHECK(combined.is_in_region(0, id));
    } else {
      CHECK(combined.is_in_region(0, id) ==
            expected_subcell_dg_only_elements.contains(id));
    }
  }
}

// With refinement 1 in xi on block 8, the inner element (index 0) is in
// Subcell and the outer element (index 1) is in "Nonconforming8" together with
// blocks 9 and 10.  The two groups can step at independent LTS rates.
void check_separable_regions() {
  const std::unique_ptr<DomainCreator<2>> domain_creator =
      std::make_unique<SeparableNonconformingCreator>();
  // Blocks 9 and 10 are squares (hypercubes) so must be forced DG-only.
  const auto subcell_opts = make_subcell_options({{"9", "10"}});

  const evolution::dg::subcell::SubcellAndNonconformingEqualRateRegions<2>
      combined(subcell_opts, domain_creator);

  // Post-processing must have left "Nonconforming8" intact
  const auto regions = combined.regions();
  CHECK(regions.size() == 2);
  REQUIRE(regions.contains("Subcell"));
  REQUIRE(regions.contains("Nonconforming8"));
  const size_t subcell_idx = regions.at("Subcell");
  const size_t nonconf_idx = regions.at("Nonconforming8");

  const std::vector<size_t> dg_only_blocks{2, 5, 8, 9, 10};

  // Subcell: block 8's inner element only (xi-index 0 at refinement_level 1).
  // Nonconforming8: block 8's outer element (xi-index 1) + blocks 9 and 10.
  const std::unordered_set<ElementId<2>> expected_subcell_dg_only{
      {2, {{{0, 0}, {0, 0}}}},
      {5, {{{2, 0}, {0, 0}}}},
      {5, {{{2, 3}, {0, 0}}}},
      {8, {{{1, 0}, {0, 0}}}}};  // inner element, index 0 at refinement 1
  const std::unordered_set<ElementId<2>> expected_nonconf{
      {8, {{{1, 1}, {0, 0}}}},  // outer element, index 1 = 2^1-1
      {9, {{{0, 0}, {0, 0}}}},
      {10, {{{0, 0}, {0, 0}}}}};

  const auto all_elements = initial_element_ids(
      SeparableNonconformingCreator{}.initial_refinement_levels());

  for (const auto& id : all_elements) {
    CAPTURE(id);
    const bool in_dg_block = alg::found(dg_only_blocks, id.block_id());
    if (not in_dg_block) {
      CHECK(combined.is_in_region(subcell_idx, id));
      CHECK(not combined.is_in_region(nonconf_idx, id));
    } else {
      CHECK(combined.is_in_region(subcell_idx, id) ==
            expected_subcell_dg_only.contains(id));
      CHECK(combined.is_in_region(nonconf_idx, id) ==
            expected_nonconf.contains(id));
    }
  }
}

// With all blocks forced DG-only (annuli via topology, squares via subcell
// options), no nonconforming interface is subcell-adjacent.  The combined
// generator must produce the same regions as NonconformingEqualRateRegions
// alone, plus an empty Subcell region - verifying the "no overlap -> same
// output" guarantee.
void check_no_overlap() {
  const std::unique_ptr<DomainCreator<2>> domain_creator =
      std::make_unique<NonconformingCreator>();
  const auto subcell_opts =
      make_subcell_options({{"0", "1", "3", "4", "6", "7"}});

  const evolution::dg::subcell::SubcellAndNonconformingEqualRateRegions<2>
      combined(subcell_opts, domain_creator);

  const auto regions = combined.regions();
  CHECK(regions.size() == 5);
  REQUIRE(regions.contains("Subcell"));
  REQUIRE(regions.contains("Nonconforming2"));
  REQUIRE(regions.contains("Nonconforming5-"));
  REQUIRE(regions.contains("Nonconforming5+"));
  REQUIRE(regions.contains("Nonconforming8"));

  const size_t subcell_idx = regions.at("Subcell");
  const size_t region2 = regions.at("Nonconforming2");
  const size_t region5m = regions.at("Nonconforming5-");
  const size_t region5p = regions.at("Nonconforming5+");
  const size_t region8 = regions.at("Nonconforming8");

  // Expected element sets match Test_NonconformingEqualRateRegions exactly.
  const std::unordered_set<ElementId<2>> expected_region2{
      {0, {{{2, 3}, {0, 0}}}},
      {1, {{{0, 0}, {2, 3}}}},
      {2, {{{0, 0}, {0, 0}}}},
      {3, {{{2, 0}, {0, 0}}}},
      {4, {{{0, 0}, {2, 0}}}}};
  const std::unordered_set<ElementId<2>> expected_region5m{
      {3, {{{2, 3}, {0, 0}}}},
      {4, {{{0, 0}, {2, 3}}}},
      {5, {{{2, 0}, {0, 0}}}}};
  const std::unordered_set<ElementId<2>> expected_region5p{
      {5, {{{2, 3}, {0, 0}}}},
      {6, {{{2, 0}, {0, 0}}}},
      {7, {{{0, 0}, {2, 0}}}}};
  const std::unordered_set<ElementId<2>> expected_region8{
      {6, {{{2, 3}, {0, 0}}}},
      {7, {{{0, 0}, {2, 3}}}},
      {8, {{{0, 0}, {0, 0}}}}};

  const auto all_elements =
      initial_element_ids(NonconformingCreator{}.initial_refinement_levels());

  for (const auto& id : all_elements) {
    CAPTURE(id);
    CHECK(not combined.is_in_region(subcell_idx, id));
    CHECK(combined.is_in_region(region2, id) == expected_region2.contains(id));
    CHECK(combined.is_in_region(region5m, id) ==
          expected_region5m.contains(id));
    CHECK(combined.is_in_region(region5p, id) ==
          expected_region5p.contains(id));
    CHECK(combined.is_in_region(region8, id) == expected_region8.contains(id));
  }
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.Subcell.SubcellAndNonconformingEqualRateRegions",
    "[Evolution][Unit]") {
  const std::unique_ptr<DomainCreator<2>> domain_creator =
      std::make_unique<NonconformingCreator>();
  const auto subcell_opts = make_subcell_options();

  const evolution::dg::subcell::SubcellAndNonconformingEqualRateRegions<2>
      combined(subcell_opts, domain_creator);
  check_regions(combined);
  check_regions(serialize_and_deserialize(combined));

  check_transitive_regions();
  check_separable_regions();
  check_no_overlap();
}
}  // namespace
