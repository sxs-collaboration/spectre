// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/AlignedLattice.hpp"
#include "Domain/Creators/Cylinder.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/BlockNeighbors.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Evolution/DiscontinuousGalerkin/OnlyDgBlockIds.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Utilities/Algorithm.hpp"

namespace {
// A domain creator with a mix of subcell-capable and subcell-incapable
// blocks, used to test auto-detection of blocks whose topology does not
// support subcell
struct MixedTopologyCreator : public DomainCreator<3> {
  Domain<3> create_domain() const override {
    std::vector<Block<3>> blocks;
    // Block 0: hypercube (I1 in all dimensions)
    blocks.emplace_back(nullptr, 0, DirectionMap<3, BlockNeighbors<3>>{},
                        "Cube", domain::topologies::hypercube<3>);
    // Block 1: spherical shell (non-hypercube, does not support subcell)
    blocks.emplace_back(nullptr, 1, DirectionMap<3, BlockNeighbors<3>>{},
                        "Shell", domain::topologies::spherical_shell);
    // Block 2: filled sphere (non-hypercube, does not support subcell)
    blocks.emplace_back(nullptr, 2, DirectionMap<3, BlockNeighbors<3>>{},
                        "Ball", domain::topologies::full_sphere);
    // Block 3: another hypercube
    blocks.emplace_back(nullptr, 3, DirectionMap<3, BlockNeighbors<3>>{},
                        "Cube2", domain::topologies::hypercube<3>);
    return Domain<3>{std::move(blocks)};
  }
  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
  external_boundary_conditions() const override {
    return {};
  }
  std::vector<std::string> block_names() const override {
    return {"Cube", "Shell", "Ball", "Cube2"};
  }
  std::vector<std::array<size_t, 3>> initial_extents() const override {
    return {};
  }
  std::vector<std::array<size_t, 3>> initial_refinement_levels()
      const override {
    return {};
  }
};

void test_compute_only_dg_block_ids() {
  const MixedTopologyCreator mixed_creator{};

  // No user-specified DG-only blocks: find Shell (block 1) and Ball (block 2)
  const std::vector<size_t> auto_detected =
      evolution::dg::compute_only_dg_block_ids(std::nullopt, mixed_creator);
  CHECK(auto_detected.size() == 2);
  CHECK(alg::found(auto_detected, size_t{1}));
  CHECK(alg::found(auto_detected, size_t{2}));
  CHECK_FALSE(alg::found(auto_detected, size_t{0}));
  CHECK_FALSE(alg::found(auto_detected, size_t{3}));

  // A user-specified DG-only block (Cube, block 0) that doesn't overlap with
  // the auto-detected blocks should be added to the list
  const std::vector<size_t> with_user_block =
      evolution::dg::compute_only_dg_block_ids(
          std::optional{std::vector<std::string>{"Cube"}}, mixed_creator);
  CHECK(with_user_block.size() == 3);
  CHECK(alg::found(with_user_block, size_t{0}));
  CHECK(alg::found(with_user_block, size_t{1}));
  CHECK(alg::found(with_user_block, size_t{2}));

  // A user-specified DG-only block that overlaps with an auto-detected block
  // should not be duplicated
  const std::vector<size_t> with_overlapping_user_block =
      evolution::dg::compute_only_dg_block_ids(
          std::optional{std::vector<std::string>{"Shell"}}, mixed_creator);
  CHECK(with_overlapping_user_block.size() == 2);
  CHECK(alg::found(with_overlapping_user_block, size_t{1}));
  CHECK(alg::found(with_overlapping_user_block, size_t{2}));
}

// The block/group names are resolved against the `DomainCreator`, and an
// unknown name is an error
void test_block_groups() {
  const domain::creators::Cylinder cylinder{2.0,   10.0, 1.0,  8.0,
                                            false, 0_st, 5_st, false};
  CHECK(evolution::dg::compute_only_dg_block_ids(
            std::optional{std::vector<std::string>{"InnerCube"}}, cylinder)
            .size() == 1);
  CHECK(evolution::dg::compute_only_dg_block_ids(
            std::optional{std::vector<std::string>{"Wedges"}}, cylinder)
            .size() == 4);
  CHECK_THROWS_WITH(
      evolution::dg::compute_only_dg_block_ids(
          std::optional{std::vector<std::string>{"blah"}}, cylinder),
      Catch::Matchers::ContainsSubstring("The block or group 'blah'"));
}

void test_option_parsing() {
  CHECK(TestHelpers::test_option_tag<
            evolution::dg::OptionTags::OnlyDgBlocksAndGroups>("None") ==
        std::nullopt);
  CHECK(
      TestHelpers::test_option_tag<
          evolution::dg::OptionTags::OnlyDgBlocksAndGroups>("[Cube, Shell]") ==
      std::optional{std::vector<std::string>{"Cube", "Shell"}});
}

void test_tag() {
  TestHelpers::db::test_simple_tag<evolution::dg::Tags::OnlyDgBlockIds<3>>(
      "OnlyDgBlockIds");

  const std::unique_ptr<DomainCreator<3>> domain_creator =
      std::make_unique<MixedTopologyCreator>();
  const std::vector<size_t> only_dg_block_ids =
      evolution::dg::Tags::OnlyDgBlockIds<3>::create_from_options(
          std::nullopt, domain_creator);
  CHECK(only_dg_block_ids.size() == 2);
  CHECK(alg::found(only_dg_block_ids, size_t{1}));
  CHECK(alg::found(only_dg_block_ids, size_t{2}));
}

void test_borders_only_dg_block() {
  // Three blocks along x (block 0 is leftmost), each with 2x2 elements
  const auto domain_creator = domain::creators::AlignedLattice<2>(
      {{{{0.0, 1.0, 2.0, 3.0}}, {{0.0, 1.0}}}}, {{{}, {}}}, {{{}, {}}},
      {{1, 1}}, {{3, 3}}, {}, {}, {});
  const auto domain = domain_creator.create_domain();
  const auto& refinement_levels = domain_creator.initial_refinement_levels();
  const auto borders = [&domain, &refinement_levels](
                           const size_t block_id, const size_t x_index,
                           const std::vector<size_t>& only_dg_block_ids) {
    const ElementId<2> element_id{block_id, {{{1, x_index}, {1, 0}}}};
    return evolution::dg::borders_only_dg_block(
        domain::create_initial_element(element_id, domain.blocks(),
                                       refinement_levels),
        only_dg_block_ids);
  };

  for (size_t block_id = 0; block_id < 3; ++block_id) {
    for (size_t x_index = 0; x_index < 2; ++x_index) {
      CHECK_FALSE(borders(block_id, x_index, {}));
    }
  }
  // Block 2 is DG-only: only the element of block 1 next to it borders it.
  // The elements of block 2 border each other.
  CHECK_FALSE(borders(0, 0, {2}));
  CHECK_FALSE(borders(0, 1, {2}));
  CHECK_FALSE(borders(1, 0, {2}));
  CHECK(borders(1, 1, {2}));
  CHECK(borders(2, 0, {2}));
  CHECK(borders(2, 1, {2}));
  // Block 1 is DG-only: the elements on both sides of it border it
  CHECK_FALSE(borders(0, 0, {1}));
  CHECK(borders(0, 1, {1}));
  CHECK(borders(2, 0, {1}));
  CHECK_FALSE(borders(2, 1, {1}));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.OnlyDgBlockIds", "[Unit][Evolution]") {
  test_compute_only_dg_block_ids();
  test_block_groups();
  test_borders_only_dg_block();
  test_option_parsing();
  test_tag();
}
