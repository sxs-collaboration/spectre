// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/Creators/Cylinder.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/BlockNeighbors.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Evolution/DgSubcell/ReconstructionMethod.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace evolution::dg::subcell {
namespace {
void test_impl(const std::vector<double>& expected_values,
               const size_t incorrect_value_index) {
  std::vector<double> values = expected_values;
  values[incorrect_value_index] += incorrect_value_index == 1 ? 1.0 : 0.1;

  const fd::ReconstructionMethod recons_method =
      fd::ReconstructionMethod::AllDimsAtOnce;

  CHECK(SubcellOptions(
            expected_values[0], static_cast<size_t>(expected_values[1]),
            expected_values[2], expected_values[3], false, false, recons_method,
            false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1) !=
        SubcellOptions(values[0], static_cast<size_t>(values[1]), values[2],
                       values[3], false, false, recons_method, false,
                       std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1));
  CHECK_FALSE(
      SubcellOptions(
          expected_values[0], static_cast<size_t>(expected_values[1]),
          expected_values[2], expected_values[3], false, false, recons_method,
          false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1) ==
      SubcellOptions(values[0], static_cast<size_t>(values[1]), values[2],
                     values[3], false, false, recons_method, false,
                     std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1));

  CHECK(SubcellOptions(
            expected_values[0], static_cast<size_t>(expected_values[1]),
            expected_values[2], expected_values[3], false, false, recons_method,
            false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1) !=
        SubcellOptions(
            expected_values[0], static_cast<size_t>(expected_values[1]),
            expected_values[2], expected_values[3], true, false, recons_method,
            false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1));
  CHECK_FALSE(
      SubcellOptions(
          expected_values[0], static_cast<size_t>(expected_values[1]),
          expected_values[2], expected_values[3], false, false, recons_method,
          false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1) ==
      SubcellOptions(
          expected_values[0], static_cast<size_t>(expected_values[1]),
          expected_values[2], expected_values[3], true, false, recons_method,
          false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1));

  CHECK(SubcellOptions(
            expected_values[0], static_cast<size_t>(expected_values[1]),
            expected_values[2], expected_values[3], false, false, recons_method,
            false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1) !=
        SubcellOptions(expected_values[0],
                       static_cast<size_t>(expected_values[1]),
                       expected_values[2], expected_values[3], false, false,
                       fd::ReconstructionMethod::DimByDim, false, std::nullopt,
                       ::fd::DerivativeOrder::Two, 1, 1, 1));
  CHECK_FALSE(SubcellOptions(expected_values[0],
                             static_cast<size_t>(expected_values[1]),
                             expected_values[2], expected_values[3], false,
                             false, recons_method, false, std::nullopt,
                             ::fd::DerivativeOrder::Two, 1, 1, 1) ==
              SubcellOptions(
                  expected_values[0], static_cast<size_t>(expected_values[1]),
                  expected_values[2], expected_values[3], false, false,
                  fd::ReconstructionMethod::DimByDim, false, std::nullopt,
                  ::fd::DerivativeOrder::Two, 1, 1, 1));
  CHECK_FALSE(
      SubcellOptions(
          expected_values[0], static_cast<size_t>(expected_values[1]),
          expected_values[2], expected_values[3], false, false, recons_method,
          false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1) ==
      SubcellOptions(
          expected_values[0], static_cast<size_t>(expected_values[1]),
          expected_values[2], expected_values[3], false, false, recons_method,
          true, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1));
  CHECK_FALSE(
      SubcellOptions(
          expected_values[0], static_cast<size_t>(expected_values[1]),
          expected_values[2], expected_values[3], false, false, recons_method,
          false, std::nullopt, ::fd::DerivativeOrder::Four, 1, 1, 1) ==
      SubcellOptions(
          expected_values[0], static_cast<size_t>(expected_values[1]),
          expected_values[2], expected_values[3], false, false, recons_method,
          false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1));
  CHECK_FALSE(
      SubcellOptions(
          expected_values[0], static_cast<size_t>(expected_values[1]),
          expected_values[2], expected_values[3], false, false, recons_method,
          false, std::nullopt, ::fd::DerivativeOrder::Two, 2, 1, 1) ==
      SubcellOptions(
          expected_values[0], static_cast<size_t>(expected_values[1]),
          expected_values[2], expected_values[3], false, false, recons_method,
          false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1));
  CHECK_FALSE(
      SubcellOptions(
          expected_values[0], static_cast<size_t>(expected_values[1]),
          expected_values[2], expected_values[3], false, false, recons_method,
          false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 2, 1) ==
      SubcellOptions(
          expected_values[0], static_cast<size_t>(expected_values[1]),
          expected_values[2], expected_values[3], false, false, recons_method,
          false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1));
  CHECK_FALSE(
      SubcellOptions(
          expected_values[0], static_cast<size_t>(expected_values[1]),
          expected_values[2], expected_values[3], false, false, recons_method,
          false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 2) ==
      SubcellOptions(
          expected_values[0], static_cast<size_t>(expected_values[1]),
          expected_values[2], expected_values[3], false, false, recons_method,
          false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1));
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.SubcellOptions",
                  "[Evolution][Unit]") {
  const std::vector<double> expected_values{4.0, static_cast<double>(1_st),
                                            2.0e-3, 2.0e-4};
  for (size_t i = 0; i < expected_values.size(); ++i) {
    test_impl(expected_values, i);
  }

  SubcellOptions options(expected_values[0],
                         static_cast<size_t>(expected_values[1]),
                         expected_values[2], expected_values[3], true, true,
                         fd::ReconstructionMethod::DimByDim, true, std::nullopt,
                         ::fd::DerivativeOrder::Four, 1, 1, 1, 2);
  const SubcellOptions deserialized_options =
      serialize_and_deserialize(options);
  CHECK(options == deserialized_options);

  CHECK(options == TestHelpers::test_option_tag<OptionTags::SubcellOptions>(
                       "TroubledCellIndicator:\n"
                       "  PerssonTci:\n"
                       "    Exponent: 4.0\n"
                       "    NumHighestModes: 1\n"
                       "  RdmpTci:\n"
                       "    Delta0: 2.0e-3\n"
                       "    Epsilon: 2.0e-4\n"
                       "  FdToDgTci:\n"
                       "    NumberOfStepsBetweenTciCalls: 1\n"
                       "    MinTciCallsAfterRollback: 1\n"
                       "    MinimumClearTcis: 1\n"
                       "  AlwaysUseSubcells: true\n"
                       "  EnableExtensionDirections: true\n"
                       "  UseHalo: true\n"
                       "  OnlyDgBlocksAndGroups: None\n"
                       "SubcellToDgReconstructionMethod: DimByDim\n"
                       "FiniteDifferenceDerivativeOrder: 4\n"
                       "FdInterpolationOrder: 2\n"));

  INFO("Test with block names and groups");
  const domain::creators::Cylinder cylinder{2.0,   10.0, 1.0,  8.0,
                                            false, 0_st, 5_st, false};
  const std::string opts_no_blocks =
      "SubcellToDgReconstructionMethod: DimByDim\n"
      "FiniteDifferenceDerivativeOrder: 4\n"
      "FdInterpolationOrder: 2\n"
      "TroubledCellIndicator:\n"
      "  PerssonTci:\n"
      "    Exponent: 4.0\n"
      "    NumHighestModes: 1\n"
      "  RdmpTci:\n"
      "    Delta0: 2.0e-3\n"
      "    Epsilon: 2.0e-4\n"
      "  FdToDgTci:\n"
      "    NumberOfStepsBetweenTciCalls: 1\n"
      "    MinTciCallsAfterRollback: 1\n"
      "    MinimumClearTcis: 1\n"
      "  AlwaysUseSubcells: true\n"
      "  EnableExtensionDirections: true\n"
      "  UseHalo: true\n";
  CHECK_THROWS_WITH(
      SubcellOptions(TestHelpers::test_option_tag<OptionTags::SubcellOptions>(
                         opts_no_blocks + "  OnlyDgBlocksAndGroups: [blah]\n"),
                     cylinder),
      Catch::Matchers::ContainsSubstring("The block or group 'blah'"));

  CHECK(SubcellOptions{
            TestHelpers::test_option_tag<OptionTags::SubcellOptions>(
                opts_no_blocks + "  OnlyDgBlocksAndGroups: [InnerCube]\n"),
            cylinder}
            .only_dg_block_ids()
            .size() == 1);
  CHECK(SubcellOptions{
            TestHelpers::test_option_tag<OptionTags::SubcellOptions>(
                opts_no_blocks + "  OnlyDgBlocksAndGroups: [Wedges]\n"),
            cylinder}
            .only_dg_block_ids()
            .size() == 4);

  INFO("Test auto-detection of non-hypercube blocks");
  {
    // Create a domain creator that produces a domain with a mix of hypercube
    // and non-hypercube blocks to test auto-detection.
    struct MixedTopologyCreator : public DomainCreator<3> {
      Domain<3> create_domain() const override {
        std::vector<Block<3>> blocks;
        // Block 0: hypercube (I1 in all dimensions)
        blocks.emplace_back(nullptr, 0, DirectionMap<3, BlockNeighbors<3>>{},
                            "Cube", domain::topologies::hypercube<3>);
        // Block 1: spherical shell (non-hypercube)
        blocks.emplace_back(nullptr, 1, DirectionMap<3, BlockNeighbors<3>>{},
                            "Shell", domain::topologies::spherical_shell);
        // Block 2: filled sphere (non-hypercube)
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
    const MixedTopologyCreator mixed_creator{};

    // No user-specified DG-only blocks: auto-detection should find Shell
    // (block 1) and Ball (block 2)
    const SubcellOptions mixed_opts{
        SubcellOptions{4.0, 1_st, 1.0e-3, 1.0e-4, false, false,
                       fd::ReconstructionMethod::DimByDim, false, std::nullopt,
                       ::fd::DerivativeOrder::Two, 1, 1, 1},
        mixed_creator};
    // Blocks 1 and 2 should be auto-detected as DG-only
    CHECK(mixed_opts.only_dg_block_ids().size() == 2);
    CHECK(alg::found(mixed_opts.only_dg_block_ids(), size_t{1}));
    CHECK(alg::found(mixed_opts.only_dg_block_ids(), size_t{2}));
    CHECK_FALSE(alg::found(mixed_opts.only_dg_block_ids(), size_t{0}));
    CHECK_FALSE(alg::found(mixed_opts.only_dg_block_ids(), size_t{3}));

    // With user-specified DG-only block that overlaps with auto-detected:
    // should not duplicate
    const SubcellOptions mixed_opts_with_user{
        SubcellOptions{4.0, 1_st, 1.0e-3, 1.0e-4, false, false,
                       fd::ReconstructionMethod::DimByDim, false,
                       std::optional{std::vector<std::string>{"Shell"}},
                       ::fd::DerivativeOrder::Two, 1, 1, 1},
        mixed_creator};
    // Shell (block 1) from user + Ball (block 2) from auto-detection = 2
    CHECK(mixed_opts_with_user.only_dg_block_ids().size() == 2);
    CHECK(alg::found(mixed_opts_with_user.only_dg_block_ids(), size_t{1}));
    CHECK(alg::found(mixed_opts_with_user.only_dg_block_ids(), size_t{2}));
  }
}
}  // namespace
}  // namespace evolution::dg::subcell
