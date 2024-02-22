// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include "Domain/Creators/Cylinder.hpp"
#include "Evolution/DgSubcell/ReconstructionMethod.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"

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
            expected_values[2], expected_values[3], false, recons_method, false,
            std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1) !=
        SubcellOptions(values[0], static_cast<size_t>(values[1]), values[2],
                       values[3], false, recons_method, false, std::nullopt,
                       ::fd::DerivativeOrder::Two, 1, 1, 1));
  CHECK_FALSE(SubcellOptions(
                  expected_values[0], static_cast<size_t>(expected_values[1]),
                  expected_values[2], expected_values[3], false, recons_method,
                  false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1) ==
              SubcellOptions(values[0], static_cast<size_t>(values[1]),
                             values[2], values[3], false, recons_method, false,
                             std::nullopt, ::fd::DerivativeOrder::Two, 1, 1,
                             1));

  CHECK(SubcellOptions(
            expected_values[0], static_cast<size_t>(expected_values[1]),
            expected_values[2], expected_values[3], false, recons_method, false,
            std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1) !=
        SubcellOptions(
            expected_values[0], static_cast<size_t>(expected_values[1]),
            expected_values[2], expected_values[3], true, recons_method, false,
            std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1));
  CHECK_FALSE(SubcellOptions(
                  expected_values[0], static_cast<size_t>(expected_values[1]),
                  expected_values[2], expected_values[3], false, recons_method,
                  false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1) ==
              SubcellOptions(
                  expected_values[0], static_cast<size_t>(expected_values[1]),
                  expected_values[2], expected_values[3], true, recons_method,
                  false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1));

  CHECK(SubcellOptions(
            expected_values[0], static_cast<size_t>(expected_values[1]),
            expected_values[2], expected_values[3], false, recons_method, false,
            std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1) !=
        SubcellOptions(expected_values[0],
                       static_cast<size_t>(expected_values[1]),
                       expected_values[2], expected_values[3], false,
                       fd::ReconstructionMethod::DimByDim, false, std::nullopt,
                       ::fd::DerivativeOrder::Two, 1, 1, 1));
  CHECK_FALSE(SubcellOptions(
                  expected_values[0], static_cast<size_t>(expected_values[1]),
                  expected_values[2], expected_values[3], false, recons_method,
                  false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1) ==
              SubcellOptions(
                  expected_values[0], static_cast<size_t>(expected_values[1]),
                  expected_values[2], expected_values[3], false,
                  fd::ReconstructionMethod::DimByDim, false, std::nullopt,
                  ::fd::DerivativeOrder::Two, 1, 1, 1));
  CHECK_FALSE(SubcellOptions(
                  expected_values[0], static_cast<size_t>(expected_values[1]),
                  expected_values[2], expected_values[3], false, recons_method,
                  false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1) ==
              SubcellOptions(
                  expected_values[0], static_cast<size_t>(expected_values[1]),
                  expected_values[2], expected_values[3], false, recons_method,
                  true, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1));
  CHECK_FALSE(SubcellOptions(
                  expected_values[0], static_cast<size_t>(expected_values[1]),
                  expected_values[2], expected_values[3], false, recons_method,
                  false, std::nullopt, ::fd::DerivativeOrder::Four, 1, 1, 1) ==
              SubcellOptions(
                  expected_values[0], static_cast<size_t>(expected_values[1]),
                  expected_values[2], expected_values[3], false, recons_method,
                  false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1));
  CHECK_FALSE(SubcellOptions(
                  expected_values[0], static_cast<size_t>(expected_values[1]),
                  expected_values[2], expected_values[3], false, recons_method,
                  false, std::nullopt, ::fd::DerivativeOrder::Two, 2, 1, 1) ==
              SubcellOptions(
                  expected_values[0], static_cast<size_t>(expected_values[1]),
                  expected_values[2], expected_values[3], false, recons_method,
                  false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1));
  CHECK_FALSE(SubcellOptions(
                  expected_values[0], static_cast<size_t>(expected_values[1]),
                  expected_values[2], expected_values[3], false, recons_method,
                  false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 2, 1) ==
              SubcellOptions(
                  expected_values[0], static_cast<size_t>(expected_values[1]),
                  expected_values[2], expected_values[3], false, recons_method,
                  false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1));
  CHECK_FALSE(SubcellOptions(
                  expected_values[0], static_cast<size_t>(expected_values[1]),
                  expected_values[2], expected_values[3], false, recons_method,
                  false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 2) ==
              SubcellOptions(
                  expected_values[0], static_cast<size_t>(expected_values[1]),
                  expected_values[2], expected_values[3], false, recons_method,
                  false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1));
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.SubcellOptions",
                  "[Evolution][Unit]") {
  const std::vector<double> expected_values{4.0, static_cast<double>(1_st),
                                            2.0e-3, 2.0e-4};
  for (size_t i = 0; i < expected_values.size(); ++i) {
    test_impl(expected_values, i);
  }

  const SubcellOptions options(
      expected_values[0], static_cast<size_t>(expected_values[1]),
      expected_values[2], expected_values[3], true,
      fd::ReconstructionMethod::DimByDim, true, std::nullopt,
      ::fd::DerivativeOrder::Four, 1, 1, 1);
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
                       "  UseHalo: true\n"
                       "  OnlyDgBlocksAndGroups: None\n"
                       "SubcellToDgReconstructionMethod: DimByDim\n"
                       "FiniteDifferenceDerivativeOrder: 4\n"));

  INFO("Test with block names and groups");
  const domain::creators::Cylinder cylinder{2.0,   10.0, 1.0,  8.0,
                                            false, 0_st, 5_st, false};
  const std::string opts_no_blocks =
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
      "  UseHalo: true\n";
  const std::string opts_end =
      "SubcellToDgReconstructionMethod: DimByDim\n"
      "FiniteDifferenceDerivativeOrder: 4\n";
  CHECK_THROWS_WITH(
      SubcellOptions(
          TestHelpers::test_option_tag<OptionTags::SubcellOptions>(
              opts_no_blocks + "  OnlyDgBlocksAndGroups: [blah]\n" + opts_end),
          cylinder),
      Catch::Matchers::ContainsSubstring("The block or group 'blah'"));

  CHECK(SubcellOptions{TestHelpers::test_option_tag<OptionTags::SubcellOptions>(
                           opts_no_blocks +
                           "  OnlyDgBlocksAndGroups: [InnerCube]\n" + opts_end),
                       cylinder}
            .only_dg_block_ids()
            .size() == 1);
  CHECK(SubcellOptions{TestHelpers::test_option_tag<OptionTags::SubcellOptions>(
                           opts_no_blocks +
                           "  OnlyDgBlocksAndGroups: [Wedges]\n" + opts_end),
                       cylinder}
            .only_dg_block_ids()
            .size() == 4);
}
}  // namespace
}  // namespace evolution::dg::subcell
