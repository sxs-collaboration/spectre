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
  CHECK(SubcellOptions(
            expected_values[0], static_cast<size_t>(expected_values[1]),
            expected_values[2], expected_values[3], false, false, recons_method,
            false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1, 1) ==
        SubcellOptions(
            expected_values[0], static_cast<size_t>(expected_values[1]),
            expected_values[2], expected_values[3], false, false, recons_method,
            false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1));
  CHECK(SubcellOptions(
            expected_values[0], static_cast<size_t>(expected_values[1]),
            expected_values[2], expected_values[3], false, false, recons_method,
            false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1, 2) !=
        SubcellOptions(
            expected_values[0], static_cast<size_t>(expected_values[1]),
            expected_values[2], expected_values[3], false, false, recons_method,
            false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1));
  CHECK(SubcellOptions(expected_values[0],
                       static_cast<size_t>(expected_values[1]),
                       expected_values[2], expected_values[3], false, false,
                       recons_method, false, std::nullopt,
                       ::fd::DerivativeOrder::Two, 1, 1, 1, 1, std::nullopt) ==
        SubcellOptions(
            expected_values[0], static_cast<size_t>(expected_values[1]),
            expected_values[2], expected_values[3], false, false, recons_method,
            false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1));
  CHECK(SubcellOptions(
            expected_values[0], static_cast<size_t>(expected_values[1]),
            expected_values[2], expected_values[3], false, false, recons_method,
            false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1, 1, 8) !=
        SubcellOptions(
            expected_values[0], static_cast<size_t>(expected_values[1]),
            expected_values[2], expected_values[3], false, false, recons_method,
            false, std::nullopt, ::fd::DerivativeOrder::Two, 1, 1, 1));
}

template <bool LocalTimeStepping>
struct Metavariables {
  static constexpr bool local_time_stepping = LocalTimeStepping;
};

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
                         ::fd::DerivativeOrder::Four, 1, 1, 1, 2, 8);
  const SubcellOptions deserialized_options =
      serialize_and_deserialize(options);
  CHECK(options == deserialized_options);

  CHECK(options == TestHelpers::test_option_tag<OptionTags::SubcellOptions,
                                                Metavariables<true>>(
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
                       "FdInterpolationOrder: 2\n"
                       "LtsStepsPerSlab: 8"));

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
      SubcellOptions(TestHelpers::test_option_tag<OptionTags::SubcellOptions,
                                                  Metavariables<false>>(
                         opts_no_blocks + "  OnlyDgBlocksAndGroups: [blah]\n"),
                     cylinder),
      Catch::Matchers::ContainsSubstring("The block or group 'blah'"));

  CHECK(SubcellOptions{
            TestHelpers::test_option_tag<OptionTags::SubcellOptions,
                                         Metavariables<false>>(
                opts_no_blocks + "  OnlyDgBlocksAndGroups: [InnerCube]\n"),
            cylinder}
            .only_dg_block_ids()
            .size() == 1);
  CHECK(SubcellOptions{
            TestHelpers::test_option_tag<OptionTags::SubcellOptions,
                                         Metavariables<false>>(
                opts_no_blocks + "  OnlyDgBlocksAndGroups: [Wedges]\n"),
            cylinder}
            .only_dg_block_ids()
            .size() == 4);

  CHECK_THROWS_WITH((TestHelpers::test_option_tag<OptionTags::SubcellOptions,
                                                  Metavariables<true>>(
                        opts_no_blocks + "  OnlyDgBlocksAndGroups: [Wedges]\n"
                                         "LtsStepsPerSlab: 0\n")),
                    Catch::Matchers::ContainsSubstring(
                        "Value 0 is below the lower bound of 1"));
  CHECK_THROWS_WITH((TestHelpers::test_option_tag<OptionTags::SubcellOptions,
                                                  Metavariables<true>>(
                        opts_no_blocks + "  OnlyDgBlocksAndGroups: [Wedges]\n"
                                         "LtsStepsPerSlab: 100000000000\n")),
                    Catch::Matchers::ContainsSubstring(
                        "Value 100000000000 is above the upper bound of"));
  CHECK_THROWS_WITH((TestHelpers::test_option_tag<OptionTags::SubcellOptions,
                                                  Metavariables<true>>(
                        opts_no_blocks + "  OnlyDgBlocksAndGroups: [Wedges]\n"
                                         "LtsStepsPerSlab: 10\n")),
                    Catch::Matchers::ContainsSubstring(
                        "LtsStepsPerSlab must be a power of 2"));
  CHECK(TestHelpers::test_option_tag<OptionTags::SubcellOptions,
                                     Metavariables<true>>(
            opts_no_blocks + "  OnlyDgBlocksAndGroups: [Wedges]\n"
                             "LtsStepsPerSlab: 16\n")
            .lts_steps_per_slab() == 16);
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      (TestHelpers::test_option_tag<OptionTags::SubcellOptions,
                                    Metavariables<false>>(
           opts_no_blocks + "  OnlyDgBlocksAndGroups: [Wedges]\n")
           .lts_steps_per_slab()),
      Catch::Matchers::ContainsSubstring("lts_steps_per_slab in GTS"));
#endif  // SPECTRE_DEBUG
}
}  // namespace
}  // namespace evolution::dg::subcell
