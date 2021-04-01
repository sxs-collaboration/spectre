// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <variant>
#include <vector>

#include "Domain/Creators/ExpandOverBlocks.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace domain {
template <typename T, size_t Dim>
void test_expand_over_blocks(
    const std::variant<T, std::array<T, Dim>, std::vector<std::array<T, Dim>>>&
        input_value,
    const size_t num_blocks,
    const std::vector<std::array<T, Dim>>& expected_expanded_value) {
  CHECK(std::visit(ExpandOverBlocks<T, Dim>{num_blocks}, input_value) ==
        expected_expanded_value);
}

SPECTRE_TEST_CASE("Unit.Domain.ExpandOverBlocks", "[Domain][Unit]") {
  test_expand_over_blocks<size_t, 1>(size_t{2}, 3, {3, {2}});
  test_expand_over_blocks<size_t, 1>(std::array<size_t, 1>{2}, 3, {3, {2}});
  test_expand_over_blocks<size_t, 1>(
      std::vector<std::array<size_t, 1>>{{2}, {3}, {4}}, 3, {{2}, {3}, {4}});
  test_expand_over_blocks<size_t, 2>(size_t{2}, 3, {3, {2, 2}});
  test_expand_over_blocks<size_t, 2>(std::array<size_t, 2>{2, 3}, 3,
                                     {3, {2, 3}});
  test_expand_over_blocks<size_t, 2>(
      std::vector<std::array<size_t, 2>>{{2, 3}, {3, 4}, {4, 5}}, 3,
      {{2, 3}, {3, 4}, {4, 5}});
  test_expand_over_blocks<size_t, 3>(size_t{2}, 3, {3, {2, 2, 2}});
  test_expand_over_blocks<size_t, 3>(std::array<size_t, 3>{2, 3, 4}, 3,
                                     {3, {2, 3, 4}});
  test_expand_over_blocks<size_t, 3>(
      std::vector<std::array<size_t, 3>>{{2, 3, 4}, {3, 4, 5}, {4, 5, 6}}, 3,
      {{2, 3, 4}, {3, 4, 5}, {4, 5, 6}});
  CHECK_THROWS_WITH(
      SINGLE_ARG(ExpandOverBlocks<size_t, 1>{3}(
          std::vector<std::array<size_t, 1>>{{2}, {3}})),
      Catch::Matchers::Contains(
          "You supplied 2 values, but the domain creator has 3 blocks."));
  {
    // [expand_over_blocks_example]
    static constexpr size_t Dim = 3;
    const size_t num_blocks = 3;
    // This is an example for a variant that represents the distribution of
    // initial refinement levels, which might be parsed from options:
    using InitialRefinementOptionType =
        std::variant<size_t, std::array<size_t, Dim>,
                     std::vector<std::array<size_t, Dim>>>;
    // In this example the user specified a single number:
    const InitialRefinementOptionType initial_refinement_from_options{
        size_t{2}};
    try {
      // Invoke `ExpandOverBlocks`:
      const auto initial_refinement =
          std::visit(ExpandOverBlocks<size_t, Dim>{num_blocks},
                     initial_refinement_from_options);
      // Since a single number was specified, we expect the vector over blocks
      // is homogeneously and isotropically filled with that number:
      std::vector<std::array<size_t, Dim>> expected_initial_refinement{
          num_blocks, {2, 2, 2}};
      CHECK(initial_refinement == expected_initial_refinement);
    } catch (const std::length_error& error) {
      // This would be a `PARSE_ERROR` in an option-parsing context
      ERROR("Invalid 'InitialRefinement': " << error.what());
    }
    // [expand_over_blocks_example]
  }
  {
    // [expand_over_blocks_named_example]
    static constexpr size_t Dim = 2;
    // In this example we name the blocks, representing a cubed-sphere domain:
    const std::vector<std::string> block_names{"InnerCube", "East", "North",
                                               "West", "South"};
    // The blocks can also be grouped:
    const std::unordered_map<std::string, std::unordered_set<std::string>>
        block_groups{{"Wedges", {"East", "North", "West", "South"}}};
    // Now we can expand values over blocks by giving their names. This can also
    // be used with a std::variant like in the other example.
    ExpandOverBlocks<size_t, Dim> expand{block_names, block_groups};
    CHECK(expand({{"West", {{3, 4}}},
                  {"InnerCube", {{2, 3}}},
                  {"South", {{3, 4}}},
                  {"North", {{5, 6}}},
                  {"East", {{1, 2}}}}) ==
          std::vector<std::array<size_t, Dim>>{
              {{2, 3}}, {{1, 2}}, {{5, 6}}, {{3, 4}}, {{3, 4}}});
    // Instead of naming all blocks individually we can also name groups:
    CHECK(expand({{"InnerCube", {{2, 3}}}, {"Wedges", {{3, 4}}}}) ==
          std::vector<std::array<size_t, Dim>>{
              {{2, 3}}, {{3, 4}}, {{3, 4}}, {{3, 4}}, {{3, 4}}});
    // [expand_over_blocks_named_example]
    CHECK_THROWS_WITH(expand({{"InnerCube", {{2, 3}}}, {"East", {{3, 4}}}}),
                      Catch::Matchers::Contains(
                          "You supplied 2 values, but the domain creator has 5 "
                          "blocks: InnerCube, East, North, West, South"));
    CHECK_THROWS_WITH(expand({{{"Wedges", {{3, 4}}}}}),
                      Catch::Matchers::Contains(
                          "You supplied 4 values, but the domain creator has 5 "
                          "blocks: InnerCube, East, North, West, South"));
    CHECK_THROWS_WITH(
        expand({{"Wedges", {{3, 4}}}, {"East", {{3, 4}}}}),
        Catch::Matchers::Contains(
            "Duplicate block name 'East' (expanded from 'Wedges')."));
    CHECK_THROWS_WITH(
        expand({{"noblock", {{3, 4}}}, {"Wedges", {{3, 4}}}}),
        Catch::Matchers::Contains("Value for block 'InnerCube' is missing. Did "
                                  "you misspell its name?"));
  }
}

}  // namespace domain
