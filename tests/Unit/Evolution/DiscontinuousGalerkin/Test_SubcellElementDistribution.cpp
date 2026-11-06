// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Domain/Creators/AlignedLattice.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementDistribution.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Tags/ElementDistribution.hpp"
#include "Evolution/DiscontinuousGalerkin/OnlyDgBlockIds.hpp"
#include "Evolution/DiscontinuousGalerkin/SubcellElementDistribution.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace {
// Returns the expected override for a domain of three blocks along x, each
// with 4x4 elements, where the elements in `blocks_with_subcell` that satisfy
// `start_on_subcell(x_index, y_index)` get an override.
template <typename F>
std::unordered_map<ElementId<2>, std::array<size_t, 2>> expected_overrides(
    const std::vector<std::pair<size_t, F>>& blocks_with_subcell) {
  std::unordered_map<ElementId<2>, std::array<size_t, 2>> expected{};
  for (const auto& [block_id, start_on_subcell] : blocks_with_subcell) {
    for (const auto& element_id :
         initial_element_ids(block_id, std::array<size_t, 2>{{2, 2}})) {
      if (start_on_subcell(element_id.segment_id(0).index(),
                           element_id.segment_id(1).index())) {
        expected.emplace(element_id, std::array<size_t, 2>{{5, 7}});
      }
    }
  }
  return expected;
}

void test_compute_weighting_extents_override() {
  // Three blocks along x (block 0 is leftmost), each with 4x4 elements and
  // 3x4 DG grid points, so a subcell element has 5x7 FD grid points. An
  // element is on an external boundary in y if its y index is 0 or 3.
  const auto domain_creator = domain::creators::AlignedLattice<2>(
      {{{{0.0, 1.0, 2.0, 3.0}}, {{0.0, 1.0}}}}, {{{}, {}}}, {{{}, {}}},
      {{2, 2}}, {{3, 4}}, {}, {}, {});
  const auto domain = domain_creator.create_domain();
  const auto& blocks = domain.blocks();
  const auto& initial_refinement_levels =
      domain_creator.initial_refinement_levels();
  const auto& initial_extents = domain_creator.initial_extents();
  const auto compute = [&](const std::vector<size_t>& only_dg_block_ids,
                           const bool subcell_enabled_at_external_boundary) {
    return evolution::dg::compute_weighting_extents_override(
        only_dg_block_ids, blocks, initial_refinement_levels, initial_extents,
        subcell_enabled_at_external_boundary);
  };
  using StartOnSubcell = std::function<bool(size_t, size_t)>;
  const StartOnSubcell all = [](const size_t /*x*/, const size_t /*y*/) {
    return true;
  };
  const StartOnSubcell interior_in_y = [](const size_t /*x*/, const size_t y) {
    return y != 0 and y != 3;
  };

  // No DG-only blocks: with subcell enabled at external boundaries, every
  // element starts on subcell
  CHECK(compute({}, true) ==
        expected_overrides<StartOnSubcell>({{0, all}, {1, all}, {2, all}}));
  // Without subcell at external boundaries, only the interior elements do
  CHECK(compute({}, false) ==
        expected_overrides<StartOnSubcell>(
            {{0, [](const size_t x,
                    const size_t y) { return x != 0 and y != 0 and y != 3; }},
             {1, interior_in_y},
             {2, [](const size_t x, const size_t y) {
                return x != 3 and y != 0 and y != 3;
              }}}));

  // Block 2 is DG-only, so it gets no override and the elements of block 1
  // that border it (x index 3) run DG
  const StartOnSubcell not_bordering_right =
      [](const size_t x, const size_t /*y*/) { return x != 3; };
  const auto overrides = compute({2}, true);
  CHECK(overrides.size() == 16 + 12);
  CHECK(overrides == expected_overrides<StartOnSubcell>(
                         {{0, all}, {1, not_bordering_right}}));
  CHECK(compute({2}, false) ==
        expected_overrides<StartOnSubcell>(
            {{0, [](const size_t x,
                    const size_t y) { return x != 0 and y != 0 and y != 3; }},
             {1, [](const size_t x, const size_t y) {
                return x != 3 and y != 0 and y != 3;
              }}}));

  // A DG-only block in the middle removes the bordering elements on both sides
  CHECK(compute({1}, true) ==
        expected_overrides<StartOnSubcell>(
            {{0, not_bordering_right},
             {2, [](const size_t x, const size_t /*y*/) { return x != 0; }}}));

  // If every block is DG-only there is nothing to override
  CHECK(compute({0, 1, 2}, true).empty());

  // The order of `only_dg_block_ids` must not matter
  CHECK(compute({2, 1}, true) == compute({1, 2}, true));

  // 1D check: a single block, so no element borders a DG-only block
  const auto domain_creator_1d = domain::creators::AlignedLattice<1>(
      {{{{0.0, 1.0}}}}, {{{}}}, {{{}}}, {{1}}, {{4}}, {}, {}, {});
  const auto overrides_1d = evolution::dg::compute_weighting_extents_override(
      std::vector<size_t>{}, domain_creator_1d.create_domain().blocks(),
      domain_creator_1d.initial_refinement_levels(),
      domain_creator_1d.initial_extents(), true);
  CHECK(overrides_1d.size() == 2);
  for (const auto& [element_id, extents] : overrides_1d) {
    CHECK(element_id.block_id() == 0);
    CHECK(extents == std::array<size_t, 1>{{7}});
  }

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      evolution::dg::compute_weighting_extents_override(
          std::vector<size_t>{}, blocks, initial_refinement_levels,
          std::vector<std::array<size_t, 2>>{{{3, 4}}, {{3, 0}}, {{3, 4}}},
          true),
      Catch::Matchers::ContainsSubstring(
          "The initial extents of block 1 in dimension 1 must be non-zero"));
  CHECK_THROWS_WITH(
      evolution::dg::compute_weighting_extents_override(
          std::vector<size_t>{}, blocks, initial_refinement_levels,
          std::vector<std::array<size_t, 2>>{{{3, 4}}}, true),
      Catch::Matchers::ContainsSubstring(
          "The number of blocks (3), initial extents (1), and initial "
          "refinement levels (3) must be the same."));
#endif  // SPECTRE_DEBUG
}

void test_tag() {
  using UseSubcellTag =
      evolution::dg::Tags::UseSubcellGridPointsForDistribution;
  TestHelpers::db::test_simple_tag<UseSubcellTag>(
      "UseSubcellGridPointsForDistribution");
  CHECK(TestHelpers::test_option_tag<
        evolution::dg::OptionTags::UseSubcellGridPointsForDistribution>(
      "true"));
  CHECK_FALSE(TestHelpers::test_option_tag<
              evolution::dg::OptionTags::UseSubcellGridPointsForDistribution>(
      "false"));

  // Weighting by the subcell grid points is only meaningful for
  // `NumGridPoints`, so any other weight must be rejected when the option is
  // enabled
  CHECK(UseSubcellTag::create_from_options(
      true, std::optional{domain::ElementWeight::NumGridPoints}));
  for (const auto& element_weight :
       {std::optional<domain::ElementWeight>{std::nullopt},
        std::optional{domain::ElementWeight::Uniform},
        std::optional{domain::ElementWeight::NumGridPointsAndGridSpacing}}) {
    // Disabled is always fine, whatever the weight is
    CHECK_FALSE(UseSubcellTag::create_from_options(false, element_weight));
    CHECK_THROWS_WITH(
        UseSubcellTag::create_from_options(true, element_weight),
        Catch::Matchers::ContainsSubstring(
            "UseSubcellGridPointsForDistribution is only supported with "
            "ElementDistribution: NumGridPoints"));
  }
}

template <bool SubcellEnabled>
struct Metavars {
  struct SubcellOptions {
    static constexpr bool subcell_enabled = SubcellEnabled;
  };
};

// Without DG-subcell support the subcell tags must be absent, since the
// options they are built from do not exist in those executables
static_assert(
    std::is_same_v<
        evolution::dg::element_distribution_cache_tags<Metavars<false>, 2>,
        tmpl::list<domain::Tags::Domain<2>,
                   domain::Tags::ElementDistribution>>);
static_assert(
    std::is_same_v<
        evolution::dg::element_distribution_cache_tags<Metavars<true>, 2>,
        tmpl::list<domain::Tags::Domain<2>, domain::Tags::ElementDistribution,
                   evolution::dg::Tags::UseSubcellGridPointsForDistribution,
                   evolution::dg::Tags::OnlyDgBlockIds<2>>>);

SPECTRE_TEST_CASE("Unit.Evolution.DG.SubcellElementDistribution",
                  "[Unit][Evolution]") {
  test_compute_weighting_extents_override();
  test_tag();
}
}  // namespace
