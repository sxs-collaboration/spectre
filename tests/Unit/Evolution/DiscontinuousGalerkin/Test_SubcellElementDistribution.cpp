// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <vector>

#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/ElementDistribution.hpp"
#include "Domain/Tags/ElementDistribution.hpp"
#include "Evolution/DiscontinuousGalerkin/OnlyDgBlockIds.hpp"
#include "Evolution/DiscontinuousGalerkin/SubcellElementDistribution.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace {
void test_compute_weighting_extents_override() {
  const std::vector<std::array<size_t, 3>> initial_extents{
      {{3, 4, 5}}, {{6, 6, 6}}, {{2, 2, 2}}, {{7, 3, 3}}};

  // Blocks 1 and 2 are DG-only, so only blocks 0 and 3 get an override, with
  // 2 * N - 1 grid points per dimension
  const auto overrides = evolution::dg::compute_weighting_extents_override(
      std::vector<size_t>{1, 2}, initial_extents);
  CHECK(overrides.size() == 2);
  CHECK(overrides.at(0) == std::array<size_t, 3>{{5, 7, 9}});
  CHECK(overrides.at(3) == std::array<size_t, 3>{{13, 5, 5}});
  CHECK(not overrides.contains(1));
  CHECK(not overrides.contains(2));

  // No DG-only blocks means every block is overridden
  const auto all_overridden = evolution::dg::compute_weighting_extents_override(
      std::vector<size_t>{}, initial_extents);
  CHECK(all_overridden.size() == initial_extents.size());
  CHECK(all_overridden.at(1) == std::array<size_t, 3>{{11, 11, 11}});
  CHECK(all_overridden.at(2) == std::array<size_t, 3>{{3, 3, 3}});

  // If every block is DG-only there is nothing to override
  CHECK(evolution::dg::compute_weighting_extents_override(
            std::vector<size_t>{0, 1, 2, 3}, initial_extents)
            .empty());

  // The order of `only_dg_block_ids` must not matter
  CHECK(evolution::dg::compute_weighting_extents_override(
            std::vector<size_t>{2, 1}, initial_extents) == overrides);

  // 1D check
  const auto overrides_1d = evolution::dg::compute_weighting_extents_override(
      std::vector<size_t>{}, std::vector<std::array<size_t, 1>>{{{4}}});
  CHECK(overrides_1d.size() == 1);
  CHECK(overrides_1d.at(0) == std::array<size_t, 1>{{7}});

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      evolution::dg::compute_weighting_extents_override(
          std::vector<size_t>{}, std::vector<std::array<size_t, 2>>{{{3, 0}}}),
      Catch::Matchers::ContainsSubstring(
          "The initial extents of block 0 in dimension 1 must be non-zero"));
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
