// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Tag.hpp"
#include "Utilities/TMPL.hpp"

class DataVector;

namespace {
// Minimal variable tag for instantiating Filters::Hypercube in this test.
struct ScalarVar : db::SimpleTag {
  using type = Scalar<DataVector>;
};

using TestTagList = tmpl::list<ScalarVar>;

void test_create_from_options() {
  using FiltersTag = Filters::Tags::SpectralFilters<3, TestTagList>;
  using FilterType = Filters::Hypercube<3, TestTagList>;

  const std::unique_ptr<DomainCreator<3>> domain_creator =
      std::make_unique<domain::creators::Brick>(
          std::array{0.0, 0.0, 0.0}, std::array{1.0, 1.0, 1.0},
          std::array{0_st, 0_st, 0_st}, std::array{2_st, 2_st, 2_st});

  const auto& block_names = domain_creator->block_names();

  FiltersTag::type input_filters;
  // Filter A: apply to all blocks (nullopt)
  input_filters.push_back(std::make_unique<FilterType>(
      4u, true, std::nullopt, false, false, std::nullopt, std::nullopt));
  // Filter B: apply only to the first (and only) block by name
  input_filters.push_back(std::make_unique<FilterType>(
      4u, true, std::vector<std::string>{block_names[0]}, false, false,
      std::nullopt, std::nullopt));

  auto result = FiltersTag::create_from_options(input_filters, domain_creator);

  // Cloning preserves the count
  CHECK(result.size() == 2);
  // All-blocks filter: blocks_to_filter() stays nullopt
  CHECK(not result[0]->blocks_to_filter().has_value());
  // Named-block filter: resolves "Brick" to index 0
  REQUIRE(result[1]->blocks_to_filter().has_value());
  CHECK(result[1]->blocks_to_filter().value() == std::vector<size_t>{0});
}

void test_invalid_block_name() {
  using FiltersTag = Filters::Tags::SpectralFilters<3, TestTagList>;
  using FilterType = Filters::Hypercube<3, TestTagList>;

  const std::unique_ptr<DomainCreator<3>> domain_creator =
      std::make_unique<domain::creators::Brick>(
          std::array{0.0, 0.0, 0.0}, std::array{1.0, 1.0, 1.0},
          std::array{0_st, 0_st, 0_st}, std::array{2_st, 2_st, 2_st});

  FiltersTag::type input_filters;
  input_filters.push_back(std::make_unique<FilterType>(
      4u, true, std::vector<std::string>{"NonExistent"}, false, false,
      std::nullopt, std::nullopt));

  CHECK_THROWS_WITH(
      FiltersTag::create_from_options(input_filters, domain_creator),
      Catch::Matchers::ContainsSubstring(
          "'NonExistent' is not one of the block names or groups"));
}

void test_databox() {
  using FiltersTag = Filters::Tags::SpectralFilters<3, TestTagList>;
  using FilterType = Filters::Hypercube<3, TestTagList>;

  const std::unique_ptr<DomainCreator<3>> domain_creator =
      std::make_unique<domain::creators::Brick>(
          std::array{0.0, 0.0, 0.0}, std::array{1.0, 1.0, 1.0},
          std::array{0_st, 0_st, 0_st}, std::array{2_st, 2_st, 2_st});

  FiltersTag::type input_filters;
  input_filters.push_back(std::make_unique<FilterType>(
      4u, true, std::nullopt, false, false, std::nullopt, std::nullopt));

  auto filters = FiltersTag::create_from_options(input_filters, domain_creator);
  auto box = db::create<db::AddSimpleTags<FiltersTag>>(std::move(filters));
  const auto& retrieved = db::get<FiltersTag>(box);
  CHECK(retrieved.size() == 1);
  CHECK(not retrieved[0]->blocks_to_filter().has_value());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.LinearOperators.Filter.Tag",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  // SpectralFilter simple tag
  TestHelpers::db::test_simple_tag<
      Filters::Tags::SpectralFilter<1, tmpl::list<>>>("SpectralFilter");
  TestHelpers::db::test_simple_tag<
      Filters::Tags::SpectralFilter<2, tmpl::list<>>>("SpectralFilter");
  TestHelpers::db::test_simple_tag<
      Filters::Tags::SpectralFilter<3, tmpl::list<>>>("SpectralFilter");

  // SpectralFilters simple tag (vector variant)
  TestHelpers::db::test_simple_tag<
      Filters::Tags::SpectralFilters<1, tmpl::list<>>>("SpectralFilters");
  TestHelpers::db::test_simple_tag<
      Filters::Tags::SpectralFilters<2, tmpl::list<>>>("SpectralFilters");
  TestHelpers::db::test_simple_tag<
      Filters::Tags::SpectralFilters<3, tmpl::list<>>>("SpectralFilters");

  // OptionTags::SpectralFilters reports "Filtering" as its name for the input
  // file key
  CHECK(Filters::OptionTags::SpectralFilters<1, tmpl::list<>>::name() ==
        "Filtering");
  CHECK(Filters::OptionTags::SpectralFilters<3, tmpl::list<>>::name() ==
        "Filtering");

  test_create_from_options();
  test_invalid_block_name();
  test_databox();
}
