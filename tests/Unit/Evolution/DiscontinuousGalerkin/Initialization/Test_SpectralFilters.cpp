// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/SpectralFilters.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/SpectralFilters.tpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Filter.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Tag.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct ScalarVar : db::SimpleTag {
  using type = Scalar<DataVector>;
};

using TagList = tmpl::list<ScalarVar>;

template <size_t Dim>
using FilterPtr = std::unique_ptr<Filters::Filter<Dim, TagList>>;

template <size_t Dim>
using FilterVec = std::vector<FilterPtr<Dim>>;

template <size_t Dim>
FilterPtr<Dim> make_hypercube(
    std::optional<std::vector<std::string>> blocks,
    const std::vector<std::string>& all_block_names = {}) {
  auto f = std::make_unique<Filters::Hypercube<Dim, TagList>>(
      4u, true, blocks, false, false, std::nullopt, std::nullopt);
  f->set_blocks_to_filter(all_block_names, {});
  return f;
}

template <size_t Dim>
auto make_box(FilterVec<Dim> filters, const size_t block_id,
              const size_t extents = 4) {
  return db::create<db::AddSimpleTags<
      Filters::Tags::SpectralFilters<Dim, TagList>, domain::Tags::Element<Dim>,
      domain::Tags::Mesh<Dim>, Filters::Tags::SpectralFilter<Dim, TagList>>>(
      std::move(filters),
      Element<Dim>{ElementId<Dim>{block_id},
                   typename Element<Dim>::Neighbors_t{}},
      Mesh<Dim>{extents, Spectral::Basis::Legendre,
                Spectral::Quadrature::GaussLobatto},
      FilterPtr<Dim>{nullptr});
}

template <size_t Dim>
void test() {
  INFO("Dim = " << Dim);
  const std::vector<std::string> block_names{"Block0", "Block1", "Block2"};

  {
    INFO("Single all-blocks filter is selected for any element");
    FilterVec<Dim> filters;
    filters.push_back(make_hypercube<Dim>(std::nullopt));
    auto box = make_box<Dim>(std::move(filters), 0);
    db::mutate_apply<
        evolution::dg::Initialization::SpectralFilters<Dim, TagList>>(
        make_not_null(&box));
    CHECK(not db::get<Filters::Tags::SpectralFilter<Dim, TagList>>(box)
                  .blocks_to_filter()
                  .has_value());
  }

  {
    INFO("Block-restricted filter selected only for its own block");
    FilterVec<Dim> filters;
    filters.push_back(
        make_hypercube<Dim>(std::vector<std::string>{"Block2"}, block_names));
    auto box = make_box<Dim>(std::move(filters), 2);
    db::mutate_apply<
        evolution::dg::Initialization::SpectralFilters<Dim, TagList>>(
        make_not_null(&box));
    CHECK(db::get<Filters::Tags::SpectralFilter<Dim, TagList>>(box)
              .blocks_to_filter()
              .value() == std::vector<size_t>{2});
  }

  {
    INFO("Two disjoint-block filters: each element gets the correct one");
    // Filter A: Block0 only; Filter B: Block1 and Block2.
    // Both filters are in a single list; each element picks only its own.
    const auto make_combined_filters = [&block_names]() {
      FilterVec<Dim> filters;
      filters.push_back(
          make_hypercube<Dim>(std::vector<std::string>{"Block0"}, block_names));
      filters.push_back(make_hypercube<Dim>(
          std::vector<std::string>{"Block1", "Block2"}, block_names));
      return filters;
    };
    {
      auto box = make_box<Dim>(make_combined_filters(), 0);
      db::mutate_apply<
          evolution::dg::Initialization::SpectralFilters<Dim, TagList>>(
          make_not_null(&box));
      CHECK(db::get<Filters::Tags::SpectralFilter<Dim, TagList>>(box)
                .blocks_to_filter()
                .value() == std::vector<size_t>{0});
    }
    {
      auto box = make_box<Dim>(make_combined_filters(), 1);
      db::mutate_apply<
          evolution::dg::Initialization::SpectralFilters<Dim, TagList>>(
          make_not_null(&box));
      CHECK(db::get<Filters::Tags::SpectralFilter<Dim, TagList>>(box)
                .blocks_to_filter()
                .value() == std::vector<size_t>{1, 2});
    }
  }

  {
    INFO("Error when no filter matches the element");
    FilterVec<Dim> filters;
    filters.push_back(
        make_hypercube<Dim>(std::vector<std::string>{"Block2"}, block_names));
    auto box = make_box<Dim>(std::move(filters), 0);
    // Wrap in a lambda to avoid macro comma parsing of template args
    const auto invoke = [&box]() {
      db::mutate_apply<
          evolution::dg::Initialization::SpectralFilters<Dim, TagList>>(
          make_not_null(&box));
    };
    CHECK_THROWS_WITH(
        invoke(),
        Catch::Matchers::ContainsSubstring("No filter found for element") and
            Catch::Matchers::ContainsSubstring("with basis"));
  }

  {
    INFO("Error when more than one filter matches the element");
    FilterVec<Dim> filters;
    filters.push_back(make_hypercube<Dim>(std::nullopt));
    filters.push_back(
        make_hypercube<Dim>(std::vector<std::string>{"Block0"}, block_names));
    auto box = make_box<Dim>(std::move(filters), 0);
    const auto invoke = [&box]() {
      db::mutate_apply<
          evolution::dg::Initialization::SpectralFilters<Dim, TagList>>(
          make_not_null(&box));
    };
    CHECK_THROWS_WITH(invoke(),
                      Catch::Matchers::ContainsSubstring(
                          "Cannot specify more than one filter for element"));
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.DG.Initialization.SpectralFilters",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<Filters::Tags::SpectralFilter<3, TagList>>(
      "SpectralFilter");

  test<1>();
  test<2>();
  test<3>();
}
}  // namespace
