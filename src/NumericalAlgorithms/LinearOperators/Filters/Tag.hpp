// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/OptionsGroup.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Filter.hpp"
#include "Options/String.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/TMPL.hpp"

namespace Filters::OptionTags {
/// \brief Option tag for the list of spectral filters read from the input file.
///
/// Parsed under the `Filtering` key in the input file, this tag holds a
/// `std::vector` of heap-allocated `Filters::Filter<Dim, TagList>` objects.
///
/// \tparam Dim Spatial dimension of the element mesh.
/// \tparam TagList `tmpl::list` of tensor tags in the `Variables` to filter.
template <size_t Dim, typename TagList>
struct SpectralFilters {
  using group = dg::OptionTags::DiscontinuousGalerkinGroup;
  static std::string name() { return "Filtering"; }
  static constexpr Options::String help =
      "A vector/list of the different filters that are applied in different "
      "elements.";
  using type = std::vector<std::unique_ptr<Filters::Filter<Dim, TagList>>>;
};
}  // namespace Filters::OptionTags

namespace Filters::Tags {
/// \brief DataBox tag for the spectral filters applied during DG time
/// integration in a single element.
///
/// Holds `Filters::Filter<Dim, TagList>`s that are applied in the element.
/// `TagList` is the `tmpl::list` of tensor tags in the `Variables` to be
/// filtered.
///
/// \tparam Dim Spatial dimension of the element mesh.
/// \tparam TagList `tmpl::list` of tensor tags in the `Variables` to filter.
template <size_t Dim, typename TagList>
struct SpectralFilter : db::SimpleTag {
  using type = std::unique_ptr<Filters::Filter<Dim, TagList>>;
};

/// \brief DataBox tag for the ordered list of all spectral filters for
/// different topologies applied during DG time integration.
///
/// Each element holds a local copy of the filters that should be applied in it.
///
/// \tparam Dim Spatial dimension of the element mesh.
/// \tparam TagList `tmpl::list` of tensor tags in the `Variables` to filter.
template <size_t Dim, typename TagList>
struct SpectralFilters : db::SimpleTag {
  using type = std::vector<std::unique_ptr<Filters::Filter<Dim, TagList>>>;
  static constexpr bool pass_metavariables = false;

  using option_tags =
      tmpl::list<Filters::OptionTags::SpectralFilters<Dim, TagList>,
                 domain::OptionTags::DomainCreator<Dim>>;

  static type create_from_options(
      const type& spectral_filters,
      const std::unique_ptr<DomainCreator<Dim>>& domain_creator) {
    auto filters = clone_unique_ptrs(spectral_filters);
    const auto& block_names = domain_creator->block_names();
    const auto& block_groups = domain_creator->block_groups();
    for (auto& filter : filters) {
      filter->set_blocks_to_filter(block_names, block_groups);
    }
    return filters;
  }
};
}  // namespace Filters::Tags
