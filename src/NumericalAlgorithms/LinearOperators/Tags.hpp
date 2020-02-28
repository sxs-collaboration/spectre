// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Options/Options.hpp"

namespace OptionTags {
/*!
 * \ingroup OptionGroupsGroup
 * \brief Groups the filtering configurations in the input file.
 */
struct FilteringGroup {
  static std::string name() noexcept { return "Filtering"; }
  static constexpr OptionString help = "Options for filtering";
};

/*!
 * \ingroup OptionTagsGroup
 * \brief The option tag that retrieves the parameters for the filter
 * from the input file
 */
template <typename FilterType>
struct Filter {
  static std::string name() noexcept { return option_name<FilterType>(); }
  static constexpr OptionString help = "Options for the filter";
  using type = FilterType;
  using group = FilteringGroup;
};
}  // namespace OptionTags

namespace Filters {
namespace Tags {
/*!
 * \brief The global cache tag for the filter
 */
template <typename FilterType>
struct Filter : db::SimpleTag {
  static std::string name() noexcept { return "Filter"; }
  using type = FilterType;
  using option_tags = tmpl::list<::OptionTags::Filter<FilterType>>;

  static constexpr bool pass_metavariables = false;
  static FilterType create_from_options(const FilterType& filter) noexcept {
    return filter;
  }
};
}  // namespace Tags
}  // namespace Filters
