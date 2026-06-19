// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "Options/String.hpp"
#include "Utilities/PrettyType.hpp"

namespace OptionTags {
/*!
 * \ingroup OptionGroupsGroup
 * \brief Groups the filtering configurations in the input file.
 */
struct FilteringGroup {
  static std::string name() { return "Filtering"; }
  static constexpr Options::String help = "Options for filtering";
};

/*!
 * \ingroup OptionTagsGroup
 * \brief The option tag that retrieves the parameters for the filter
 * from the input file
 */
template <typename FilterType>
struct Filter {
  static std::string name() { return pretty_type::name<FilterType>(); }
  static constexpr Options::String help = "Options for the filter";
  using type = FilterType;
  using group = FilteringGroup;
};
}  // namespace OptionTags
