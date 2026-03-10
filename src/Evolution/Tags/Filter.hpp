// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <pup_stl.h>
#include <string>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Domain.hpp"
#include "NumericalAlgorithms/LinearOperators/Filter.hpp"
#include "Options/String.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/StdHelpers.hpp"

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
 * \brief Option tag that constructs a list of filters from the input file.
 */
template <typename ListLabel>
struct FilterList {
  static std::string name() { return pretty_type::short_name<ListLabel>(); }
  static constexpr Options::String help =
      "A list of filters applied in the order specified.";
  using type = std::vector<std::unique_ptr<::Filters::Filter>>;
  using group = FilteringGroup;
};
}  // namespace OptionTags

namespace Filters::Tags {
/*!
 * \brief The DataBox tag for a list of filters.
 *
 * Also checks if the specified blocks are actually in the domain.
 */
template <typename ListLabel>
struct FilterList : db::SimpleTag {
  using type = std::vector<std::unique_ptr<::Filters::Filter>>;
  template <typename Metavariables>
  using option_tags =
      tmpl::list<::OptionTags::FilterList<ListLabel>,
                 domain::OptionTags::DomainCreator<Metavariables::volume_dim>>;

  static constexpr bool pass_metavariables = true;
  template <typename Metavariables>
  static type create_from_options(
      const type& filters_in,
      const std::unique_ptr<DomainCreator<Metavariables::volume_dim>>&
          domain_creator) {
    auto filters = deserialize<type>(serialize<type>(filters_in).data());
    for (const auto& filter : filters) {
      const auto& blocks_to_filter = filter->blocks_to_filter();
      if (not blocks_to_filter.has_value()) {
        continue;
      }
      const auto& block_names = domain_creator->block_names();
      const auto& block_groups = domain_creator->block_groups();

      if (block_names.size() == 0) {
        ERROR(
            "The domain chosen doesn't use block names, but the Filter tag has "
            "specified block names to use.");
      }

      // The name must either be a block or a block group
      for (const std::string& block_to_filter : blocks_to_filter.value()) {
        const auto block_name_iter = alg::find(block_names, block_to_filter);
        if (block_name_iter == block_names.end() and
            block_groups.count(block_to_filter) == 0) {
          ERROR("Specified block (group) name '"
                << block_to_filter
                << "' is not a block name or a block "
                   "group. Existing blocks are:\n"
                << block_names << "\nExisting block groups are:\n"
                << keys_of(block_groups));
        }
      }
    }
    return filters;
  }
};
}  // namespace Filters::Tags
