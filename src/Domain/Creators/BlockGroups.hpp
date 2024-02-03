// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Utilities to work with block names and groups

#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace domain {

/*!
 * \brief Check if a block is in the given group
 *
 * \param block_name The name of the block to check
 * \param block_or_group_name The group name we're testing. Returns true if
 * the block is in this group. This can also be the name of the block itself.
 * \param block_groups All block groups.
 */
bool block_is_in_group(
    const std::string& block_name, const std::string& block_or_group_name,
    const std::unordered_map<std::string, std::unordered_set<std::string>>&
        block_groups);

/*!
 * \brief Expand a list of block or group names into a list of block names
 *
 * \param block_or_group_names Block or group names to expand
 * \param all_block_names All block names in the domain
 * \param block_groups Block groups used to expand the names
 * \return std::unordered_set<std::string> List of block names that appear in
 * \p all_block_names. If one of the input names was a group, then all block
 * names from that group are included. Overlaps between groups are allowed.
 */
std::unordered_set<std::string> expand_block_groups_to_block_names(
    const std::vector<std::string>& block_or_group_names,
    const std::vector<std::string>& all_block_names,
    const std::unordered_map<std::string, std::unordered_set<std::string>>&
        block_groups);

}  // namespace domain
