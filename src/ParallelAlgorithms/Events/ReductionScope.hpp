// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/// \cond
namespace Options {
class Option;
template <typename T>
struct create_from_yaml;
}  // namespace Options
/// \endcond

namespace Events {
/*!
 * \brief How reduction events like `Events::ObserveNorms` group elements into
 * reductions
 *
 * - `WholeDomain`: a single reduction over all observed elements.
 * - `ByBlock`: a separate reduction for each observed block.
 * - `ByBlockOrGroup`: a separate reduction for each entry of the
 *   `BlocksToObserve` option, where a block group is reduced as a whole.
 *
 * See `Events::reduction_subfile_suffix` for how the reductions are named.
 */
enum class ReductionScope : int8_t { WholeDomain, ByBlock, ByBlockOrGroup };

std::ostream& operator<<(std::ostream& os, ReductionScope reduction_scope);

/*!
 * \brief The suffix to append to the subfile path for an element in the block
 * \p block_name, or `std::nullopt` if the block is not observed.
 *
 * - `WholeDomain`: the suffix is empty, so all observed elements contribute to
 *   the same subfile.
 * - `ByBlock`: the suffix is `/<BlockName>`. Block groups listed in
 *   \p blocks_to_observe are expanded to their blocks.
 * - `ByBlockOrGroup`: the suffix is `/<BlockOrGroupName>`, the entry of
 *   \p blocks_to_observe that contains the block. Since each element
 *   contributes to only one reduction, it is an error if the block is
 *   contained in more than one entry. When \p blocks_to_observe is
 *   `std::nullopt` (i.e. all blocks) this is the same as `ByBlock`.
 *
 * \param reduction_scope How to group elements into reductions
 * \param blocks_to_observe The block or group names to observe, or
 * `std::nullopt` to observe all blocks
 * \param block_name The name of the block the element is in
 * \param all_block_names All block names in the domain
 * \param block_groups All block groups in the domain
 */
std::optional<std::string> reduction_subfile_suffix(
    ReductionScope reduction_scope,
    const std::optional<std::vector<std::string>>& blocks_to_observe,
    const std::string& block_name,
    const std::vector<std::string>& all_block_names,
    const std::unordered_map<std::string, std::unordered_set<std::string>>&
        block_groups);
}  // namespace Events

template <>
struct Options::create_from_yaml<Events::ReductionScope> {
  template <typename Metavariables>
  static Events::ReductionScope create(const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
Events::ReductionScope
Options::create_from_yaml<Events::ReductionScope>::create<void>(
    const Options::Option& options);
