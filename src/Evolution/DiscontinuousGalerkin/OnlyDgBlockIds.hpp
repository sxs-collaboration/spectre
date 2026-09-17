// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Evolution/DiscontinuousGalerkin/OptionTags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t VolumeDim>
class Block;
template <size_t VolumeDim>
class DomainCreator;
/// \endcond

namespace evolution::dg {
/*!
 * \brief Returns the IDs of the `Block`s on which only the DG scheme is
 * allowed to run, i.e. that are not subcell-capable.
 *
 * This is the union of the blocks and groups listed in
 * `evolution::dg::OptionTags::OnlyDgBlocksAndGroups` and the blocks whose
 * topology does not support subcell (see `block_supports_subcell`).
 *
 * \note This is not inside the `DgSubcell` library so that it can be used by
 * code (like the element distribution) that must not depend on `DgSubcell`.
 */
template <size_t Dim>
std::vector<size_t> compute_only_dg_block_ids(
    const std::optional<std::vector<std::string>>&
        only_dg_block_and_group_names,
    const std::vector<std::string>& block_names,
    const std::unordered_map<std::string, std::unordered_set<std::string>>&
        block_groups,
    const std::vector<Block<Dim>>& blocks);

/*!
 * \brief Returns the IDs of the `Block`s on which only the DG scheme is
 * allowed to run, i.e. that are not subcell-capable.
 *
 * Convenience overload that builds the `Domain` from the \p domain_creator.
 * If you already have a `Domain`, use the overload taking the blocks
 * directly to avoid constructing a second one.
 */
template <size_t Dim>
std::vector<size_t> compute_only_dg_block_ids(
    const std::optional<std::vector<std::string>>&
        only_dg_block_and_group_names,
    const DomainCreator<Dim>& domain_creator);

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// See `evolution::dg::compute_only_dg_block_ids`.
template <size_t Dim>
struct OnlyDgBlockIds : db::SimpleTag {
  using type = std::vector<size_t>;
  using option_tags =
      tmpl::list<evolution::dg::OptionTags::OnlyDgBlocksAndGroups,
                 domain::OptionTags::DomainCreator<Dim>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(
      const std::optional<std::vector<std::string>>&
          only_dg_block_and_group_names,
      const std::unique_ptr<DomainCreator<Dim>>& domain_creator) {
    return compute_only_dg_block_ids(only_dg_block_and_group_names,
                                     *domain_creator);
  }
};
}  // namespace Tags
}  // namespace evolution::dg
