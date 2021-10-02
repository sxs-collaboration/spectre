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
#include "Domain/OptionTags.hpp"
#include "Utilities/TMPL.hpp"

namespace domain::Tags {

/// A human-readable name for every block in the domain.
///
/// \warning Not all domain creators support block names yet, so the list may be
/// empty.
/// \see DomainCreator::block_names
template <size_t Dim>
struct BlockNames : db::SimpleTag {
  using type = std::vector<std::string>;

  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<domain::OptionTags::DomainCreator<Dim>>;
  static type create_from_options(
      const std::unique_ptr<::DomainCreator<Dim>>& domain_creator) {
    return domain_creator->block_names();
  }
};

/// Labeled groups of blocks
///
/// \see DomainCreator::block_groups
template <size_t Dim>
struct BlockGroups : db::SimpleTag {
  using type = std::unordered_map<std::string, std::unordered_set<std::string>>;

  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<domain::OptionTags::DomainCreator<Dim>>;
  static type create_from_options(
      const std::unique_ptr<::DomainCreator<Dim>>& domain_creator) {
    return domain_creator->block_groups();
  }
};

}  // namespace domain::Tags
