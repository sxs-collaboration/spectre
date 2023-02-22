// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Domain.hpp"
#include "Utilities/TMPL.hpp"

namespace domain::Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// The ::Domain.
template <size_t VolumeDim>
struct Domain : db::SimpleTag {
  using type = ::Domain<VolumeDim>;
  using option_tags = tmpl::list<domain::OptionTags::DomainCreator<VolumeDim>>;

  static constexpr bool pass_metavariables = false;
  static ::Domain<VolumeDim> create_from_options(
      const std::unique_ptr<::DomainCreator<VolumeDim>>& domain_creator);
};
}  // namespace domain::Tags
