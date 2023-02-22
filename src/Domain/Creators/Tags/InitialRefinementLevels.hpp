// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Domain.hpp"
#include "Utilities/TMPL.hpp"

namespace domain::Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// The initial refinement level per dimension for all elements in each block of
/// the initial computational domain
template <size_t Dim>
struct InitialRefinementLevels : db::SimpleTag {
  using type = std::vector<std::array<size_t, Dim>>;
  using option_tags = tmpl::list<domain::OptionTags::DomainCreator<Dim>>;

  static constexpr bool pass_metavariables = false;
  static std::vector<std::array<size_t, Dim>> create_from_options(
      const std::unique_ptr<::DomainCreator<Dim>>& domain_creator);
};

}  // namespace domain::Tags
