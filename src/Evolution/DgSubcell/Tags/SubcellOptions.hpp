// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/OptionsGroup.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::subcell {
namespace OptionTags {
/// System-agnostic options for DG-subcell
struct SubcellOptions {
  static std::string name() { return "Subcell"; }
  using type = evolution::dg::subcell::SubcellOptions;
  static constexpr Options::String help =
      "System-agnostic options for DG-subcell";
  using group = ::dg::OptionTags::DiscontinuousGalerkinGroup;
};
}  // namespace OptionTags

namespace Tags {
/// System-agnostic options for DG-subcell
template <size_t Dim>
struct SubcellOptions : db::SimpleTag {
  using type = evolution::dg::subcell::SubcellOptions;

  using option_tags = tmpl::list<OptionTags::SubcellOptions,
                                 ::domain::OptionTags::DomainCreator<Dim>>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(
      const type& subcell_options,
      const std::unique_ptr<DomainCreator<Dim>>& domain_creator) {
    return {subcell_options, *domain_creator};
  }
};
}  // namespace Tags
}  // namespace evolution::dg::subcell
