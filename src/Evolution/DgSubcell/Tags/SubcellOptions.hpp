// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/OptionsGroup.hpp"
#include "Options/Options.hpp"
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
struct SubcellOptions : db::SimpleTag {
  using type = evolution::dg::subcell::SubcellOptions;

  using option_tags = tmpl::list<OptionTags::SubcellOptions>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& subcell_options) {
    return subcell_options;
  }
};
}  // namespace Tags
}  // namespace evolution::dg::subcell
