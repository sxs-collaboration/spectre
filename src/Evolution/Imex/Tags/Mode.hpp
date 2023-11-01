// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "Evolution/Imex/Mode.hpp"
#include "Evolution/Imex/Tags/OptionGroup.hpp"
#include "Options/Options.hpp"

namespace imex {
namespace OptionTags {
/// Tag for IMEX implementation to use
struct Mode {
  static constexpr Options::String help{"IMEX implementation to use"};
  using type = ::imex::Mode;
  using group = Group;
};
}  // namespace OptionTags

namespace Tags {
/// Tag for IMEX implementation to use
struct Mode : db::SimpleTag {
  using type = ::imex::Mode;
  using option_tags = tmpl::list<::imex::OptionTags::Mode>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& mode) { return mode; }
};
}  // namespace Tags
}  // namespace imex
