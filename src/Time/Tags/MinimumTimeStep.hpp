// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "Time/OptionTags/MinimumTimeStep.hpp"
#include "Utilities/TMPL.hpp"

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief The minimum step size without triggering an error
struct MinimumTimeStep : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<::OptionTags::MinimumTimeStep>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type option) { return option; }
};
}  // namespace Tags
