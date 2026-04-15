// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "Time/LtsMode.hpp"
#include "Utilities/TMPL.hpp"

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief The version of local time-stepping in use
struct LtsMode : db::SimpleTag {
  using type = ::LtsMode;

  static constexpr bool pass_metavariables = true;

  template <typename Metavars>
  using option_tags = tmpl::list<>;

  template <typename Metavars>
  static type create_from_options() {
    return Metavars::local_time_stepping ? ::LtsMode::Conservative
                                         : ::LtsMode::Off;
  }
};
}  // namespace Tags
