// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "Time/LtsMode.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
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

/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// Version of LtsMode that forces a specific value, primarily for
/// executables without LTS support.
template <::LtsMode Mode>
struct LtsModeForced : LtsMode {
  using base = LtsMode;

  template <typename Metavars>
  static type create_from_options() {
    // Check consistency rather than just returning the forced mode so
    // that other tag creation functions can rely on the parsed value
    // being correct.
    const auto lts_mode = Metavars::local_time_stepping
                              ? ::LtsMode::Conservative
                              : ::LtsMode::Off;
    if (lts_mode != Mode) {
      ERROR_NO_TRACE(
          "This executable only supports LocalTimeStepping: " << Mode);
    }
    return lts_mode;
  }
};
}  // namespace Tags
