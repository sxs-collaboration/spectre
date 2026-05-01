// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "Time/LtsMode.hpp"
#include "Time/OptionTags/LocalTimeStepping.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/TMPL.hpp"

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief The version of local time-stepping in use
struct LtsMode : db::SimpleTag {
  using type = ::LtsMode;

  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<::OptionTags::LocalTimeStepping>;

  static type create_from_options(const type lts_mode) { return lts_mode; }
};

/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// Version of LtsMode that forces a specific value, primarily for
/// executables without LTS support.
template <::LtsMode Mode>
struct LtsModeForced : LtsMode {
  using base = LtsMode;

  static type create_from_options(const type lts_mode) {
    // Check consistency rather than just returning the forced mode so
    // that other tag creation functions can rely on the parsed value
    // being correct.
    if (lts_mode != Mode) {
      ERROR_NO_TRACE(
          "This executable only supports LocalTimeStepping: " << Mode);
    }
    return lts_mode;
  }
};
}  // namespace Tags
