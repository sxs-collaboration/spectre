// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "Time/OptionTags/InitialTime.hpp"
#include "Utilities/TMPL.hpp"

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief Tag for the current time as a double
///
/// The meaning of "current time" varies during the algorithm, but
/// generally is whatever time is appropriate for the calculation
/// being run.  Usually this is the substep time, but things such as
/// dense-output calculations may temporarily change the value.
struct Time : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::InitialTime>;

  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double initial_time) {
    return initial_time;
  }
};
}  // namespace Tags
