// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "Time/OptionTags/InitialSlabSize.hpp"
#include "Time/OptionTags/InitialTimeStep.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/TMPL.hpp"

namespace Initialization {
/// \ingroup InitializationGroup
/// \brief %Tags used during initialization of parallel components.
namespace Tags {
struct InitialTimeDelta : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::InitialTimeStep>;

  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double initial_time_step) {
    if (initial_time_step == 0.0) {
      ERROR_NO_TRACE("InitialTimeStep must be nonzero");
    }
    return initial_time_step;
  }
};

struct InitialSlabSize : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::InitialSlabSize>;

  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double initial_slab_size) {
    if (initial_slab_size == 0.0) {
      ERROR_NO_TRACE("InitialSlabSize must be nonzero");
    }
    return initial_slab_size;
  }
};
}  // namespace Tags
}  // namespace Initialization
