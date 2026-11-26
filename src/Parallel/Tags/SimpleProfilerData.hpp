// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "Utilities/SimpleProfiler.hpp"
#include "Utilities/TMPL.hpp"

namespace Parallel::Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ParallelGroup
/// Tag to retrieve the `profiling::SimpleProfilerData` used for
/// profiling parallel executables.
struct SimpleProfilerData : db::SimpleTag {
  using type = std::vector<profiling::SimpleProfilerData>;
  constexpr static bool pass_metavariables = false;
  using option_tags = tmpl::list<>;
  // This is somewhat an abuse, since this doesn't actually take any
  // options, but we need it initialized really early before anyone
  // tries to profile anything.
  static type create_from_options();
};
}  // namespace Parallel::Tags
