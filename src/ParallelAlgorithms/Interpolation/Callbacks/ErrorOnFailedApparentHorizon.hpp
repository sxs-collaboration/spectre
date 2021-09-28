// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ApparentHorizons/FastFlow.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/PrettyType.hpp"

/// \cond
namespace db {
template <typename DbTags>
class DataBox;
}  // namespace db
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
/// \endcond

namespace intrp::callbacks {

/// \brief Callback for a failed apparent horizon find that simply errors.
struct ErrorOnFailedApparentHorizon {
  template <typename InterpolationTargetTag, typename DbTags,
            typename Metavariables, typename TemporalId>
  static void apply(const db::DataBox<DbTags>& /*box*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const TemporalId& /*temporal_id*/,
                    const FastFlow::Status failure_reason) {
    ERROR("Apparent horizon finder "
          << pretty_type::short_name<InterpolationTargetTag>()
          << " failed, reason = " << failure_reason);
  }
};

}  // namespace intrp::callbacks
