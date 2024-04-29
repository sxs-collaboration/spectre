// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "Utilities/PrettyType.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace logging::Tags {
template <typename OptionsGroup>
struct Verbosity;
}  // namespace logging::Tags
/// \endcond

namespace intrp::callbacks {

/// \brief Callback for a failed apparent horizon find that prints a
/// message (if sufficient Verbosity is enabled) but does not
/// terminate the executable.
struct IgnoreFailedApparentHorizon {
  template <typename InterpolationTargetTag, typename DbTags,
            typename Metavariables, typename TemporalId>
  static void apply(const db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const TemporalId& /*temporal_id*/,
                    const FastFlow::Status failure_reason) {
    const auto& verbosity =
        db::get<logging::Tags::Verbosity<InterpolationTargetTag>>(box);
    if (verbosity >= ::Verbosity::Quiet) {
      Parallel::printf("Remark: Horizon finder %s failed, reason = %s\n",
                       pretty_type::name<InterpolationTargetTag>(),
                       failure_reason);
    }
  }
};

}  // namespace intrp::callbacks
