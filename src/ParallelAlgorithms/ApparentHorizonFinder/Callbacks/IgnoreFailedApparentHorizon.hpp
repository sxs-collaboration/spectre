// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
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
                    const TemporalId& temporal_id,
                    const FastFlow::Status failure_reason) {
    const auto& verbosity =
        db::get<logging::Tags::Verbosity<InterpolationTargetTag>>(box);
    if (verbosity >= ::Verbosity::Quiet) {
      const auto& invalid_indices =
          db::get<::intrp::Tags::IndicesOfInvalidInterpPoints<TemporalId>>(box);
      std::ostringstream os;
      if (invalid_indices.contains(temporal_id) and
          not invalid_indices.at(temporal_id).empty()) {
        // There are invalid points (i.e. points that could not be
        // interpolated). Print info about those points.

        // First get the actual points
        const auto coords =
            InterpolationTargetTag::compute_target_points::points(
                box, tmpl::type_<Metavariables>{}, temporal_id);

        const auto& fast_flow = db::get<::ah::Tags::FastFlow>(box);

        // Now output some information about them
        os << "Invalid points (in "
           << pretty_type::name<typename InterpolationTargetTag::
                                    compute_target_points::frame>()
           << " frame) at time "
           << InterpolationTarget_detail::get_temporal_id_value(temporal_id)
           << " at fast-flow iteration " << fast_flow.current_iteration()
           << " are:\n";
        for (const auto index : invalid_indices.at(temporal_id)) {
          os << " (" << get<0>(coords)[index] << "," << get<1>(coords)[index]
             << "," << get<2>(coords)[index] << ")\n";
        }
      }

      Parallel::printf(
          "Remark: Horizon finder %s failed. Number of interpolation retries: "
          "%u, reason = %s\n%s",
          pretty_type::name<InterpolationTargetTag>(),
          db::get<ah::Tags::FailedInterpolationIterations>(box), failure_reason,
          os.str());
    }
  }
};

}  // namespace intrp::callbacks
