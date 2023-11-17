// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <sstream>

#include "DataStructures/DataBox/DataBox.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
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
  static void apply(const db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const TemporalId& temporal_id,
                    const FastFlow::Status failure_reason) {
    if (failure_reason == FastFlow::Status::InterpolationFailure) {
      const auto& invalid_indices =
          db::get<::intrp::Tags::IndicesOfInvalidInterpPoints<TemporalId>>(box);
      if (invalid_indices.find(temporal_id) != invalid_indices.end() and
          not invalid_indices.at(temporal_id).empty()) {
        // There are invalid points (i.e. points that could not be
        // interpolated). Print info about those points.

        // First get the actual points
        const auto coords =
            InterpolationTargetTag::compute_target_points::points(
                box, tmpl::type_<Metavariables>{}, temporal_id);

        // Now output some information about them
        std::ostringstream os;
        os << "\nInvalid points (in Strahlkorper frame) at time "
           << InterpolationTarget_detail::get_temporal_id_value(temporal_id)
           << " are:\n";
        for (const auto index : invalid_indices.at(temporal_id)) {
          os << "(" << get<0>(coords)[index] << "," << get<1>(coords)[index]
             << "," << get<2>(coords)[index] << ")\n";
        }
        ERROR("Apparent horizon finder "
              << pretty_type::name<InterpolationTargetTag>()
              << " failed, reason = " << failure_reason << os.str());
      }
    }
    ERROR("Apparent horizon finder "
          << pretty_type::name<InterpolationTargetTag>()
          << " failed, reason = " << failure_reason << " at time "
          << InterpolationTarget_detail::get_temporal_id_value(temporal_id));
  }
};

}  // namespace intrp::callbacks
