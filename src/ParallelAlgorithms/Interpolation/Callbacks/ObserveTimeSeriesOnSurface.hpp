// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/TagName.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/PostInterpolationCallback.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
namespace observers {
template <class Metavariables>
struct ObserverWriter;
}  // namespace observers
/// \endcond

namespace intrp {
namespace callbacks {

namespace detail {

template <typename... Ts>
auto make_legend(tmpl::list<Ts...> /* meta */) {
  return std::vector<std::string>{"Time", db::tag_name<Ts>()...};
}

template <typename DbTags, typename... Ts>
auto make_reduction_data(const db::DataBox<DbTags>& box, double time,
                         tmpl::list<Ts...> /* meta */) {
  return std::make_tuple(time, get<Ts>(box)...);
}

}  // namespace detail

/// \brief post_interpolation_callback that outputs
/// a time series on a surface.
///
/// Uses:
/// - Metavariables
///   - `temporal_id`
/// - DataBox:
///   - `TagsToObserve`
///
/// Conforms to the intrp::protocols::PostInterpolationCallback protocol
///
/// For requirements on InterpolationTargetTag, see
/// intrp::protocols::InterpolationTargetTag
template <typename TagsToObserve, typename InterpolationTargetTag>
struct ObserveTimeSeriesOnSurface
    : tt::ConformsTo<intrp::protocols::PostInterpolationCallback> {
  static constexpr double fill_invalid_points_with =
      std::numeric_limits<double>::quiet_NaN();

  template <typename DbTags, typename Metavariables, typename TemporalId>
  static void apply(const db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const TemporalId& temporal_id) {
    auto& proxy = Parallel::get_parallel_component<
        observers::ObserverWriter<Metavariables>>(cache);

    // We call this on proxy[0] because the 0th element of a NodeGroup is
    // always guaranteed to be present.
    Parallel::threaded_action<
        observers::ThreadedActions::WriteReductionDataRow>(
        proxy[0],
        std::string{"/" + pretty_type::name<InterpolationTargetTag>()},
        detail::make_legend(TagsToObserve{}),
        detail::make_reduction_data(
            box, InterpolationTarget_detail::get_temporal_id_value(temporal_id),
            TagsToObserve{}));
  }
};
}  // namespace callbacks
}  // namespace intrp
