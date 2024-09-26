// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <vector>

#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/AddTemporalIdsToInterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorReceiveVolumeData.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t VolumeDim>
class ElementId;
namespace intrp {
template <typename Metavariables, typename Tag>
struct InterpolationTarget;
template <typename Metavariables>
struct Interpolator;
}  // namespace intrp
/// \endcond

namespace intrp {
/// \brief Send data to the interpolator for interpolation.
///
/// \note if `interpolator_id` is not `std::nullopt` then we send to the
/// `interpolator_id.value()` index of the `Interpolator` parallel
/// component. This can be used to keep a specific element always sending to
/// the same `Interpolator` element.
template <typename InterpolationTargetTag, size_t VolumeDim,
          typename Metavariables, typename... InterpolatorSourceVars>
void interpolate(
    const typename InterpolationTargetTag::temporal_id::type& temporal_id,
    const Mesh<VolumeDim>& mesh, Parallel::GlobalCache<Metavariables>& cache,
    const ElementId<VolumeDim>& array_index,
    const std::optional<int> interpolator_id,
    const InterpolatorSourceVars&... interpolator_source_vars_input) {
  Variables<typename Metavariables::interpolator_source_vars>
      interpolator_source_vars(mesh.number_of_grid_points());
  const std::tuple<const InterpolatorSourceVars&...>
      interpolator_source_vars_tuple{interpolator_source_vars_input...};
  tmpl::for_each<
      tmpl::make_sequence<tmpl::size_t<0>, sizeof...(InterpolatorSourceVars)>>(
      [&interpolator_source_vars,
       &interpolator_source_vars_tuple](auto index_v) {
        constexpr size_t index = tmpl::type_from<decltype(index_v)>::value;
        get<tmpl::at_c<typename Metavariables::interpolator_source_vars,
                       index>>(interpolator_source_vars) =
            get<index>(interpolator_source_vars_tuple);
      });

  // Send volume data to the Interpolator, to trigger interpolation.
  if (interpolator_id.has_value()) {
    auto interpolator =
        ::Parallel::get_parallel_component<Interpolator<Metavariables>>(
            cache)[interpolator_id.value()];
    Parallel::simple_action<Actions::InterpolatorReceiveVolumeData<
        typename InterpolationTargetTag::temporal_id>>(
        interpolator, temporal_id, array_index, mesh, interpolator_source_vars);
  } else {
    auto& interpolator = *Parallel::local_branch(
        ::Parallel::get_parallel_component<Interpolator<Metavariables>>(cache));
    Parallel::simple_action<Actions::InterpolatorReceiveVolumeData<
        typename InterpolationTargetTag::temporal_id>>(
        interpolator, temporal_id, array_index, mesh, interpolator_source_vars);
  }

  // Tell the interpolation target that it should interpolate.
  auto& target = Parallel::get_parallel_component<
      InterpolationTarget<Metavariables, InterpolationTargetTag>>(cache);
  Parallel::simple_action<
      Actions::AddTemporalIdsToInterpolationTarget<InterpolationTargetTag>>(
      target, temporal_id);
}
}  // namespace intrp
