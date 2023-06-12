// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <deque>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/TryToInterpolate.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace intrp {
namespace Actions {
namespace detail {
template <typename Metavariables, typename Tag, typename = std::void_t<>>
struct get_interpolating_component_or_interpolator {
  using type = Interpolator<Metavariables>;
};
template <typename Metavariables, typename Tag>
struct get_interpolating_component_or_interpolator<
    Metavariables, Tag,
    std::void_t<
        typename Tag::template interpolating_component<Metavariables>>> {
  using type = typename Tag::template interpolating_component<Metavariables>;
};
template <typename Metavariables, typename Tag>
using get_interpolating_component_or_interpolator_t =
    typename get_interpolating_component_or_interpolator<Metavariables,
                                                         Tag>::type;

template <typename Metavariables, typename Tag>
constexpr bool using_interpolator_component_v = std::is_same_v<
    get_interpolating_component_or_interpolator_t<Metavariables, Tag>,
    Interpolator<Metavariables>>;
}  // namespace detail

/// \ingroup ActionsGroup
/// \brief Adds volume data from an `Element`.
///
/// Attempts to interpolate if it already has received target points from
/// any InterpolationTargets.
///
/// Uses:
/// - DataBox:
///   - `Tags::NumberOfElements`
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - `Tags::VolumeVarsInfo<Metavariables>`
///   - `Tags::InterpolatedVarsHolders<Metavariables>`
template <typename TemporalId>
struct InterpolatorReceiveVolumeData {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex, size_t VolumeDim>
  static void apply(
      db::DataBox<DbTags>& box, Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/,
      const typename TemporalId::type& temporal_id,
      const ElementId<VolumeDim>& element_id, const ::Mesh<VolumeDim>& mesh,
      Variables<typename Metavariables::interpolator_source_vars>&&
          interpolator_source_vars) {
    // Determine if we have already finished interpolating on this
    // temporal_id.  If so, then we simply return, ignore the incoming
    // data, and do not interpolate.
    //
    // This scenario can happen if there is an element that is not
    // used or needed for any InterpolationTarget, and if that element
    // calls InterpolatorReceiveVolumeData so late that all the
    // InterpolationTargets for the current temporal_id have already
    // finished.
    bool this_temporal_id_is_done = true;
    const auto& holders =
        db::get<Tags::InterpolatedVarsHolders<Metavariables>>(box);
    tmpl::for_each<typename Metavariables::interpolation_target_tags>(
        [&holders, &this_temporal_id_is_done, &temporal_id](auto tag_v) {
          using tag = typename decltype(tag_v)::type;
          // Here we decide whether this interpolation target is "done" (i.e. it
          // does not need to interpolate) at this temporal_id. If it is "done",
          // then we don't need to store the volume data. Usually "done" means
          // that it has already done its interpolation at this temporal_id. But
          // note that if an interpolation target is not using the Interpolator
          // at all, it is considered "done", because we don't need any volume
          // data for it and should not store it. Similarly, interpolation
          // targets whose temporal_id has a different type than TemporalId are
          // considered "done" because they too do not use the Interpolator and
          // don't need volume data to be saved.
          if constexpr (detail::using_interpolator_component_v<Metavariables,
                                                               tag> and
                        std::is_same_v<TemporalId, typename tag::temporal_id>) {
            const auto& finished_temporal_ids =
                get<Vars::HolderTag<tag, Metavariables>>(holders)
                    .temporal_ids_when_data_has_been_interpolated;
            if (not alg::found(finished_temporal_ids, temporal_id)) {
              this_temporal_id_is_done = false;
            }
          }
        });

    if (this_temporal_id_is_done) {
      return;
    }

    // Add to the VolumeVarsInfo for this TemporalId type.  (Note that
    // multiple VolumeVarsInfos, each with a different TemporalId
    // type, can be in the databox.  Note also that the above check
    // for this_temporal_id_is_done and the interpolation below are
    // done only for this TemporalId type and not for any other
    // VolumeVarsInfos that might be in the DataBox.)
    db::mutate<Tags::VolumeVarsInfo<Metavariables, TemporalId>>(
        [&temporal_id, &element_id, &mesh, &interpolator_source_vars](
            const gsl::not_null<
                typename Tags::VolumeVarsInfo<Metavariables, TemporalId>::type*>
                container) {
          if (container->find(temporal_id) == container->end()) {
            container->emplace(
                temporal_id,
                std::unordered_map<ElementId<VolumeDim>,
                                   typename Tags::VolumeVarsInfo<
                                       Metavariables, TemporalId>::Info>{});
          }
          container->at(temporal_id)
              .emplace(std::make_pair(
                  element_id,
                  typename Tags::VolumeVarsInfo<Metavariables,
                                                TemporalId>::Info{
                      mesh, std::move(interpolator_source_vars), {}}));
        },
        make_not_null(&box));

    // Try to interpolate data for all InterpolationTargets for this
    // temporal_id.
    tmpl::for_each<typename Metavariables::interpolation_target_tags>(
        [&box, &cache, &temporal_id](auto tag_v) {
          using tag = typename decltype(tag_v)::type;
          if constexpr (std::is_same_v<typename tag::temporal_id, TemporalId>) {
            try_to_interpolate<tag>(make_not_null(&box), make_not_null(&cache),
                                    temporal_id);
          }
        });
  }
};

}  // namespace Actions
}  // namespace intrp
