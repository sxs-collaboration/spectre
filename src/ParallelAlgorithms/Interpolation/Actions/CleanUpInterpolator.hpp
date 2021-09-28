// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
// IWYU pragma: no_forward_declare db::DataBox
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace intrp {
namespace Tags {
struct NumberOfElements;
template <typename Metavariables>
struct InterpolatedVarsHolders;
template <typename Metavariables, typename TemporalId>
struct VolumeVarsInfo;
}  // namespace Tags
namespace Vars {
template <typename InterpolationTargetTag, typename Metavariables>
struct HolderTag;
}  // namespace Vars
}  // namespace intrp
/// \endcond

namespace intrp {
namespace Actions {

/// \ingroup ActionsGroup
/// \brief Cleans up stored volume data that is no longer needed.
///
/// Called by InterpolationTargetReceiveVars.
///
/// Uses:
/// - Databox:
///   - `Tags::InterpolatedVarsHolders<Metavariables>`
///   - `Tags::VolumeVarsInfo<Metavariables>`
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - `Tags::InterpolatedVarsHolders<Metavariables>`
///   - `Tags::VolumeVarsInfo<Metavariables>`
///
/// For requirements on InterpolationTargetTag, see InterpolationTarget
template <typename InterpolationTargetTag>
struct CleanUpInterpolator {
  template <
      typename ParallelComponent, typename DbTags, typename Metavariables,
      typename ArrayIndex,
      Requires<tmpl::list_contains_v<DbTags, Tags::NumberOfElements>> = nullptr>
  static void apply(
      db::DataBox<DbTags>& box,  // HorizonManager's box
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const typename InterpolationTargetTag::temporal_id::type& temporal_id) {
    // Signal that this InterpolationTarget is done at this time.
    db::mutate<Tags::InterpolatedVarsHolders<Metavariables>>(
        make_not_null(&box),
        [&temporal_id](
            const gsl::not_null<
                typename Tags::InterpolatedVarsHolders<Metavariables>::type*>
                holders) {
          get<Vars::HolderTag<InterpolationTargetTag, Metavariables>>(*holders)
              .temporal_ids_when_data_has_been_interpolated.insert(temporal_id);
        });

    // If we don't need any of the volume data anymore for this
    // temporal_id, we will remove them.
    bool this_temporal_id_is_done = true;
    const auto& holders =
        db::get<Tags::InterpolatedVarsHolders<Metavariables>>(box);
    tmpl::for_each<typename Metavariables::interpolation_target_tags>(
        [&](auto tag) {
          using Tag = typename decltype(tag)::type;
          if constexpr (std::is_same_v<
                            typename InterpolationTargetTag::temporal_id,
                            typename Tag::temporal_id>) {
            const auto& found =
                get<Vars::HolderTag<Tag, Metavariables>>(holders)
                    .temporal_ids_when_data_has_been_interpolated;
            if (found.count(temporal_id) == 0) {
              this_temporal_id_is_done = false;
            }
          }
        });

    // We don't need any more volume data for this temporal_id,
    // so remove it.
    if (this_temporal_id_is_done) {
      db::mutate<Tags::VolumeVarsInfo<
          Metavariables, typename InterpolationTargetTag::temporal_id>>(
          make_not_null(&box),
          [&temporal_id](
              const gsl::not_null<typename Tags::VolumeVarsInfo<
                  Metavariables,
                  typename InterpolationTargetTag::temporal_id>::type*>
                  volume_vars_info) { volume_vars_info->erase(temporal_id); });

      // Clean up temporal_ids_when_data_has_been_interpolated
      db::mutate<Tags::InterpolatedVarsHolders<Metavariables>>(
          make_not_null(&box),
          [&temporal_id](
              const gsl::not_null<
                  typename Tags::InterpolatedVarsHolders<Metavariables>::type*>
                  holders_l) {
            tmpl::for_each<typename Metavariables::interpolation_target_tags>(
                [&](auto tag) {
                  using Tag = typename decltype(tag)::type;
                  if constexpr (std::is_same_v<typename InterpolationTargetTag::
                                                   temporal_id,
                                               typename Tag::temporal_id>) {
                    get<Vars::HolderTag<Tag, Metavariables>>(*holders_l)
                        .temporal_ids_when_data_has_been_interpolated.erase(
                            temporal_id);
                  }
                });
          });
    }
  }
};
}  // namespace Actions
}  // namespace intrp
