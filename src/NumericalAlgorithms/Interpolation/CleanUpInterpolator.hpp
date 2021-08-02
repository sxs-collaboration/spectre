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
template <typename Metavariables>
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
  static void apply(db::DataBox<DbTags>& box,  // HorizonManager's box
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const double time) noexcept {
    // Signal that this InterpolationTarget is done at this time.
    db::mutate<Tags::InterpolatedVarsHolders<Metavariables>>(
        make_not_null(&box),
        [&time](const gsl::not_null<
                typename Tags::InterpolatedVarsHolders<Metavariables>::type*>
                    holders) noexcept {
          get<Vars::HolderTag<InterpolationTargetTag, Metavariables>>(*holders)
              .times_when_data_has_been_interpolated.insert(time);
        });

    // If we don't need any of the volume data anymore for this
    // time, we will remove them.
    bool this_time_is_done = true;
    const auto& holders =
        db::get<Tags::InterpolatedVarsHolders<Metavariables>>(box);
    tmpl::for_each<typename Metavariables::interpolation_target_tags>(
        [&](auto tag) noexcept {
          using Tag = typename decltype(tag)::type;
          const auto& found = get<Vars::HolderTag<Tag, Metavariables>>(holders)
                                  .times_when_data_has_been_interpolated;
          if (found.count(time) == 0) {
            this_time_is_done = false;
          }
        });

    // We don't need any more volume data for this time,
    // so remove it.
    if (this_time_is_done) {
      db::mutate<Tags::VolumeVarsInfo<Metavariables>>(
          make_not_null(&box),
          [&time](const gsl::not_null<
                  typename Tags::VolumeVarsInfo<Metavariables>::type*>
                      volume_vars_info) noexcept {
            volume_vars_info->erase(time);
          });

      // Clean up times_when_data_has_been_interpolated
      db::mutate<Tags::InterpolatedVarsHolders<Metavariables>>(
          make_not_null(&box),
          [&time](const gsl::not_null<
                  typename Tags::InterpolatedVarsHolders<Metavariables>::type*>
                      holders_l) noexcept {
            tmpl::for_each<typename Metavariables::interpolation_target_tags>(
                [&](auto tag) noexcept {
                  using Tag = typename decltype(tag)::type;
                  get<Vars::HolderTag<Tag, Metavariables>>(*holders_l)
                      .times_when_data_has_been_interpolated.erase(time);
                });
          });
    }
  }
};
}  // namespace Actions
}  // namespace intrp
