// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/ElementLogicalCoordinates.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace intrp {
template <typename Metavariables, typename InterpolationTargetTag>
struct InterpolationTarget;
namespace Actions {
template <typename InterpolationTargetTag>
struct InterpolationTargetReceiveVars;
}  // namespace Actions
}  // namespace intrp
/// \endcond

namespace intrp {

namespace interpolator_detail {

// Interpolates data onto a set of points desired by an InterpolationTarget.
template <typename InterpolationTargetTag, typename Metavariables,
          typename DbTags>
void interpolate_data(
    const gsl::not_null<db::DataBox<DbTags>*> box,
    const typename Metavariables::temporal_id::type& temporal_id) noexcept {
  db::mutate_apply<tmpl::list<Tags::InterpolatedVarsHolders<Metavariables>>,
                   tmpl::list<Tags::VolumeVarsInfo<Metavariables>>>(
      [&temporal_id](
          const gsl::not_null<
              db::item_type<Tags::InterpolatedVarsHolders<Metavariables>>*>
              holders,
          const db::item_type<Tags::VolumeVarsInfo<Metavariables>>&
              volume_vars_info) noexcept {
        auto& interp_info =
            get<Vars::HolderTag<InterpolationTargetTag, Metavariables>>(
                *holders)
                .infos.at(temporal_id);

        for (const auto& volume_info_outer : volume_vars_info) {
          // Are we at the right time?
          if (volume_info_outer.first != temporal_id) {
            continue;
          }

          // Get list of ElementIds that have the correct temporal_id and that
          // have not yet been interpolated.
          std::vector<
              ElementId<Metavariables::domain_dim>>
              element_ids;

          for (const auto& volume_info_inner : volume_info_outer.second) {
            // Have we interpolated this element before?
            if (interp_info.interpolation_is_done_for_these_elements.find(
                    volume_info_inner.first) ==
                interp_info.interpolation_is_done_for_these_elements.end()) {
              interp_info.interpolation_is_done_for_these_elements.emplace(
                  volume_info_inner.first);
              element_ids.push_back(volume_info_inner.first);
            }
          }

          // Get element logical coordinates.
          const auto element_coord_holders = element_logical_coordinates(
              element_ids, interp_info.block_coord_holders);

          // Construct local vars and interpolate.
          for (const auto& element_coord_pair : element_coord_holders) {
            const auto& element_id = element_coord_pair.first;
            const auto& element_coord_holder = element_coord_pair.second;
            const auto& volume_info = volume_info_outer.second.at(element_id);

            // Construct local_vars which is some set of variables
            // derived from volume_info.vars plus an arbitrary set
            // of compute items in
            // InterpolationTargetTag::compute_items_on_source.

            auto new_box = db::create<
                db::AddSimpleTags<::Tags::Variables<
                    typename Metavariables::interpolator_source_vars>>,
                db::AddComputeTags<
                    typename InterpolationTargetTag::compute_items_on_source>>(
                volume_info.vars);

            Variables<
                typename InterpolationTargetTag::vars_to_interpolate_to_target>
                local_vars(volume_info.mesh.number_of_grid_points());

            tmpl::for_each<
                typename InterpolationTargetTag::vars_to_interpolate_to_target>(
                [&new_box, &local_vars](auto x) noexcept {
                  using tag = typename decltype(x)::type;
                  get<tag>(local_vars) = db::get<tag>(new_box);
                });

            // Now interpolate.
            intrp::Irregular<Metavariables::domain_dim>
                interpolator(volume_info.mesh,
                             element_coord_holder.element_logical_coords);
            interp_info.vars.emplace_back(interpolator.interpolate(local_vars));
            interp_info.global_offsets.emplace_back(
                element_coord_holder.offsets);
          }
        }
      },
      box);
}
}  // namespace interpolator_detail

/// Check if we have enough information to interpolate.  If so, do the
/// interpolation and send data to the InterpolationTarget.
template <typename InterpolationTargetTag, typename Metavariables,
          typename DbTags>
void try_to_interpolate(
    const gsl::not_null<db::DataBox<DbTags>*> box,
    const gsl::not_null<Parallel::ConstGlobalCache<Metavariables>*> cache,
    const typename Metavariables::temporal_id::type& temporal_id) noexcept {
  const auto& holders =
      db::get<Tags::InterpolatedVarsHolders<Metavariables>>(*box);
  const auto& vars_infos =
      get<Vars::HolderTag<InterpolationTargetTag, Metavariables>>(holders)
          .infos;

  // If we don't yet have any points for this InterpolationTarget at
  // this temporal_id, we should exit (we can't interpolate anyway).
  if (vars_infos.count(temporal_id) == 0) {
    return;
  }

  interpolator_detail::interpolate_data<InterpolationTargetTag, Metavariables>(
      box, temporal_id);

  // Send interpolated data only if interpolation has been done on all
  // of the local elements.
  const auto& num_elements = db::get<Tags::NumberOfElements>(*box);
  if (vars_infos.at(temporal_id)
          .interpolation_is_done_for_these_elements.size() == num_elements) {
    // Send data to InterpolationTarget, but only if the list of points is
    // non-empty.
    if (not vars_infos.at(temporal_id).global_offsets.empty()) {
      const auto& info = vars_infos.at(temporal_id);
      auto& receiver_proxy = Parallel::get_parallel_component<
          InterpolationTarget<Metavariables, InterpolationTargetTag>>(*cache);
      Parallel::simple_action<
          Actions::InterpolationTargetReceiveVars<InterpolationTargetTag>>(
          receiver_proxy, info.vars, info.global_offsets);
    }

    // Clear interpolated data, since we don't need it anymore.
    db::mutate<Tags::InterpolatedVarsHolders<Metavariables>>(
        box, [&temporal_id](const gsl::not_null<db::item_type<
                                Tags::InterpolatedVarsHolders<Metavariables>>*>
                                holders_l) noexcept {
          get<Vars::HolderTag<InterpolationTargetTag, Metavariables>>(
              *holders_l)
              .infos.erase(temporal_id);
        });
  }
}

}  // namespace intrp
