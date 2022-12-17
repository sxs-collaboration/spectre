// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/ElementLogicalCoordinates.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
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
    Parallel::GlobalCache<Metavariables>& cache,
    const typename InterpolationTargetTag::temporal_id::type& temporal_id) {
  db::mutate_apply<
      tmpl::list<
          ::intrp::Tags::InterpolatedVarsHolders<Metavariables>,
          ::intrp::Tags::VolumeVarsInfo<
              Metavariables, typename InterpolationTargetTag::temporal_id>>,
      tmpl::list<domain::Tags::Domain<Metavariables::volume_dim>>>(
      [&cache, &temporal_id](
          const gsl::not_null<typename ::intrp::Tags::InterpolatedVarsHolders<
              Metavariables>::type*>
              holders,
          const gsl::not_null<typename ::intrp::Tags::VolumeVarsInfo<
              Metavariables,
              typename InterpolationTargetTag::temporal_id>::type*>
              volume_vars_info,
          const Domain<Metavariables::volume_dim>& domain) {
        auto& interp_info =
            get<Vars::HolderTag<InterpolationTargetTag, Metavariables>>(
                *holders)
                .infos.at(temporal_id);

        // Avoid compiler warning for unused variable in some 'if
        // constexpr' branches.
        (void)cache;

        for (auto& volume_info_outer : *volume_vars_info) {
          // Are we at the right time?
          if (volume_info_outer.first != temporal_id) {
            continue;
          }

          // Get list of ElementIds that have the correct temporal_id and that
          // have not yet been interpolated.
          std::vector<ElementId<Metavariables::volume_dim>> element_ids;

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
            auto& volume_info = volume_info_outer.second.at(element_id);
            auto& vars_to_interpolate =
                get<::intrp::Tags::VarsToInterpolateToTarget<
                    InterpolationTargetTag>>(volume_info.vars_to_interpolate);

            if constexpr (InterpolationTarget_detail::
                              has_compute_vars_to_interpolate_v<
                                  InterpolationTargetTag>) {
              if (vars_to_interpolate.size() == 0) {
                // vars_to_interpolate has not been filled for
                // this element at this temporal_id.  So fill it.
                vars_to_interpolate.initialize(
                    volume_info.source_vars_from_element
                        .number_of_grid_points());

                InterpolationTarget_detail::compute_dest_vars_from_source_vars<
                    InterpolationTargetTag>(
                    make_not_null(&vars_to_interpolate),
                    volume_info.source_vars_from_element, domain,
                    volume_info.mesh, element_id, cache, temporal_id);
              }
            }

            // Now interpolate.
            const auto& element_coord_holder = element_coord_pair.second;
            intrp::Irregular<Metavariables::volume_dim> interpolator(
                volume_info.mesh, element_coord_holder.element_logical_coords);
            // This first branch is used if compute_vars_to_interpolate exists
            // or if the vars_to_interpolate_to_target is a subset of the
            // interpolator_source_vars.
            if constexpr (
                InterpolationTarget_detail::has_compute_vars_to_interpolate_v<
                    InterpolationTargetTag> or
                not std::is_same_v<
                    tmpl::list_difference<
                        typename Metavariables::interpolator_source_vars,
                        typename InterpolationTargetTag::
                            vars_to_interpolate_to_target>,
                    tmpl::list<>>) {
              interp_info.vars.emplace_back(
                  interpolator.interpolate(vars_to_interpolate));
            } else {
              // If compute_vars_to_interpolate does not exist and
              // vars_to_interpolate_to_target isn't a subset of
              // interpolator_source_vars, then
              // volume_info.source_vars_from_element is the same as
              // volume_info.vars_to_interpolate.
              interp_info.vars.emplace_back(interpolator.interpolate(
                  volume_info.source_vars_from_element));
            }
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
    const gsl::not_null<Parallel::GlobalCache<Metavariables>*> cache,
    const typename InterpolationTargetTag::temporal_id::type& temporal_id) {
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
      box, *cache, temporal_id);

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
          receiver_proxy, info.vars, info.global_offsets, temporal_id);
    }

    // Clear interpolated data, since we don't need it anymore.
    db::mutate<Tags::InterpolatedVarsHolders<Metavariables>>(
        box,
        [&temporal_id](
            const gsl::not_null<
                typename Tags::InterpolatedVarsHolders<Metavariables>::type*>
                holders_l) {
          get<Vars::HolderTag<InterpolationTargetTag, Metavariables>>(
              *holders_l)
              .infos.erase(temporal_id);
        });
  }
}

}  // namespace intrp
