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
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits/CreateHasTypeAlias.hpp"

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

namespace detail {

CREATE_HAS_TYPE_ALIAS(compute_vars_to_interpolate)
CREATE_HAS_TYPE_ALIAS_V(compute_vars_to_interpolate)

template <typename Tag, typename Frame>
using any_index_in_frame_impl =
    TensorMetafunctions::any_index_in_frame<typename Tag::type, Frame>;
}  // namespace detail

// Returns true if any of the tensors in TagList have any of their
// indices in the given frame.
template <typename TagList, typename Frame>
constexpr bool any_index_in_frame_v =
    tmpl::any<TagList, tmpl::bind<detail::any_index_in_frame_impl, tmpl::_1,
                                  Frame>>::value;

// Interpolates data onto a set of points desired by an InterpolationTarget.
template <typename InterpolationTargetTag, typename Metavariables,
          typename DbTags>
void interpolate_data(const gsl::not_null<db::DataBox<DbTags>*> box,
                      Parallel::GlobalCache<Metavariables>& cache,
                      const double time) noexcept {
  db::mutate_apply<
      tmpl::list<::intrp::Tags::InterpolatedVarsHolders<Metavariables>,
                 ::intrp::Tags::VolumeVarsInfo<Metavariables>>,
      tmpl::list<domain::Tags::Domain<Metavariables::volume_dim>>>(
      [&cache, &time](
          const gsl::not_null<typename ::intrp::Tags::InterpolatedVarsHolders<
              Metavariables>::type*>
              holders,
          const gsl::not_null<
              typename ::intrp::Tags::VolumeVarsInfo<Metavariables>::type*>
              volume_vars_info,
          const Domain<Metavariables::volume_dim>& domain) noexcept {
        auto& interp_info =
            get<Vars::HolderTag<InterpolationTargetTag, Metavariables>>(
                *holders)
                .infos.at(time);

        // Avoid compiler warning for unused variable in some 'if
        // constexpr' branches.
        (void)cache;

        for (auto& volume_info_outer : *volume_vars_info) {
          // Are we at the right time?
          if (volume_info_outer.first != time) {
            continue;
          }

          // Get list of ElementIds that have the correct time and that
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

            if constexpr (detail::has_compute_vars_to_interpolate_v<
                              InterpolationTargetTag>) {
              if (vars_to_interpolate.size() == 0) {
                // vars_to_interpolate has not been filled for
                // this element at this time.  So fill it. How we
                // fill it will depend on whether we need to change frames.
                vars_to_interpolate.initialize(
                    volume_info.vars_from_element.number_of_grid_points());

                if constexpr (any_index_in_frame_v<typename Metavariables::
                                                       interpolator_source_vars,
                                                   Frame::Inertial> and
                              any_index_in_frame_v<
                                  typename InterpolationTargetTag::
                                      vars_to_interpolate_to_target,
                                  Frame::Grid>) {
                  // Need to do frame transformations.

                  // The functions of time are always guaranteed to be
                  // up-to-date here, because they are guaranteed to be
                  // up-to-date before calling SendPointsToInterpolator
                  // (which is guaranteed to be called before
                  // interpolate_data is called).
                  const auto& functions_of_time =
                      get<domain::Tags::FunctionsOfTime>(cache);
                  const auto& block = domain.blocks().at(element_id.block_id());
                  ElementMap<3, ::Frame::Grid> map_logical_to_grid{
                      element_id,
                      block.moving_mesh_logical_to_grid_map().get_clone()};
                  const auto invjac_logical_to_grid =
                      map_logical_to_grid.inv_jacobian(
                          logical_coordinates(volume_info.mesh));
                  const auto jac_grid_to_inertial =
                      block.moving_mesh_grid_to_inertial_map().jacobian(
                          map_logical_to_grid(
                              logical_coordinates(volume_info.mesh)),
                          time, functions_of_time);
                  InterpolationTargetTag::compute_vars_to_interpolate::apply(
                      make_not_null(&vars_to_interpolate),
                      volume_info.vars_from_element, volume_info.mesh,
                      jac_grid_to_inertial, invjac_logical_to_grid);
                } else {
                  InterpolationTargetTag::compute_vars_to_interpolate::apply(
                      make_not_null(&vars_to_interpolate),
                      volume_info.vars_from_element, volume_info.mesh);
                }
              }
            }

            // Now interpolate.
            const auto& element_coord_holder = element_coord_pair.second;
            intrp::Irregular<Metavariables::volume_dim> interpolator(
                volume_info.mesh, element_coord_holder.element_logical_coords);
            if constexpr (detail::has_compute_vars_to_interpolate_v<
                              InterpolationTargetTag>) {
              interp_info.vars.emplace_back(
                  interpolator.interpolate(vars_to_interpolate));
            } else {
              // If compute_vars_to_interpolate does not exist, then
              // volume_info.vars_from_element is the same as
              // volume_info.vars_to_interpolate.
              interp_info.vars.emplace_back(
                  interpolator.interpolate(volume_info.vars_from_element));
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
    const double time) noexcept {
  const auto& holders =
      db::get<Tags::InterpolatedVarsHolders<Metavariables>>(*box);
  const auto& vars_infos =
      get<Vars::HolderTag<InterpolationTargetTag, Metavariables>>(holders)
          .infos;

  // If we don't yet have any points for this InterpolationTarget at
  // this time, we should exit (we can't interpolate anyway).
  if (vars_infos.count(time) == 0) {
    return;
  }

  interpolator_detail::interpolate_data<InterpolationTargetTag, Metavariables>(
      box, *cache, time);

  // Send interpolated data only if interpolation has been done on all
  // of the local elements.
  const auto& num_elements = db::get<Tags::NumberOfElements>(*box);
  if (vars_infos.at(time).interpolation_is_done_for_these_elements.size() ==
      num_elements) {
    // Send data to InterpolationTarget, but only if the list of points is
    // non-empty.
    if (not vars_infos.at(time).global_offsets.empty()) {
      const auto& info = vars_infos.at(time);
      auto& receiver_proxy = Parallel::get_parallel_component<
          InterpolationTarget<Metavariables, InterpolationTargetTag>>(*cache);
      Parallel::simple_action<
          Actions::InterpolationTargetReceiveVars<InterpolationTargetTag>>(
          receiver_proxy, info.vars, info.global_offsets, time);
    }

    // Clear interpolated data, since we don't need it anymore.
    db::mutate<Tags::InterpolatedVarsHolders<Metavariables>>(
        box,
        [&time](const gsl::not_null<
                typename Tags::InterpolatedVarsHolders<Metavariables>::type*>
                    holders_l) noexcept {
          get<Vars::HolderTag<InterpolationTargetTag, Metavariables>>(
              *holders_l)
              .infos.erase(time);
        });
  }
}

}  // namespace intrp
