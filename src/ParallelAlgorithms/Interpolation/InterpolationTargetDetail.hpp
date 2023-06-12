// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/IdPair.hpp"
#include "DataStructures/Tensor/Metafunctions.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/CoordinateMaps/Composition.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/ElementToBlockLogicalMap.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "ParallelAlgorithms/Interpolation/TagsMetafunctions.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"
#include "Utilities/TypeTraits/CreateHasStaticMemberVariable.hpp"
#include "Utilities/TypeTraits/CreateHasTypeAlias.hpp"

/// \cond
// IWYU pragma: no_forward_declare db::DataBox
namespace intrp {
template <class Metavariables>
struct Interpolator;
template <typename Metavariables, typename InterpolationTargetTag>
class InterpolationTarget;
namespace Actions {
template <typename InterpolationTargetTag>
struct CleanUpInterpolator;
template <typename InterpolationTargetTag>
struct ReceivePoints;
}  // namespace Actions
namespace Tags {
template <typename TemporalId>
struct IndicesOfFilledInterpPoints;
template <typename TemporalId>
struct IndicesOfInvalidInterpPoints;
template <typename InterpolationTargetTag, typename TemporalId>
struct InterpolatedVars;
template <typename TemporalId>
struct CompletedTemporalIds;
template <typename TemporalId>
struct PendingTemporalIds;
template <typename TemporalId>
struct TemporalIds;
}  // namespace Tags
}  // namespace intrp
template <typename Id>
struct LinkedMessageId;
template <typename TagsList>
struct Variables;
/// \endcond

namespace intrp {

namespace InterpolationTarget_detail {
double get_temporal_id_value(double time);
double get_temporal_id_value(const LinkedMessageId<double>& id);
double get_temporal_id_value(const TimeStepId& time_id);

// apply_callback accomplishes the overload for the
// two signatures of callback functions.
// Uses SFINAE on return type.
template <typename T, typename DbTags, typename Metavariables,
          typename TemporalId>
auto apply_callback(
    const gsl::not_null<db::DataBox<DbTags>*> box,
    const gsl::not_null<Parallel::GlobalCache<Metavariables>*> cache,
    const TemporalId& temporal_id)
    -> decltype(T::post_interpolation_callback::apply(box, cache, temporal_id),
                bool()) {
  return T::post_interpolation_callback::apply(box, cache, temporal_id);
}

template <typename T, typename DbTags, typename Metavariables,
          typename TemporalId>
auto apply_callback(
    const gsl::not_null<db::DataBox<DbTags>*> box,
    const gsl::not_null<Parallel::GlobalCache<Metavariables>*> cache,
    const TemporalId& temporal_id)
    -> decltype(T::post_interpolation_callback::apply(*box, *cache,
                                                      temporal_id),
                bool()) {
  T::post_interpolation_callback::apply(*box, *cache, temporal_id);
  // For the simpler callback function, we will always clean up volume data, so
  // we return true here.
  return true;
}

CREATE_HAS_STATIC_MEMBER_VARIABLE(fill_invalid_points_with)
CREATE_HAS_STATIC_MEMBER_VARIABLE_V(fill_invalid_points_with)

// Fills invalid points with some constant value.
template <typename InterpolationTargetTag, typename TemporalId, typename DbTags,
          Requires<not has_fill_invalid_points_with_v<
              typename InterpolationTargetTag::post_interpolation_callback>> =
              nullptr>
void fill_invalid_points(const gsl::not_null<db::DataBox<DbTags>*> /*box*/,
                         const TemporalId& /*temporal_id*/) {}

template <typename InterpolationTargetTag, typename TemporalId, typename DbTags,
          Requires<has_fill_invalid_points_with_v<
              typename InterpolationTargetTag::post_interpolation_callback>> =
              nullptr>
void fill_invalid_points(const gsl::not_null<db::DataBox<DbTags>*> box,
                         const TemporalId& temporal_id) {
  const auto& invalid_indices =
      db::get<Tags::IndicesOfInvalidInterpPoints<TemporalId>>(*box);
  if (invalid_indices.find(temporal_id) != invalid_indices.end() and
      not invalid_indices.at(temporal_id).empty()) {
    db::mutate<Tags::IndicesOfInvalidInterpPoints<TemporalId>,
               Tags::InterpolatedVars<InterpolationTargetTag, TemporalId>>(
        [&temporal_id](
            const gsl::not_null<
                std::unordered_map<TemporalId, std::unordered_set<size_t>>*>
                indices_of_invalid_points,
            const gsl::not_null<std::unordered_map<
                TemporalId, Variables<typename InterpolationTargetTag::
                                          vars_to_interpolate_to_target>>*>
                vars_dest_all_times) {
          auto& vars_dest = vars_dest_all_times->at(temporal_id);
          const size_t npts_dest = vars_dest.number_of_grid_points();
          const size_t nvars = vars_dest.number_of_independent_components;
          for (auto index : indices_of_invalid_points->at(temporal_id)) {
            for (size_t v = 0; v < nvars; ++v) {
              // clang-tidy: no pointer arithmetic
              vars_dest.data()[index + v * npts_dest] =  // NOLINT
                  InterpolationTargetTag::post_interpolation_callback::
                      fill_invalid_points_with;
            }
          }
          // Further functions may test if there are invalid points.
          // Clear the invalid points now, since we have filled them.
          indices_of_invalid_points->erase(temporal_id);
        },
        box);
  }
}

/// Wraps calling the callback function on an InterpolationTarget.
///
/// First prepares for the callback, then calls the callback,
/// and returns true if the callback is done and
/// false if the callback is not done (e.g. if the callback is an
/// apparent horizon search and it needs another iteration).
///
/// call_callback is called by an Action of InterpolationTarget.
///
/// Currently two Actions call call_callback:
/// - InterpolationTargetReceiveVars (called by Interpolator ParallelComponent)
/// - InterpolationTargetVarsFromElement (called by DgElementArray)
template <typename InterpolationTargetTag, typename DbTags,
          typename Metavariables, typename TemporalId>
bool call_callback(
    const gsl::not_null<db::DataBox<DbTags>*> box,
    const gsl::not_null<Parallel::GlobalCache<Metavariables>*> cache,
    const TemporalId& temporal_id) {
  // Before doing anything else, deal with the possibility that some
  // of the points might be outside of the Domain.
  fill_invalid_points<InterpolationTargetTag>(box, temporal_id);

  // Fill ::Tags::Variables<typename
  //      InterpolationTargetTag::vars_to_interpolate_to_target>
  // with variables from correct temporal_id.
  db::mutate_apply<
      tmpl::list<::Tags::Variables<
          typename InterpolationTargetTag::vars_to_interpolate_to_target>>,
      tmpl::list<Tags::InterpolatedVars<InterpolationTargetTag, TemporalId>>>(
      [&temporal_id](
          const gsl::not_null<Variables<
              typename InterpolationTargetTag::vars_to_interpolate_to_target>*>
              vars,
          const std::unordered_map<
              TemporalId, Variables<typename InterpolationTargetTag::
                                        vars_to_interpolate_to_target>>&
              vars_at_all_times) { *vars = vars_at_all_times.at(temporal_id); },
      box);

  // apply_callback should return true if we are done with this
  // temporal_id.  It should return false only if the callback
  // calls another `intrp::Action` that needs the volume data at this
  // same temporal_id.
  return apply_callback<InterpolationTargetTag>(box, cache, temporal_id);
}

/// Frees InterpolationTarget's memory associated with the supplied
/// temporal_id.
///
/// clean_up_interpolation_target is called by an Action of InterpolationTarget.
///
/// Currently two Actions call clean_up_interpolation_target:
/// - InterpolationTargetReceiveVars (called by Interpolator ParallelComponent)
/// - InterpolationTargetVarsFromElement (called by DgElementArray)
template <typename InterpolationTargetTag, typename DbTags, typename TemporalId>
void clean_up_interpolation_target(
    const gsl::not_null<db::DataBox<DbTags>*> box,
    const TemporalId& temporal_id) {
  // We are now done with this temporal_id, so we can pop it and
  // clean up data associated with it.
  db::mutate<Tags::TemporalIds<TemporalId>,
             Tags::CompletedTemporalIds<TemporalId>,
             Tags::IndicesOfFilledInterpPoints<TemporalId>,
             Tags::IndicesOfInvalidInterpPoints<TemporalId>,
             Tags::InterpolatedVars<InterpolationTargetTag, TemporalId>>(
      [&temporal_id](
          const gsl::not_null<std::deque<TemporalId>*> ids,
          const gsl::not_null<std::deque<TemporalId>*> completed_ids,
          const gsl::not_null<
              std::unordered_map<TemporalId, std::unordered_set<size_t>>*>
              indices_of_filled,
          const gsl::not_null<
              std::unordered_map<TemporalId, std::unordered_set<size_t>>*>
              indices_of_invalid,
          const gsl::not_null<std::unordered_map<
              TemporalId, Variables<typename InterpolationTargetTag::
                                        vars_to_interpolate_to_target>>*>
              interpolated_vars) {
        completed_ids->push_back(temporal_id);
        ASSERT(std::find(ids->begin(), ids->end(), temporal_id) != ids->end(),
               "Temporal id " << temporal_id << " does not exist in ids");
        ids->erase(std::remove(ids->begin(), ids->end(), temporal_id),
                   ids->end());
        // We want to keep track of all completed temporal_ids to deal with
        // the possibility of late calls to
        // AddTemporalIdsToInterpolationTarget.  We could keep all
        // completed_ids forever, but we probably don't want it to get too
        // large, so we limit its size.  We assume that
        // asynchronous calls to AddTemporalIdsToInterpolationTarget do not span
        // more than 1000 temporal_ids.
        if (completed_ids->size() > 1000) {
          completed_ids->pop_front();
        }
        indices_of_filled->erase(temporal_id);
        indices_of_invalid->erase(temporal_id);
        interpolated_vars->erase(temporal_id);
      },
      box);
}

/// Returns true if this InterpolationTarget has received data
/// at all its points.
///
/// have_data_at_all_points is called by an Action of InterpolationTarget.
///
/// Currently two Actions call have_data_at_all_points:
/// - InterpolationTargetReceiveVars (called by Interpolator ParallelComponent)
/// - InterpolationTargetVarsFromElement (called by DgElementArray)
template <typename InterpolationTargetTag, typename DbTags, typename TemporalId>
bool have_data_at_all_points(const db::DataBox<DbTags>& box,
                             const TemporalId& temporal_id) {
  const size_t filled_size =
      db::get<Tags::IndicesOfFilledInterpPoints<TemporalId>>(box)
          .at(temporal_id)
          .size();
  const size_t invalid_size = [&box, &temporal_id]() {
    const auto& invalid_indices =
        db::get<Tags::IndicesOfInvalidInterpPoints<TemporalId>>(box);
    if (invalid_indices.count(temporal_id) > 0) {
      return invalid_indices.at(temporal_id).size();
    }
    return 0_st;
  }();
  const size_t interp_size =
      db::get<Tags::InterpolatedVars<InterpolationTargetTag, TemporalId>>(box)
          .at(temporal_id)
          .number_of_grid_points();
  return (invalid_size + filled_size == interp_size);
}

/// Tells an InterpolationTarget that it should interpolate at
/// the supplied temporal_ids.  Changes the InterpolationTarget's DataBox
/// accordingly.
///
/// Returns the temporal_ids that have actually been newly flagged
/// (since some of them may have been flagged already).
///
/// flag_temporal_ids_for_interpolation is called by an Action
/// of InterpolationTarget
///
/// Currently one Action calls flag_temporal_ids_for_interpolation:
/// - InterpolationTargetVarsFromElement (called by DgElementArray)
template <typename InterpolationTargetTag, typename DbTags, typename TemporalId>
std::vector<TemporalId> flag_temporal_ids_for_interpolation(
    const gsl::not_null<db::DataBox<DbTags>*> box,
    const std::vector<TemporalId>& temporal_ids) {
  // We allow this function to be called multiple times with the same
  // temporal_ids (e.g. from each element, or from each node of a
  // NodeGroup ParallelComponent such as Interpolator). If multiple
  // calls occur, we care only about the first one, and ignore the
  // others.  The first call will often begin interpolation.  So if
  // multiple calls occur, it is possible that some of them may arrive
  // late, even after interpolation has been completed on one or more
  // of the temporal_ids (and after that id has already been removed
  // from `ids`).  If this happens, we don't want to add the
  // temporal_ids again. For that reason we keep track of the
  // temporal_ids that we have already completed interpolation on.  So
  // here we do not add any temporal_ids that are already present in
  // `ids` or `completed_ids`.
  std::vector<TemporalId> new_temporal_ids{};

  db::mutate_apply<tmpl::list<Tags::TemporalIds<TemporalId>>,
                   tmpl::list<Tags::CompletedTemporalIds<TemporalId>>>(
      [&temporal_ids, &new_temporal_ids](
          const gsl::not_null<std::deque<TemporalId>*> ids,
          const std::deque<TemporalId>& completed_ids) {
        for (auto& id : temporal_ids) {
          if (std::find(completed_ids.begin(), completed_ids.end(), id) ==
                  completed_ids.end() and
              std::find(ids->begin(), ids->end(), id) == ids->end()) {
            ids->push_back(id);
            new_temporal_ids.push_back(id);
          }
        }
      },
      box);

  return new_temporal_ids;
}

/// Tells an InterpolationTarget that it should interpolate at
/// the supplied temporal_ids.  Changes the InterpolationTarget's DataBox
/// accordingly.
///
/// Returns the temporal_ids that have actually been newly flagged
/// (since some of them may have been flagged already).
///
/// flag_temporal_ids_as_pending is called by an Action
/// of InterpolationTarget
///
/// Currently one Action calls flag_temporal_ids_as_pending:
/// - AddTemporalIdsToInterpolationTarget (called by Events::Interpolate)
template <typename InterpolationTargetTag, typename DbTags, typename TemporalId>
std::vector<TemporalId> flag_temporal_ids_as_pending(
    const gsl::not_null<db::DataBox<DbTags>*> box,
    const std::vector<TemporalId>& temporal_ids) {
  // We allow this function to be called multiple times with the same
  // temporal_ids (e.g. from each element, or from each node of a
  // NodeGroup ParallelComponent such as Interpolator). If multiple
  // calls occur, we care only about the first one, and ignore the
  // others.  The first call will often begin interpolation.  So if
  // multiple calls occur, it is possible that some of them may arrive
  // late, even after interpolation has been completed on one or more
  // of the temporal_ids (and after that id has already been removed
  // from `ids`).  If this happens, we don't want to add the
  // temporal_ids again. For that reason we keep track of the
  // temporal_ids that we have already completed interpolation on.  So
  // here we do not add any temporal_ids that are already present in
  // `ids` or `completed_ids`.
  std::vector<TemporalId> new_temporal_ids{};

  db::mutate_apply<tmpl::list<Tags::PendingTemporalIds<TemporalId>>,
                   tmpl::list<Tags::TemporalIds<TemporalId>,
                              Tags::CompletedTemporalIds<TemporalId>>>(
      [&temporal_ids, &new_temporal_ids](
          const gsl::not_null<std::deque<TemporalId>*> pending_ids,
          const std::deque<TemporalId>& ids,
          const std::deque<TemporalId>& completed_ids) {
        for (auto& id : temporal_ids) {
          if (std::find(completed_ids.begin(), completed_ids.end(), id) ==
                  completed_ids.end() and
              std::find(ids.begin(), ids.end(), id) == ids.end() and
              std::find(pending_ids->begin(), pending_ids->end(), id) ==
                  pending_ids->end()) {
            pending_ids->push_back(id);
            new_temporal_ids.push_back(id);
          }
        }
      },
      box);

  return new_temporal_ids;
}

/// Adds the supplied interpolated variables and offsets to the
/// InterpolationTarget's internal DataBox.
///
/// Note that the template argument to Variables in vars_src is called
/// InterpolationTargetTag::vars_to_interpolate_to_target.  This is a list
/// of tags, and is used for both the interpolated variables (in
/// this function add_received_variables) and for the source variables
/// (in other functions). The source and interpolated quantities are
/// the same set of variables (but at different points).
///
/// add_received_variables is called by an Action of InterpolationTarget.
///
/// Currently two Actions call add_received_variables:
/// - InterpolationTargetReceiveVars (called by Interpolator ParallelComponent)
/// - InterpolationTargetVarsFromElement (called by DgElementArray)
template <typename InterpolationTargetTag, typename DbTags, typename TemporalId>
void add_received_variables(
    const gsl::not_null<db::DataBox<DbTags>*> box,
    const std::vector<Variables<
        typename InterpolationTargetTag::vars_to_interpolate_to_target>>&
        vars_src,
    const std::vector<std::vector<size_t>>& global_offsets,
    const TemporalId& temporal_id) {
  db::mutate<Tags::IndicesOfFilledInterpPoints<TemporalId>,
             Tags::InterpolatedVars<InterpolationTargetTag, TemporalId>>(
      [&temporal_id, &vars_src, &global_offsets](
          const gsl::not_null<
              std::unordered_map<TemporalId, std::unordered_set<size_t>>*>
              indices_of_filled,
          const gsl::not_null<std::unordered_map<
              TemporalId, Variables<typename InterpolationTargetTag::
                                        vars_to_interpolate_to_target>>*>
              vars_dest_all_times) {
        auto& vars_dest = vars_dest_all_times->at(temporal_id);
        // Here we assume that vars_dest has been allocated to the correct
        // size (but could contain garbage, since below we are filling it).
        const size_t npts_dest = vars_dest.number_of_grid_points();
        const size_t nvars = vars_dest.number_of_independent_components;
        for (size_t j = 0; j < global_offsets.size(); ++j) {
          const size_t npts_src = global_offsets[j].size();
          for (size_t i = 0; i < npts_src; ++i) {
            // If a point is on the boundary of two (or more)
            // elements, it is possible that we have received data
            // for this point from more than one Interpolator.
            // This will rarely occur, but it does occur, e.g. when
            // a point is exactly on some symmetry
            // boundary (such as the x-y plane) and this symmetry
            // boundary is exactly the boundary between two
            // elements.  If this happens, we accept the first
            // duplicated point, and we ignore subsequent
            // duplicated points.  The points are easy to keep track
            // of because global_offsets uniquely identifies them.
            if ((*indices_of_filled)[temporal_id]
                    .insert(global_offsets[j][i])
                    .second) {
              for (size_t v = 0; v < nvars; ++v) {
                // clang-tidy: no pointer arithmetic
                vars_dest.data()[global_offsets[j][i] +    // NOLINT
                                 v * npts_dest] =          // NOLINT
                    vars_src[j].data()[i + v * npts_src];  // NOLINT
              }
            }
          }
        }
      },
      box);
}

/// Computes the block logical coordinates of an InterpolationTarget.
///
/// block_logical_coords is called by an Action of InterpolationTarget.
///
/// Currently one Action directly calls this version of block_logical_coords:
/// - InterpolationTargetSendTimeIndepPointsToElements
///   (in InterpolationTarget ActionList)
/// and one Action indirectly calls this version of block_logical_coords:
/// - SendPointsToInterpolator (called by AddTemporalIdsToInterpolationTarget
///                             and by FindApparentHorizon)
template <typename InterpolationTargetTag, typename Metavariables,
          typename TemporalId>
auto block_logical_coords(
    const Parallel::GlobalCache<Metavariables>& cache,
    const tnsr::I<
        DataVector, Metavariables::volume_dim,
        typename InterpolationTargetTag::compute_target_points::frame>& coords,
    const TemporalId& temporal_id) {
  const auto& domain =
      get<domain::Tags::Domain<Metavariables::volume_dim>>(cache);
  if constexpr (std::is_same_v<typename InterpolationTargetTag::
                                   compute_target_points::frame,
                               ::Frame::Grid>) {
    // Frame is grid frame, so don't need any FunctionsOfTime,
    // whether or not the maps are time_dependent.
    return ::block_logical_coordinates(domain, coords);
  }

  if (domain.is_time_dependent()) {
    if constexpr (Parallel::is_in_mutable_global_cache<
                      Metavariables, domain::Tags::FunctionsOfTime>) {
      // Whoever calls block_logical_coords when the maps are
      // time-dependent is responsible for ensuring
      // that functions_of_time are up to date at temporal_id.
      const auto& functions_of_time = get<domain::Tags::FunctionsOfTime>(cache);
      return ::block_logical_coordinates(
          domain, coords,
          InterpolationTarget_detail::get_temporal_id_value(temporal_id),
          functions_of_time);
    } else {
      // We error here because the maps are time-dependent, yet
      // the cache does not contain FunctionsOfTime.  It would be
      // nice to make this a compile-time error; however, we want
      // the code to compile for the completely time-independent
      // case where there are no FunctionsOfTime in the cache at
      // all.  Unfortunately, checking whether the maps are
      // time-dependent is currently not constexpr.
      ERROR(
          "There is a time-dependent CoordinateMap in at least one "
          "of the Blocks, but FunctionsOfTime are not in the "
          "GlobalCache.  If you intend to use a time-dependent "
          "CoordinateMap, please add FunctionsOfTime to the GlobalCache.");
    }
  }

  // Time-independent case.
  return ::block_logical_coordinates(domain, coords);
}

/// Version of block_logical_coords that computes the interpolation
/// points on the fly.  This version is called when the interpolation
/// points are (or might be) time-dependent in the
/// InterpolationTargetTag's frame.
///
/// This version of block_logical_coordinates is called when there
/// is an Interpolator ParallelComponent.
///
/// Currently one Action directly calls this version of block_logical_coords:
/// - SendPointsToInterpolator (called by AddTemporalIdsToInterpolationTarget
///                             and by FindApparentHorizon)
template <typename InterpolationTargetTag, typename DbTags,
          typename Metavariables, typename TemporalId>
auto block_logical_coords(const db::DataBox<DbTags>& box,
                          const Parallel::GlobalCache<Metavariables>& cache,
                          const TemporalId& temporal_id) {
  return block_logical_coords<InterpolationTargetTag>(
      cache,
      InterpolationTargetTag::compute_target_points::points(
          box, tmpl::type_<Metavariables>{}, temporal_id),
      temporal_id);
}

/// Version of block_logical_coords for when the coords are
/// time-independent.
template <typename InterpolationTargetTag, typename DbTags,
          typename Metavariables>
auto block_logical_coords(const db::DataBox<DbTags>& box,
                          const tmpl::type_<Metavariables>& meta) {
  const auto& domain =
      db::get<domain::Tags::Domain<Metavariables::volume_dim>>(box);
  return ::block_logical_coordinates(
      domain, InterpolationTargetTag::compute_target_points::points(box, meta));
}

/// Initializes InterpolationTarget's variables storage and lists of indices
/// corresponding to the supplied block logical coordinates and `temporal_id`.
///
/// set_up_interpolation is called by an Action of InterpolationTarget.
///
/// Currently two Actions call set_up_interpolation:
/// - SendPointsToInterpolator (called by AddTemporalIdsToInterpolationTarget
///                             and by FindApparentHorizon)
/// - InterpolationTargetVarsFromElement (called by DgElementArray)
template <typename InterpolationTargetTag, typename DbTags, size_t VolumeDim,
          typename TemporalId>
void set_up_interpolation(
    const gsl::not_null<db::DataBox<DbTags>*> box,
    const TemporalId& temporal_id,
    const std::vector<std::optional<
        IdPair<domain::BlockId,
               tnsr::I<double, VolumeDim, typename ::Frame::BlockLogical>>>>&
        block_logical_coords) {
  db::mutate<Tags::IndicesOfFilledInterpPoints<TemporalId>,
             Tags::IndicesOfInvalidInterpPoints<TemporalId>,
             Tags::InterpolatedVars<InterpolationTargetTag, TemporalId>>(
      [&block_logical_coords, &temporal_id](
          const gsl::not_null<
              std::unordered_map<TemporalId, std::unordered_set<size_t>>*>
              indices_of_filled,
          const gsl::not_null<
              std::unordered_map<TemporalId, std::unordered_set<size_t>>*>
              indices_of_invalid_points,
          const gsl::not_null<std::unordered_map<
              TemporalId, Variables<typename InterpolationTargetTag::
                                        vars_to_interpolate_to_target>>*>
              vars_dest_all_times) {
        // Because we are sending new points to the interpolator,
        // we know that none of these points have been interpolated to,
        // so clear the list.
        indices_of_filled->erase(temporal_id);

        // Set the indices of invalid points.
        indices_of_invalid_points->erase(temporal_id);
        for (size_t i = 0; i < block_logical_coords.size(); ++i) {
          if (not block_logical_coords[i].has_value()) {
            (*indices_of_invalid_points)[temporal_id].insert(i);
          }
        }

        // At this point we don't know if vars_dest exists in the map;
        // if it doesn't then we want to default construct it.
        auto& vars_dest = (*vars_dest_all_times)[temporal_id];

        // We will be filling vars_dest with interpolated data.
        // Here we make sure it is allocated to the correct size.
        if (vars_dest.number_of_grid_points() != block_logical_coords.size()) {
          vars_dest = Variables<
              typename InterpolationTargetTag::vars_to_interpolate_to_target>(
              block_logical_coords.size());
        }
      },
      box);
}

CREATE_HAS_TYPE_ALIAS(compute_vars_to_interpolate)
CREATE_HAS_TYPE_ALIAS_V(compute_vars_to_interpolate)

namespace detail {
template <typename Tag, typename Frame>
using any_index_in_frame_impl =
    TensorMetafunctions::any_index_in_frame<typename Tag::type, Frame>;
}  // namespace detail

/// Returns true if any of the tensors in TagList have any of their
/// indices in the given frame.
template <typename TagList, typename Frame>
constexpr bool any_index_in_frame_v =
    tmpl::any<TagList, tmpl::bind<detail::any_index_in_frame_impl, tmpl::_1,
                                  Frame>>::value;

/// Calls compute_vars_to_interpolate to compute
/// InterpolationTargetTag::vars_to_interpolate_to_target from the source
/// variables.  Does any frame tranformations needed.
template <typename InterpolationTargetTag, typename SourceTags,
          typename Metavariables, typename ElementId>
void compute_dest_vars_from_source_vars(
    const gsl::not_null<Variables<
        typename InterpolationTargetTag::vars_to_interpolate_to_target>*>
        dest_vars,
    const Variables<SourceTags>& source_vars,
    const Domain<Metavariables::volume_dim>& domain,
    const Mesh<Metavariables::volume_dim>& mesh, const ElementId& element_id,
    const Parallel::GlobalCache<Metavariables>& cache,
    const typename InterpolationTargetTag::temporal_id::type& temporal_id) {
  const auto& block = domain.blocks().at(element_id.block_id());
  if (block.is_time_dependent()) {
    // The functions of time are always guaranteed to be
    // up-to-date here.
    // For interpolation without an Interpolator ParallelComponent,
    // this is because the InterpWithoutInterpComponent event will be called
    // after the Action that keeps functions of time up to date.
    // For interpolation with an Interpolator ParallelCompoent,
    // this is because the functions of time are made up to date before
    // calling SendPointsToInterpolator.
    if constexpr (any_index_in_frame_v<SourceTags, Frame::Inertial> and
                  any_index_in_frame_v<typename InterpolationTargetTag::
                                           vars_to_interpolate_to_target,
                                       Frame::Grid>) {
      // Need to do frame transformations to Grid frame.
      const auto& functions_of_time = get<domain::Tags::FunctionsOfTime>(cache);
      ElementMap<3, ::Frame::Grid> map_logical_to_grid{
          element_id, block.moving_mesh_logical_to_grid_map().get_clone()};
      const auto logical_coords = logical_coordinates(mesh);
      const auto time =
          InterpolationTarget_detail::get_temporal_id_value(temporal_id);
      const auto jac_logical_to_grid =
          map_logical_to_grid.jacobian(logical_coords);
      const auto invjac_logical_to_grid =
          map_logical_to_grid.inv_jacobian(logical_coords);
      const auto [inertial_coords, invjac_grid_to_inertial,
                  jac_grid_to_inertial, inertial_mesh_velocity] =
          block.moving_mesh_grid_to_inertial_map()
              .coords_frame_velocity_jacobians(
                  map_logical_to_grid(logical_coords), time, functions_of_time);
      InterpolationTargetTag::compute_vars_to_interpolate::apply(
          dest_vars, source_vars, mesh, jac_grid_to_inertial,
          invjac_grid_to_inertial, jac_logical_to_grid, invjac_logical_to_grid,
          inertial_mesh_velocity, tnsr::I<DataVector, 3, Frame::Grid>{});
    } else if constexpr (any_index_in_frame_v<SourceTags, Frame::Inertial> and
                         any_index_in_frame_v<typename InterpolationTargetTag::
                                                  vars_to_interpolate_to_target,
                                              Frame::Distorted>) {
      // Need to do frame transformations to Distorted frame.
      const auto& functions_of_time = get<domain::Tags::FunctionsOfTime>(cache);
      ASSERT(block.has_distorted_frame(),
             "Cannot interpolate to distorted frame in a block that does not "
             "have a distorted frame");
      const domain::CoordinateMaps::Composition
          element_logical_to_distorted_map{
              domain::element_to_block_logical_map(element_id),
              block.moving_mesh_logical_to_grid_map().get_clone(),
              block.moving_mesh_grid_to_distorted_map().get_clone()};
      const domain::CoordinateMaps::Composition element_logical_to_grid_map{
          domain::element_to_block_logical_map(element_id),
          block.moving_mesh_logical_to_grid_map().get_clone()};
      const auto logical_coords = logical_coordinates(mesh);
      const auto time =
          InterpolationTarget_detail::get_temporal_id_value(temporal_id);
      const auto [inertial_coords, invjac_distorted_to_inertial,
                  jac_distorted_to_inertial,
                  distorted_to_inertial_mesh_velocity] =
          block.moving_mesh_distorted_to_inertial_map()
              .coords_frame_velocity_jacobians(
                  element_logical_to_distorted_map(logical_coords, time,
                                                   functions_of_time),
                  time, functions_of_time);
      const auto grid_to_distorted_mesh_velocity =
          get<3>(block.moving_mesh_grid_to_distorted_map()
                     .coords_frame_velocity_jacobians(
                         element_logical_to_grid_map(logical_coords, time,
                                                     functions_of_time),
                         time, functions_of_time));
      InterpolationTargetTag::compute_vars_to_interpolate::apply(
          dest_vars, source_vars, mesh, jac_distorted_to_inertial,
          invjac_distorted_to_inertial,
          element_logical_to_distorted_map.jacobian(logical_coords, time,
                                                    functions_of_time),
          element_logical_to_distorted_map.inv_jacobian(logical_coords, time,
                                                        functions_of_time),
          distorted_to_inertial_mesh_velocity, grid_to_distorted_mesh_velocity);
    } else {
      // No frame transformations needed.
      InterpolationTargetTag::compute_vars_to_interpolate::apply(
          dest_vars, source_vars, mesh);
    }
  } else {
    // No frame transformations needed, since the maps are time-independent
    // and therefore all the frames are the same.
    //
    // Sometimes dest_vars and source_vars have different Frame tags
    // in the time-independent case, even though the frames are really
    // the same.  The source vars should all be in the Inertial frame,
    // so we create a new non-owning Variables called
    // dest_vars_in_inertial_frame that points to dest_vars but is
    // tagged as the inertial frame, and we pass
    // dest_vars_in_inertial_frame to
    // compute_vars_to_interpolate::apply.
    using dest_vars_tags =
        typename InterpolationTargetTag::vars_to_interpolate_to_target;
    using dest_tags_in_inertial_frame =
        TensorMetafunctions::replace_frame_in_taglist<dest_vars_tags,
                                                      ::Frame::Inertial>;
    Variables<dest_tags_in_inertial_frame> dest_vars_in_inertial_frame(
        dest_vars->data(), dest_vars->size());
    InterpolationTargetTag::compute_vars_to_interpolate::apply(
        make_not_null(&dest_vars_in_inertial_frame), source_vars, mesh);
  }
}

}  // namespace InterpolationTarget_detail
}  // namespace intrp
