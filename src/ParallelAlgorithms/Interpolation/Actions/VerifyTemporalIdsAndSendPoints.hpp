// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/SendPointsToInterpolator.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace intrp::Actions {

template <typename InterpolationTargetTag>
struct VerifyTemporalIdsAndSendPoints;

namespace detail {
template <typename InterpolationTargetTag, typename ParallelComponent,
          typename DbTags, typename Metavariables>
void verify_temporal_ids_and_send_points_time_independent(
    const gsl::not_null<db::DataBox<DbTags>*> box,
    Parallel::GlobalCache<Metavariables>& cache) {
  using TemporalId = typename InterpolationTargetTag::temporal_id::type;

  // Move all PendingTemporalIds to TemporalIds, provided
  // that they are not already there, and fill new_temporal_ids
  // with the temporal_ids that were so moved.
  std::vector<TemporalId> new_temporal_ids{};
  db::mutate_apply<tmpl::list<Tags::TemporalIds<TemporalId>,
                              Tags::PendingTemporalIds<TemporalId>>,
                   tmpl::list<Tags::CompletedTemporalIds<TemporalId>>>(
      [&new_temporal_ids](
          const gsl::not_null<std::deque<TemporalId>*> ids,
          const gsl::not_null<std::deque<TemporalId>*> pending_ids,
          const std::deque<TemporalId>& completed_ids) {
        for (auto& id : *pending_ids) {
          if (std::find(completed_ids.begin(), completed_ids.end(), id) ==
                  completed_ids.end() and
              std::find(ids->begin(), ids->end(), id) == ids->end()) {
            ids->push_back(id);
            new_temporal_ids.push_back(id);
          }
        }
        pending_ids->clear();
      },
      box);
  if (InterpolationTargetTag::compute_target_points::is_sequential::value) {
    // Sequential: start interpolation only for the first new_temporal_id.
    if (not new_temporal_ids.empty()) {
      auto& my_proxy =
          Parallel::get_parallel_component<ParallelComponent>(cache);
      Parallel::simple_action<
          Actions::SendPointsToInterpolator<InterpolationTargetTag>>(
          my_proxy, new_temporal_ids.front());
    }
  } else {
    // Non-sequential: start interpolation for all new_temporal_ids.
    auto& my_proxy = Parallel::get_parallel_component<ParallelComponent>(cache);
    for (const auto& id : new_temporal_ids) {
      Parallel::simple_action<
          Actions::SendPointsToInterpolator<InterpolationTargetTag>>(my_proxy,
                                                                     id);
    }
  }
}

template <typename InterpolationTargetTag, typename ParallelComponent,
          typename DbTags, typename Metavariables>
void verify_temporal_ids_and_send_points_time_dependent(
    const gsl::not_null<db::DataBox<DbTags>*> box,
    Parallel::GlobalCache<Metavariables>& cache) {
  using TemporalId = typename InterpolationTargetTag::temporal_id::type;

  const auto& pending_temporal_ids =
      db::get<Tags::PendingTemporalIds<TemporalId>>(*box);
  if (pending_temporal_ids.empty()) {
    return; // Nothing to do if there are no pending temporal_ids.
  }

  auto& this_proxy = Parallel::get_parallel_component<ParallelComponent>(cache);
  double min_expiration_time = std::numeric_limits<double>::max();
  const bool at_least_one_pending_temporal_id_is_ready =
      ::Parallel::mutable_cache_item_is_ready<domain::Tags::FunctionsOfTime>(
          cache,
          [&this_proxy, &pending_temporal_ids, &min_expiration_time](
              const std::unordered_map<
                  std::string,
                  std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
                  functions_of_time) -> std::unique_ptr<Parallel::Callback> {
            min_expiration_time =
                std::min_element(functions_of_time.begin(),
                                 functions_of_time.end(),
                                 [](const auto& a, const auto& b) {
                                   return a.second->time_bounds()[1] <
                                          b.second->time_bounds()[1];
                                 })
                    ->second->time_bounds()[1];
            for (const auto& pending_id : pending_temporal_ids) {
              if (InterpolationTarget_detail::
                      evaluate_temporal_id_for_expiration(pending_id) <=
                  min_expiration_time) {
                // Success: at least one pending_temporal_id is ok.
                return std::unique_ptr<Parallel::Callback>{};
              }
            }
            // Failure: none of the pending_temporal_ids are ok.
            return std::unique_ptr<Parallel::Callback>(
                new Parallel::SimpleActionCallback<
                    VerifyTemporalIdsAndSendPoints<InterpolationTargetTag>,
                    decltype(this_proxy)>(this_proxy));
          });

  if (not at_least_one_pending_temporal_id_is_ready) {
    // A callback has been set so that VerifyTemporalIdsAndSendPoints will
    // be called by MutableGlobalCache when domain::Tags::FunctionsOfTime
    // is updated.  So we can exit now.
    return;
  }

  // Move up-to-date PendingTemporalIds to TemporalIds, provided
  // that they are not already there, and fill new_temporal_ids
  // with the temporal_ids that were so moved.
  std::vector<TemporalId> new_temporal_ids{};
  db::mutate_apply<tmpl::list<Tags::TemporalIds<TemporalId>,
                              Tags::PendingTemporalIds<TemporalId>>,
                   tmpl::list<Tags::CompletedTemporalIds<TemporalId>>>(
      [&min_expiration_time, &new_temporal_ids](
          const gsl::not_null<std::deque<TemporalId>*> ids,
          const gsl::not_null<std::deque<TemporalId>*> pending_ids,
          const std::deque<TemporalId>& completed_ids) {
        for (auto it = pending_ids->begin(); it != pending_ids->end();) {
          if (InterpolationTarget_detail::evaluate_temporal_id_for_expiration(
                  *it) <= min_expiration_time and
              std::find(completed_ids.begin(), completed_ids.end(), *it) ==
                  completed_ids.end() and
              std::find(ids->begin(), ids->end(), *it) == ids->end()) {
            ids->push_back(*it);
            new_temporal_ids.push_back(*it);
            it = pending_ids->erase(it);
          } else {
            ++it;
          }
        }
      },
      box);

  if (InterpolationTargetTag::compute_target_points::is_sequential::value) {
    // Sequential: start interpolation only for the first new_temporal_id.
    if (not new_temporal_ids.empty()) {
      auto& my_proxy =
          Parallel::get_parallel_component<ParallelComponent>(cache);
      Parallel::simple_action<
          Actions::SendPointsToInterpolator<InterpolationTargetTag>>(
          my_proxy, new_temporal_ids.front());
    }
  } else {
    // Non-sequential: start interpolation for all new_temporal_ids.
    auto& my_proxy = Parallel::get_parallel_component<ParallelComponent>(cache);
    for (const auto& id : new_temporal_ids) {
      Parallel::simple_action<
          Actions::SendPointsToInterpolator<InterpolationTargetTag>>(my_proxy,
                                                                     id);
    }
    // If there are still pending temporal_ids, call
    // VerifyTemporalIdsAndSendPoints again, so that those pending
    // temporal_ids can be waited for.
    if (not db::get<Tags::PendingTemporalIds<TemporalId>>(*box).empty()) {
      Parallel::simple_action<
          VerifyTemporalIdsAndSendPoints<InterpolationTargetTag>>(my_proxy);
    }
  }
}
}  // namespace detail

/// \ingroup ActionsGroup
/// \brief Sends points to an Interpolator for verified temporal_ids.
///
/// VerifyTemporalIdsAndSendPoints is invoked on an InterpolationTarget.
///
/// In more detail, does the following:
/// - If any map is time-dependent:
///   - Moves verified PendingTemporalIds to TemporalIds, where
///     verified means that the FunctionsOfTime in the GlobalCache
///     are up-to-date for that TemporalId.  If no PendingTemporalIds are
///     moved, then VerifyTemporalIdsAndSendPoints sets itself as a
///     callback in the GlobalCache so that it is called again when the
///     FunctionsOfTime are mutated.
///   - If the InterpolationTarget is sequential, invokes
///     intrp::Actions::SendPointsToInterpolator for the first TemporalId.
///     (when interpolation is complete,
///      intrp::Actions::InterpolationTargetReceiveVars will begin interpolation
///     on the next TemporalId)
///   - If the InterpolationTarget is not sequential, invokes
///     intrp::Actions::SendPointsToInterpolator for all valid TemporalIds,
///     and then if PendingTemporalIds is non-empty it invokes itself.
///
/// - If all maps are time-independent:
///   - Moves all PendingTemporalIds to TemporalIds
///   - If the InterpolationTarget is sequential, invokes
///     intrp::Actions::SendPointsToInterpolator for the first TemporalId.
///     (when interpolation is complete,
///      intrp::Actions::InterpolationTargetReceiveVars will begin interpolation
///     on the next TemporalId)
///   - If the InterpolationTarget is not sequential, invokes
///     intrp::Actions::SendPointsToInterpolator for all TemporalIds.
///
/// Uses:
/// - DataBox:
///   - `intrp::Tags::PendingTeporalIds`
///   - `intrp::Tags::TeporalIds`
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - `intrp::Tags::PendingTeporalIds`
///   - `intrp::Tags::TeporalIds`
///
/// For requirements on InterpolationTargetTag, see InterpolationTarget
template <typename InterpolationTargetTag>
struct VerifyTemporalIdsAndSendPoints {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex,
            Requires<tmpl::list_contains_v<
                DbTags, Tags::TemporalIds<typename InterpolationTargetTag::
                                              temporal_id::type>>> = nullptr>
  static void apply(db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/) {
    if constexpr (std::is_same_v<typename InterpolationTargetTag::
                                     compute_target_points::frame,
                                 ::Frame::Grid>) {
      detail::verify_temporal_ids_and_send_points_time_independent<
          InterpolationTargetTag, ParallelComponent>(make_not_null(&box),
                                                     cache);
    } else {
      if (InterpolationTarget_detail::maps_are_time_dependent<
              InterpolationTargetTag>(box, tmpl::type_<Metavariables>{})) {
        if constexpr (InterpolationTarget_detail::
                          cache_contains_functions_of_time<
                              Metavariables>::value) {
          detail::verify_temporal_ids_and_send_points_time_dependent<
              InterpolationTargetTag, ParallelComponent>(make_not_null(&box),
                                                         cache);
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
      } else {
        detail::verify_temporal_ids_and_send_points_time_independent<
            InterpolationTargetTag, ParallelComponent>(make_not_null(&box),
                                                       cache);
      }
    }
  }
};
}  // namespace intrp::Actions
