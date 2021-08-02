// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/Gsl.hpp"

namespace intrp::Actions {

template <typename InterpolationTargetTag>
struct VerifyTimesAndSendPoints;

namespace detail {
template <typename InterpolationTargetTag, typename ParallelComponent,
          typename DbTags, typename Metavariables>
void verify_times_and_send_points_time_independent(
    const gsl::not_null<db::DataBox<DbTags>*> box,
    Parallel::GlobalCache<Metavariables>& cache) noexcept {
  // Move all PendingTimes to Times, provided
  // that they are not already there, and fill new_times
  // with the times that were so moved.
  std::vector<double> new_times{};
  db::mutate_apply<tmpl::list<Tags::Times, Tags::PendingTimes>,
                   tmpl::list<Tags::CompletedTimes>>(
      [&new_times](const gsl::not_null<std::deque<double>*> ids,
                   const gsl::not_null<std::deque<double>*> pending_ids,
                   const std::deque<double>& completed_ids) noexcept {
        for (auto& id : *pending_ids) {
          if (std::find(completed_ids.begin(), completed_ids.end(), id) ==
                  completed_ids.end() and
              std::find(ids->begin(), ids->end(), id) == ids->end()) {
            ids->push_back(id);
            new_times.push_back(id);
          }
        }
        pending_ids->clear();
      },
      box);
  if (InterpolationTargetTag::compute_target_points::is_sequential::value) {
    // Sequential: start interpolation only for the first new_time.
    if (not new_times.empty()) {
      auto& my_proxy =
          Parallel::get_parallel_component<ParallelComponent>(cache);
      Parallel::simple_action<
          Actions::SendPointsToInterpolator<InterpolationTargetTag>>(
          my_proxy, new_times.front());
    }
  } else {
    // Non-sequential: start interpolation for all new_times.
    auto& my_proxy = Parallel::get_parallel_component<ParallelComponent>(cache);
    for (const auto& time : new_times) {
      Parallel::simple_action<
          Actions::SendPointsToInterpolator<InterpolationTargetTag>>(my_proxy,
                                                                     time);
    }
  }
}

template <typename InterpolationTargetTag, typename ParallelComponent,
          typename DbTags, typename Metavariables>
void verify_times_and_send_points_time_dependent(
    const gsl::not_null<db::DataBox<DbTags>*> box,
    Parallel::GlobalCache<Metavariables>& cache) noexcept {
  const auto& pending_times = db::get<Tags::PendingTimes>(*box);
  if (pending_times.empty()) {
    return;  // Nothing to do if there are no pending times.
  }

  auto& this_proxy = Parallel::get_parallel_component<ParallelComponent>(cache);
  double min_expiration_time = std::numeric_limits<double>::max();
  const bool at_least_one_pending_time_is_ready =
      ::Parallel::mutable_cache_item_is_ready<domain::Tags::FunctionsOfTime>(
          cache,
          [&this_proxy, &pending_times, &min_expiration_time](
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
            for (const auto& pending_id : pending_times) {
              if (pending_id <= min_expiration_time) {
                // Success: at least one pending_time is ok.
                return std::unique_ptr<Parallel::Callback>{};
              }
            }
            // Failure: none of the pending_times are ok.
            return std::unique_ptr<Parallel::Callback>(
                new Parallel::SimpleActionCallback<
                    VerifyTimesAndSendPoints<InterpolationTargetTag>,
                    decltype(this_proxy)>(this_proxy));
          });

  if (not at_least_one_pending_time_is_ready) {
    // A callback has been set so that VerifyTimesAndSendPoints will
    // be called by MutableGlobalCache when domain::Tags::FunctionsOfTime
    // is updated.  So we can exit now.
    return;
  }

  // Move up-to-date PendingTimes to Times, provided
  // that they are not already there, and fill new_times
  // with the times that were so moved.
  std::vector<double> new_times{};
  db::mutate_apply<tmpl::list<Tags::Times, Tags::PendingTimes>,
                   tmpl::list<Tags::CompletedTimes>>(
      [&min_expiration_time, &new_times](
          const gsl::not_null<std::deque<double>*> ids,
          const gsl::not_null<std::deque<double>*> pending_ids,
          const std::deque<double>& completed_ids) noexcept {
        for (auto it = pending_ids->begin(); it != pending_ids->end();) {
          if (*it <= min_expiration_time and
              std::find(completed_ids.begin(), completed_ids.end(), *it) ==
                  completed_ids.end() and
              std::find(ids->begin(), ids->end(), *it) == ids->end()) {
            ids->push_back(*it);
            new_times.push_back(*it);
            it = pending_ids->erase(it);
          } else {
            ++it;
          }
        }
      },
      box);

  if (InterpolationTargetTag::compute_target_points::is_sequential::value) {
    // Sequential: start interpolation only for the first new_time.
    if (not new_times.empty()) {
      auto& my_proxy =
          Parallel::get_parallel_component<ParallelComponent>(cache);
      Parallel::simple_action<
          Actions::SendPointsToInterpolator<InterpolationTargetTag>>(
          my_proxy, new_times.front());
    }
  } else {
    // Non-sequential: start interpolation for all new_times.
    auto& my_proxy = Parallel::get_parallel_component<ParallelComponent>(cache);
    for (const auto& time : new_times) {
      Parallel::simple_action<
          Actions::SendPointsToInterpolator<InterpolationTargetTag>>(my_proxy,
                                                                     time);
    }
    // If there are still pending times, call
    // VerifyTimesAndSendPoints again, so that those pending
    // times can be waited for.
    if (not db::get<Tags::PendingTimes>(*box).empty()) {
      Parallel::simple_action<VerifyTimesAndSendPoints<InterpolationTargetTag>>(
          my_proxy);
    }
  }
}
}  // namespace detail

/// \ingroup ActionsGroup
/// \brief Sends points to an Interpolator for verified times.
///
/// VerifyTimesAndSendPoints is invoked on an InterpolationTarget.
///
/// In more detail, does the following:
/// - If any map is time-dependent:
///   - Moves verified PendingTimes to Times, where
///     verified means that the FunctionsOfTime in the GlobalCache
///     are up-to-date for that time.  If no PendingTimes are
///     moved, then VerifyTimesAndSendPoints sets itself as a
///     callback in the GlobalCache so that it is called again when the
///     FunctionsOfTime are mutated.
///   - If the InterpolationTarget is sequential, invokes
///     intrp::Actions::SendPointsToInterpolator for the first Time.
///     (when interpolation is complete,
///      intrp::Actions::InterpolationTargetReceiveVars will begin interpolation
///     on the next Time)
///   - If the InterpolationTarget is not sequential, invokes
///     intrp::Actions::SendPointsToInterpolator for all valid times,
///     and then if PendingTimes is non-empty it invokes itself.
///
/// - If all maps are time-independent:
///   - Moves all PendingTimes to Times
///   - If the InterpolationTarget is sequential, invokes
///     intrp::Actions::SendPointsToInterpolator for the first Time.
///     (when interpolation is complete,
///      intrp::Actions::InterpolationTargetReceiveVars will begin interpolation
///     on the next Time)
///   - If the InterpolationTarget is not sequential, invokes
///     intrp::Actions::SendPointsToInterpolator for all Times.
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
struct VerifyTimesAndSendPoints {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex,
            Requires<tmpl::list_contains_v<DbTags, Tags::Times>> = nullptr>
  static void apply(db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/) noexcept {
    if constexpr (std::is_same_v<typename InterpolationTargetTag::
                                     compute_target_points::frame,
                                 ::Frame::Grid>) {
      detail::verify_times_and_send_points_time_independent<
          InterpolationTargetTag, ParallelComponent>(make_not_null(&box),
                                                     cache);
    } else {
      if (InterpolationTarget_detail::maps_are_time_dependent<
              InterpolationTargetTag>(box, tmpl::type_<Metavariables>{})) {
        if constexpr (InterpolationTarget_detail::
                          cache_contains_functions_of_time<
                              Metavariables>::value) {
          detail::verify_times_and_send_points_time_dependent<
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
        detail::verify_times_and_send_points_time_independent<
            InterpolationTargetTag, ParallelComponent>(make_not_null(&box),
                                                       cache);
      }
    }
  }
};
}  // namespace intrp::Actions
