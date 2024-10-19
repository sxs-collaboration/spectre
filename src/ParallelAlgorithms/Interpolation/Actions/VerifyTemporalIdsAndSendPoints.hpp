// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <deque>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Domain.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Parallel/ArrayComponentId.hpp"
#include "Parallel/Callback.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/SendPointsToInterpolator.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"

namespace intrp::Actions {
/// \ingroup ActionsGroup
/// \brief Sends points to an Interpolator for verified temporal_ids.
///
/// VerifyTemporalIdsAndSendPoints is invoked on an InterpolationTarget.
///
/// In more detail, does the following:
/// - If any map is time-dependent:
///   - If there is a CurrentTemporalId or no PendingTemporalIds, does nothing
///   - If any map is time-dependent:
///     Moves first verified PendingTemporalId to CurrentTemporalId, where
///     verified means that the FunctionsOfTime in the GlobalCache
///     are up-to-date for that TemporalId.  If no PendingTemporalIds are
///     moved, then VerifyTemporalIdsAndSendPoints sets itself as a
///     callback in the GlobalCache so that it is called again when the
///     FunctionsOfTime are mutated.
///   - If maps are time-independent:
///     Verified means just the first id in PendingTemporalIds
///   - If the temporal id type is a LinkedMessageId, determines if this
///     verified id is in order. If it isn't, then this just returns and does
///     nothing
///   - Calls intrp::Actions::SendPointsToInterpolator directly for this
///     verified TemporalId. (when interpolation is complete,
///      intrp::Actions::InterpolationTargetReceiveVars will begin interpolation
///     on the next verified Id)
///
/// Uses:
/// - DataBox:
///   - `intrp::Tags::PendingTemporalIds`
///   - `intrp::Tags::CompletedTemporalIds`
///   - `intrp::Tags::CurrentTemporalId`
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - `intrp::Tags::PendingTemporalIds`
///   - `intrp::Tags::CurrentTemporalId`
///
/// For requirements on InterpolationTargetTag, see InterpolationTarget
template <typename InterpolationTargetTag>
struct VerifyTemporalIdsAndSendPoints {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index) {
    static_assert(
        InterpolationTargetTag::compute_target_points::is_sequential::value,
        "Actions::VerifyTemporalIdsAndSendPoints can be used only with "
        "sequential targets.");

    using TemporalId = typename InterpolationTargetTag::temporal_id::type;

    std::stringstream ss{};
    const ::Verbosity& verbosity = Parallel::get<intrp::Tags::Verbosity>(cache);
    const bool verbose_print = verbosity >= ::Verbosity::Verbose;
    if (verbose_print) {
      ss << InterpolationTarget_detail::target_output_prefix<
                VerifyTemporalIdsAndSendPoints<InterpolationTargetTag>,
                InterpolationTargetTag>()
         << ", ";
    }

    const auto& pending_temporal_ids =
        db::get<Tags::PendingTemporalIds<TemporalId>>(box);
    // Nothing to do if there's an interpolation in progress or there are no
    // pending temporal_ids.
    if (db::get<Tags::CurrentTemporalId<TemporalId>>(box).has_value() or
        pending_temporal_ids.empty()) {
      if (verbose_print) {
        if (db::get<Tags::CurrentTemporalId<TemporalId>>(box).has_value()) {
          ss << "Interpolation already in progess at id "
             << db::get<Tags::CurrentTemporalId<TemporalId>>(box).value();
        } else {
          ss << "No pending temporal ids to send points at.";
        }
        Parallel::printf("%s\n", ss.str());
      }

      return;
    }

    const auto& domain =
        get<domain::Tags::Domain<Metavariables::volume_dim>>(cache);

    // Account for time dependent maps. If the points are in the grid frame or
    // if we don't have FunctionsOfTime in the cache, then we don't have to
    // check the FunctionsOfTime.
    if constexpr (not std::is_same_v<typename InterpolationTargetTag::
                                         compute_target_points::frame,
                                     ::Frame::Grid> and
                  Parallel::is_in_global_cache<Metavariables,
                                               domain::Tags::FunctionsOfTime>) {
      if (domain.is_time_dependent()) {
        const auto& next_pending_id = pending_temporal_ids.front();

        auto& this_proxy =
            Parallel::get_parallel_component<ParallelComponent>(cache);
        double min_expiration_time = std::numeric_limits<double>::max();
        const Parallel::ArrayComponentId array_component_id =
            Parallel::make_array_component_id<ParallelComponent>(array_index);
        const bool next_pending_id_is_ready = [&cache, &array_component_id,
                                               &this_proxy, &next_pending_id,
                                               &min_expiration_time]() {
          return ::Parallel::mutable_cache_item_is_ready<
              domain::Tags::FunctionsOfTime>(
              cache, array_component_id,
              [&this_proxy, &next_pending_id, &min_expiration_time](
                  const std::unordered_map<
                      std::string,
                      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
                      functions_of_time)
                  -> std::unique_ptr<Parallel::Callback> {
                min_expiration_time =
                    std::min_element(functions_of_time.begin(),
                                     functions_of_time.end(),
                                     [](const auto& a, const auto& b) {
                                       return a.second->time_bounds()[1] <
                                              b.second->time_bounds()[1];
                                     })
                        ->second->time_bounds()[1];

                // Success: the next pending_temporal_id is ok.
                // Failure: the next pending_temporal_id is not ok.
                return InterpolationTarget_detail::get_temporal_id_value(
                           next_pending_id) <= min_expiration_time
                           ? std::unique_ptr<Parallel::Callback>{}
                           : std::unique_ptr<Parallel::Callback>(
                                 new Parallel::SimpleActionCallback<
                                     VerifyTemporalIdsAndSendPoints<
                                         InterpolationTargetTag>,
                                     decltype(this_proxy)>(this_proxy));
              });
        }();

        if (not next_pending_id_is_ready) {
          if (verbose_print) {
            ss << "The next pending temporal id " << next_pending_id
               << " is not ready.";
            Parallel::printf("%s\n", ss.str());
          }

          // A callback has been set so that VerifyTemporalIdsAndSendPoints will
          // be called by the GlobalCache when domain::Tags::FunctionsOfTime is
          // updated.  So we can exit now.
          return;
        }
      }  // if (domain.is_time_dependent())
    } else if constexpr (not std::is_same_v<typename InterpolationTargetTag::
                                                compute_target_points::frame,
                                            ::Frame::Grid> and
                         not Parallel::is_in_global_cache<
                             Metavariables, domain::Tags::FunctionsOfTime>) {
      if (domain.is_time_dependent()) {
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

    // Move the most recent PendingTemporalId to CurrentTemporalId. If this is a
    // LinkedMessageId, make sure this is the next id before sending points
    db::mutate_apply<tmpl::list<Tags::CurrentTemporalId<TemporalId>,
                                Tags::PendingTemporalIds<TemporalId>>,
                     tmpl::list<Tags::CompletedTemporalIds<TemporalId>>>(
        [](const gsl::not_null<std::optional<TemporalId>*> current_id,
           const gsl::not_null<std::deque<TemporalId>*> pending_ids,
           const std::deque<TemporalId>& completed_ids) {
          auto& next_pending_id = pending_ids->front();
          bool use_next_pending_id = true;

          if constexpr (std::is_same_v<LinkedMessageId<double>, TemporalId>) {
            // If completed ids is empty (and at this point so is temporal ids)
            // then we must check if this is the first id.
            use_next_pending_id = completed_ids.empty()
                                      ? not next_pending_id.previous.has_value()
                                      : next_pending_id.previous.value() ==
                                            completed_ids.back().id;
          }

          if (use_next_pending_id) {
            *current_id = next_pending_id;
            pending_ids->pop_front();
          }
        },
        make_not_null(&box));

    // Send points to interpolator if the next id is ready
    if (const auto& current_id =
            db::get<Tags::CurrentTemporalId<TemporalId>>(box);
        current_id.has_value()) {
      if (verbose_print) {
        ss << "Going to send points to interpolator at temporal id "
           << current_id.value();
      }
      Actions::SendPointsToInterpolator<InterpolationTargetTag>::template apply<
          ParallelComponent>(box, cache, array_index, current_id.value());
    } else if (verbose_print) {
      ss << "No temporal ids to send points at.";
    }

    if (verbose_print) {
      Parallel::printf("%s\n", ss.str());
    }
  }
};
}  // namespace intrp::Actions
